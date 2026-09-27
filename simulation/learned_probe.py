"""One real learned Isaac episode and evaluator-only measurements; no reference fallback."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from learning.common import canonical, digest, file_digest, integer, read_json, require
from learning.contract import Scope
from learning.paused import FrozenCameraSample
from learning.paused.capture import PausedEpisodeBudget, validate_dataset
from learning.paused.evaluation import _trial
from learning.paused.rollout import derive_trial
from learning.paused.task import TaskState, evaluate_task_states
from simulation.batch_learned import BatchLearnedSpec, load_inputs
from simulation.paused_learned import PausedLearnedRuntime

SCHEMA = "physicalai.paused-learned-attempt/v1"


class LearnedTrace:
    def __init__(self, core, hardware, request) -> None:
        self.core, self.hardware, self.request = core, hardware, request
        self.states: list[TaskState] = []
        self.heartbeat_ns: list[int] = []

    def observe(self) -> None:
        driver = self.hardware.paused_driver
        if driver is None:
            return
        require(
            isinstance(driver, PausedLearnedRuntime)
            and driver.request == self.request
            and driver.binding.epoch == self.core.epoch
            and self.hardware.controller is None
            and self.hardware.route is None,
            "Learned trace binding/epoch changed or a reference controller was constructed.",
        )
        measured = self.hardware.frozen_physics_state(self.core.epoch)
        metrics = driver.metrics()
        require(
            metrics.reference_route_calls == 0
            and metrics.applied_model_sha256
            == (self.request.model_sha256 if metrics.applied_action_count else None),
            "Actual learned model or reference-route binding changed.",
        )
        now = self.core.clock_ns()
        self.heartbeat_ns.append(now)
        require(len(self.heartbeat_ns) <= 1_000_000, "Bounded heartbeat trace exhausted.")
        if self.states and measured.physics_step == self.states[-1].physics_step:
            return
        require(
            len(self.states) <= self.core.paused_profile.max_simulation_steps
            and measured.physics_step == driver.episode.initial.physics_step + len(self.states)
            and metrics.applied_action_count == len(self.states),
            "Skipped actual task tick; partial poses cannot be padded into a complete trace.",
        )
        if not self.states:
            require(metrics.policy_predict_calls == 0, "Task trace started after model actuation.")
        state = TaskState(
            captured_at_utc=self.core.clock_utc().isoformat().replace("+00:00", "Z"),
            monotonic_ns=now,
            physics_step=measured.physics_step,
            object_position_m=measured.object_position,
            object_linear_velocity_m_s=measured.object_linear_velocity,
            tcp_position_m=self.hardware._measured_tcp(),
            joint_positions=measured.joint_positions,
            command_id=str(self.request.command_id),
            epoch=str(measured.epoch),
            policy_predict_calls=metrics.policy_predict_calls,
            applied_action_count=metrics.applied_action_count,
            reference_route_calls=metrics.reference_route_calls,
            applied_model_sha256=metrics.applied_model_sha256,
        )
        state.validate()
        self.states.append(state)

    def recording(self, *, end_ns: int) -> dict:
        episode = self.hardware.paused_driver.episode
        return {
            "task_states": [asdict(state) for state in self.states],
            "heartbeat_ns": [stamp for stamp in self.heartbeat_ns if stamp <= end_ns],
            "episode_budget": asdict(
                PausedEpisodeBudget(
                    episode.started_ns,
                    episode.wall_deadline_ns,
                    episode.initial.physics_step,
                    episode.initial.physics_step + episode.max_simulation_steps,
                )
            ),
        }


def measured_runtime_type():
    # Offline evidence verification must not import the live simulator lifecycle.
    from simulation.run_isaac import SimulatorRuntime

    class MeasuredLearnedRuntime(SimulatorRuntime):
        trace: LearnedTrace | None = None

        def _publish_policy_metrics(self) -> None:
            super()._publish_policy_metrics()
            if self.trace is not None:
                self.trace.observe()

    return MeasuredLearnedRuntime


def encode_image(image: FrozenCameraSample) -> dict:
    return {
        **{key: value for key, value in asdict(image).items() if key != "png"},
        "png_base64": base64.b64encode(image.png).decode("ascii"),
    }


def decode_image(value: dict) -> FrozenCameraSample:
    require(
        set(value) == (set(FrozenCameraSample.__dataclass_fields__) - {"png"}) | {"png_base64"},
        "Incomplete original final-camera metadata.",
    )
    require(len(value["png_base64"]) <= 1024**2, "Final 320x320 camera exceeds its byte bound.")
    image = FrozenCameraSample(
        png=base64.b64decode(value["png_base64"], validate=True),
        **{key: item for key, item in value.items() if key != "png_base64"},
    )
    metadata = image.metadata()
    require(metadata["width"] == metadata["height"] == 320, "Both real cameras must be 320x320.")
    return image


def physical_case(spec, scene, grant) -> dict:
    return {
        "episode_id": str(spec.attempt_id),
        "seed": scene.seed,
        "attempt": 0,
        "environment_id": grant.authorization.environment_id,
        "revision": grant.authorization.revision,
        "expected_destination_id": grant.authorization.task.goal_id,
        "expected_pose_m": list(scene.station(grant.authorization.task.goal_id).position),
        "initial_pose_m": list(scene.part_position),
        "tolerance_m": 0.04,
        "scene_builder_sha256": scene.scene_builder_sha256,
    }


def rescore(report: dict, spec: BatchLearnedSpec, scene, profile, grant) -> dict:
    require(
        report.get("schema") == SCHEMA
        and report.get("controller") == "learned"
        and report.get("policy_type") == "smolvla"
        and report.get("execution_timing") == "paused_simulation"
        and report.get("real_time_admission") is False
        and report.get("source_revision") == spec.source_revision
        and report.get("simulator_image_digest") == spec.platform.container_image.split("@", 1)[1]
        and report.get("model_sha256") == spec.model.manifest.sha256
        and report.get("control_profile_sha256") == profile.sha256 == spec.control_profile_sha256
        and report.get("criteria_sha256") == spec.criteria_canonical_sha256
        and report.get("frozen_plan_sha256") == spec.conditions_canonical_sha256
        and report.get("operator_grant_sha256") == spec.grant.sha256
        and report.get("command_id") == str(spec.attempt_id)
        and report.get("environment_id") == grant.authorization.environment_id
        and report.get("revision") == grant.authorization.revision
        and report.get("model_weights_loaded") is True
        and report.get("learning_quality_proven") is False,
        "Native learned report changed its actual model, task, owner or source binding.",
    )
    values = report.get("task_states")
    require(
        isinstance(values, list) and 2 <= len(values) <= profile.max_simulation_steps + 1,
        "Missing bounded actual learned task measurements.",
    )
    states = [TaskState(**value) for value in values]
    case = physical_case(spec, scene, grant)
    task = evaluate_task_states(
        states, initial=case["initial_pose_m"], goal=case["expected_pose_m"], profile=profile
    )
    first, last = states[0], states[-1]
    metrics = report.get("metrics") or {}
    require(
        metrics.get("controller") == "learned"
        and metrics.get("applied_model_sha256") == last.applied_model_sha256
        and metrics.get("applied_action_count")
        == metrics.get("simulation_steps")
        == last.applied_action_count
        and metrics.get("policy_predict_calls") == last.policy_predict_calls
        and metrics.get("reference_route_calls") == 0,
        "Actual actuator model metrics differ from the measured learned task trace.",
    )
    from simulation.paused_gripper_servo import validate_servo_receipt

    validate_servo_receipt(
        report.get("gripper_servo"),
        command_id=str(spec.attempt_id),
        environment_id=grant.authorization.environment_id,
        revision=grant.authorization.revision,
    )
    require(
        report["gripper_servo"]["frozen_state"]["before"]["physics_step"] == first.physics_step,
        "Gripper calibration differs from the original learned task state.",
    )
    require(
        first.command_id == last.command_id == report["command_id"]
        and first.applied_action_count
        == first.policy_predict_calls
        == first.reference_route_calls
        == 0
        and first.applied_model_sha256 is None
        and all(
            state.reference_route_calls == 0
            and state.applied_model_sha256 == spec.model.manifest.sha256
            and state.policy_predict_calls == (index + profile.hold_steps - 1) // profile.hold_steps
            for index, state in enumerate(states[1:], 1)
        ),
        "The measured task trace contains foreign model/reference controls.",
    )
    trial = report["trial"]
    require(
        trial["episode_id"] == case["episode_id"]
        and trial["policy"] == spec.role
        and trial["model_sha256"] == spec.model.manifest.sha256
        and trial["task_evidence"] == task
        and tuple(trial["final_pose_m"]) == tuple(last.object_position_m)
        and trial["applied_action_count"] == last.applied_action_count
        and trial["policy_predict_calls"] == last.policy_predict_calls
        and trial["reference_route_calls"] == 0,
        "Declared learned trial differs from native per-tick rescore.",
    )
    passed, _, _, resource_violations = _trial(trial, case, learned=True, profile=profile)
    capture = report.get("capture") or {}
    receipt = capture.get("receipt") or {}
    require(
        capture.get("status") == "ready"
        and receipt.get("status") == "uploaded"
        and receipt.get("episode_id") == report["command_id"]
        and receipt.get("frame_count") * profile.hold_steps == last.applied_action_count
        and last.applied_action_count <= grant.authorization.max_simulation_steps,
        "Learned trial lacks a complete native capture receipt.",
    )
    from learning.common import utc

    require(
        set(report["final_images"]) == {"overview", "inspection"},
        "Both original final-camera publications are required.",
    )
    for name, value in report["final_images"].items():
        image = decode_image(value)
        require(
            image.physics_step == last.physics_step
            and digest(image.png) == trial["final_images"][name]
            and 0
            <= image.monotonic_ns - last.monotonic_ns
            <= profile.max_observation_wall_ms * 1_000_000
            and utc(last.captured_at_utc)
            <= utc(image.captured_at_utc)
            < grant.authorization.wall_expires_at,
            "The actual final camera evidence differs from the native trial.",
        )
    budget = PausedEpisodeBudget(**report["episode_budget"])
    budget.validate(profile)
    end = max(value["monotonic_ns"] for value in report["final_images"].values())
    require(
        budget.started_ns
        <= first.monotonic_ns
        <= last.monotonic_ns
        <= end
        <= budget.wall_deadline_ns
        and first.physics_step == budget.initial_physics_step
        and last.physics_step <= budget.simulation_step_deadline
        and trial["wall_duration_ms"] == (end - budget.started_ns) / 1e6,
        "Learned trial changed the original episode budget or measured duration.",
    )
    heartbeats = report.get("heartbeat_ns")
    require(
        isinstance(heartbeats, list) and 1 <= len(heartbeats) <= 1_000_000,
        "Missing bounded original main-thread heartbeat samples.",
    )
    previous, gaps = budget.started_ns, []
    for stamp in heartbeats:
        integer(stamp, "original heartbeat", previous, end)
        if stamp != previous:
            gaps.append((stamp - previous) / 1e6)
        previous = stamp
    gaps.append((end - previous) / 1e6)
    require(
        gaps == trial["heartbeat_gap_ms"], "Declared heartbeat gaps differ from actual samples."
    )

    require(
        grant.issued_at
        <= utc(first.captured_at_utc)
        <= utc(last.captured_at_utc)
        < grant.authorization.wall_expires_at,
        "Measured learned episode escaped the original operator grant.",
    )
    accepted = passed and report.get("physical_status") == "succeeded" and resource_violations == 0
    return {
        "accepted": accepted,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "model_sha256": spec.model.manifest.sha256,
        "applied_model_sha256": last.applied_model_sha256,
        "policy_predict_calls": last.policy_predict_calls,
        "reference_route_calls": 0,
        "applied_action_count": last.applied_action_count,
        "manifest_sha256": receipt["manifest_sha256"],
        "task_evidence": task,
        "learning_quality_proven": False,
    }


def run(
    spec: BatchLearnedSpec, *, paths, model_root: Path, socket_path: Path, output: Path
) -> dict:
    from simulation.capture_status import CaptureStatusStore
    from simulation.core import SimulationCore
    from simulation.extensions import SceneRegistry
    from simulation.paused_contracts import SimulationEpisodeCommand
    from simulation.paused_deployment import InstalledPausedPolicyProvider
    from simulation.probe_control import (
        _close_application,
        _persist_receipt,
        initialize_probe_assets,
    )
    from simulation.run_isaac import create_simulation_app

    environment, scene, profile, grant = load_inputs(spec, paths)
    permit = grant.authorization
    require(not output.exists(), "Never overwrite a learned physical attempt.")
    catalogue = {
        "schema": "physicalai.paused-policy-catalog/v1",
        "policies": [
            {
                "grant": grant.model_dump(mode="json", by_alias=True),
                "model_root": str(model_root),
                "socket_path": str(socket_path),
                "expected_peer_uid": os.geteuid(),
            }
        ],
    }
    catalogue_path = output.parent / "catalogue.json"
    catalogue_path.write_bytes(canonical(catalogue))
    provider = InstalledPausedPolicyProvider.load(
        catalogue_path, file_digest(catalogue_path), profile
    )
    application = hardware = runtime = core = trace = None
    report = None
    phase = "asset_preparation"
    try:
        initialize_probe_assets()
        phase = "application_initialization"
        application = create_simulation_app(sensor_only=True)
        from simulation.isaac_adapter import IsaacWorkcell

        hardware = IsaacWorkcell()
        store = CaptureStatusStore(Path("/data/demonstrations/.capture-status"))
        core = SimulationCore(
            SceneRegistry(),
            paused_profile=profile,
            paused_authorizer=provider,
            paused_policy_provider=provider,
            tenant_id=str(grant.tenant_id),
            capture_status_reader=store.get,
        )
        runtime = measured_runtime_type()(
            core,
            hardware,
            heartbeat=output.with_suffix(".heartbeat"),
            capture_store=store,
        )
        phase = "scene_preparation"
        core.activate(permit.owner, environment)
        preparation_deadline = min(
            time.monotonic() + 120,
            time.monotonic() + (grant.expires_at - core.clock_utc()).total_seconds(),
        )
        while not core.ready:
            require(
                application.is_running() and time.monotonic() < preparation_deadline,
                "The original learned preparation budget expired.",
            )
            runtime.tick()
            require(core.error is None, core.error or "Learned scene preparation failed.")
        publication = hardware.paused_publications.publication
        require(publication is not None, "No actual original frozen publication.")
        observation = core.observe(
            permit.owner, environment.environment_id, environment.revision, "inspection"
        )
        request = SimulationEpisodeCommand(
            schema="physicalai.simulation-episode-command/v1",
            execution_timing="paused_simulation",
            real_time_admission=False,
            profile_id=profile.profile_id,
            command_id=spec.attempt_id,
            environment_id=environment.environment_id,
            revision=environment.revision,
            epoch=core.epoch,
            state_revision=core.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=permit.task.goal_id,
            task=permit.task,
            wall_expires_at=permit.wall_expires_at,
            max_simulation_steps=permit.max_simulation_steps,
            controller="learned",
            policy_type="smolvla",
            model_sha256=spec.model.manifest.sha256,
            authorization_kind="evaluation_grant",
            authorization_id=permit.authorization_id,
        )
        trace = LearnedTrace(core, hardware, request)
        runtime.trace = trace
        phase = "learned_episode"
        core.dispatch_simulation_episode(permit.owner, request)
        while core.command(permit.owner, request.command_id).status in {
            "queued",
            "running",
            "cancelling",
        }:
            require(application.is_running(), "Isaac application closed during learned execution.")
            runtime.tick()
            time.sleep(0.001)
        result = core.command(permit.owner, request.command_id)
        phase = "final_frozen_cameras"
        final = hardware._paused_publication(
            core, deadline_ns=core.monotonic_deadlines[(permit.owner, request.command_id)]
        )
        final_images = dict(final.images)
        phase = "capture_publication"
        capture = core.captures.get((permit.owner, request.command_id))
        while capture is not None and capture.status not in {"ready", "invalid"}:
            if core.clock_utc() >= request.wall_expires_at:
                runtime.capture_worker.invalidate(
                    "Original learned grant expired before publication."
                )
            runtime.tick()
            capture = core.capture(permit.owner, request.command_id)
            time.sleep(0.001)
        report = {
            "schema": SCHEMA,
            "execution_timing": "paused_simulation",
            "real_time_admission": False,
            "controller": "learned",
            "policy_type": "smolvla",
            "command_id": str(request.command_id),
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "source_revision": spec.source_revision,
            "simulator_image_digest": spec.platform.container_image.split("@", 1)[1],
            "model_sha256": spec.model.manifest.sha256,
            "control_profile_sha256": profile.sha256,
            "criteria_sha256": spec.criteria_canonical_sha256,
            "frozen_plan_sha256": spec.conditions_canonical_sha256,
            "operator_grant_sha256": spec.grant.sha256,
            "physical_status": result.status,
            "model_weights_loaded": bool(trace.states and trace.states[-1].policy_predict_calls),
            "metrics": result.simulation_runtime.model_dump(mode="json"),
            "learning_quality_proven": False,
            "initial_publication_record": publication.private_evidence(),
            **trace.recording(end_ns=max(image.monotonic_ns for image in final_images.values())),
            "final_images": {name: encode_image(image) for name, image in final_images.items()},
            "capture": capture.model_dump(mode="json") if capture is not None else None,
            "gripper_servo": hardware.paused_gripper_servo_evidence(),
            "error": result.error.model_dump() if result.error else None,
        }
        if capture is not None and capture.status == "ready":
            raw = validate_dataset(
                Path("/data/demonstrations") / spec.owner_id / str(request.command_id),
                expected_scope=Scope(str(grant.tenant_id), permit.owner),
                require_live=True,
                expected_manifest_sha256=capture.receipt.manifest_sha256,
            )
            require(
                all(
                    image["width"] == image["height"] == 320
                    for episode in raw.episodes
                    for frame in episode.frames
                    for image in frame["images"].values()
                ),
                "Both actual learned cameras must remain 320x320.",
            )
            report["trial"] = derive_trial(
                raw,
                trace.states,
                case=physical_case(spec, scene, grant),
                role=spec.role,
                model_sha256=spec.model.manifest.sha256,
                profile=profile,
                final_images=final_images,
                heartbeat_ns=tuple(report["heartbeat_ns"]),
                destination_id=permit.task.goal_id,
                failure_reason=str(result.error.message)[:512] if result.error else None,
            )
            report["native_acceptance"] = rescore(report, spec, scene, profile, grant)
        return report
    finally:
        error = sys.exc_info()[1]
        try:
            if runtime is not None:
                runtime.close()
            if report is None:
                report = {
                    "schema": "physicalai.paused-learned-failure/v1",
                    "phase": phase,
                    "physical_status": "not_completed",
                    "failure_type": type(error).__name__,
                    "failure": str(error),
                    "model_sha256": spec.model.manifest.sha256,
                    "learning_quality_proven": False,
                    "task_states": [asdict(state) for state in trace.states] if trace else [],
                }
            if hardware is not None:
                report["gripper_servo"] = hardware.paused_gripper_servo_evidence()
                report["camera_publication_evidence"] = hardware.paused_camera_evidence()
            _persist_receipt(output, report)
        finally:
            if application is not None:
                _close_application(application, error or sys.exc_info()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = BatchLearnedSpec.model_validate(read_json(args.spec, max_bytes=1024**2))
    report = run(
        spec,
        paths={
            name: args.inputs / (name + ".json")
            for name in ("environment", "grant", "criteria", "conditions")
        },
        model_root=args.model_root,
        socket_path=args.socket_path,
        output=args.output,
    )
    if (report.get("native_acceptance") or {}).get("accepted") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
