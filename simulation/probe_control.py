"""Operator-only, isolated GPU mechanics probe; never a model or learning-quality gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from uuid import UUID, uuid4

from azure.identity import ManagedIdentityCredential

from apps.api.models import EnvironmentRecord, Execution, MotionCommand, utcnow
from learning.common import canonical, digest, read_json, require
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.inference import GuardedPolicyAdapter
from simulation.assets import configure_asset_environment
from simulation.core import SimulationCore
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry
from simulation.reference_teaching import ReferenceTeacher
from simulation.run_isaac import SimulatorRuntime, create_simulation_app
from simulation.runtime_configuration import servo_profile_sha256
from simulation.runtime_contracts import (
    CaptureStatus,
    PolicyCommand,
    TeachingInput,
    TeachingLease,
    TeachingStart,
)

FIXTURE_SHA = digest(b"physicalai-isolated-actuation-fixture/v1")


def initialize_probe_assets() -> None:
    with ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"]) as credential:
        configure_asset_environment(credential)


def _persist_receipt(path: Path, report: dict) -> None:
    temporary = path.with_suffix(f".{uuid4()}.new")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(report) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "PHYSICALAI_OPERATOR_RECEIPT "
        + json.dumps(
            {
                "output": str(path),
                "schema": report["schema"],
                "probe_completed": report.get("probe_completed", False),
                "physical_status": report["physical_status"],
                "failure_type": report.get("failure_type"),
            }
        ),
        flush=True,
    )


def _close_application(application, pending_error: BaseException | None) -> None:
    try:
        application.close()
    except SystemExit as exc:
        # Kit may exit Python during teardown; it must not replace the actual run outcome.
        if pending_error is None and exc.code not in (None, 0):
            raise


def outcome_fields(mode: str, result: Execution | None, capture: CaptureStatus | None) -> dict:
    return {
        "model_weights_loaded": False,
        "learning_quality_proven": False,
        "production_ready": False,
        "physical_task_success": (
            mode == "collect-reference" and result is not None and result.status == "succeeded"
        ),
        "physical_status": result.status if result is not None else "not_started",
        "error": result.error.model_dump(mode="json")
        if result is not None and result.error
        else None,
        "capture": capture.model_dump(mode="json") if capture is not None else None,
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument(
        "--mode", choices=("teaching", "policy-fixture", "collect-reference"), required=True
    )
    parser.add_argument("--intervals", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-isolated-simulator", action="store_true", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--instruction")
    parser.add_argument("--goal-id")
    args = parser.parse_args(argv)
    if not 100 <= args.intervals <= 150:
        parser.error("The probe is bounded to 100..150 complete control intervals.")
    if args.mode == "collect-reference" and not all((args.task_id, args.instruction, args.goal_id)):
        parser.error("Reference collection requires the approved task ID, instruction and goal.")
    return args


class _FixturePolicy:
    fps, physics_hz, chunk_size, n_action_steps = 10, 60, 1, 1
    model_sha256 = FIXTURE_SHA
    metadata = {"policy_type": "smolvla", "test_fixture": True}

    def __init__(self, scope: Scope) -> None:
        self.scope, self.predict_calls, self.target = scope, 0, None

    def reset(self) -> None:
        self.target = None

    def predict_chunk(self, observation):
        self.predict_calls += 1
        if self.target is None:
            self.target = (
                observation.joint_positions[0] + 0.0005,
                *observation.joint_positions[1:],
            )
        return (self.target,)


class _FixtureProvider:
    """Constructed only by this explicit CLI, never by the HTTP/deployment catalogue."""

    def __init__(self, scope: Scope, authorization_id: UUID) -> None:
        self.scope, self.authorization_id = scope, authorization_id

    def authorize(self, owner, request, profile) -> None:
        require(
            owner == self.scope.owner_id
            and request.policy_release_id == self.authorization_id
            and request.model_sha256 == FIXTURE_SHA
            and request.policy_type == "smolvla",
            "The isolated fixture authorization does not match this probe",
        )

    def create(self, owner, request, profile):
        self.authorize(owner, request, profile)
        return GuardedPolicyAdapter(_FixturePolicy(self.scope))


def run_probe(args) -> dict:
    scope = Scope(str(args.tenant_id), args.owner)
    scope.validate()
    require(
        scope.tenant_id == os.environ.get("ENTRA_TENANT_ID"),
        "The isolated probe tenant must match the trusted simulator deployment",
    )
    environment = EnvironmentRecord.model_validate(read_json(args.environment_record))
    execution = environment.document["execution"]
    collecting = args.mode == "collect-reference"
    require(
        execution.get("record_demonstration") is True
        and (collecting or execution.get("demonstration_split") == "test"),
        "Approve capture and a prechosen split; mechanics probes require the TEST split",
    )
    if collecting:
        require(
            environment.document["scene"]["template_id"] == "inspection-cell-learning-v1",
            "Bootstrap collection requires the reviewed pose-varying learning scene",
        )
    require(not args.output.exists(), "Refusing to overwrite earlier probe evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = args.output.with_suffix(".heartbeat")
    command_id, session_id, lease_id = uuid4(), uuid4(), uuid4()
    heartbeat_times = []
    initial = None
    task = None
    application = None
    runtime = None
    core = None
    hardware = None
    report = None
    phase = "asset_preparation"
    try:
        initialize_probe_assets()
        phase = "application_initialization"
        application = create_simulation_app()
        from simulation.isaac_adapter import IsaacWorkcell

        phase = "hardware_initialization"
        hardware = IsaacWorkcell()
        profile = ControlProfile(servo_profile_sha256())
        authorization_id = uuid4()
        provider = (
            _FixtureProvider(scope, authorization_id) if args.mode == "policy-fixture" else None
        )
        core = SimulationCore(
            SceneRegistry(),
            control_profile=profile,
            policy_provider=provider,
            tenant_id=scope.tenant_id,
        )

        def capture_factory(binding):
            request = Demonstration.prepare(core, binding.command_id)
            source = DemonstrationSource(
                "reference_controller",
                task["task_id"],
                task["instruction"],
                task["goal_id"],
            )
            return Demonstration(replace(request, control_profile=profile, demonstration=source))

        runtime = SimulatorRuntime(
            core, hardware, heartbeat=heartbeat, capture_factory=capture_factory
        )
        phase = "scene_activation"
        core.activate(scope.owner_id, environment)
        startup_deadline = time.monotonic() + 120
        while not core.ready:
            require(
                application.is_running() and time.monotonic() < startup_deadline,
                "Isaac probe scene did not become ready",
            )
            runtime.tick()
            require(core.error is None, f"Isaac probe failed: {core.error}")
        initial = hardware.initial_state_evidence()
        observation = core.observe(
            scope.owner_id, environment.environment_id, environment.revision, "inspection"
        )
        scene = dict(
            command_id=command_id,
            environment_id=environment.environment_id,
            revision=environment.revision,
            epoch=core.epoch,
            state_revision=core.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=args.goal_id if collecting else core.spec.rejected_id,
        )
        task = dict(
            task_id=args.task_id if collecting else "runtime-mechanics-probe",
            instruction=(
                args.instruction
                if collecting
                else "Hold a measured pose during a bounded mechanics probe."
            ),
            goal_id=args.goal_id if collecting else core.spec.rejected_id,
        )
        motion_seconds = min(30, execution["max_step_seconds"])
        phase = "command_admission"
        if args.mode != "policy-fixture":
            request = TeachingStart(
                **scene,
                session_id=session_id,
                lease_id=lease_id,
                session_expires_at=utcnow() + timedelta(seconds=motion_seconds),
                control_profile_id=profile.profile_id,
                task=task,
                demonstrator_kind="reference_controller",
                split=execution["demonstration_split"],
            )
            core.start_teaching(scope.owner_id, request)
        else:
            core.dispatch_policy(
                scope.owner_id,
                PolicyCommand(
                    command=MotionCommand(
                        **scene, deadline=utcnow() + timedelta(seconds=motion_seconds)
                    ),
                    policy_type="smolvla",
                    policy_release_id=authorization_id,
                    model_sha256=FIXTURE_SHA,
                    control_profile_id=profile.profile_id,
                    task=task,
                ),
            )
        jog_sent = False
        teacher = None
        finish_sent = False
        phase = "control"
        while core.command(scope.owner_id, command_id).status in {"queued", "running"}:
            if (
                collecting
                and not finish_sent
                and hardware.control_mode == "human_teaching"
                and not hardware.warmup_steps
                and not core.deadline_expired()
                and core.clock_ns() >= hardware.control_next_ns
                and (hardware.held_targets is None or hardware.hold_offset == profile.hold_steps)
            ):
                if teacher is None:
                    teacher = ReferenceTeacher(
                        request, core.spec, tcp=hardware._measured_tcp(), part=hardware.position()
                    )
                try:
                    intent = teacher.next_input(
                        now=core.clock_utc(),
                        tcp=hardware._measured_tcp(),
                        finger_gap=float(sum(hardware.robot.get_joint_positions()[7:])),
                        part=hardware.position(),
                    )
                    if intent is None:
                        core.finish_teaching(
                            scope.owner_id,
                            session_id,
                            TeachingLease(lease_id=lease_id, epoch=core.epoch),
                        )
                        finish_sent = True
                    else:
                        core.teaching_input(scope.owner_id, session_id, intent)
                except (ValueError, RuntimeError) as exc:
                    runtime.finish("failed", str(exc))
            if (
                args.mode == "teaching"
                and not jog_sent
                and hardware.control_mode == "human_teaching"
                and not hardware.warmup_steps
            ):
                now = utcnow()
                core.teaching_input(
                    scope.owner_id,
                    session_id,
                    TeachingInput(
                        lease_id=lease_id,
                        epoch=core.epoch,
                        sequence=1,
                        deadman=True,
                        delta_xyz_m=(0, 0, 0.001),
                        gripper="hold",
                        expires_at=now + timedelta(milliseconds=250),
                        grant_id=uuid4(),
                        grant_expires_at=now + timedelta(seconds=1),
                    ),
                )
                jog_sent = True
            runtime.tick()
            if heartbeat.is_file():
                timestamp = float(heartbeat.read_text())
                if not heartbeat_times or timestamp != heartbeat_times[-1]:
                    heartbeat_times.append(timestamp)
            if not collecting and len(getattr(hardware, "control_timings", ())) >= args.intervals:
                core.cancel(scope.owner_id, command_id)
                runtime.tick()
                break
            time.sleep(0.001)
        persistence_deadline = time.monotonic() + 120
        phase = "capture_publication"
        capture = core.captures.get((scope.owner_id, command_id))
        while capture is not None and capture.status not in {"ready", "invalid"}:
            if time.monotonic() >= persistence_deadline:
                runtime.capture_worker.invalidate("Capture publication probe timed out.")
            runtime.tick()
            capture = core.capture(scope.owner_id, command_id)
            timestamp = float(heartbeat.read_text())
            if not heartbeat_times or timestamp != heartbeat_times[-1]:
                heartbeat_times.append(timestamp)
            time.sleep(0.001)
        result = core.command(scope.owner_id, command_id)
        fixture = None
        if args.mode == "policy-fixture" and result.policy_runtime is not None:
            fixture = result.policy_runtime.model_dump(mode="json")
            fixture["fixture_authorization_id"] = fixture.pop("policy_release_id")
            fixture["fixture_compatibility_type"] = fixture.pop("policy_type")
            fixture["applied_fixture_sha256"] = fixture.pop("applied_model_sha")
        report = {
            "schema": (
                "physicalai.reference-teaching-receipt/v1"
                if collecting
                else "physicalai.gpu-control-probe/v1"
            ),
            "mode": args.mode,
            "demonstrator_kind": "reference_controller",
            **outcome_fields(args.mode, result, capture),
            "command_id": str(command_id),
            "task": task,
            "split": execution["demonstration_split"],
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "initial_state": initial,
            "control_profile_sha256": profile.sha256,
            "control_intervals": getattr(hardware, "control_timings", []),
            "camera_observation": getattr(hardware, "camera_observation_metadata", None),
            "probe_completed": (
                not collecting
                and len(getattr(hardware, "control_timings", [])) >= args.intervals
                and capture is not None
                and capture.status == "ready"
            ),
            "max_heartbeat_gap_ms": max(
                ((b - a) * 1000 for a, b in pairwise(heartbeat_times)),
                default=0,
            ),
            "fixture_actuation": fixture,
        }
        return report
    finally:
        pending_error = sys.exc_info()[1]
        try:
            if report is None:
                result = None
                capture = None
                if core is not None:
                    with core.lock:
                        result = core.commands.get((scope.owner_id, command_id))
                        capture = core.captures.get((scope.owner_id, command_id))
                report = {
                    "schema": "physicalai.operator-attempt-failure/v1",
                    "mode": args.mode,
                    "environment_id": environment.environment_id,
                    "revision": environment.revision,
                    "initial_state": initial,
                    "command_id": str(command_id),
                    "phase": phase,
                    "failure_type": (
                        type(pending_error).__name__
                        if pending_error is not None
                        else "MissingTerminalOutcome"
                    ),
                    "failure": str(pending_error)
                    if pending_error is not None
                    else "No terminal outcome",
                    "probe_completed": False,
                    "control_intervals": (
                        getattr(hardware, "control_timings", []) if hardware is not None else []
                    ),
                    "camera_observation": (
                        getattr(hardware, "camera_observation_metadata", None)
                        if hardware is not None
                        else None
                    ),
                    **outcome_fields(args.mode, result, capture),
                }
            _persist_receipt(args.output, report)
            if pending_error is not None:
                traceback.print_exception(
                    type(pending_error), pending_error, pending_error.__traceback__, file=sys.stderr
                )
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                try:
                    if application is not None:
                        _close_application(application, pending_error or sys.exc_info()[1])
                finally:
                    heartbeat.unlink(missing_ok=True)
        if isinstance(pending_error, SystemExit) and pending_error.code in (None, 0):
            raise SystemExit(1) from pending_error


def main() -> None:
    args = arguments()
    report = run_probe(args)
    print(
        "PHYSICALAI_GPU_CONTROL_PROBE "
        + json.dumps(
            {
                "output": str(args.output),
                "mode": report["mode"],
                "model_weights_loaded": False,
                "learning_quality_proven": False,
            }
        ),
        flush=True,
    )
    if args.mode == "collect-reference":
        if not report["physical_task_success"] or report["capture"]["status"] != "ready":
            raise SystemExit(1)
    elif not report["probe_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
