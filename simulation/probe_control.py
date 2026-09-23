"""Operator-only, isolated GPU mechanics probe; never a model or learning-quality gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from uuid import UUID, uuid4

from apps.api.models import EnvironmentRecord, MotionCommand, utcnow
from learning.common import canonical, digest, read_json, require
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.inference import GuardedPolicyAdapter
from simulation.core import SimulationCore
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry
from simulation.run_isaac import SimulatorRuntime, create_simulation_app
from simulation.runtime_configuration import servo_profile_sha256
from simulation.runtime_contracts import PolicyCommand, TeachingInput, TeachingStart

FIXTURE_SHA = digest(b"physicalai-isolated-actuation-fixture/v1")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--tenant-id", type=UUID, required=True)
    parser.add_argument("--mode", choices=("teaching", "policy-fixture"), required=True)
    parser.add_argument("--intervals", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-isolated-simulator", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not 100 <= args.intervals <= 150:
        parser.error("The probe is bounded to 100..150 complete control intervals.")
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
    require(
        execution.get("record_demonstration") is True
        and execution.get("demonstration_split") == "test",
        "Save and explicitly approve a TEST-split capture environment for this mechanics probe",
    )
    require(not args.output.exists(), "Refusing to overwrite earlier probe evidence")
    application = create_simulation_app()
    from simulation.isaac_adapter import IsaacWorkcell

    hardware = IsaacWorkcell()
    profile = ControlProfile(servo_profile_sha256())
    authorization_id = uuid4()
    provider = _FixtureProvider(scope, authorization_id) if args.mode == "policy-fixture" else None
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
            "runtime-mechanics-probe",
            "Hold a measured pose during a bounded mechanics probe.",
            core.spec.rejected_id,
        )
        return Demonstration(replace(request, control_profile=profile, demonstration=source))

    heartbeat = args.output.with_suffix(".heartbeat")
    runtime = SimulatorRuntime(core, hardware, heartbeat=heartbeat, capture_factory=capture_factory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command_id, session_id, lease_id = uuid4(), uuid4(), uuid4()
    heartbeat_times = []
    try:
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
            target_station_id=core.spec.rejected_id,
        )
        task = dict(
            task_id="runtime-mechanics-probe",
            instruction="Hold a measured pose during a bounded mechanics probe.",
            goal_id=core.spec.rejected_id,
        )
        if args.mode == "teaching":
            request = TeachingStart(
                **scene,
                session_id=session_id,
                lease_id=lease_id,
                session_expires_at=utcnow() + timedelta(seconds=30),
                control_profile_id=profile.profile_id,
                task=task,
                demonstrator_kind="reference_controller",
            )
            core.start_teaching(scope.owner_id, request)
        else:
            core.dispatch_policy(
                scope.owner_id,
                PolicyCommand(
                    command=MotionCommand(**scene, deadline=utcnow() + timedelta(seconds=30)),
                    policy_type="smolvla",
                    policy_release_id=authorization_id,
                    model_sha256=FIXTURE_SHA,
                    control_profile_id=profile.profile_id,
                    task=task,
                ),
            )
        jog_sent = False
        while core.command(scope.owner_id, command_id).status in {"queued", "running"}:
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
            require(core.error is None, f"Isaac probe failed: {core.error}")
            if len(getattr(hardware, "control_timings", ())) >= args.intervals:
                core.cancel(scope.owner_id, command_id)
                runtime.tick()
                break
            time.sleep(0.001)
        require(
            len(hardware.control_timings) >= args.intervals, "Probe ended before its interval goal"
        )
        persistence_deadline = time.monotonic() + 120
        while core.capture(scope.owner_id, command_id).status not in {"ready", "invalid"}:
            require(time.monotonic() < persistence_deadline, "Capture publication probe timed out")
            runtime.tick()
            timestamp = float(heartbeat.read_text())
            if not heartbeat_times or timestamp != heartbeat_times[-1]:
                heartbeat_times.append(timestamp)
            time.sleep(0.001)
        capture = core.capture(scope.owner_id, command_id)
        require(capture.status == "ready", "The real probe capture was not published")
        result = core.command(scope.owner_id, command_id)
        fixture = None
        if args.mode == "policy-fixture":
            fixture = result.policy_runtime.model_dump(mode="json")
            fixture["fixture_authorization_id"] = fixture.pop("policy_release_id")
            fixture["fixture_compatibility_type"] = fixture.pop("policy_type")
            fixture["applied_fixture_sha256"] = fixture.pop("applied_model_sha")
        report = {
            "schema": "physicalai.gpu-control-probe/v1",
            "mode": args.mode,
            "demonstrator_kind": "reference_controller",
            "model_weights_loaded": False,
            "learning_quality_proven": False,
            "production_ready": False,
            "physical_task_success": False,
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "initial_state": initial,
            "control_profile_sha256": profile.sha256,
            "control_intervals": hardware.control_timings,
            "max_heartbeat_gap_ms": max(
                ((b - a) * 1000 for a, b in pairwise(heartbeat_times)),
                default=0,
            ),
            "fixture_actuation": fixture,
            "capture": capture.model_dump(mode="json"),
            "physical_status": result.status,
        }
        args.output.write_bytes(canonical(report) + b"\n")
        return report
    finally:
        try:
            runtime.close()
        finally:
            application.close()
            heartbeat.unlink(missing_ok=True)


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


if __name__ == "__main__":
    main()
