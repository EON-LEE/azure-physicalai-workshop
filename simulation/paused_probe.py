"""Operator-authorized NON_REALTIME_SIMULATION reference capture, never real-time admission."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from apps.api.models import EnvironmentRecord, utcnow
from learning.common import read_json, require
from learning.paused import PausedControlProfile, protocol_schemas
from simulation.capture_status import CaptureStatusStore
from simulation.core import SimulationCore
from simulation.extensions import SceneRegistry
from simulation.paused_configuration import OperatorPausedAuthority, paused_servo_sha256
from simulation.paused_contracts import SimulationEpisodeCommand
from simulation.probe_control import _close_application, _persist_receipt, initialize_probe_assets
from simulation.run_isaac import SimulatorRuntime, create_simulation_app


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--operator-grant", type=Path, required=True)
    parser.add_argument("--grant-sha256", required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--criteria-sha256", required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--conditions-sha256", required=True)
    parser.add_argument("--mode", choices=("mechanics", "reference-task"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-isolated-simulator", action="store_true", required=True)
    return parser.parse_args(argv)


def run(args) -> dict:
    environment = EnvironmentRecord.model_validate(read_json(args.environment_record))
    profile = PausedControlProfile(paused_servo_sha256())
    authority = OperatorPausedAuthority.load(
        args.operator_grant,
        args.grant_sha256,
        environment=environment,
        profile=profile,
        criteria=args.criteria,
        criteria_sha256=args.criteria_sha256,
        conditions=args.conditions,
        conditions_sha256=args.conditions_sha256,
    )
    permit = authority.grant.authorization
    require(not args.output.exists(), "Never overwrite an existing paused attempt receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = args.output.with_suffix(".heartbeat")
    application = hardware = runtime = core = None
    command_id = uuid4()
    result = report = None
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
            paused_authorizer=authority,
            tenant_id=str(authority.grant.tenant_id),
            capture_status_reader=store.get,
        )
        runtime = SimulatorRuntime(core, hardware, heartbeat=heartbeat, capture_store=store)
        phase = "scene_preparation"
        core.activate(permit.owner, environment)
        preparation_deadline = min(
            time.monotonic() + 120,
            time.monotonic() + (authority.grant.expires_at - utcnow()).total_seconds(),
        )
        while not core.ready:
            require(
                application.is_running() and time.monotonic() < preparation_deadline,
                "The original operator preparation budget expired",
            )
            runtime.tick()
            require(core.error is None, core.error or "Paused scene preparation failed")
        publication = hardware.paused_publications.publication
        require(publication is not None, "No actual contemporaneous first publication")
        observation = core.observe(
            permit.owner, environment.environment_id, environment.revision, "inspection"
        )
        request = SimulationEpisodeCommand(
            schema="physicalai.simulation-episode-command/v1",
            execution_timing="paused_simulation",
            real_time_admission=False,
            profile_id=profile.profile_id,
            command_id=command_id,
            environment_id=environment.environment_id,
            revision=environment.revision,
            epoch=core.epoch,
            state_revision=core.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=permit.task.goal_id,
            task=permit.task,
            wall_expires_at=min(
                permit.wall_expires_at,
                utcnow() + timedelta(seconds=permit.max_episode_wall_seconds),
            ),
            max_simulation_steps=permit.max_simulation_steps,
            controller="reference_controller",
            authorization_kind=permit.authorization_kind,
            authorization_id=permit.authorization_id,
        )
        phase = "episode"
        core.dispatch_simulation_episode(permit.owner, request)
        while core.command(permit.owner, command_id).status in {"queued", "running", "cancelling"}:
            runtime.tick()
            driver = hardware.paused_driver
            if args.mode == "mechanics" and driver is not None and driver.complete_intervals >= 100:
                core.cancel(permit.owner, command_id)
            time.sleep(0.001)
        result = core.command(permit.owner, command_id)
        phase = "capture_publication"
        capture = core.captures.get((permit.owner, command_id))
        while capture is not None and capture.status not in {"ready", "invalid"}:
            if utcnow() >= request.wall_expires_at:
                runtime.capture_worker.invalidate(
                    "Original paused wall deadline expired before publication."
                )
            runtime.tick()
            capture = core.capture(permit.owner, command_id)
            time.sleep(0.001)
        driver = hardware.paused_driver
        metrics = driver.episode.metrics() if driver is not None else None
        if metrics is not None:
            metrics["intervals"] = [
                {
                    **item,
                    "applied_controls": [asdict(control) for control in item["applied_controls"]],
                }
                for item in metrics["intervals"]
            ]
        report = {
            "schema": "physicalai.paused-reference-attempt/v1",
            "execution_timing": "paused_simulation",
            "real_time_admission": False,
            "display_label": "NON_REALTIME_SIMULATION",
            "mode": args.mode,
            "physical_status": result.status,
            "command_id": str(command_id),
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "task": permit.task.model_dump(),
            "source_kind": "reference_controller",
            "source_revision": os.environ["SOURCE_REVISION"],
            "simulator_image_digest": os.environ["SIMULATOR_IMAGE"].split("@")[-1],
            "control_profile": asdict(profile),
            "control_profile_sha256": profile.sha256,
            "criteria_sha256": args.criteria_sha256,
            "frozen_plan_sha256": args.conditions_sha256,
            "operator_grant_sha256": args.grant_sha256,
            "protocol_schemas": protocol_schemas(),
            "initial_state": hardware.initial_state_evidence(),
            "initial_publication_record": publication.private_evidence(),
            "final_position": result.final_position,
            "grasp_verified": hardware.grasp_verified,
            "peak_tcp_speed_m_s": hardware.peak_tcp_speed,
            "metrics": metrics,
            "unarmed_warmup": hardware.control_warmup_timings,
            "physics_scheduling": hardware.physics_scheduling,
            "capture": capture.model_dump(mode="json") if capture is not None else None,
            "physical_task_success": result.status == "succeeded" and args.mode == "reference-task",
            "model_weights_loaded": False,
            "learning_quality_proven": False,
            "error": result.error.model_dump() if result.error is not None else None,
        }
        return report
    finally:
        error = sys.exc_info()[1]
        try:
            if report is None:
                report = {
                    "schema": "physicalai.paused-reference-failure/v1",
                    "execution_timing": "paused_simulation",
                    "real_time_admission": False,
                    "display_label": "NON_REALTIME_SIMULATION",
                    "physical_status": result.status if result is not None else "not_completed",
                    "command_id": str(command_id),
                    "phase": phase,
                    "failure_type": type(error).__name__
                    if error is not None
                    else "IncompleteAttempt",
                    "failure": str(error),
                    "environment_id": environment.environment_id,
                    "revision": environment.revision,
                    "criteria_sha256": args.criteria_sha256,
                    "frozen_plan_sha256": args.conditions_sha256,
                    "model_weights_loaded": False,
                    "learning_quality_proven": False,
                }
            _persist_receipt(args.output, report)
        finally:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                if application is not None:
                    _close_application(application, error or sys.exc_info()[1])
                heartbeat.unlink(missing_ok=True)


def main() -> None:
    args = arguments()
    report = run(args)
    print(
        "PHYSICALAI_PAUSED_ATTEMPT "
        + json.dumps(
            {
                "output": str(args.output),
                "physical_status": report["physical_status"],
                "real_time_admission": False,
            }
        ),
        flush=True,
    )
    passed = (report.get("capture") or {}).get("status") == "ready"
    if args.mode == "reference-task":
        passed = passed and report.get("physical_task_success") is True
    else:
        passed = (
            passed
            and report["physical_status"] == "cancelled"
            and len((report.get("metrics") or {}).get("intervals", [])) == 100
        )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
