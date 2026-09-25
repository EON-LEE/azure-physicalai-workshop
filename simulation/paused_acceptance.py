"""Validate one actual paused reference capture without granting real-time or model quality."""

from __future__ import annotations

import argparse
import json
from math import dist
from pathlib import Path

from apps.api.errors import Problem
from apps.api.models import EnvironmentRecord
from learning.common import canonical, digest, finite, integer, read_json, require, sha256, vector
from learning.contract import Scope
from learning.paused.capture import validate_dataset
from simulation.extensions import SceneRegistry


def validate_attempt(report: dict, *, environment: EnvironmentRecord, mode: str) -> dict:
    spec = SceneRegistry(load_installed=False).build(environment)
    authority = spec.require_paused_authority()
    require(
        report.get("schema") == "physicalai.paused-reference-attempt/v1"
        and report.get("execution_timing") == "paused_simulation"
        and report.get("real_time_admission") is False
        and report.get("mode") == mode
        and report.get("environment_id") == environment.environment_id
        and report.get("revision") == environment.revision
        and report.get("source_kind") == "reference_controller"
        and report.get("model_weights_loaded") is False
        and report.get("learning_quality_proven") is False,
        "Wrong reference mode, case or unsupported learned/real-time admission claim",
    )
    capture = report.get("capture")
    require(isinstance(capture, dict) and capture.get("status") == "ready", "Capture is not ready")
    receipt = capture.get("receipt")
    require(
        isinstance(receipt, dict) and receipt.get("status") == "uploaded",
        "A manifest-last upload receipt is required",
    )
    metrics = report.get("metrics")
    require(isinstance(metrics, dict), "Missing actual paused timing metrics")
    steps = integer(metrics.get("simulation_steps"), "actual simulation steps", 12, 1800)
    require(
        0
        < finite(metrics.get("wall_elapsed_ms"), "wall duration")
        <= authority.max_wall_seconds * 1000
        and steps <= authority.max_simulation_steps
        and steps % 6 == 0,
        "The recorded wall/simulation work exceeds the independently approved scene budgets",
    )
    require(
        abs(finite(metrics.get("simulation_elapsed_seconds"), "simulation duration") - steps / 60)
        <= 1e-6,
        "Simulation time does not match the actual 60 Hz ticks",
    )
    intervals = metrics.get("intervals")
    require(isinstance(intervals, list) and len(intervals) * 6 == steps, "Missing actual intervals")
    require(receipt.get("frame_count") == len(intervals), "Capture omitted an actual interval")
    prior = None
    for interval in intervals:
        start = integer(interval.get("observation_physics_step"), "interval start step")
        end = integer(interval.get("completed_physics_step"), "interval end step")
        require(
            end - start == 6 and (prior is None or start == prior), "Skipped actual physics ticks"
        )
        for field, limit in (
            ("observation_wall_ms", 2000),
            ("policy_wall_ms", 2000),
            ("hold_wall_ms", 2000),
            ("interval_wall_ms", 5000),
        ):
            require(
                0 <= finite(interval.get(field), field) <= limit,
                f"{field} exceeded its wall budget",
            )
        prior = end
    initial = report.get("initial_state") or {}
    require(
        initial.get("scene_builder_sha256") == spec.scene_builder_sha256
        and dist(
            vector(initial.get("observed_initial_pose_m"), 3, "actual initial pose"),
            spec.part_position,
        )
        <= 0.001,
        "Actual initial pose or scene builder differs from the frozen case",
    )
    require(
        0
        <= finite(report.get("peak_tcp_speed_m_s"), "peak TCP speed")
        <= min(0.2, spec.requested_speed),
        "The physical TCP speed guard did not pass",
    )
    if mode == "reference-task":
        goal = spec.station(report["task"]["goal_id"]).position
        actual = vector(report.get("final_position"), 3, "final measured position")
        require(
            report.get("physical_status") == "succeeded"
            and report.get("physical_task_success") is True
            and report.get("grasp_verified") is True
            and all(abs(a - b) <= 0.04 for a, b in zip(actual, goal, strict=True)),
            "Measured reference pick/lift/place did not pass its physical checks",
        )
    else:
        require(
            mode == "mechanics"
            and len(intervals) == 100
            and report.get("physical_status") == "cancelled"
            and report.get("physical_task_success") is False,
            "Mechanics evidence is not a complete bounded 100-interval attempt",
        )
    return {
        "accepted": True,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "mode": mode,
        "actual_intervals": len(intervals),
        "simulation_steps": steps,
        "wall_elapsed_ms": metrics["wall_elapsed_ms"],
        "learning_quality_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--criteria-sha256", required=True)
    parser.add_argument("--conditions-sha256", required=True)
    parser.add_argument("--mode", choices=("mechanics", "reference-task"), required=True)
    args = parser.parse_args()
    try:
        report = read_json(args.report)
        environment = EnvironmentRecord.model_validate(read_json(args.environment_record))
        result = validate_attempt(report, environment=environment, mode=args.mode)
        raw = validate_dataset(
            args.dataset_root,
            expected_scope=Scope(args.tenant_id, args.owner),
            require_live=True,
            expected_manifest_sha256=report["capture"]["receipt"]["manifest_sha256"],
        )
        require(len(raw.episodes) == 1, "One operator episode must have one raw capture")
        metadata = raw.episodes[0].metadata
        for value in (args.profile_sha256, args.criteria_sha256, args.conditions_sha256):
            sha256(value)
        require(
            raw.manifest["criteria_sha256"] == report["criteria_sha256"] == args.criteria_sha256
            and raw.manifest["frozen_plan_sha256"]
            == report["frozen_plan_sha256"]
            == args.conditions_sha256
            and digest(canonical(raw.manifest["control_profile"]))
            == report["control_profile_sha256"]
            == args.profile_sha256
            and metadata["provenance"]["code_revision"]
            == report["source_revision"]
            == args.source_revision
            and metadata["provenance"]["simulator_image_digest"]
            == report["simulator_image_digest"]
            == args.image_digest
            and metadata["episode_id"] == report["command_id"]
            and metadata["environment_id"] == environment.environment_id
            and metadata["revision"] == environment.revision
            and metadata["demonstration"]["kind"] == "reference_controller"
            and metadata["frame_count"] == result["actual_intervals"],
            "Raw capture, image, source, criteria, profile or owner/episode binding differs",
        )
        require(
            all(
                image["width"] == image["height"] == 320
                for frame in raw.episodes[0].frames
                for image in frame["images"].values()
            ),
            "Both actual cameras must remain 320 by 320",
        )
        result["manifest_sha256"] = raw.manifest_sha256
        print("PHYSICALAI_PAUSED_ACCEPTANCE " + json.dumps(result), flush=True)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, Problem) as exc:
        print(
            "PHYSICALAI_PAUSED_ACCEPTANCE "
            + json.dumps(
                {
                    "accepted": False,
                    "real_time_admission": False,
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
