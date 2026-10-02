"""Fail-closed operator evidence checks; process exit status alone is never GPU proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.common import canonical, digest, finite, integer, read_json, require, sha256, vector
from learning.contract import Scope, validate_dataset


def validate_probe_report(
    report: dict, *, mode: str, environment_id: str, revision: str, intervals: int = 100
) -> dict:
    require(
        report.get("schema") == "physicalai.gpu-control-probe/v1"
        and report.get("mode") == mode
        and report.get("environment_id") == environment_id
        and report.get("revision") == revision,
        "The receipt is not the expected mechanics case",
    )
    require(report.get("probe_completed") is True, "The GPU probe did not complete")
    require(
        report.get("model_weights_loaded") is False
        and report.get("learning_quality_proven") is False
        and report.get("production_ready") is False
        and report.get("physical_task_success") is False
        and report.get("demonstrator_kind") == "reference_controller",
        "Mechanics evidence cannot claim human input, model weights or learning/task success",
    )
    require(report.get("physical_status") == "cancelled", "The bounded probe did not stop normally")
    heartbeat = finite(report.get("max_heartbeat_gap_ms"), "heartbeat gap")
    require(0 < heartbeat <= 2000, "Missing or excessive main-thread heartbeat gap")
    initial = report.get("initial_state")
    require(isinstance(initial, dict), "Missing measured initial state")
    vector(initial.get("observed_initial_pose_m"), 3, "initial part pose")
    sha256(initial.get("scene_builder_sha256"), "scene builder")
    sha256(report.get("control_profile_sha256"), "control profile")
    controls = report.get("control_intervals")
    require(
        isinstance(controls, list) and len(controls) == intervals, "Wrong actual interval count"
    )
    previous = None
    for control in controls:
        require(isinstance(control, dict), "Invalid control interval")
        start = integer(control.get("observation_step"), "observed physics step")
        end = integer(control.get("completed_step"), "completed physics step")
        require(
            end - start == 6 and (previous is None or start == previous),
            "Missing, repeated or resampled physical control ticks",
        )
        require(
            0 <= finite(control.get("control_cycle_ms"), "control cycle") < 100,
            "The control work exceeded the unchanged 100 ms profile",
        )
        require(
            0 <= finite(control.get("inference_latency_ms"), "inference latency") <= 80,
            "Inference exceeded the unchanged 80 ms budget",
        )
        previous = end
    capture = report.get("capture")
    require(isinstance(capture, dict) and capture.get("status") == "ready", "Capture is not ready")
    receipt = capture.get("receipt")
    require(
        isinstance(receipt, dict) and receipt.get("status") == "uploaded",
        "Missing validated manifest-last upload receipt",
    )
    require(
        integer(receipt.get("frame_count"), "capture frames") == intervals,
        "Capture frame count does not match actual complete intervals",
    )
    sha256(receipt.get("manifest_sha256"), "uploaded manifest")
    result = {
        "accepted": True,
        "control_intervals": intervals,
        "max_control_cycle_ms": max(item["control_cycle_ms"] for item in controls),
        "max_inference_latency_ms": max(item["inference_latency_ms"] for item in controls),
        "max_heartbeat_gap_ms": heartbeat,
        "manifest_sha256": receipt["manifest_sha256"],
        "control_profile_sha256": report["control_profile_sha256"],
        "learning_quality_proven": False,
    }
    if mode == "policy-fixture":
        fixture = report.get("fixture_actuation")
        require(isinstance(fixture, dict), "Missing actual fixture actuator evidence")
        sha256(fixture.get("applied_fixture_sha256"), "applied fixture")
        require(
            fixture.get("reference_route_calls") == 0
            and integer(fixture.get("policy_predict_calls"), "prediction attempts") == intervals
            and integer(fixture.get("applied_action_count"), "applied ticks") == intervals * 6,
            "Fixture predictions did not exclusively drive all actual actuator ticks",
        )
        result["applied_action_count"] = fixture["applied_action_count"]
    else:
        require(
            report.get("fixture_actuation") is None, "Teaching unexpectedly used a policy fixture"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--mode", choices=("teaching", "policy-fixture"), required=True)
    parser.add_argument("--intervals", type=int, default=100)
    args = parser.parse_args()
    try:
        report = read_json(args.report)
        environment = read_json(args.environment_record)
        summary = validate_probe_report(
            report,
            mode=args.mode,
            environment_id=environment["environment_id"],
            revision=environment["revision"],
            intervals=args.intervals,
        )
        raw = validate_dataset(
            args.dataset_root,
            expected_scope=Scope(args.tenant_id, args.owner),
            expected_manifest_sha256=summary["manifest_sha256"],
            require_live=True,
        )
        require(
            raw.manifest["schema"] == "physicalai.demonstrations/v2",
            "G0 capture must use actual v2 held-action semantics",
        )
        require(len(raw.episodes) == 1, "A mechanics probe must contain exactly one bound episode")
        episode = raw.episodes[0]
        metadata = episode.metadata
        require(
            metadata["environment_id"] == environment["environment_id"]
            and metadata["revision"] == environment["revision"]
            and metadata["seed"] == environment["document"]["scene"]["seed"]
            and metadata["split"] == "test"
            and metadata["demonstration"]["kind"] == "reference_controller",
            "Raw capture does not match the preapproved test case and scripted source",
        )
        provenance = metadata["provenance"]
        require(
            provenance["simulator_image_digest"] == args.image_digest
            and provenance["code_revision"] == args.source_revision
            and provenance["scene_builder_sha256"]
            == report["initial_state"]["scene_builder_sha256"],
            "Actual capture image, source or scene-builder provenance differs",
        )
        require(
            digest(canonical(raw.manifest["control_profile"])) == summary["control_profile_sha256"],
            "Dataset and probe control profiles differ",
        )
        require(
            all(
                image["width"] == image["height"] == 320
                for frame in episode.frames
                for image in frame["images"].values()
            ),
            "Both actual cameras must be 320 by 320",
        )
        summary.update(episode_id=metadata["episode_id"], source_revision=args.source_revision)
        print("PHYSICALAI_G0_ACCEPTANCE " + json.dumps(summary), flush=True)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(
            "PHYSICALAI_G0_ACCEPTANCE "
            + json.dumps(
                {
                    "accepted": False,
                    "error": str(exc),
                    "learning_quality_proven": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
