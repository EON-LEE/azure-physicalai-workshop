"""Local integrity/claim verification for an independently retrieved vendor-only proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.common import file_digest, finite, integer, read_json, require, sha256, vector
from learning.smolvla import UPSTREAM
from learning.smolvla.prepare import VENDOR_GIT_BLOBS, VENDOR_SHA256


def verify_proof(path: Path, expected_sha256: str, specification: dict) -> dict:
    require(file_digest(path) == sha256(expected_sha256), "Retrieved proof checksum mismatch")
    report = read_json(path, max_bytes=64 * 1024)
    require(
        report.get("schema") == "physicalai.smolvla-vendor-compatibility-proof/v1"
        and report.get("purpose") == "vendor_weight_and_native_runtime_compatibility_only"
        and report.get("source") == "actual_azure_ml"
        and report.get("scope") == specification["scope"]
        and report.get("azure_job_id")
        == specification["workspace_id"] + "/jobs/" + specification["job_name"],
        "Proof does not belong to the exact authorized diagnostic",
    )
    for name in (
        "image",
        "image_code_snapshot_sha256",
        "diagnostic_code_sha256",
        "vendor_inventory_sha256",
    ):
        require(report.get(name) == specification[name], "Diagnostic artifact identity changed")
    require(report.get("upstream") == UPSTREAM, "Vendor model/backbone/source/license pins changed")
    require(
        report.get("passed") is True
        and report.get("phase") == "completed"
        and report.get("vendor_weights_loaded") is True,
        "Actual vendor-weight compatibility did not complete",
    )
    require(
        report.get("observation_source") == "fixture" and report.get("test_only") is True,
        "Synthetic observations must remain explicitly identified",
    )
    for name in ("optimizer_steps", "actuator_calls"):
        require(
            type(report.get(name)) is int and report[name] == 0, "Unauthorized training/actuation"
        )
    for name in (
        "franka_adaptation_verified",
        "latency_admission_verified",
        "learning_quality_verified",
        "ready_for_live_execution",
    ):
        require(report.get(name) is False, "Vendor compatibility cannot claim physical admission")
    require(
        report.get("declared_state_dimensions") == report.get("declared_action_dimensions") == 6
        and report.get("declared_camera_count") == 3
        and report.get("fixture_input_shapes")
        == {
            "observation.state": [6],
            **{f"observation.images.camera{i}": [3, 256, 256] for i in (1, 2, 3)},
        },
        "Diagnostic did not preserve the published six-dimensional/three-camera inputs",
    )
    files = report.get("verified_assets")
    require(
        isinstance(files, dict) and set(files) == {"model", "backbone"},
        "Missing exact vendor inventories",
    )
    for kind in ("model", "backbone"):
        expected = set(VENDOR_SHA256[kind]) | set(VENDOR_GIT_BLOBS[kind])
        require(set(files[kind]) == expected, "Verified vendor inventory is incomplete")
        for name, checksum in files[kind].items():
            sha256(checksum)
            if name in VENDOR_SHA256[kind]:
                require(
                    checksum == VENDOR_SHA256[kind][name],
                    "Wrong actual vendor weight/license bytes",
                )
    gpu = report.get("gpu")
    require(
        isinstance(gpu, dict) and isinstance(gpu.get("name"), str) and bool(gpu["name"]),
        "Missing actual CUDA device evidence",
    )
    memory = integer(gpu.get("total_memory_bytes"), "GPU memory", 1)
    require(
        gpu.get("torch_version") == "2.7.1+cu126" and gpu.get("cuda_version") == "12.6",
        "Runtime differs from the approved CUDA dependency image",
    )
    require(
        isinstance(gpu.get("nvidia_smi"), str) and bool(gpu["nvidia_smi"]),
        "Missing actual driver evidence",
    )
    allocated = integer(report.get("peak_cuda_allocated_bytes"), "peak allocated memory", 1, memory)
    integer(report.get("peak_cuda_reserved_bytes"), "peak reserved memory", allocated, memory)
    require(
        0
        < finite(report.get("load_seconds"), "vendor model load time")
        <= finite(report.get("elapsed_seconds"), "diagnostic elapsed time")
        <= specification["execution_timeout_seconds"],
        "Unbounded/invalid actual diagnostic timing",
    )
    integer(report.get("parameters"), "actual model parameters", 1)
    require(
        report.get("actual_forward_calls") == 3
        and isinstance(report.get("forward_calls"), list)
        and len(report["forward_calls"]) == 3,
        "Missing actual native forward calls",
    )
    for index, call in enumerate(report["forward_calls"], start=1):
        require(
            call.get("call") == index
            and call.get("shape") == [1, 50, 6]
            and call.get("finite") is True,
            "Actual native output shape/finiteness mismatch",
        )
        require(
            finite(call.get("latency_ms"), "measured native latency") > 0, "Invalid actual latency"
        )
        sha256(call.get("output_tensor_sha256"))
        vector(call.get("first_action_fixture_only"), 6, "fixture-only sample")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    report = verify_proof(args.proof, args.sha256, read_json(args.plan)["specification"])
    print(
        json.dumps(
            {
                "proof_sha256": args.sha256,
                "job": report["azure_job_id"].rsplit("/", 1)[1],
                "passed": report["passed"],
                "gpu": report["gpu"],
                "load_seconds": report["load_seconds"],
                "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": report["peak_cuda_reserved_bytes"],
                "forward_calls": [
                    {key: call[key] for key in ("call", "shape", "latency_ms", "finite")}
                    for call in report["forward_calls"]
                ],
                "test_only": True,
                "optimizer_steps": 0,
                "actuator_calls": 0,
                "franka_adaptation_verified": False,
                "latency_admission_verified": False,
                "learning_quality_verified": False,
            },
            indent=2,
        )
    )
