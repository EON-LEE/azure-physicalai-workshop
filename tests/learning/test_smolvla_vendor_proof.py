from copy import deepcopy

import pytest

from learning.common import ContractError, canonical, digest, write_json
from learning.smolvla import UPSTREAM
from learning.smolvla.prepare import VENDOR_GIT_BLOBS, VENDOR_SHA256


def proof():
    spec = {
        "scope": {"tenant_id": "11111111-1111-4111-8111-111111111111", "owner_id": "a" * 64},
        "workspace_id": (
            "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/test-rg"
            "/providers/Microsoft.MachineLearningServices/workspaces/existing-ml"
        ),
        "job_name": "unit-fixture",
        "image": "testregistry.azurecr.io/physicalai-smolvla@sha256:" + "b" * 64,
        "image_code_snapshot_sha256": "c" * 64,
        "diagnostic_code_sha256": "d" * 64,
        "vendor_inventory_sha256": "e" * 64,
        "execution_timeout_seconds": 600,
    }
    value = {
        "schema": "physicalai.smolvla-vendor-compatibility-proof/v1",
        "purpose": "vendor_weight_and_native_runtime_compatibility_only",
        "source": "actual_azure_ml",
        "scope": spec["scope"],
        "azure_job_id": spec["workspace_id"] + "/jobs/" + spec["job_name"],
        **{
            key: spec[key]
            for key in (
                "image",
                "image_code_snapshot_sha256",
                "diagnostic_code_sha256",
                "vendor_inventory_sha256",
            )
        },
        "upstream": UPSTREAM,
        "passed": True,
        "phase": "completed",
        "test_only": True,
        "observation_source": "fixture",
        "optimizer_steps": 0,
        "actuator_calls": 0,
        "franka_adaptation_verified": False,
        "latency_admission_verified": False,
        "learning_quality_verified": False,
        "ready_for_live_execution": False,
        "vendor_weights_loaded": True,
        "actual_forward_calls": 3,
        "declared_state_dimensions": 6,
        "declared_action_dimensions": 6,
        "declared_camera_count": 3,
        "fixture_input_shapes": {
            "observation.state": [6],
            **{f"observation.images.camera{i}": [3, 256, 256] for i in (1, 2, 3)},
        },
        "verified_assets": {
            kind: {**VENDOR_SHA256[kind], **{name: "f" * 64 for name in VENDOR_GIT_BLOBS[kind]}}
            for kind in ("model", "backbone")
        },
        "gpu": {
            "name": "unit-test GPU, not live evidence",
            "total_memory_bytes": 80_000_000_000,
            "torch_version": "2.7.1+cu126",
            "cuda_version": "12.6",
            "nvidia_smi": "unit-test driver",
            "compute_capability": [8, 0],
        },
        "load_seconds": 12.0,
        "elapsed_seconds": 50.0,
        "parameters": 450000000,
        "peak_cuda_allocated_bytes": 4_000_000_000,
        "peak_cuda_reserved_bytes": 5_000_000_000,
        "forward_calls": [
            {
                "call": i,
                "shape": [1, 50, 6],
                "latency_ms": 125.0,
                "finite": True,
                "output_tensor_sha256": "9" * 64,
                "first_action_fixture_only": [0.0] * 6,
            }
            for i in (1, 2, 3)
        ],
    }
    return value, spec


def test_successful_vendor_proof_is_checked_against_the_frozen_plan_and_bytes(tmp_path):
    from learning.checks.verify_smolvla_vendor_proof import verify_proof

    value, spec = proof()
    path = tmp_path / "proof.json"
    write_json(path, value)
    assert verify_proof(path, digest(canonical(value) + b"\n"), spec)["actual_forward_calls"] == 3


@pytest.mark.parametrize(
    "change", ["image", "harness", "state", "model", "optimizer", "admission", "calls", "nonfinite"]
)
def test_wrong_or_overclaimed_vendor_proofs_are_rejected(tmp_path, change):
    from learning.checks.verify_smolvla_vendor_proof import verify_proof

    value, spec = proof()
    value = deepcopy(value)
    if change == "image":
        value["image"] = value["image"].replace("b" * 64, "1" * 64)
    elif change == "harness":
        value["diagnostic_code_sha256"] = "2" * 64
    elif change == "state":
        value["declared_state_dimensions"] = 9
    elif change == "model":
        value["verified_assets"]["model"]["model.safetensors"] = "3" * 64
    elif change == "optimizer":
        value["optimizer_steps"] = 1
    elif change == "admission":
        value["ready_for_live_execution"] = True
    elif change == "calls":
        value["actual_forward_calls"] = 2
    else:
        value["forward_calls"][0]["finite"] = False
    path = tmp_path / "proof.json"
    write_json(path, value)
    with pytest.raises(ContractError):
        verify_proof(path, digest(canonical(value) + b"\n"), spec)
