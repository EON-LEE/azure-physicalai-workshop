from copy import deepcopy

import pytest

from learning.checks.fixtures import SCOPE
from learning.common import ContractError


def vendor_config():
    return {
        "type": "smolvla",
        "n_obs_steps": 1,
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "empty_cameras": 0,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            **{
                f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]}
                for i in (1, 2, 3)
            },
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }


def specification():
    workspace = (
        "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/test-rg"
        "/providers/Microsoft.MachineLearningServices/workspaces/existing-ml"
    )
    prefix = f"tenants/{SCOPE.tenant_id}/owners/{SCOPE.owner_id}/learning"
    return {
        "schema": "physicalai.smolvla-vendor-diagnostic/v1",
        "scope": {"tenant_id": SCOPE.tenant_id, "owner_id": SCOPE.owner_id},
        "workspace_id": workspace,
        "job_name": "smol-vendor-compatibility-01",
        "compute_name": "existing-single-gpu",
        "managed_identity_client_id": "33333333-3333-4333-8333-333333333333",
        "storage_account_url": "https://teststorage.blob.core.windows.net",
        "container": "artifacts",
        "vendor_prefix": prefix + "/vendor/exact-version",
        "vendor_inventory_sha256": "b" * 64,
        "output_prefix": prefix + "/outputs/vendor-compatibility-01",
        "image": "testregistry.azurecr.io/physicalai-smolvla@sha256:" + "c" * 64,
        "image_code_snapshot_sha256": "d" * 64,
        "diagnostic_code_sha256": "e" * 64,
        "execution_timeout_seconds": 600,
    }


def test_vendor_diagnostic_keeps_published_six_dimensions_and_three_camera_slots():
    from learning.checks.smolvla_vendor_diagnostic import validate_vendor_configuration

    value = vendor_config()
    validate_vendor_configuration(value)
    assert value["input_features"]["observation.state"]["shape"] == [6]
    assert len(value["input_features"]) == 4


@pytest.mark.parametrize("change", ["franka", "missing-camera", "gr00t", "aloha", "chunk"])
def test_vendor_diagnostic_rejects_relabeling_or_changing_the_published_inputs(change):
    from learning.checks.smolvla_vendor_diagnostic import validate_vendor_configuration

    value = deepcopy(vendor_config())
    if change == "franka":
        value["input_features"]["observation.state"]["shape"] = [9]
        value["output_features"]["action"]["shape"] = [9]
    elif change == "missing-camera":
        value["input_features"].pop("observation.images.camera3")
    elif change == "gr00t":
        value["type"] = "gr00t_n1_5"
    elif change == "aloha":
        value["adapt_to_pi_aloha"] = True
    else:
        value["chunk_size"] = 16
    with pytest.raises(ContractError):
        validate_vendor_configuration(value)


@pytest.mark.parametrize("change", ["owner", "output", "timeout", "image", "credentials"])
def test_one_shot_diagnostic_scope_and_budget_are_explicit(change):
    from learning.checks.smolvla_vendor_diagnostic import validate_specification

    value = specification()
    validate_specification(value)
    if change == "owner":
        value["vendor_prefix"] = value["vendor_prefix"].replace(SCOPE.owner_id, "f" * 64)
    elif change == "output":
        value["output_prefix"] = value["vendor_prefix"]
    elif change == "timeout":
        value["execution_timeout_seconds"] = 601
    elif change == "image":
        value["image"] = "testregistry.azurecr.io/physicalai-smolvla:latest"
    else:
        value["hf_token"] = "not-a-real-token"
    with pytest.raises(ContractError):
        validate_specification(value)


def test_vendor_diagnostic_proof_cannot_be_mistaken_for_training_or_admission():
    from learning.checks.smolvla_vendor_diagnostic import initial_report

    report = initial_report(specification())
    assert report["observation_source"] == "fixture"
    assert report["test_only"] is True
    assert report["declared_state_dimensions"] == report["declared_action_dimensions"] == 6
    assert report["declared_camera_count"] == 3
    assert report["optimizer_steps"] == report["actuator_calls"] == 0
    assert report["franka_adaptation_verified"] is False
    assert report["latency_admission_verified"] is False
    assert report["learning_quality_verified"] is False
    assert report["ready_for_live_execution"] is False
