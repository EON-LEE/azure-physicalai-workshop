from dataclasses import replace

import pytest

from learning.common import ContractError


def test_real_training_options_are_bounded_and_incremental_on_spot():
    from learning.gr00t.train import Gr00tTrainOptions

    options = Gr00tTrainOptions(max_steps=100, checkpoint_steps=10, compute_tier="LowPriority")
    options.validate()
    for changes in (
        {"max_steps": 0},
        {"batch_size": 0},
        {"timeout_seconds": 0},
        {"compute_tier": "auto"},
        {"checkpoint_steps": 101},
        {"compute_tier": "LowPriority", "checkpoint_steps": 100},
        {"learning_rate": float("nan")},
        {"resume_mode": "pickle"},
    ):
        with pytest.raises(ContractError):
            replace(options, **changes).validate()


def test_upstream_training_argument_contract_is_real_and_does_not_publish(tmp_path):
    from learning.gr00t.train import Gr00tTrainOptions, training_arguments

    arguments = training_arguments(tmp_path, Gr00tTrainOptions(max_steps=100, checkpoint_steps=10))
    assert arguments["save_strategy"] == "steps"
    assert arguments["save_steps"] == 10
    assert arguments["save_safetensors"] is True and arguments["save_only_model"] is True
    assert arguments["push_to_hub"] is False and arguments["report_to"] == []
    assert arguments["bf16"] is True
    assert arguments["max_steps"] == 100


def test_physical_action_decode_never_slices_wrong_embodiment_outputs():
    from learning.gr00t.inference import physical_actions

    result = physical_actions(
        {"action.arm": [[0, -0.5, 0, -2, 0, 1.5, 0.8]] * 16, "action.fingers": [[0.02, 0.02]] * 16}
    )
    assert len(result) == 16 and len(result[0]) == 9
    for invalid in (
        {"action": [[0] * 32] * 16},
        {"action.arm": [[0] * 6] * 16, "action.fingers": [[0.02]] * 16},
        {"action.arm": [[0] * 7] * 15, "action.fingers": [[0.02, 0.02]] * 15},
        {"action.arm": [[float("nan")] * 7] * 16, "action.fingers": [[0.02, 0.02]] * 16},
    ):
        with pytest.raises(ContractError):
            physical_actions(invalid)


def test_gr00t_training_cannot_start_without_real_azure_gpu_context(monkeypatch):
    from learning.gr00t.train import azure_job_identity

    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    with pytest.raises(ContractError, match="Azure ML"):
        azure_job_identity(
            {
                "subscription_id": "22222222-2222-4222-8222-222222222222",
                "resource_group": "rg",
                "workspace": "ml",
            }
        )
