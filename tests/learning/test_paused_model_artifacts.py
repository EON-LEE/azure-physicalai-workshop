from pathlib import Path

import pytest

from learning.checks.fixtures import SCOPE
from learning.checks.groot_fixtures import TASK
from learning.common import ContractError, file_digest, read_json, write_json
from learning.smolvla.adaptation import adapted_config
from tests.learning.test_paused_contract import profile
from tests.learning.test_smolvla_contract import base_config


def fixture(root):
    from learning.paused.artifacts import model_contract

    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"codec-unit-test-not-real-model-weights")
    write_json(
        checkpoint / "config.json",
        adapted_config(base_config(), backbone_path=Path("/local/backbone")),
    )
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        write_json(checkpoint / name, {"steps": [{"registry_name": "normalizer_processor"}]})
    job = (
        "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/unit/"
        "providers/Microsoft.MachineLearningServices/workspaces/unit/jobs/"
    )
    training = {
        key: "a" * 64
        for key in (
            "parent_model_sha256",
            "parent_weights_sha256",
            "raw_manifest_sha256",
            "conversion_sha256",
            "config_sha256",
            "code_snapshot_sha256",
            "specification_sha256",
            "updated_parameter_sample_before",
        )
    }
    training.update(
        updated_parameter_sample_after="b" * 64,
        optimizer_steps=1,
        cumulative_optimizer_steps=1,
        checkpoint_step=1,
        resume_mode="new",
        test_only=False,
        episodes=[{"episode_id": "unit-episode"}],
        azure_job_id=job + "component",
        azure_pipeline_job_id=job + "pipeline",
        gpu={"cuda": True, "name": "unit-metadata-double-not-hardware"},
        loss=None,
        execution_timing="paused_simulation",
        real_time_admission=False,
        raw_schema="physicalai.demonstrations/v3",
        conversion_schema="physicalai.lerobot-conversion/v3",
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64,
    )
    value = model_contract(
        checkpoint=checkpoint,
        scope=SCOPE,
        profile=profile(),
        task=TASK,
        backbone_manifest_sha256="f" * 64,
        role="candidate",
        training=training,
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64,
    )
    write_json(root / "model.json", value)
    return file_digest(root / "model.json")


def test_model_v2_binds_paused_stats_profile_and_real_training_lineage_shape(tmp_path):
    from learning.paused.artifacts import validate_model

    checksum = fixture(tmp_path)
    value = validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)
    assert value["schema"] == "physicalai.smolvla-checkpoint/v2"
    assert value["execution_timing"] == "paused_simulation"
    assert value["real_time_admission"] is False
    assert value["timestamp_basis"] == "simulation_time"
    assert value["criteria_sha256"] == "d" * 64
    config = read_json(tmp_path / "checkpoint" / "config.json")
    assert config["output_features"]["action"]["shape"] == [9]
    assert (
        len([key for key in config["input_features"] if key.startswith("observation.images.")]) == 2
    )


@pytest.mark.parametrize(
    "change",
    ["rt-flag", "clock", "raw-version", "conversion-version", "criteria", "plan", "no-update"],
)
def test_paused_model_cannot_relabel_real_time_or_unupdated_weights(tmp_path, change):
    from learning.checks.fixtures import overwrite
    from learning.paused.artifacts import validate_model

    fixture(tmp_path)
    value = read_json(tmp_path / "model.json")
    if change == "rt-flag":
        value["real_time_admission"] = True
    elif change == "clock":
        value["timestamp_basis"] = "wall_time"
    elif change == "raw-version":
        value["training"]["raw_schema"] = "physicalai.demonstrations/v2"
    elif change == "conversion-version":
        value["training"]["conversion_schema"] = "physicalai.lerobot-conversion/v2"
    elif change == "criteria":
        value["training"]["criteria_sha256"] = "0" * 64
    elif change == "plan":
        value["training"]["frozen_plan_sha256"] = "0" * 64
    else:
        value["training"]["updated_parameter_sample_after"] = value["training"][
            "updated_parameter_sample_before"
        ]
    overwrite(tmp_path / "model.json", value)
    with pytest.raises(ContractError):
        validate_model(
            tmp_path,
            expected_scope=SCOPE,
            expected_model_sha256=file_digest(tmp_path / "model.json"),
        )


def test_real_time_model_validator_refuses_paused_checkpoint(tmp_path):
    from learning.smolvla.artifacts import validate_model

    checksum = fixture(tmp_path)
    with pytest.raises(ContractError):
        validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)


def test_paused_loader_uses_v2_validator_before_actual_native_weight_loading(tmp_path, monkeypatch):
    from learning.paused.model import LocalPausedSmolVLAPolicy
    from learning.smolvla.inference import LocalSmolVLAPolicy

    checksum = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(
        LocalSmolVLAPolicy, "_load", lambda self, root, **kwargs: calls.append(kwargs)
    )
    LocalPausedSmolVLAPolicy(
        tmp_path,
        backbone_root=tmp_path / "backbone",
        scope=SCOPE,
        model_sha256=checksum,
        expected_control_profile_sha256=profile().sha256,
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
    )
    assert len(calls) == 1
    assert calls[0]["metadata"]["schema"] == "physicalai.smolvla-checkpoint/v2"
    assert calls[0]["profile"].execution_timing == "paused_simulation"
    with pytest.raises(ContractError, match="criteria"):
        LocalPausedSmolVLAPolicy(
            tmp_path,
            backbone_root=tmp_path / "backbone",
            scope=SCOPE,
            model_sha256=checksum,
            expected_control_profile_sha256=profile().sha256,
            expected_criteria_sha256="0" * 64,
            expected_frozen_plan_sha256="e" * 64,
        )
    assert len(calls) == 1
