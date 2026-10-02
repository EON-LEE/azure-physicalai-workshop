"""Explicit model admission/producer fixtures; no real model weights or GPU are used."""

from dataclasses import asdict
from functools import partial

import pytest

from learning.checks.fixtures import SCOPE, overwrite
from learning.common import ContractError, file_digest, read_json, write_json
from tests.learning.test_direct_command_provenance import command_model
from tests.learning.test_paused_contract import profile
from tests.learning.test_paused_evaluation import plan
from tests.learning.test_paused_model_artifacts import fixture


def test_new_native_constructor_validates_original_v3_and_inherits_unmodified_actuator_methods(
    tmp_path,
    monkeypatch,
):
    from learning.paused.command_model import LocalCommandPausedSmolVLAPolicy
    from learning.paused.model import LocalPausedSmolVLAPolicy
    from learning.smolvla.inference import LocalSmolVLAPolicy

    metadata = command_model(tmp_path)
    calls = []
    monkeypatch.setattr(
        LocalSmolVLAPolicy, "_load", lambda self, root, **kwargs: calls.append(kwargs)
    )
    kwargs = dict(
        backbone_root=tmp_path / "backbone",
        scope=SCOPE,
        model_sha256=file_digest(tmp_path / "model.json"),
        expected_control_profile_sha256=profile().sha256,
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
    )
    policy = LocalCommandPausedSmolVLAPolicy(tmp_path, **kwargs)
    assert policy.model_admission == "azureml_command_v3"
    assert calls[0]["metadata"] == metadata
    assert calls[0]["metadata"]["schema"] == "physicalai.smolvla-checkpoint/v3"
    assert "azure_pipeline_job_id" not in calls[0]["metadata"]["training"]
    assert LocalCommandPausedSmolVLAPolicy.predict_chunk is LocalPausedSmolVLAPolicy.predict_chunk
    assert LocalCommandPausedSmolVLAPolicy.reset is LocalSmolVLAPolicy.reset
    with pytest.raises(ContractError):
        LocalPausedSmolVLAPolicy(tmp_path, **kwargs)
    with pytest.raises(ContractError, match="criteria"):
        LocalCommandPausedSmolVLAPolicy(
            tmp_path,
            **{**kwargs, "expected_criteria_sha256": "0" * 64},
        )
    assert len(calls) == 1


def test_new_entry_keeps_original_profile_guard_before_any_native_weight_load(tmp_path):
    from learning.paused.command_model import LocalCommandPausedSmolVLAPolicy

    command_model(tmp_path)
    with pytest.raises(ContractError, match="servo profile"):
        LocalCommandPausedSmolVLAPolicy(
            tmp_path,
            backbone_root=tmp_path / "not-loaded",
            scope=SCOPE,
            model_sha256=file_digest(tmp_path / "model.json"),
            expected_control_profile_sha256="0" * 64,
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="e" * 64,
        )


def test_new_admission_never_implicitly_accepts_legacy_model(tmp_path):
    from learning.paused.command_artifacts import validate_model

    checksum = fixture(tmp_path)
    with pytest.raises(ContractError, match="command"):
        validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)


@pytest.mark.parametrize("change", [None, "lineage", "heldout", "criteria", "task", "missing-role"])
def test_new_paired_gate_preserves_model_conditions_ancestry_and_heldout_leak_rules(
    tmp_path, change
):
    from learning.paused.command_artifacts import validate_models

    roots = {"before": tmp_path / "before", "after": tmp_path / "after"}
    before = command_model(roots["before"])
    after = command_model(roots["after"])
    before_sha = file_digest(roots["before"] / "model.json")
    after["training"]["parent_model_sha256"] = before_sha
    after["training"]["azure_job_id"] += "-after"
    if change == "lineage":
        after["training"]["parent_model_sha256"] = "0" * 64
    elif change == "heldout":
        after["training"]["episodes"][0]["seed"] = 30001
    elif change == "criteria":
        after["criteria_sha256"] = after["training"]["criteria_sha256"] = "0" * 64
    elif change == "task":
        after["task"]["instruction"] = "Different task"
    overwrite(roots["after"] / "model.json", after)
    value = plan()
    value.update(
        policy_before_sha256=before_sha,
        policy_after_sha256=file_digest(roots["after"] / "model.json"),
    )
    if change == "missing-role":
        roots.pop("after")
    if change is None:
        assert validate_models(value, roots, scope=SCOPE) == {"before": before, "after": after}
    else:
        with pytest.raises(ContractError):
            validate_models(value, roots, scope=SCOPE)


@pytest.mark.parametrize("resumed", [False, True])
def test_same_native_sealer_produces_real_single_job_fields_and_exact_new_update_counts(
    tmp_path,
    monkeypatch,
    resumed,
):
    from learning.paused.command_artifacts import model_contract, validate_model
    from learning.smolvla import train

    source_root = tmp_path / "codec-source"
    source = command_model(source_root)
    step_dir = tmp_path / "native" / "001000"
    step_dir.mkdir(parents=True)
    import shutil

    shutil.copytree(source_root / "checkpoint", step_dir / "pretrained_model")
    (step_dir / "training_state").mkdir()
    write_json(step_dir / "training_state" / "training_step.json", {"step": 1000})
    parent = {**source, "role": "pretrained", "training": None, "weights_sha256": "a" * 64}
    config_profile = profile()
    metadata = {
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
    }
    converted = {
        "schema": "physicalai.lerobot-conversion/v3",
        "raw_manifest_sha256": "c" * 64,
        "control_profile": asdict(config_profile),
        "episodes": source["training"]["episodes"],
    }
    binding = {
        "azure_job_id": source["training"]["azure_job_id"] + "-new",
        "azure_job_type": "command",
        "specification_sha256": "d" * 64,
    }
    previous = (
        {
            "step": 400,
            "cumulative_optimizer_steps": 700,
            "origin": {
                "azure_job_id": source["training"]["azure_job_id"],
                "azure_job_type": "command",
            },
            "files": {"pretrained_model/model.safetensors": {"sha256": "a" * 64}},
        }
        if resumed
        else None
    )
    monkeypatch.setattr(train, "parameter_fingerprint", lambda path: "f" * 64)
    output = tmp_path / "sealed"
    train._seal_checkpoint(
        step_dir,
        output,
        step=1000,
        scope=SCOPE,
        parent=parent,
        converted=converted,
        profile=config_profile,
        options=train.TrainOptions(
            max_steps=1000, checkpoint_steps=100, resume_mode="full_state" if resumed else "new"
        ),
        binding=binding,
        gpu={"cuda": True, "device_count": 1, "name": "codec-double-not-GPU"},
        before="e" * 64,
        parent_model_sha256="a" * 64,
        conversion_sha256="b" * 64,
        code_snapshot_sha256="c" * 64,
        model_builder=partial(
            model_contract, criteria_sha256="d" * 64, frozen_plan_sha256="e" * 64
        ),
        model_validator=validate_model,
        mode_metadata=metadata,
        resumed=previous,
        resume_manifest_sha256="f" * 64 if resumed else None,
    )
    result = read_json(output / "result.json")
    model = validate_model(
        output / result["candidate"],
        expected_scope=SCOPE,
        expected_model_sha256=result["model_manifest_sha256"],
    )
    assert result["schema"] == "physicalai.smolvla-command-training-result/v1"
    assert result["azure_job_id"] == binding["azure_job_id"]
    assert result["azure_job_type"] == "command"
    assert result["optimizer_steps"] == (600 if resumed else 1000)
    assert model["training"]["cumulative_optimizer_steps"] == (1300 if resumed else 1000)
    assert model["training"]["checkpoint_step"] == 1000
    assert "azure_pipeline_job_id" not in model["training"]
    assert "azure_component_job_id" not in result
    assert model["training_execution"] == "azureml_command"
    assert result["learning_quality_verified"] is False
