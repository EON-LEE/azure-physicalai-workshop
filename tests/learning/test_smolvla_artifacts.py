from pathlib import Path

import pytest

from learning.checks.fixtures import SCOPE
from learning.checks.groot_fixtures import teaching
from learning.common import ContractError, file_digest, write_json


def test_smol_model_checks_scope_digest_before_optional_ml_imports(tmp_path):
    from learning.smolvla.artifacts import validate_model

    write_json(tmp_path / "model.json", {"schema": "not-smol"})
    with pytest.raises(ContractError, match="checksum"):
        validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256="0" * 64)


def test_offline_vendor_import_rejects_wrong_file_before_copy_or_network(tmp_path):
    from learning.smolvla.prepare import verify_vendor_files

    (tmp_path / "model.safetensors").write_bytes(b"not-the-reviewed-smol-weights")
    with pytest.raises(ContractError):
        verify_vendor_files(tmp_path, kind="model")


def test_smol_training_command_calls_real_upstream_and_saves_incrementally():
    from learning.smolvla.train import TrainOptions, training_command

    command = training_command(
        Path("/data"),
        Path("/seed"),
        Path("/output"),
        TrainOptions(max_steps=100, checkpoint_steps=10, compute_tier="LowPriority"),
    )
    assert command[1:3] == ["-m", "lerobot.scripts.lerobot_train"]
    assert "--policy.push_to_hub=false" in command
    assert "--save_freq=10" in command and "--steps=100" in command
    assert "--wandb.enable=false" in command and "--eval_freq=0" in command
    assert not any("gr00t" in arg for arg in command)


def test_smol_export_requires_actual_ten_hz_teacher_profile(tmp_path, monkeypatch):
    from learning.smolvla.dataset import convert_dataset

    raw = teaching(tmp_path / "raw")
    with pytest.raises(ContractError, match="not live"):
        convert_dataset(
            raw,
            tmp_path / "export",
            expected_scope=SCOPE,
            expected_manifest_sha256=file_digest(raw / "manifest.json"),
        )
