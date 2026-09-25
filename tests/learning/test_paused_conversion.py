import json

import pytest

from learning.checks.fixtures import SCOPE
from learning.common import ContractError, file_digest, read_json
from tests.learning.test_conversion_training import mock_converter as mock_converter
from tests.learning.test_paused_capture import complete


def test_paused_conversion_uses_actual_simulation_time_and_keeps_original_wall_sidecar(
    tmp_path, mock_converter
):
    from learning.paused.dataset import convert_dataset, validate_conversion

    source = complete(tmp_path / "raw")
    output = tmp_path / "converted"
    value = convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
        allow_test_fixture=True,
    )
    assert value["schema"] == "physicalai.lerobot-conversion/v3"
    assert value["timestamp_basis"] == "simulation_time"
    assert value["real_time_admission"] is False
    assert value["test_only"] is True
    frames = mock_converter[0].episodes[0]
    assert mock_converter[0].options["fps"] == 10
    assert set(frames[0]) == {
        "observation.state",
        "action",
        "task",
        "observation.images.inspection",
        "observation.images.overview",
    }
    timing = [
        json.loads(line) for line in (output / "source-timing.jsonl").read_text().splitlines()
    ]
    assert timing[1]["monotonic_ns"] - timing[0]["monotonic_ns"] == 2_000_000_000
    assert timing[1]["physics_step"] - timing[0]["physics_step"] == 6
    assert timing[1]["simulation_time_numerator"] == 11
    assert timing[1]["simulation_time_denominator"] == 10
    assert validate_conversion(output, SCOPE) == read_json(output / "conversion.json")


def test_wrong_frozen_plan_rejected_before_optional_ml_import(tmp_path, monkeypatch):
    from learning import convert
    from learning.paused.dataset import convert_dataset

    source = complete(tmp_path / "raw")
    monkeypatch.setattr(convert, "require_lerobot", lambda: pytest.fail("No ML before binding"))
    with pytest.raises(ContractError, match="frozen"):
        convert_dataset(
            source,
            tmp_path / "out",
            expected_scope=SCOPE,
            expected_manifest_sha256=file_digest(source / "manifest.json"),
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="0" * 64,
            allow_test_fixture=True,
        )
    assert not (tmp_path / "out").exists()


def test_legacy_training_conversion_validator_rejects_paused_data(tmp_path, mock_converter):
    from learning.paused.dataset import convert_dataset
    from learning.train import validate_conversion

    source = complete(tmp_path / "raw")
    output = tmp_path / "converted"
    convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
        allow_test_fixture=True,
    )
    with pytest.raises(ContractError):
        validate_conversion(output, SCOPE)


def test_final_heldout_seed_cannot_enter_training_by_changing_its_split(tmp_path, monkeypatch):
    from learning import convert
    from learning.paused.dataset import convert_dataset
    from tests.learning.test_paused_capture import shifted_sample, writer

    captured = writer(tmp_path / "raw", split="train", seed=30001)
    captured.append(shifted_sample(0))
    captured.append(shifted_sample(1, terminal=True))
    captured.finalize()
    monkeypatch.setattr(
        convert, "require_lerobot", lambda: pytest.fail("No heldout training import")
    )
    with pytest.raises(ContractError, match="held.out"):
        convert_dataset(
            captured.root,
            tmp_path / "model-input",
            expected_scope=SCOPE,
            expected_manifest_sha256=file_digest(captured.root / "manifest.json"),
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="e" * 64,
            allow_test_fixture=True,
        )
