from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from learning import LEROBOT_VERSION, convert, offline, train
from learning.checks.fixtures import SCOPE, overwrite
from learning.common import ContractError, file_digest, read_json

pytest_plugins = ("learning.checks.pytest_fixtures",)


@pytest.fixture
def mock_converter(monkeypatch):
    instances = []
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    numpy = ModuleType("numpy")
    numpy.asarray = lambda value, dtype: value
    numpy.float32 = "float32"
    numpy.uint8 = "uint8"

    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            instance = cls()
            instance.options = kwargs
            instance.frames = []
            instance.episodes = []
            instance.finalized = False
            kwargs["root"].mkdir(parents=True)
            (kwargs["root"] / "metadata.json").write_text('{"test_double":true}')
            instances.append(instance)
            return instance

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.episodes.append(self.frames)
            self.frames = []

        def finalize(self):
            self.finalized = True

    dataset_module.LeRobotDataset = Dataset
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)
    monkeypatch.setattr(convert, "require_lerobot", lambda: None)
    return instances


def test_converter_uses_real_api_shape_and_train_split_only(make_dataset, tmp_path, mock_converter):
    source = make_dataset()
    output = tmp_path / "converted"
    manifest = convert.convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        image_size=32,
        allow_test_fixture=True,
    )
    dataset = mock_converter[0]
    assert dataset.finalized is True
    assert dataset.options["use_videos"] is False
    assert dataset.options["repo_id"] == "azure-local/physicalai-reference-arm"
    assert len(dataset.episodes) == 1
    assert len(dataset.episodes[0]) == 3
    assert set(dataset.episodes[0][0]) == {
        "observation.state",
        "action",
        "observation.images.inspection",
        "observation.images.overview",
        "task",
    }
    assert manifest["test_only"] is True
    assert manifest["episodes"][0]["episode_id"] == "train-episode"
    assert manifest["lerobot_version"] == LEROBOT_VERSION
    assert train.validate_conversion(output, SCOPE) == read_json(output / "conversion.json")


def test_converter_rejects_fixture_without_explicit_test_opt_in(make_capture, tmp_path):
    source = make_capture()
    with pytest.raises(ContractError, match="not live"):
        convert.convert_dataset(
            source,
            tmp_path / "converted",
            expected_scope=SCOPE,
            expected_manifest_sha256=file_digest(source / "manifest.json"),
        )
    assert not (tmp_path / "converted").exists()


def test_converter_validates_hash_before_optional_imports(make_capture, tmp_path, monkeypatch):
    source = make_capture()
    monkeypatch.setattr(convert, "require_lerobot", lambda: pytest.fail("Imported optional ML"))
    with pytest.raises(ContractError, match="Manifest checksum"):
        convert.convert_dataset(
            source,
            tmp_path / "converted",
            expected_scope=SCOPE,
            expected_manifest_sha256="0" * 64,
            allow_test_fixture=True,
        )


def test_heldout_conversion_cannot_enter_training(make_dataset, tmp_path, mock_converter):
    source = make_dataset()
    output = tmp_path / "converted"
    convert.convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        split="test",
        allow_test_fixture=True,
    )
    with pytest.raises(ContractError, match="Only the train split"):
        train.validate_conversion(output, SCOPE)


def test_tampered_converted_data_rejected(make_capture, tmp_path, mock_converter):
    source = make_capture()
    output = tmp_path / "converted"
    convert.convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        allow_test_fixture=True,
    )
    (output / "metadata.json").write_text('{"changed":true}')
    with pytest.raises(ContractError, match="inventory/checksum"):
        train.validate_conversion(output, SCOPE)


def test_training_calls_supported_pinned_entrypoint_with_cloud_disabled(tmp_path):
    options = train.TrainOptions(steps=123, timeout_seconds=456, n_action_steps=1)
    command = train.training_command(tmp_path / "dataset", tmp_path / "output", options)
    assert command[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    assert "--policy.type=act" in command
    assert "--policy.pretrained_backbone_weights=null" in command
    assert "--policy.push_to_hub=false" in command
    assert "--wandb.enable=false" in command
    assert "--eval_freq=0" in command
    assert "--steps=123" in command and "--save_freq=123" in command
    assert not any("hf://" in value or "https://" in value for value in command)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("steps", 1000001),
        ("timeout_seconds", 0),
        ("timeout_seconds", 86401),
        ("n_action_steps", 11),
        ("chunk_size", 0),
        ("device", "auto"),
        ("batch_size", 65),
        ("seed", -1),
        ("device", "cpu"),
    ],
)
def test_training_limits_fail_closed(field, value):
    with pytest.raises(ContractError):
        replace(train.TrainOptions(), **{field: value}).validate()


def test_cpu_training_is_explicitly_smoke_only():
    train.TrainOptions(device="cpu", smoke_test=True, steps=1, batch_size=2).validate()
    with pytest.raises(ContractError, match="test-only budget"):
        train.TrainOptions(device="cpu", smoke_test=True, steps=100, batch_size=2).validate()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "lerobot-train"),
        subprocess.TimeoutExpired("lerobot-train", 1),
    ],
)
def test_training_failures_never_publish_model_manifest(
    make_capture, tmp_path, mock_converter, monkeypatch, failure
):
    source = make_capture()
    converted = tmp_path / "converted"
    convert.convert_dataset(
        source,
        converted,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        allow_test_fixture=True,
    )
    monkeypatch.setattr(train, "require_lerobot", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    def run(*args, **kwargs):
        assert kwargs["check"] is True and kwargs["timeout"] == 1
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        raise failure

    monkeypatch.setattr(train.subprocess, "run", run)
    output = tmp_path / "trained"
    with pytest.raises(type(failure)):
        train.train_policy(
            converted,
            output,
            expected_scope=SCOPE,
            expected_conversion_sha256=file_digest(converted / "conversion.json"),
            model_name="act",
            model_version="1",
            code_snapshot_sha256="f" * 64,
            options=train.TrainOptions(
                device="cpu", smoke_test=True, steps=1, batch_size=2, timeout_seconds=1
            ),
        )
    assert not (output / "model.json").exists()


def test_conversion_split_metadata_cannot_be_relabeled(make_capture, tmp_path, mock_converter):
    source = make_capture()
    output = tmp_path / "converted"
    convert.convert_dataset(
        source,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        allow_test_fixture=True,
    )
    manifest = read_json(output / "conversion.json")
    manifest["episodes"][0]["split"] = "test"
    overwrite(output / "conversion.json", manifest)
    with pytest.raises(ContractError, match="Held-out"):
        train.validate_conversion(output, SCOPE)


def test_offline_flags_are_not_overridable(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    offline.enforce_offline()
    assert offline.os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setenv("HF_TOKEN", "test-not-a-real-token")
    with pytest.raises(ContractError, match="must not be present"):
        offline.enforce_offline()


def test_lerobot_version_is_checked(monkeypatch):
    monkeypatch.setattr(offline, "version", lambda name: "0.6.1")
    with pytest.raises(ContractError, match="locked LeRobot"):
        offline.require_lerobot()


def test_root_learning_imports_do_not_load_optional_ml_dependencies():
    script = (
        "import sys;"
        "import learning.capture, learning.inference, learning.azure, learning.components;"
        "assert not any(name in sys.modules for name in "
        "('torch','lerobot','numpy','azure.ai.ml')); print('lightweight imports only')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, text=True, capture_output=True
    )
    assert result.stdout.strip() == "lightweight imports only"
