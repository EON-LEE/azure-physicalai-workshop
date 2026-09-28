"""P1 transport/native-trainer wiring and old-v3 compatibility; no optimizer is run."""

import shutil
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import SCOPE, overwrite
from learning.common import ContractError, file_digest, read_json, write_json
from learning.paused.blob_transfer import PrivateBlobTransfer, input_prefix
from learning.paused.command_artifacts import validate_model
from learning.smolvla.train import TrainOptions
from tests.learning.test_direct_blob_transfer import MemoryContainer
from tests.learning.test_direct_command_jobs import live_job
from tests.learning.test_direct_command_runtime import raw_cohort, upload_fixture
from tests.learning.test_p1_training_cohort import SELECTOR, p0_parent, p1_config, p1_manifest


@pytest.mark.parametrize("missing_selector", [False, True])
def test_additional_raw_fixture_is_never_promoted_to_live_demonstrations(
    tmp_path, missing_selector
):
    from learning.paused.command import _download_inputs

    config, raw = raw_cohort(tmp_path, seeds=range(11001, 11021))
    config["parameters"]["resume_mode"] = "weights_only"
    if not missing_selector:
        config["training_cohort"] = dict(SELECTOR)
    store = MemoryContainer()
    upload_fixture(store, input_prefix(config, "demonstrations"), raw)
    with pytest.raises(ContractError, match="twenty TRAIN seeds" if missing_selector else "live"):
        _download_inputs(
            PrivateBlobTransfer(store, config, remaining=lambda: 60),
            config,
            tmp_path / "download",
        )
    assert not any("/parent_model/" in name for action, name in store.operations)


def test_p1_uses_unchanged_native_validation_before_parent_overlap_admission(tmp_path, monkeypatch):
    from learning.paused import capture
    from learning.paused.command import _download_inputs

    config, raw = raw_cohort(tmp_path, seeds=range(11001, 11021))
    config.update(training_cohort=dict(SELECTOR), run_id="p1-new-command")
    config["parameters"]["resume_mode"] = "weights_only"
    parent_root = tmp_path / "p0"
    parent = p0_parent(parent_root, config)
    manifest = read_json(raw / "manifest.json")
    # Codec metadata aligns with the real raw-schema task; no fixture is admitted outside this test.
    parent["task"] = {
        name: manifest["episodes"][0]["demonstration"][name]
        for name in ("task_id", "instruction", "goal_id")
    }
    parent["training"]["episodes"][0]["episode_id"] = manifest["episodes"][0]["episode_id"]
    overwrite(parent_root / "model.json", parent)
    config["inputs"]["parent_model"]["sha256"] = file_digest(parent_root / "model.json")
    from learning.common import canonical, digest

    config["task_sha256"] = digest(canonical(parent["task"]))
    store = MemoryContainer()
    upload_fixture(store, input_prefix(config, "demonstrations"), raw)
    upload_fixture(store, input_prefix(config, "parent_model"), parent_root)
    original = capture.validate_dataset
    calls = []

    def fixture_only(root, **kwargs):
        calls.append(kwargs)
        assert kwargs["require_live"] is True and kwargs["require_demonstrations"] is True
        return original(root, **{**kwargs, "require_live": False})

    monkeypatch.setattr(capture, "validate_dataset", fixture_only)
    with pytest.raises(ContractError, match="overlaps or relabels"):
        _download_inputs(
            PrivateBlobTransfer(store, config, remaining=lambda: 60),
            config,
            tmp_path / "download",
        )
    assert len(calls) == 1
    assert not any("/backbone/" in name for action, name in store.operations)


def test_p1_native_warm_start_preserves_all_forty_episodes_and_old_v3_shape(tmp_path, monkeypatch):
    from learning.paused import train as paused_train
    from learning.smolvla import checkpoint_runner, train

    config = p1_config()
    config["parameters"].update(max_steps=1000, checkpoint_steps=100)
    parent_root = tmp_path / "p0"
    parent = p0_parent(parent_root, config)
    converted = {
        **p1_manifest(config, parent),
        "schema": "physicalai.lerobot-conversion/v3",
        "raw_manifest_sha256": config["inputs"]["demonstrations"]["sha256"],
        "test_only": False,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
    }
    dataset, backbone = tmp_path / "dataset", tmp_path / "backbone"
    dataset.mkdir()
    backbone.mkdir()
    write_json(dataset / "conversion.json", converted)
    monkeypatch.setattr(paused_train, "validate_conversion", lambda root, scope: converted)
    monkeypatch.setattr(train, "validate_backbone", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(train, "require_lerobot", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda: "unit-double-not-hardware",
            ),
        ),
    )
    monkeypatch.setattr(checkpoint_runner, "native_runtime", lambda **kwargs: {"test_only": True})
    before, after = "1" * 64, "2" * 64
    monkeypatch.setattr(
        train,
        "parameter_fingerprint",
        lambda path: before if "initialization" in str(path) else after,
    )
    stages = []

    def initialize(actual_parent, actual_backbone, actual_dataset, seed):
        stages.append("fresh-seed")
        assert (actual_parent, actual_backbone, actual_dataset) == (parent_root, backbone, dataset)
        shutil.copytree(parent_root / "checkpoint", seed)

    monkeypatch.setattr(train, "prepare_seed", initialize)
    root_job = live_job(config)
    original_config = p1_config()
    original_config.pop("training_cohort")
    original_config.update(
        run_id="p0-completed-command",
        task_sha256=config["task_sha256"],
        control_profile_sha256=config["control_profile_sha256"],
        specification_sha256=parent["training"]["specification_sha256"],
    )
    parent_job = live_job(original_config)
    parent_job.status = "Completed"
    parent_job.tags["code_snapshot_sha256"] = parent["training"]["code_snapshot_sha256"]
    jobs = {root_job.name: root_job, parent_job.name: parent_job}
    client = SimpleNamespace(jobs=SimpleNamespace(get=lambda name: jobs[name]))
    monkeypatch.setenv("AZUREML_RUN_ID", config["run_id"])
    output = tmp_path / "trained"
    invocation = []

    def native_cli(command, **kwargs):
        invocation.append(command)
        assert "--resume=true" not in command and not any(
            "--config_path=" in arg for arg in command
        )
        assert f"--policy.path={(output / 'initialization').resolve()}" in command
        assert command[2] == "learning.smolvla.checkpoint_runner"
        context = read_json(output / "checkpoint-run.json")
        assert context["prior_optimizer_steps"] == 1000
        assert context["resume_checkpoint_root"] is None
        assert context["resume_from_checkpoint_sha256"] is None
        assert context["origin"]["azure_job_id"] == root_job.id
        assert context["origin"]["azure_job_type"] == "command"
        assert context["training_parameters"] == asdict(TrainOptions(**config["parameters"]))
        step = output / "training" / "checkpoints" / "001000"
        shutil.copytree(output / "initialization", step / "pretrained_model")
        (step / "pretrained_model" / "model.safetensors").write_bytes(b"codec-updated-not-weights")
        (step / "training_state").mkdir()
        write_json(step / "training_state" / "training_step.json", {"step": 1000})

    monkeypatch.setattr(train.subprocess, "run", native_cli)
    model = paused_train.run_training(
        dataset,
        parent_root,
        backbone,
        output,
        scope=SCOPE,
        parent_model_sha256=config["inputs"]["parent_model"]["sha256"],
        conversion_sha256=file_digest(dataset / "conversion.json"),
        code_snapshot_sha256="b" * 64,
        config=config,
        client=client,
        options=TrainOptions(**config["parameters"]),
    )
    assert stages == ["fresh-seed"] and len(invocation) == 1
    result = read_json(output / "result.json")
    assert (
        validate_model(
            output / result["candidate"],
            expected_scope=SCOPE,
            expected_model_sha256=result["model_manifest_sha256"],
        )
        == model
    )
    training = model["training"]
    assert len(training["episodes"]) == len({ep["episode_id"] for ep in training["episodes"]}) == 40
    assert {ep["seed"] for ep in training["episodes"]} == set(range(10001, 10021)) | set(
        range(11001, 11021)
    )
    assert training["parent_model_sha256"] == config["inputs"]["parent_model"]["sha256"]
    assert training["parent_weights_sha256"] == parent["weights_sha256"]
    assert (
        training["resume_mode"] == "weights_only" and training["optimizer_state_restored"] is False
    )
    assert training["bitwise_continuation_claimed"] is False
    assert (
        training["resume_from_checkpoint_sha256"] is training["resume_from_checkpoint_step"] is None
    )
    assert training["resume_from_job_id"] == parent_job.id
    assert training["resume_from_job_type"] == "command"
    assert (
        training["optimizer_steps"]
        == training["checkpoint_step"]
        == result["optimizer_steps"]
        == 1000
    )
    assert training["cumulative_optimizer_steps"] == 2000
    assert training["azure_job_id"] == root_job.id
    assert "azure_pipeline_job_id" not in training and "azure_component_job_id" not in training
    assert "training_cohort" not in model and "training_cohort" not in result
    assert result["learning_quality_verified"] is False
    from learning.paused.command_artifacts import validate_models
    from tests.learning.test_paused_evaluation import plan

    comparison = plan()
    comparison.update(
        policy_before_sha256=config["inputs"]["parent_model"]["sha256"],
        policy_after_sha256=result["model_manifest_sha256"],
    )
    models = validate_models(
        comparison,
        {"before": parent_root, "after": output / result["candidate"]},
        scope=SCOPE,
    )
    assert models["before"] == parent and models["after"] == model
