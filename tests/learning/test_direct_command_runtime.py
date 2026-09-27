"""Local storage/native-boundary doubles; fixtures cannot be admitted as live TRAIN."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import SCOPE, overwrite
from learning.common import (
    ContractError,
    canonical,
    digest,
    file_digest,
    inventory,
    read_json,
    write_json,
)
from learning.paused.blob_transfer import PrivateBlobTransfer, input_prefix
from tests.learning.test_direct_blob_transfer import MemoryContainer
from tests.learning.test_direct_command_jobs import command_config, live_job
from tests.learning.test_paused_capture import shifted_sample, writer
from tests.learning.test_training_checkpoints import native_bundle


def raw_cohort(tmp_path):
    from learning.paused.capture import assemble_dataset

    roots = []
    for seed in range(10001, 10021):
        name = f"codec-unit-{seed}"
        root = tmp_path / name
        capture = writer(root, episode_id=name, seed=seed)
        for index in range(2):
            sample = shifted_sample(index, terminal=index == 1)
            capture.append(
                replace(sample, observation=replace(sample.observation, episode_id=name))
            )
        capture.finalize()
        roots.append(root)
    output = tmp_path / "raw"
    assemble_dataset(
        roots, output, dataset_id="codec-unit-data", expected_scope=SCOPE, require_live=False
    )
    config = command_config()
    config["inputs"]["demonstrations"]["sha256"] = file_digest(output / "manifest.json")
    manifest = read_json(output / "manifest.json")
    from learning.paused.contract import PausedControlProfile

    config["control_profile_sha256"] = PausedControlProfile(**manifest["control_profile"]).sha256
    config["task_sha256"] = digest(
        canonical(
            {
                name: manifest["episodes"][0]["demonstration"][name]
                for name in ("task_id", "instruction", "goal_id")
            }
        )
    )
    return config, output


def upload_fixture(store, prefix, root):
    for name in inventory(root):
        store.data[prefix + "/" + name] = (root / name).read_bytes()


@pytest.mark.parametrize("change", ["fixture", "holdout", "missing-episode", "task", "criteria"])
def test_all_twenty_raw_episodes_are_native_validated_before_parent_or_optimizer(tmp_path, change):
    from learning.paused.command import _download_inputs

    config, raw = raw_cohort(tmp_path)
    manifest = read_json(raw / "manifest.json")
    if change == "holdout":
        manifest["episodes"][0]["seed"] = 30001
    elif change == "missing-episode":
        manifest["episodes"].pop()
    elif change == "task":
        config["task_sha256"] = "0" * 64
    elif change == "criteria":
        config["criteria_sha256"] = "0" * 64
    overwrite(raw / "manifest.json", manifest)
    config["inputs"]["demonstrations"]["sha256"] = file_digest(raw / "manifest.json")
    store = MemoryContainer()
    prefix = input_prefix(config, "demonstrations")
    upload_fixture(store, prefix, raw)
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 60)
    with pytest.raises(ContractError):
        _download_inputs(transfer, config, tmp_path / "download")
    assert not any("/parent_model/" in name for _, name in store.operations)
    if change == "fixture":
        assert (tmp_path / "download" / "raw" / "manifest.json").is_file()
        assert (
            len(
                [
                    name
                    for action, name in store.operations
                    if action == "read" and name.endswith(".png")
                ]
            )
            == 80
        )


def test_input_transfer_consumes_only_signed_original_raw_parent_and_backbone(
    tmp_path, monkeypatch
):
    """Model/camera doubles test byte transport and real header guards, not learning quality."""
    from learning.paused import capture, command_artifacts
    from learning.paused.command import _download_inputs
    from learning.smolvla import UPSTREAM
    from learning.smolvla import artifacts as shared

    config, raw = raw_cohort(tmp_path)
    store = MemoryContainer()
    upload_fixture(store, input_prefix(config, "demonstrations"), raw)
    manifest = read_json(raw / "manifest.json")
    parent_root, backbone_root = tmp_path / "parent-fixture", tmp_path / "backbone-fixture"
    (parent_root / "checkpoint").mkdir(parents=True)
    (parent_root / "checkpoint" / "model.safetensors").write_bytes(b"unit-not-model-weights")
    # Candidate input has no implicit unsigned extra license; pretrained license remains pinned.
    parent = {
        "schema": "physicalai.smolvla-checkpoint/v3",
        "role": "candidate",
        "scope": manifest["scope"],
        "control_profile": manifest["control_profile"],
        "criteria_sha256": config["criteria_sha256"],
        "frozen_plan_sha256": config["frozen_plan_sha256"],
        "checkpoint_files": inventory(parent_root / "checkpoint"),
    }
    backbone_root.mkdir()
    (backbone_root / "fixture.bin").write_bytes(b"unit-not-backbone")
    write_json(
        backbone_root / "backbone.json",
        {
            "schema": "physicalai.smolvla-backbone/v1",
            "scope": manifest["scope"],
            "upstream": UPSTREAM,
            "files": inventory(backbone_root),
        },
    )
    config["inputs"]["backbone"]["sha256"] = file_digest(backbone_root / "backbone.json")
    parent["backbone_manifest_sha256"] = config["inputs"]["backbone"]["sha256"]
    write_json(parent_root / "model.json", parent)
    config["inputs"]["parent_model"]["sha256"] = file_digest(parent_root / "model.json")
    upload_fixture(store, input_prefix(config, "parent_model"), parent_root)
    upload_fixture(store, input_prefix(config, "backbone"), backbone_root)
    calls = []

    def raw_validator(root, **kwargs):
        calls.append(("raw", kwargs))
        assert kwargs["require_live"] is True and kwargs["require_demonstrations"] is True
        assert inventory(root) == inventory(raw)
        return SimpleNamespace(manifest=manifest)

    def parent_validator(root, **kwargs):
        calls.append(("parent", kwargs))
        assert inventory(root) == inventory(parent_root)
        return parent

    def backbone_validator(root, **kwargs):
        calls.append(("backbone", kwargs))
        assert inventory(root) == inventory(backbone_root)
        return root

    monkeypatch.setattr(capture, "validate_dataset", raw_validator)
    monkeypatch.setattr(command_artifacts, "validate_parent", parent_validator)
    monkeypatch.setattr(shared, "validate_backbone", backbone_validator)
    transferred = _download_inputs(
        PrivateBlobTransfer(store, config, remaining=lambda: 60),
        config,
        tmp_path / "task",
    )
    assert [name for name, _ in calls] == ["raw", "parent", "backbone"]
    assert transferred[3] is None
    assert calls[1][1]["for_inference"] is False


def test_registered_inputs_are_checked_without_code_or_data_asset_creation():
    from learning.paused.command import _registered_inputs

    config = command_config()
    seen = []

    def read(name, *, version):
        seen.append((name, version))
        expected = next(asset for asset in config["inputs"].values() if asset["name"] == name)
        return SimpleNamespace(type="uri_folder", path=expected["uri"] + "/")

    store = SimpleNamespace(
        type="AzureBlob",
        account_name=config["storage_account_name"],
        container_name=config["blob_container"],
        credentials=SimpleNamespace(type="None"),
    )
    client = SimpleNamespace(
        data=SimpleNamespace(get=read), datastores=SimpleNamespace(get=lambda name: store)
    )
    _registered_inputs(client, config)
    assert len(seen) == 3
    store.credentials.type = "AccountKey"
    with pytest.raises(ContractError, match="keyless"):
        _registered_inputs(client, config)


def test_candidate_publication_omits_mutable_native_tree_and_seals_result_last(tmp_path):
    config = command_config()
    root = tmp_path / "output"
    candidate = root / "candidates" / "step-001000"
    (candidate / "checkpoint").mkdir(parents=True)
    (candidate / "checkpoint" / "model.safetensors").write_bytes(b"unit-fixture-only")
    write_json(candidate / "model.json", {"unit_test": True})
    native_bundle(root / "training" / "checkpoints" / "001000", step=1000)
    (root / "training" / "checkpoints" / "last").symlink_to("001000", target_is_directory=True)
    write_json(root / "training-context.json", {"unit_test": True})
    (root / "training.log").write_bytes(b"fixture trainer boundary")
    write_json(
        root / "result.json",
        {
            "candidate": "candidates/step-001000",
            "model_manifest_sha256": file_digest(candidate / "model.json"),
        },
    )
    store = MemoryContainer()
    target = config["output_prefix"] + "/" + config["run_id"] + "/model"
    PrivateBlobTransfer(store, config, remaining=lambda: 60).publish_candidate(root, target)
    assert not any("/training/" in name or "/initialization/" in name for name in store.data)
    writes = [name for action, name in store.operations if action == "write"]
    assert writes[-1] == target + "/result.json"
    assert all(("read", name) in store.operations for name in writes)


def test_failed_training_preserves_error_log_without_success_shaped_model(tmp_path, capsys):
    from learning.paused.command import _preserve_failure

    config = command_config()
    root = tmp_path / "task"
    (root / "model").mkdir(parents=True)
    (root / "model" / "training.log").write_text("native optimizer failed: unit test")
    write_json(root / "model" / "training-context.json", {"unit_test": True})
    store = MemoryContainer()
    _preserve_failure(
        PrivateBlobTransfer(store, config, remaining=lambda: 60),
        config,
        root,
        {"azure_job_id": live_job(config).id, "azure_job_type": "command"},
        RuntimeError("fixture"),
    )
    assert "optimizer failed" in capsys.readouterr().err
    assert any(name.endswith("/failure.json") for name in store.data)
    assert any(name.endswith("/training.log") for name in store.data)
    assert not any(name.endswith(("/result.json", "/model.json")) for name in store.data)


@pytest.mark.parametrize("wrong_conversion", [False, True])
def test_one_command_wires_existing_native_conversion_training_and_private_publication(
    tmp_path,
    monkeypatch,
    wrong_conversion,
):
    from learning.paused import command, dataset, train
    from tests.learning.test_direct_command_provenance import command_model

    config, raw = raw_cohort(tmp_path)
    raw_manifest = read_json(raw / "manifest.json")
    directory = tmp_path / "task"
    directory.mkdir()
    store = MemoryContainer()
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 60)
    job = live_job(config)
    client = SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job))
    monkeypatch.setenv("AZUREML_RUN_ID", config["run_id"])
    monkeypatch.setattr(command, "_registered_inputs", lambda client, config: None)
    monkeypatch.setattr(
        command,
        "_download_inputs",
        lambda transfer, config, root: (raw, root / "parent", root / "backbone", None),
    )
    stages = []

    def convert(source, output, **kwargs):
        stages.append("convert")
        assert (
            source == raw
            and kwargs["expected_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"]
        )
        output.mkdir(parents=True)
        (output / "data.bin").write_bytes(b"unit-converted-double")
        value = {
            "scope": raw_manifest["scope"],
            "control_profile": raw_manifest["control_profile"],
            "raw_manifest_sha256": "0" * 64
            if wrong_conversion
            else config["inputs"]["demonstrations"]["sha256"],
            "episodes": raw_manifest["episodes"],
        }
        write_json(output / "conversion.json", value)
        return value

    def native(dataset_root, parent, backbone, output, **kwargs):
        stages.append("native")
        assert kwargs["config"] == config and kwargs["client"] is client
        assert kwargs["resume_checkpoint"] is None and kwargs["code_snapshot_sha256"] == "b" * 64
        assert dataset_root == directory / "converted" / "dataset"
        assert kwargs["options"].max_steps == config["parameters"]["max_steps"]
        candidate = output / "candidates" / "step-000100"
        command_model(candidate)
        write_json(output / "training-context.json", {"test_only": True})
        (output / "training.log").write_bytes(b"Native-boundary double; no model trained.")
        write_json(
            output / "result.json",
            {
                "schema": command.RESULT_SCHEMA,
                "azure_job_id": job.id,
                "azure_job_type": "command",
                "candidate": "candidates/step-000100",
                "optimizer_steps": 100,
                "model_manifest_sha256": file_digest(candidate / "model.json"),
                "specification_sha256": config["specification_sha256"],
                "learning_quality_verified": False,
            },
        )

    monkeypatch.setattr(dataset, "convert_dataset", convert)
    monkeypatch.setattr(
        dataset, "validate_conversion", lambda root, scope: read_json(root / "conversion.json")
    )
    monkeypatch.setattr(train, "run_training", native)
    if wrong_conversion:
        with pytest.raises(ContractError, match="raw dataset/profile"):
            command.run_workload(config, client, transfer, directory, "b" * 64)
        assert stages == ["convert"]
        assert not any(name.endswith("/model/result.json") for name in store.data)
    else:
        result = command.run_workload(config, client, transfer, directory, "b" * 64)
        assert result["azure_job_type"] == "command" and stages == ["convert", "native"]
        prefix = config["output_prefix"] + "/" + config["run_id"]
        assert prefix + "/dataset/dataset/conversion.json" in store.data
        assert prefix + "/model/result.json" in store.data
        assert prefix + "/transfer/completion.json" in store.data
