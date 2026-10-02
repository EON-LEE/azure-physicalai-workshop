import json
import struct
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import SCOPE
from learning.common import ContractError, read_json, write_json


def tensor_file(path):
    header = json.dumps(
        {"unit": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


def binding():
    return {
        "scope": asdict(SCOPE),
        "raw_manifest_sha256": "a" * 64,
        "conversion_sha256": "b" * 64,
        "parent_model_sha256": "c" * 64,
        "parent_weights_sha256": "d" * 64,
        "control_profile_sha256": "e" * 64,
        "task_sha256": "f" * 64,
        "criteria_sha256": "1" * 64,
        "frozen_plan_sha256": "2" * 64,
        "training_config_sha256": "3" * 64,
        "training_code_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
    }


def origin():
    workspace = (
        "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/approved/"
        "providers/Microsoft.MachineLearningServices/workspaces/learning/jobs/"
    )
    return {
        "azure_job_id": workspace + "original-component",
        "azure_pipeline_job_id": workspace + "original-pipeline",
        "specification_sha256": "6" * 64,
        "code_snapshot_sha256": "7" * 64,
        "job_deadline_utc": "2026-09-26T16:00:00Z",
        "test_only": True,
    }


def native_bundle(root, *, step=10, full=True):
    model, state = root / "pretrained_model", root / "training_state"
    model.mkdir(parents=True)
    state.mkdir()
    tensor_file(model / "model.safetensors")
    for name in (
        "config.json",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        write_json(model / name, {"test_only": True})
    write_json(state / "training_step.json", {"step": step})
    if full:
        for name in (
            "optimizer_state.safetensors",
            "rng_state.safetensors",
            "continuation.safetensors",
        ):
            tensor_file(state / name)
        (state / "optimizer_param_groups.json").write_text('[{"params":[0],"lr":0.0001}]')
        write_json(state / "scheduler_state.json", {"last_epoch": step})
        write_json(
            state / "continuation.json",
            {
                "schema": "physicalai.smolvla-continuation/v1",
                "step": step,
                "sampler": {"schema": "physicalai.smolvla-data-order/v1", "batches": step},
            },
        )
    return root


def seal(root, *, step=10, kind="full_state"):
    from learning.smolvla.checkpoints import seal_checkpoint

    return seal_checkpoint(root, step=step, binding=binding(), origin=origin(), state_kind=kind)


class MemoryBlob:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def get_blob_properties(self, **kwargs):
        if self.name not in self.owner.data:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("missing")
        return SimpleNamespace(size=len(self.owner.data[self.name]), etag="unchanged")

    def download_blob(self, **kwargs):
        self.owner.operations.append(("read", self.name))
        data = self.owner.data[self.name]
        if self.name == self.owner.corrupt_read:
            data = data[:-1] + b"!"
        return SimpleNamespace(chunks=lambda: iter((data,)))


class MemoryContainer:
    def __init__(self):
        self.data, self.operations = {}, []
        self.fail_on = None
        self.corrupt_read = None

    def upload_blob(self, name, data, **kwargs):
        assert kwargs["overwrite"] is False
        self.operations.append(("write", name))
        if name == self.fail_on:
            raise TimeoutError("upload acknowledgement unknown")
        if name in self.data:
            from azure.core.exceptions import ResourceExistsError

            raise ResourceExistsError("already exists")
        self.data[name] = data.read() if hasattr(data, "read") else data

    def get_blob_client(self, name):
        return MemoryBlob(self, name)

    def list_blobs(self, **kwargs):
        return (
            SimpleNamespace(name=name)
            for name in sorted(self.data)
            if name.startswith(kwargs["name_starts_with"])
        )


def prefix():
    return (
        f"tenants/{SCOPE.tenant_id}/owners/{SCOPE.owner_id}/learning/outputs/"
        "approved-job/checkpoints"
    )


def test_only_complete_safe_native_bundle_gets_manifest_last(tmp_path):
    from learning.smolvla.checkpoints import validate_checkpoint

    root = native_bundle(tmp_path / "checkpoint")
    checksum = seal(root)
    manifest = validate_checkpoint(root, expected_sha256=checksum, expected_binding=binding())
    assert manifest["step"] == 10
    assert manifest["state_kind"] == "full_state"
    assert manifest["origin"]["test_only"] is True
    assert manifest["learning_quality_verified"] is False
    assert manifest["files"]["training_state/optimizer_state.safetensors"]["sha256"]
    assert not any("pickle" in name for name in manifest["files"])


@pytest.mark.parametrize(
    "missing",
    [
        "pretrained_model/model.safetensors",
        "training_state/optimizer_param_groups.json",
        "training_state/scheduler_state.json",
        "training_state/continuation.json",
    ],
)
def test_partial_step_marker_never_implies_complete_state(tmp_path, missing):
    root = native_bundle(tmp_path / "partial")
    (root / missing).unlink()
    with pytest.raises(ContractError):
        seal(root)
    assert not (root / "checkpoint.json").exists()


@pytest.mark.parametrize("bad", ["optimizer.pt", "payload.pkl", "../escape", "unsafe.py"])
def test_checkpoint_inventory_rejects_non_native_or_unsafe_files(tmp_path, bad):
    root = native_bundle(tmp_path / "bad")
    target = root / "training_state" / bad
    target.write_bytes(b"not a safe state")
    with pytest.raises(ContractError):
        seal(root)
    assert not (root / "checkpoint.json").exists()


def test_corrupted_tensor_header_is_rejected_without_torch_load(tmp_path):
    root = native_bundle(tmp_path / "bad")
    (root / "training_state" / "optimizer_state.safetensors").write_bytes(
        b"pickle-shaped-not-tensors"
    )
    with pytest.raises(ContractError):
        seal(root)


def test_weights_only_is_explicit_and_cannot_be_promoted_to_full_state(tmp_path):
    from learning.smolvla.checkpoints import validate_checkpoint

    root = native_bundle(tmp_path / "weights", full=False)
    checksum = seal(root, kind="weights_only")
    with pytest.raises(ContractError, match="full.state"):
        validate_checkpoint(
            root, expected_sha256=checksum, expected_binding=binding(), require_full_state=True
        )
    manifest = read_json(root / "checkpoint.json")
    assert manifest["state_kind"] == "weights_only"
    assert manifest["bitwise_continuation_claimed"] is False


@pytest.mark.parametrize("field", list(binding()))
def test_checkpoint_resume_rejects_changed_training_identity(tmp_path, field):
    from learning.smolvla.checkpoints import validate_checkpoint

    root = native_bundle(tmp_path / "bound")
    checksum = seal(root)
    changed = {
        **binding(),
        field: {"tenant_id": SCOPE.tenant_id, "owner_id": "0" * 64}
        if field == "scope"
        else "0" * 64,
    }
    with pytest.raises(ContractError, match="binding"):
        validate_checkpoint(root, expected_sha256=checksum, expected_binding=changed)


def test_publisher_reads_back_every_file_before_atomic_complete_marker(tmp_path):
    from learning.smolvla.checkpoints import publish_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    receipt = publish_checkpoint(
        root,
        store,
        prefix=prefix(),
        expected_sha256=checksum,
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    marker = receipt["manifest_blob"]
    marker_write = store.operations.index(("write", marker))
    assert marker.endswith("/step-000010/checkpoint.json")
    assert all(
        store.operations.index(("read", name)) < marker_write
        for name in store.data
        if name != marker
    )
    assert receipt["readback_verified"] is True
    assert receipt["checkpoint_sha256"] == checksum
    assert store.data[marker] == (root / "checkpoint.json").read_bytes()
    assert len([op for op in store.operations if op == ("write", marker)]) == 1


def test_failed_upload_or_checksum_never_publishes_complete_marker(tmp_path):
    from learning.smolvla.checkpoints import publish_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    for failure in ("timeout", "checksum"):
        store = MemoryContainer()
        target = prefix() + "/step-000010/pretrained_model/model.safetensors"
        if failure == "timeout":
            store.fail_on = target
        else:
            store.corrupt_read = target
        with pytest.raises((TimeoutError, ContractError)):
            publish_checkpoint(
                root,
                store,
                prefix=prefix(),
                expected_sha256=checksum,
                expected_binding=binding(),
                check_deadline=lambda: 60.0,
            )
        assert not any(name.endswith("/checkpoint.json") for name in store.data)


def test_verified_complete_scan_ignores_unfinished_newer_checkpoint_but_not_corrupt_complete(
    tmp_path,
):
    from learning.smolvla.checkpoints import latest_complete_checkpoint

    complete = native_bundle(tmp_path / "step-000010")
    checksum = seal(complete)
    native_bundle(tmp_path / "step-000020", step=20)
    selected = latest_complete_checkpoint(tmp_path, expected_binding=binding())
    assert selected["step"] == 10 and selected["sha256"] == checksum
    (complete / "pretrained_model" / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ContractError):
        latest_complete_checkpoint(tmp_path, expected_binding=binding())


def test_publication_keeps_existing_complete_bytes_immutable(tmp_path):
    from learning.smolvla.checkpoints import publish_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    kwargs = dict(
        prefix=prefix(),
        expected_sha256=checksum,
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    first = publish_checkpoint(root, store, **kwargs)
    second = publish_checkpoint(root, store, **kwargs)
    assert first == second
    store.data[first["manifest_blob"]] = b"changed remote manifest"
    with pytest.raises(ContractError):
        publish_checkpoint(root, store, **kwargs)


def test_expired_job_cannot_start_checkpoint_upload(tmp_path):
    from learning.smolvla.checkpoints import publish_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    with pytest.raises(ContractError, match="deadline"):
        publish_checkpoint(
            root,
            store,
            prefix=prefix(),
            expected_sha256=checksum,
            expected_binding=binding(),
            check_deadline=lambda: 0.0,
        )
    assert store.operations == []


def test_remote_checkpoint_survives_loss_of_local_source_and_rehydrates_only_after_readback(
    tmp_path,
):
    import shutil

    from learning.smolvla.checkpoints import (
        publish_checkpoint,
        restore_checkpoint,
        validate_checkpoint,
    )

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    receipt = publish_checkpoint(
        root,
        store,
        prefix=prefix(),
        expected_sha256=checksum,
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    assert receipt["blob_etags"]["checkpoint.json"] == "unchanged"
    shutil.rmtree(root)
    destination = tmp_path / "fresh-managed-job"
    restored = restore_checkpoint(
        store,
        checkpoint_prefix=prefix() + "/step-000010",
        expected_sha256=checksum,
        expected_scope=SCOPE,
        destination=destination,
        check_deadline=lambda: 60.0,
    )
    assert restored["step"] == 10
    validate_checkpoint(destination, expected_sha256=checksum, expected_binding=binding())


def test_torn_remote_bundle_does_not_create_local_complete_marker(tmp_path):
    from learning.smolvla.checkpoints import publish_checkpoint, restore_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    publish_checkpoint(
        root,
        store,
        prefix=prefix(),
        expected_sha256=checksum,
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    store.corrupt_read = prefix() + "/step-000010/training_state/optimizer_state.safetensors"
    with pytest.raises(ContractError, match="checksum"):
        restore_checkpoint(
            store,
            checkpoint_prefix=prefix() + "/step-000010",
            expected_sha256=checksum,
            expected_scope=SCOPE,
            destination=tmp_path / "failed",
            check_deadline=lambda: 60.0,
        )
    assert not (tmp_path / "failed" / "checkpoint.json").exists()


def test_native_boundary_publishes_periodically_as_weights_only_until_full_state_is_verified(
    tmp_path,
):
    from learning.smolvla.checkpoint_runner import NativeCheckpointPublisher
    from learning.smolvla.checkpoints import DEFAULT_LIMITS

    store = MemoryContainer()
    context = {
        "binding": binding(),
        "origin": origin(),
        "limits": asdict(DEFAULT_LIMITS),
        "blob_prefix": prefix(),
        "resume_from_checkpoint_sha256": None,
    }

    def save_checkpoint(*, checkpoint_dir, step):
        native_bundle(checkpoint_dir, step=step)

    publisher = NativeCheckpointPublisher(
        original_save=save_checkpoint,
        context=context,
        container=store,
        check_deadline=lambda: 60.0,
    )
    for step in (10, 20):
        publisher(checkpoint_dir=tmp_path / f"step-{step}", step=step)
    assert [receipt["step"] for receipt in publisher.receipts] == [10, 20]
    assert all(receipt["resume_capability"] == "weights_only" for receipt in publisher.receipts)
    assert all(receipt["readback_verified"] for receipt in publisher.receipts)


def test_restart_contract_binds_new_authorization_and_exact_original_checkpoint(tmp_path):
    from learning.checks.smolvla_aml_check import example_config
    from learning.smolvla.checkpoint_runner import POLICY_SCHEMA, validate_resume_checkpoint
    from learning.smolvla.checkpoints import DEFAULT_LIMITS

    root = native_bundle(tmp_path / "checkpoint")
    checksum = seal(root)
    config = example_config()
    config["parameters"]["resume_mode"] = "weights_only"
    config["checkpointing"] = {
        "schema": POLICY_SCHEMA,
        "limits": asdict(DEFAULT_LIMITS),
        "resume": {
            "checkpoint_sha256": checksum,
            "source_azure_job_id": origin()["azure_job_id"],
            "source_azure_pipeline_job_id": origin()["azure_pipeline_job_id"],
            "step": 10,
        },
    }
    current = {
        **origin(),
        "azure_job_id": origin()["azure_job_id"] + "-new",
        "azure_pipeline_job_id": origin()["azure_pipeline_job_id"] + "-new",
        "specification_sha256": "8" * 64,
        "job_deadline_utc": "2026-09-27T00:00:00Z",
    }
    value = validate_resume_checkpoint(
        root, config=config, expected_binding=binding(), current_origin=current
    )
    assert value["origin"]["job_deadline_utc"] == origin()["job_deadline_utc"]
    assert current["job_deadline_utc"] != value["origin"]["job_deadline_utc"]
    with pytest.raises(ContractError, match="newly authorized"):
        validate_resume_checkpoint(
            root, config=config, expected_binding=binding(), current_origin=origin()
        )


def test_remote_scan_uses_only_last_complete_owned_bundle_and_rejects_corruption(tmp_path):
    from learning.smolvla.checkpoints import latest_remote_checkpoint, publish_checkpoint

    root = native_bundle(tmp_path / "native")
    checksum = seal(root)
    store = MemoryContainer()
    publish_checkpoint(
        root,
        store,
        prefix=prefix(),
        expected_sha256=checksum,
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    store.data[prefix() + "/step-000020/pretrained_model/model.safetensors"] = b"unfinished"
    selected = latest_remote_checkpoint(
        store,
        prefix=prefix(),
        expected_binding=binding(),
        check_deadline=lambda: 60.0,
    )
    assert selected["step"] == 10 and selected["checkpoint_sha256"] == checksum
    assert selected["jobs_submitted"] == 0
    store.corrupt_read = prefix() + "/step-000010/training_state/optimizer_state.safetensors"
    with pytest.raises(ContractError, match="checksum"):
        latest_remote_checkpoint(
            store,
            prefix=prefix(),
            expected_binding=binding(),
            check_deadline=lambda: 60.0,
        )
