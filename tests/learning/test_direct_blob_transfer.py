import pytest

from learning.common import ContractError, write_json
from tests.learning.test_direct_command_jobs import command_config
from tests.learning.test_training_checkpoints import MemoryContainer as BaseMemoryContainer


class MemoryContainer(BaseMemoryContainer):
    def list_blobs(self, **kwargs):
        from types import SimpleNamespace

        return (
            SimpleNamespace(name=name, size=len(body), etag="unchanged")
            for name, body in sorted(self.data.items())
            if name.startswith(kwargs["name_starts_with"])
        )


def test_input_prefix_is_derived_only_from_exact_approved_datastore_uri():
    from learning.paused.blob_transfer import input_prefix

    value = command_config()
    prefix = input_prefix(value, "parent_model")
    assert prefix.startswith(f"tenants/{value['tenant_id']}/owners/{value['owner_id']}/")
    assert "://" not in prefix and not prefix.endswith("/")
    assert not prefix.startswith("artifacts/")


@pytest.mark.parametrize(
    "change", ["account", "owner", "other-datastore", "query", "traversal", "slash"]
)
def test_unapproved_location_is_rejected_before_storage_io(change):
    from learning.paused.blob_transfer import input_prefix

    value = command_config()
    asset = value["inputs"]["demonstrations"]
    if change == "account":
        asset["uri"] = "https://other.blob.core.windows.net/artifacts/data"
    elif change == "owner":
        asset["uri"] = asset["uri"].replace(value["owner_id"], "0" * 64)
    elif change == "other-datastore":
        asset["uri"] = asset["uri"].replace(
            f"/datastores/{value['datastore']}/", "/datastores/other/"
        )
    elif change == "query":
        asset["uri"] += "?sig=not-allowed"
    elif change == "traversal":
        asset["uri"] += "/../outside"
    else:
        asset["uri"] += "/"
    with pytest.raises(ContractError):
        input_prefix(value, "demonstrations")


def test_manifest_download_verifies_etag_hash_and_closed_inventory(tmp_path):
    from learning.common import canonical, digest
    from learning.paused.blob_transfer import PrivateBlobTransfer, input_prefix

    store = MemoryContainer()
    config = command_config()
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 60)
    prefix = input_prefix(config, "parent_model")
    body = canonical({"files": {"payload.bin": digest(b"frozen")}}) + b"\n"
    store.data[prefix + "/manifest.json"] = body
    store.data[prefix + "/payload.bin"] = b"frozen"
    value, raw = transfer.manifest(prefix, "manifest.json", digest(body))
    assert value["files"] == {"payload.bin": digest(b"frozen")} and raw == body
    transfer.download_files(
        prefix, value["files"], tmp_path / "payload", metadata={"manifest.json": raw}
    )
    assert (tmp_path / "payload" / "payload.bin").read_bytes() == b"frozen"
    assert (tmp_path / "payload" / "manifest.json").read_bytes() == body


def test_extra_or_corrupt_blob_is_never_downloaded_as_valid_input(tmp_path):
    from learning.common import digest
    from learning.paused.blob_transfer import PrivateBlobTransfer, input_prefix

    store = MemoryContainer()
    config = command_config()
    prefix = input_prefix(config, "parent_model")
    store.data[prefix + "/file"] = b"expected"
    store.data[prefix + "/extra"] = b"unreviewed"
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 60)
    with pytest.raises(ContractError, match="inventory"):
        transfer.download_files(prefix, {"file": digest(b"expected")}, tmp_path / "extra")
    del store.data[prefix + "/extra"]
    store.corrupt_read = prefix + "/file"
    with pytest.raises(ContractError, match="checksum"):
        transfer.download_files(prefix, {"file": digest(b"expected")}, tmp_path / "corrupt")


def test_result_publication_is_manifest_last_and_partial_failure_has_no_success_marker(tmp_path):
    from learning.paused.blob_transfer import PrivateBlobTransfer

    config = command_config()
    prefix = config["output_prefix"] + "/direct-p0/model"
    root = tmp_path / "output"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"unit-test-payload")
    write_json(root / "result.json", {"actual_optimizer_steps": 1})
    store = MemoryContainer()
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 60)
    receipt = transfer.publish_files(root, prefix, marker="result.json")
    writes = [name for verb, name in store.operations if verb == "write"]
    assert writes[-1] == prefix + "/result.json"
    assert receipt["readback_verified"] is True
    broken = MemoryContainer()
    broken.fail_on = prefix + "/model.safetensors"
    with pytest.raises(TimeoutError):
        PrivateBlobTransfer(broken, config, remaining=lambda: 60).publish_files(
            root, prefix, marker="result.json"
        )
    assert prefix + "/result.json" not in broken.data


def test_expired_transfer_performs_no_storage_read(tmp_path):
    from learning.paused.blob_transfer import PrivateBlobTransfer

    store = MemoryContainer()
    config = command_config()
    transfer = PrivateBlobTransfer(store, config, remaining=lambda: 0)
    with pytest.raises(ContractError, match="deadline"):
        transfer.download_files(config["output_prefix"], {"file": "a" * 64}, tmp_path / "expired")
    assert store.operations == []


def test_missing_etag_rejects_payload_before_download(tmp_path, monkeypatch):
    from learning.common import digest
    from learning.paused.blob_transfer import PrivateBlobTransfer, input_prefix
    from tests.learning.test_training_checkpoints import MemoryBlob

    store = MemoryContainer()
    config = command_config()
    prefix = input_prefix(config, "parent_model")
    store.data[prefix + "/model.bin"] = b"fixture"
    original = MemoryBlob.get_blob_properties

    def no_etag(self, **kwargs):
        properties = original(self, **kwargs)
        properties.etag = None
        return properties

    monkeypatch.setattr(MemoryBlob, "get_blob_properties", no_etag)
    with pytest.raises(ContractError, match="ETag"):
        PrivateBlobTransfer(store, config, remaining=lambda: 60).download_files(
            prefix,
            {"model.bin": digest(b"fixture")},
            tmp_path / "partial",
        )
    assert store.operations == []
