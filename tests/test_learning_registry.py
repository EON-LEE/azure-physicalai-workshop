import json
from types import SimpleNamespace
from uuid import uuid4

from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.registry import BlobRegistry
from tests.runtime_support import ACTOR


def test_registry_files_use_the_native_tenant_opaque_owner_datastore_prefix():
    path = BlobRegistry.key(ACTOR, "artifacts/test/files/manifest.json")
    assert path.startswith(f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/")
    assert str(ACTOR.object_id) not in path


def test_repeated_verified_parent_registration_returns_the_original_immutable_receipt(
    tmp_path, monkeypatch
):
    model = {
        "policy_type": "smolvla",
        "role": "pretrained",
        "training": None,
        "processor_sha256": "b" * 64,
        "upstream": {"source_commit": "c" * 40, "model_revision": "d" * 40},
    }
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")
    saved = {}

    def put(actor, key, value):
        if key in saved:
            assert saved[key] == value, "Repeated registration must not invent a new created_at."
        saved[key] = value
        return True

    registry = SimpleNamespace(
        get=lambda actor, key: saved.get(key),
        put=put,
        upload=lambda *args: None,
    )
    artifacts = VerifiedArtifacts(
        registry,
        None,
        "https://test.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    monkeypatch.setattr(artifacts, "_model", lambda *args, **kwargs: model)
    first = artifacts.register_parent(ACTOR, tmp_path, "a" * 64, uuid4())
    second = artifacts.register_parent(ACTOR, tmp_path, "a" * 64, first.registered_by)
    assert first == second
