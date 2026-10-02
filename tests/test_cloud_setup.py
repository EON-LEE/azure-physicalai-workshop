import json
from uuid import uuid4

import pytest

from scripts.cloud_build import snapshot, source_files
from scripts.provision_identity import provision


def test_cloud_snapshot_omits_tests_dependencies_and_credentials_outside_sources(tmp_path):
    for name in ("Dockerfile", "pyproject.toml", "uv.lock", ".python-version", ".dockerignore"):
        (tmp_path / name).write_text("test")
    for name in (
        "apps/api/main.py",
        "apps/web/src/main.ts",
        "apps/web/tests/fake.ts",
        "apps/web/node_modules/thirdparty.js",
        "apps/web/dist/main.js",
        "test-results/token.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    selected = {path.relative_to(tmp_path).as_posix() for path in source_files(tmp_path)}
    assert "apps/api/main.py" in selected
    assert not any(
        "tests/" in name or "node_modules/" in name or "dist/" in name for name in selected
    )
    assert len(snapshot(tmp_path, tmp_path / "context.tar.gz")) == 64


def test_cloud_snapshot_refuses_private_key_in_included_source(tmp_path):
    (tmp_path / "apps").mkdir()
    (tmp_path / "apps" / "secret.key").write_text("test key, not a credential")
    with pytest.raises(ValueError, match="Credential-shaped"):
        source_files(tmp_path)


def test_cloud_snapshot_refuses_symlinks_in_included_source(tmp_path):
    (tmp_path / "apps").mkdir()
    outside = tmp_path / "outside"
    outside.write_text("fixture")
    (tmp_path / "apps" / "link.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        source_files(tmp_path)


def test_identity_resume_scopes_exist_before_preauthorization(tmp_path, monkeypatch):
    tenant = uuid4()
    applications = [
        {"name": f"physicalai-factory-{kind}", "object_id": str(uuid4()), "client_id": str(uuid4())}
        for kind in ("api", "spa")
    ]
    manifest = tmp_path / "identity.json"
    manifest.write_text(
        json.dumps(
            {
                "tenant_id": str(tenant),
                "status": "incomplete",
                "applications": applications,
            }
        )
    )
    patches = []

    def azure(*args):
        if args[:2] == ("account", "show"):
            return {"tenantId": str(tenant)}
        if args[:3] == ("ad", "app", "show"):
            item = next(app for app in applications if app["client_id"] == args[-1])
            return {"id": item["object_id"], "displayName": item["name"]}
        if args[:3] == ("ad", "sp", "list"):
            return [{"id": "existing-service-principal"}]
        pytest.fail(f"Unexpected or duplicate identity write: {args}")

    monkeypatch.setattr("scripts.provision_identity.azure", azure)
    monkeypatch.setattr(
        "scripts.provision_identity.graph_patch", lambda key, body: patches.append(body)
    )
    result = provision(tenant, "factory", manifest, include_cli=True, resume=True)
    assert "oauth2PermissionScopes" in patches[0]["api"]
    assert "preAuthorizedApplications" not in patches[0]["api"]
    assert "preAuthorizedApplications" in patches[1]["api"]
    scope = patches[0]["api"]["oauth2PermissionScopes"][0]["id"]
    assert all(
        app["delegatedPermissionIds"] == [scope]
        for app in patches[1]["api"]["preAuthorizedApplications"]
    )
    assert result["secrets_created"] is False
    assert result["directory_api_permissions_added"] is False
    assert json.loads(manifest.read_text())["scope_id"] == scope


def test_identity_resume_refuses_foreign_tenant(tmp_path, monkeypatch):
    tenant = uuid4()
    monkeypatch.setattr("scripts.provision_identity.azure", lambda *args: {"tenantId": str(tenant)})
    manifest = tmp_path / "identity.json"
    manifest.write_text(json.dumps({"tenant_id": str(uuid4()), "applications": []}))
    with pytest.raises(ValueError, match="different tenant"):
        provision(tenant, "factory", manifest, resume=True)
