"""Installed-catalogue/real Unix IPC contract tests with metadata doubles, not trained weights."""

import os
import socket
from dataclasses import asdict
from datetime import timedelta

import pytest
from runtime_support import ACTOR
from test_paused_dispatch import paused_core as paused_core

from apps.api.errors import Problem
from apps.api.models import utcnow
from learning.common import canonical, file_digest, read_json
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.paused_configuration import PausedOperatorGrant
from simulation.paused_contracts import ResolvedSimulationAuthorization
from simulation.paused_deployment import (
    InstalledPausedPolicyProvider,
    load_paused_policy_deployment,
)
from tests.learning.test_paused_model_artifacts import fixture as model_fixture


@pytest.fixture
def installed(paused_core, monkeypatch, tmp_path):
    core, _, original = paused_core
    model_root = tmp_path / "model"
    model_fixture(model_root)
    metadata = read_json(model_root / "model.json")
    metadata.update(
        scope={"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key},
        control_profile=asdict(core.paused_profile),
        task=original.task.model_dump(),
    )
    (model_root / "model.json").write_bytes(canonical(metadata))
    model_sha = file_digest(model_root / "model.json")
    now = utcnow()
    request = original.model_copy(
        update={
            "controller": "learned",
            "authorization_kind": "evaluation_grant",
            "policy_type": "smolvla",
            "model_sha256": model_sha,
            "wall_expires_at": now + timedelta(seconds=590),
        }
    )
    permission = ResolvedSimulationAuthorization(
        authorization_id=request.authorization_id,
        authorization_kind=request.authorization_kind,
        owner=ACTOR.owner_key,
        environment_id=request.environment_id,
        revision=request.revision,
        controller="learned",
        task=request.task,
        control_profile_sha256=core.paused_profile.sha256,
        wall_expires_at=request.wall_expires_at,
        max_episode_wall_seconds=600,
        max_simulation_steps=1800,
        purpose="evaluation",
        criteria_sha256=metadata["criteria_sha256"],
        frozen_plan_sha256=metadata["frozen_plan_sha256"],
        policy_type="smolvla",
        model_sha256=model_sha,
    )
    grant = PausedOperatorGrant(
        schema="physicalai.paused-operator-grant/v1",
        execution_timing="paused_simulation",
        real_time_admission=False,
        tenant_id=ACTOR.tenant_id,
        source_revision="a" * 40,
        simulator_image_digest="sha256:" + "b" * 64,
        issued_at=now,
        expires_at=now + timedelta(seconds=600),
        authorization=permission,
    )
    monkeypatch.setenv("SOURCE_REVISION", grant.source_revision)
    monkeypatch.setenv(
        "SIMULATOR_IMAGE", "test.azurecr.io/simulator@" + grant.simulator_image_digest
    )
    monkeypatch.setenv("ENTRA_TENANT_ID", str(ACTOR.tenant_id))
    for key in ("PAUSED_POLICY_CATALOG_FILE", "PAUSED_POLICY_CATALOG_SHA256"):
        monkeypatch.delenv(key, raising=False)
    sock_path = tmp_path / "policy.sock"
    catalogue = {
        "schema": "physicalai.paused-policy-catalog/v1",
        "policies": [
            {
                "grant": grant.model_dump(mode="json", by_alias=True),
                "model_root": str(model_root),
                "socket_path": str(sock_path),
                "expected_peer_uid": os.geteuid(),
            }
        ],
    }
    path = tmp_path / "catalogue.json"
    path.write_bytes(canonical(catalogue))
    return core, request, permission, path, catalogue, sock_path, metadata


def load(installed):
    core, _, _, path, _, _, _ = installed
    return InstalledPausedPolicyProvider.load(path, file_digest(path), core.paused_profile)


def test_absent_paused_catalogue_keeps_learned_deployment_disabled(installed):
    assert load_paused_policy_deployment() == (None, None)


def test_actual_installed_v2_candidate_opens_only_a_protected_guarded_model_port(installed):
    core, request, permission, _, _, sock_path, _ = installed
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(sock_path))
        os.chmod(sock_path, 0o660)
        provider = load(installed)
        assert provider.authorize(ACTOR.owner_key, request, core.paused_profile) == permission
        port = provider.create(
            ACTOR.owner_key,
            request,
            core.paused_profile,
            publication_guard=lambda *args: False,
        )
        assert isinstance(port, PausedGuardedPolicyAdapter)
        assert port.policy.model_sha256 == request.model_sha256
        assert port.policy.profile == core.paused_profile
        assert port.policy.task.source_policy_sha256 == request.model_sha256
        assert port.predict_calls == port.policy.predict_calls == 0
        assert port.policy.expected_peer_uid == os.geteuid()


@pytest.mark.parametrize("field", ["owner", "model", "profile", "task", "authority"])
def test_installed_authority_rejects_rebound_learned_request(installed, field):
    core, request, _, _, _, _, _ = installed
    provider = load(installed)
    owner = ACTOR.owner_key
    if field == "owner":
        owner = "f" * 64
    elif field == "model":
        request = request.model_copy(update={"model_sha256": "f" * 64})
    elif field == "profile":
        from learning.paused import PausedControlProfile

        core.paused_profile = PausedControlProfile("f" * 64)
    elif field == "task":
        request = request.model_copy(
            update={"task": request.task.model_copy(update={"instruction": "Unapproved task"})}
        )
    else:
        from uuid import uuid4

        request = request.model_copy(update={"authorization_id": uuid4()})
    with pytest.raises((ValueError, RuntimeError, Problem)):
        provider.authorize(owner, request, core.paused_profile)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "image",
        "tenant",
        "expired",
        "renewed",
        "kind",
        "purpose",
        "model-clock",
        "model-profile",
        "criteria",
    ],
)
def test_catalogue_cannot_mint_authority_for_wrong_runtime_or_relabel_a_checkpoint(
    installed, change
):
    core, _, _, path, value, _, metadata = installed
    entry = value["policies"][0]
    grant, permit = entry["grant"], entry["grant"]["authorization"]
    if change == "source":
        grant["source_revision"] = "f" * 40
    elif change == "image":
        grant["simulator_image_digest"] = "sha256:" + "f" * 64
    elif change == "tenant":
        grant["tenant_id"] = "99999999-9999-4999-8999-999999999999"
    elif change == "expired":
        grant["issued_at"] = (utcnow() - timedelta(seconds=600)).isoformat()
        grant["expires_at"] = (utcnow() - timedelta(seconds=1)).isoformat()
    elif change == "renewed":
        grant["expires_at"] = (utcnow() + timedelta(seconds=900)).isoformat()
    elif change == "kind":
        permit["authorization_kind"] = "reference_collection"
    elif change == "purpose":
        permit["purpose"] = "demonstration"
    elif change == "criteria":
        permit["criteria_sha256"] = "f" * 64
    else:
        if change == "model-clock":
            metadata["training"]["timestamp_basis"] = "wall_time"
        else:
            metadata["control_profile"]["servo_profile_sha256"] = "f" * 64
        from pathlib import Path

        model_path = Path(entry["model_root"]) / "model.json"
        model_path.write_bytes(canonical(metadata))
        permit["model_sha256"] = file_digest(model_path)
    path.write_bytes(canonical(value))
    with pytest.raises(ValueError):
        load(installed)


def test_missing_model_socket_fails_without_constructing_a_scripted_substitute(installed):
    core, request, _, _, _, _, _ = installed
    provider = load(installed)
    with pytest.raises((OSError, RuntimeError), match="socket|Socket|No such file"):
        provider.create(
            ACTOR.owner_key, request, core.paused_profile, publication_guard=lambda *args: False
        )


def test_rewritten_model_manifest_cannot_replace_the_admitted_candidate(installed):
    core, request, _, _, value, _, _ = installed
    provider = load(installed)
    from pathlib import Path

    (Path(value["policies"][0]["model_root"]) / "model.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="manifest|model|checksum"):
        provider.create(
            ACTOR.owner_key, request, core.paused_profile, publication_guard=lambda *args: False
        )


def test_deployment_switch_requires_both_file_and_immutable_bytes_hash(installed, monkeypatch):
    _, _, _, path, _, _, _ = installed
    monkeypatch.setenv("PAUSED_POLICY_CATALOG_FILE", str(path))
    with pytest.raises(ValueError, match="checksum|pinned"):
        load_paused_policy_deployment()


def test_runtime_catalogue_and_driver_imports_do_not_load_isaac_or_model_frameworks():
    import subprocess
    import sys

    code = (
        "import sys; import simulation.paused_deployment; import simulation.paused_learned; "
        "assert not {'isaacsim','carb','omni','pxr','torch','lerobot'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
