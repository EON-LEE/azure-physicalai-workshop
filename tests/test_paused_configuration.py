"""CPU pinning tests for operator-only non-real-time authority."""

import json
from datetime import timedelta

import pytest
from test_paused_dispatch import paused_core as paused_core

from apps.api.models import utcnow
from learning.common import canonical, digest, file_digest
from simulation import paused_configuration
from simulation.paused_configuration import OperatorPausedAuthority


def test_paused_fingerprint_includes_all_new_driver_protocol_and_capture_code(monkeypatch):
    files = []
    monkeypatch.setattr(
        paused_configuration, "file_digest", lambda path: files.append(path.name) or "a" * 64
    )
    digest_one = paused_configuration.paused_servo_sha256()
    assert {
        "paused_control.py",
        "paused_worker.py",
        "paused_teacher.py",
        "paused_runtime.py",
        "paused_capture.py",
        "paused_observation.py",
        "paused_contracts.py",
        "isaac_adapter.py",
        "capture_worker.py",
        "run_isaac.py",
    } <= set(files)
    monkeypatch.setattr(
        paused_configuration,
        "file_digest",
        lambda path: ("b" if path.name == "paused_control.py" else "a") * 64,
    )
    assert paused_configuration.paused_servo_sha256() != digest_one


def test_protocol_versions_are_bound_into_the_new_fingerprint(monkeypatch):
    monkeypatch.setattr(paused_configuration, "file_digest", lambda path: "a" * 64)
    before = paused_configuration.paused_servo_sha256()
    monkeypatch.setattr(paused_configuration, "protocol_schemas", lambda: {"raw": "wrong"})
    assert paused_configuration.paused_servo_sha256() != before


def test_operator_grant_binds_real_saved_case_profile_criteria_and_source(
    paused_core, monkeypatch, tmp_path
):
    core, environment, request = paused_core
    now = utcnow()
    monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("SIMULATOR_IMAGE", "test.azurecr.io/simulator@sha256:" + "c" * 64)
    monkeypatch.setenv("ENTRA_TENANT_ID", "11111111-1111-4111-8111-111111111111")
    criteria_value = {
        "schema": "physicalai.operator-paused-criteria/v2",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "profile_id": core.paused_profile.profile_id,
        "task": request.task.model_dump(),
    }
    criteria = tmp_path / "criteria.json"
    criteria.write_bytes(canonical(criteria_value))
    criteria_sha = digest(canonical(criteria_value))
    conditions_value = {
        "schema": "physicalai.fixed-paused-scene-conditions/v2",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "criteria_canonical_sha256": criteria_sha,
        "task": request.task.model_dump(),
        "cases": [
            {
                "environment_id": environment.environment_id,
                "revision": environment.revision,
                "seed": core.spec.seed,
                "split": "test",
                "scene_builder_sha256": core.spec.scene_builder_sha256,
                "expected_initial_position_m": list(core.spec.part_position),
                "goal_position_m": list(core.spec.station(request.target_station_id).position),
            }
        ],
    }
    conditions = tmp_path / "conditions.json"
    conditions.write_bytes(canonical(conditions_value))
    conditions_sha = digest(canonical(conditions_value))
    authority = core.paused_authorizer.authorize(
        core.owner, request, core.paused_profile
    ).model_copy(update={"criteria_sha256": criteria_sha, "frozen_plan_sha256": conditions_sha})
    grant_value = {
        "schema": "physicalai.paused-operator-grant/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "tenant_id": "11111111-1111-4111-8111-111111111111",
        "source_revision": "b" * 40,
        "simulator_image_digest": "sha256:" + "c" * 64,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=600)).isoformat(),
        "authorization": authority.model_dump(mode="json"),
    }
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps(grant_value))
    loaded = OperatorPausedAuthority.load(
        grant,
        file_digest(grant),
        environment=environment,
        profile=core.paused_profile,
        criteria=criteria,
        criteria_sha256=criteria_sha,
        conditions=conditions,
        conditions_sha256=conditions_sha,
    )
    assert loaded.authorize(core.owner, request, core.paused_profile) == authority
    monkeypatch.setenv("SOURCE_REVISION", "d" * 40)
    with pytest.raises(ValueError, match="source|runtime"):
        OperatorPausedAuthority.load(
            grant,
            file_digest(grant),
            environment=environment,
            profile=core.paused_profile,
            criteria=criteria,
            criteria_sha256=criteria_sha,
            conditions=conditions,
            conditions_sha256=conditions_sha,
        )
