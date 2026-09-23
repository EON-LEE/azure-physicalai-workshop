"""CPU validation of trusted configuration, using fabricated and explicitly local evidence."""

import json
from dataclasses import asdict
from uuid import uuid4

import pytest
from runtime_support import ACTOR
from test_policy_runtime import request_for
from test_teaching_runtime import teaching as teaching

from apps.api.errors import Problem
from learning.common import file_digest
from learning.contract import ControlProfile
from simulation import runtime_configuration as configuration


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMULATOR_IMAGE", "test.azurecr.io/isaac@sha256:" + "a" * 64)
    monkeypatch.setenv("FRANKA_ASSET_SHA256", "b" * 64)
    monkeypatch.setenv("SOURCE_REVISION", "c" * 40)
    monkeypatch.setenv("ISAAC_SIM_VERSION", "6.0.0")
    return tmp_path


def write_evidence(root, **changes):
    profile = ControlProfile(configuration.servo_profile_sha256())
    payload = {
        "schema": "physicalai.control-timing/v1",
        "evidence_kind": "azure_isaac_gpu",
        "simulator_image_digest": "sha256:" + "a" * 64,
        "simulator_version": "6.0.0",
        "robot_asset_sha256": "b" * 64,
        "source_revision": "c" * 40,
        "control_profile": asdict(profile),
        "model_sha256": None,
        "policy_type": None,
        "control_intervals": 100,
        "camera_pairs": 100,
        "max_control_cycle_ms": 95.0,
        "max_inference_latency_ms": 0.0,
        "max_heartbeat_gap_ms": 1100.0,
        "trace_sha256": "d" * 64,
        "gpu_model": "CPU test payload, not actual GPU evidence",
    } | changes
    path = root / f"timing-{uuid4()}.json"
    path.write_text(json.dumps(payload))
    return path, profile


def test_profile_requires_pinned_deployment_evidence_not_request_supplied_defaults(evidence):
    path, profile = write_evidence(evidence)
    assert configuration.load_control_profile(path, file_digest(path)) == profile
    with pytest.raises(ValueError, match="checksum"):
        configuration.load_control_profile(path, "f" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_kind": "test_fixture"},
        {"max_control_cycle_ms": 101},
        {"max_inference_latency_ms": 81},
        {"control_intervals": 1},
        {"camera_pairs": 99},
        {"source_revision": "e" * 40},
        {"robot_asset_sha256": "e" * 64},
        {"simulator_image_digest": "sha256:" + "e" * 64},
        {"control_profile": asdict(ControlProfile("f" * 64))},
    ],
)
def test_unmeasured_or_wrong_runtime_timing_cannot_enable_control(evidence, changes):
    path, _ = write_evidence(evidence, **changes)
    with pytest.raises(ValueError):
        configuration.load_control_profile(path, file_digest(path))


def test_model_specific_timing_cannot_be_replaced_by_servo_only_proof(evidence):
    path, _ = write_evidence(evidence)
    with pytest.raises(ValueError, match="model"):
        configuration.load_control_profile(path, file_digest(path), expected_model_sha256="b" * 64)


def test_actual_cpu_scheduling_implementation_is_part_of_the_immutable_servo_digest(monkeypatch):
    files = []

    def file_digest(path):
        files.append(path.name)
        return "a" * 64

    monkeypatch.setattr(configuration, "file_digest", file_digest)
    before = configuration.servo_profile_sha256()
    assert "physics_scheduling.py" in files
    monkeypatch.setattr(
        configuration,
        "file_digest",
        lambda path: ("b" if path.name == "physics_scheduling.py" else "a") * 64,
    )
    assert configuration.servo_profile_sha256() != before


def test_release_catalogue_binds_model_task_owner_and_explicit_environment_cases(
    evidence, teaching
):
    core, request, _ = request_for(teaching)
    path, profile = write_evidence(
        evidence,
        model_sha256=request.model_sha256,
        max_inference_latency_ms=30.0,
        policy_type="smolvla",
    )
    entry = {
        "policy_release_id": str(request.policy_release_id),
        "policy_type": request.policy_type,
        "scope": {"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key},
        "model_sha256": request.model_sha256,
        "control_profile_sha256": profile.sha256,
        "task": request.task.model_dump(),
        "environment_cases": [
            {
                "environment_id": request.command.environment_id,
                "revision": request.command.revision,
            }
        ],
        "socket_path": str(evidence / "policy.sock"),
        "expected_peer_uid": 1000,
        "timing_evidence_file": str(path),
        "timing_evidence_sha256": file_digest(path),
    }
    catalog = evidence / "releases.json"
    catalog.write_text(json.dumps({"schema": "physicalai.policy-catalog/v1", "releases": [entry]}))
    provider = configuration.DeploymentPolicyProvider.load(catalog, file_digest(catalog), profile)
    provider.authorize(ACTOR.owner_key, request, profile)
    for changed in (
        request.model_copy(update={"model_sha256": "e" * 64}),
        request.model_copy(
            update={"command": request.command.model_copy(update={"revision": "e" * 64})}
        ),
        request.model_copy(
            update={"task": request.task.model_copy(update={"instruction": "Changed"})}
        ),
    ):
        with pytest.raises(Problem):
            provider.authorize(ACTOR.owner_key, changed, profile)
    with pytest.raises(Problem):
        provider.authorize("f" * 64, request, profile)
    for family in ("gr00t_n1_5", "gr00t_n1_7"):
        with pytest.raises(ValueError, match="commercial|license"):
            configuration._require_approved_policy_family(family)
