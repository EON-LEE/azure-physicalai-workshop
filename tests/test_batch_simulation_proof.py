"""CPU fixtures exercise wrapper integrity only; they are not GPU or training evidence."""

import json
from types import SimpleNamespace

import pytest
from test_batch_simulation_task import gpu
from test_paused_acceptance import report
from test_paused_dispatch import paused_core as paused_core
from test_simulation_batch import spec as spec

from learning.common import canonical, digest
from learning.paused.contract import RAW_SCHEMA
from simulation.batch import BatchSimulationSpec
from simulation.batch_task import PrivateArtifacts, validate_success_evidence
from simulation.paused_acceptance import validate_attempt


@pytest.fixture
def evidence(spec, paused_core):
    core, _, _ = paused_core
    environment, native = report(paused_core)
    values = spec.model_dump(mode="json", by_alias=True)
    values.update(
        owner_id=core.owner,
        source_revision=native["source_revision"],
        control_profile_sha256=core.paused_profile.sha256,
        profile_id=core.paused_profile.profile_id,
    )
    values["platform"]["container_image"] = (
        "unit.azurecr.io/sim@" + native["simulator_image_digest"]
    )
    documents = {
        "inputs/environment.json": canonical(environment.model_dump(mode="json")),
        "inputs/grant.json": b'{"fixture":"CPU only; no actual authority"}',
        "inputs/criteria.json": b'{"fixture":"CPU criteria"}',
        "inputs/conditions.json": b'{"fixture":"CPU conditions"}',
    }
    for name in ("environment", "grant", "criteria", "conditions"):
        payload = documents[f"inputs/{name}.json"]
        values[name].update(sha256=digest(payload), size_bytes=len(payload))
    spec = BatchSimulationSpec.model_validate(values)
    native.update(
        criteria_sha256=spec.criteria_canonical_sha256,
        frozen_plan_sha256=spec.conditions_canonical_sha256,
    )
    manifest = {
        "schema": RAW_SCHEMA,
        "scope": {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id},
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "control_profile": native["control_profile"],
        "criteria_sha256": spec.criteria_canonical_sha256,
        "frozen_plan_sha256": spec.conditions_canonical_sha256,
        "episodes": [
            {
                "episode_id": native["command_id"],
                "environment_id": environment.environment_id,
                "revision": environment.revision,
                "frame_count": 2,
                "demonstration": {"kind": "reference_controller"},
                "provenance": {
                    "code_revision": spec.source_revision,
                    "simulator_image_digest": native["simulator_image_digest"],
                    "robot_asset_sha256": spec.asset_bundle.sha256,
                },
            }
        ],
    }
    documents["raw-manifest.json"] = canonical(manifest)
    raw_uri = (
        f"{spec.storage_account_url}/{spec.output_container}/"
        f"{spec.owner_id}/{native['command_id']}/manifest.json"
    )
    native["capture"]["receipt"].update(
        manifest_sha256=digest(documents["raw-manifest.json"]),
        manifest_uri=raw_uri,
        episode_id=native["command_id"],
    )
    native_acceptance = validate_attempt(native, environment=environment, mode="reference-task")
    native_acceptance["manifest_sha256"] = digest(documents["raw-manifest.json"])
    documents.update(
        {
            "preflight.json": canonical(gpu()),
            "probe.json": canonical(native),
            "acceptance.json": canonical(native_acceptance),
            "probe.log": b"CPU fixture, not a live simulator",
            "acceptance.log": b"CPU fixture, not a dataset qualification",
            "inputs/spec.json": canonical(spec.model_dump(mode="json", by_alias=True)),
        }
    )
    proof = {
        "schema": "physicalai.batch-simulation-result/v1",
        "attempt_id": str(spec.attempt_id),
        "previous_attempt_id": None,
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "spec_sha256": spec.sha256,
        "spec_file_sha256": digest(documents["inputs/spec.json"]),
        "source_revision": spec.source_revision,
        "image": spec.platform.container_image,
        "control_profile_sha256": spec.control_profile_sha256,
        "profile_id": spec.profile_id,
        "accepted": True,
        "outcome": "accepted",
        "native_acceptance": "accepted",
        "physical_status": "succeeded",
        "capture_status": "ready",
        "learning_quality_proven": False,
        "raw_manifest": native["capture"]["receipt"],
        "artifacts": [
            {"path": name, "bytes": len(payload), "sha256": digest(payload)}
            for name, payload in documents.items()
        ],
    }
    return spec, documents, proof


class Blob:
    def __init__(self, payload):
        self.payload = payload

    def get_blob_properties(self):
        return SimpleNamespace(size=len(self.payload))

    def download_blob(self):
        return SimpleNamespace(chunks=lambda: iter([self.payload]))


def private_store(spec, documents, proof):
    blobs = {spec.artifact_prefix + "/" + name: payload for name, payload in documents.items()}
    blobs[spec.artifact_prefix + "/completion.json"] = canonical(proof)
    raw_name = f"{spec.owner_id}/{proof['raw_manifest']['episode_id']}/manifest.json"
    blobs[raw_name] = documents["raw-manifest.json"]
    store = object.__new__(PrivateArtifacts)
    store.spec = spec
    store.client = SimpleNamespace(get_blob_client=lambda container, name: Blob(blobs[name]))
    return store, blobs


def test_status_rechecks_native_semantics_and_all_private_artifact_hashes(evidence):
    spec, documents, proof = evidence
    store, _ = private_store(spec, documents, proof)
    assert store.read_completion()["accepted"] is True


@pytest.mark.parametrize(
    "artifact",
    [
        "inputs/spec.json",
        "inputs/environment.json",
        "probe.json",
        "acceptance.json",
        "raw-manifest.json",
        "probe.log",
    ],
)
def test_tampered_private_artifact_cannot_qualify(evidence, artifact):
    spec, documents, proof = evidence
    store, blobs = private_store(spec, documents, proof)
    blobs[spec.artifact_prefix + "/" + artifact] += b"tampered"
    with pytest.raises(ValueError, match="size|checksum"):
        store.read_completion()


def test_remote_raw_manifest_must_still_match_after_task_completion(evidence):
    spec, documents, proof = evidence
    store, blobs = private_store(spec, documents, proof)
    blobs[f"{spec.owner_id}/{proof['raw_manifest']['episode_id']}/manifest.json"] = b"{}"
    with pytest.raises(ValueError, match="Remote raw manifest"):
        store.read_completion()


def test_terminal_receipt_cannot_redirect_to_a_copy_under_another_episode(evidence):
    spec, documents, proof = evidence
    proof["raw_manifest"] = {
        **proof["raw_manifest"],
        "episode_id": str(spec.attempt_id),
        "manifest_uri": f"{spec.storage_account_url}/{spec.output_container}/"
        f"{spec.owner_id}/{spec.attempt_id}/manifest.json",
    }
    store, _ = private_store(spec, documents, proof)
    with pytest.raises(ValueError, match="receipt|episode"):
        store.read_completion()


@pytest.mark.parametrize(
    "key,value",
    [
        ("physical_status", "timed_out"),
        ("peak_tcp_speed_m_s", 0.201),
        ("model_weights_loaded", True),
        ("source_revision", "f" * 40),
    ],
)
def test_native_acceptance_flag_cannot_override_the_real_report(evidence, key, value):
    spec, documents, _ = evidence
    native = json.loads(documents["probe.json"])
    documents["probe.json"] = canonical(native | {key: value})
    with pytest.raises(ValueError):
        validate_success_evidence(spec, documents)
