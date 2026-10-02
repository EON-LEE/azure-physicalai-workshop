"""CPU receipt tests for separately attested admission, not command-training or GPU proof."""

import base64
import json

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_command_runtime_admission import command_runtime_value as command_runtime_value
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_simulation_batch import spec as spec

from learning.common import canonical, digest
from simulation import batch_learned, learned_probe


@pytest.fixture
def command_proof(evidence, command_runtime_value):
    spec, scene, profile, grant, report, documents = evidence
    runtime = batch_learned.parse_model_runtime(
        {
            **command_runtime_value,
            "control_profile_sha256": profile.sha256,
            "legacy_servo_sha256": profile.servo_profile_sha256,
            "simulator_image": spec.platform.container_image,
            "simulator_source_revision": spec.source_revision,
        }
    )
    raw = canonical(runtime.model_dump(mode="json", by_alias=True))
    spec = spec.model_copy(
        update={
            "model_runtime": spec.model_runtime.model_copy(
                update={
                    "sha256": digest(raw),
                    "size_bytes": len(raw),
                }
            )
        }
    )
    admission = batch_learned.command_admission_proof(
        runtime, runtime_sha256=spec.model_runtime.sha256
    )
    report["model_admission"] = admission
    server = json.loads(documents["preflight.json"])
    server["model_process"].update(
        runtime_sha256=spec.model_runtime.sha256,
        runtime_descriptor_base64=base64.b64encode(raw).decode("ascii"),
        admission=admission,
    )
    documents.update(
        {
            "preflight.json": canonical(server),
            "probe.json": canonical(report),
            "acceptance.json": canonical(
                learned_probe.rescore(report, spec, scene, profile, grant)
            ),
            "inputs/spec.json": canonical(spec.model_dump(mode="json", by_alias=True)),
        }
    )
    return spec, documents, admission


def test_actual_report_and_server_bind_identical_v3_admission_and_original_descriptor(
    command_proof,
):
    spec, documents, admission = command_proof
    receipt, result = batch_learned.rescore_evidence(spec, documents)
    assert result["accepted"] is True
    assert receipt["episode_id"] == str(spec.attempt_id)
    assert admission["runtime_sha256"] == spec.model_runtime.sha256
    assert result["learning_quality_proven"] is False


@pytest.mark.parametrize("target", ["server", "report"])
@pytest.mark.parametrize(
    "field",
    [
        "runtime_sha256",
        "provider_entrypoint",
        "server_entrypoint",
        "artifact_schema",
        "request_schema",
        "response_schema",
        "legacy_servo_sha256",
        "simulator_sources_sha256",
        "native_sources_sha256",
    ],
)
def test_receipt_cannot_rebind_v3_entrypoints_or_source_identity(command_proof, target, field):
    spec, documents, _ = command_proof
    if target == "server":
        value = json.loads(documents["preflight.json"])
        value["model_process"]["admission"][field] = "wrong"
        documents["preflight.json"] = canonical(value)
    else:
        value = json.loads(documents["probe.json"])
        value["model_admission"][field] = "wrong"
        documents["probe.json"] = canonical(value)
    with pytest.raises(ValueError, match="admission"):
        batch_learned.rescore_evidence(spec, documents)


def test_legacy_runtime_descriptor_does_not_admit_v3_receipt_flags(evidence):
    spec, _, _, _, report, documents = evidence
    report["model_admission"] = {"admission_kind": "azureml_command_v3"}
    documents["probe.json"] = canonical(report)
    with pytest.raises(ValueError, match="Legacy"):
        batch_learned.rescore_evidence(spec, documents)


def test_v3_receipt_requires_new_descriptor_not_relabelled_legacy_bytes(command_proof):
    spec, documents, _ = command_proof
    value = json.loads(documents["preflight.json"])
    original = base64.b64decode(value["model_process"]["runtime_descriptor_base64"])
    descriptor = json.loads(original)
    descriptor["schema"] = "physicalai.paused-model-runtime/v1"
    changed = canonical(descriptor)
    spec = spec.model_copy(
        update={
            "model_runtime": spec.model_runtime.model_copy(
                update={
                    "sha256": digest(changed),
                    "size_bytes": len(changed),
                }
            )
        }
    )
    value["model_process"].update(
        runtime_sha256=spec.model_runtime.sha256,
        runtime_descriptor_base64=base64.b64encode(changed).decode("ascii"),
    )
    documents["preflight.json"] = canonical(value)
    with pytest.raises(ValueError):
        batch_learned.rescore_evidence(spec, documents)
