"""CPU model metadata verifies explicit paired admission, never GPU/model-quality qualification."""

import json
from copy import deepcopy

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_command_runtime_admission import command_runtime_value as command_runtime_value
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_paired_artifacts import recording as recording
from test_paired_evaluation import mapping_value as mapping_value
from test_simulation_batch import spec as spec

from learning.common import canonical, digest, file_digest, read_json
from learning.contract import Scope
from simulation import batch_learned, paired_evaluation


@pytest.fixture
def command_pair(recording, command_runtime_value, tmp_path):
    _, _, mapping, _, _, roots, spec, snapshot = recording
    for role in ("before", "after"):
        path = roots[role] / "model.json"
        metadata = read_json(path)
        metadata.update(
            schema="physicalai.smolvla-checkpoint/v3", training_execution="azureml_command"
        )
        metadata["training"]["azure_job_type"] = "command"
        metadata["training"]["gpu"]["device_count"] = 1
        metadata["training"]["episodes"] = [
            {
                "episode_id": "cpu-training-10001",
                "environment_id": "cpu-training-case",
                "revision": "a" * 64,
                "seed": 10001,
            }
        ]
        job_prefix = metadata["training"]["azure_job_id"].rsplit("/", 1)[0]
        metadata["training"]["azure_job_id"] = job_prefix + "/cpu-" + role
        metadata["training"].pop("azure_pipeline_job_id", None)
        metadata["training"].pop("azure_component_job_id", None)
        if role == "after":
            metadata["training"]["parent_model_sha256"] = file_digest(
                roots["before"] / "model.json"
            )
        path.write_bytes(canonical(metadata))
    value = mapping.model_dump(mode="json", by_alias=True)
    for role in ("before", "after"):
        value["evaluation_plan"][f"policy_{role}_sha256"] = file_digest(roots[role] / "model.json")
    runtime = batch_learned.parse_model_runtime(
        {
            **command_runtime_value,
            "legacy_servo_sha256": value["evaluation_plan"]["control_profile"][
                "servo_profile_sha256"
            ],
            "control_profile_sha256": value["evaluation_plan"]["control_profile_sha256"],
            "simulator_image": spec.platform.container_image,
            "simulator_source_revision": spec.source_revision,
        }
    )
    encoded = canonical(runtime.model_dump(mode="json", by_alias=True))
    value["model_runtime_sha256"] = digest(encoded)
    mapping = paired_evaluation.PairingPlan.model_validate(value)
    evidence = paired_evaluation.PairingEvidence(
        schema="physicalai.managed-paired-evidence/v1",
        mapping_sha256="a" * 64,
        snapshot_at_utc=snapshot,
        attempts=[],
    )
    root = tmp_path / "unreceived-command-trials"
    root.mkdir()
    return root, mapping, evidence, roots, encoded


def test_command_pair_requires_exact_v2_descriptor_and_native_v3_model_validation(command_pair):
    root, mapping, evidence, models, runtime = command_pair
    result = paired_evaluation.aggregate(
        root,
        mapping,
        evidence,
        mapping_sha256="a" * 64,
        evidence_sha256="b" * 64,
        before_root=models["before"],
        after_root=models["after"],
        model_runtime=runtime,
    )
    assert not result["complete"] and result["quality_gate_passed"] is False
    assert result["model_admission"]["artifact_schema"] == "physicalai.smolvla-checkpoint/v3"
    assert result["model_admission"]["runtime_sha256"] == mapping.model_runtime_sha256
    assert len(result["missing"]) == 40
    with pytest.raises(ValueError, match="v2|provenance"):
        paired_evaluation.aggregate(
            root,
            mapping,
            evidence,
            mapping_sha256="a" * 64,
            evidence_sha256="b" * 64,
            before_root=models["before"],
            after_root=models["after"],
        )


def test_legacy_descriptor_and_command_candidate_cannot_be_mixed(command_pair):
    root, mapping, evidence, models, original = command_pair
    value = json.loads(original)
    legacy = {
        name: value[field.alias or name]
        for name, field in batch_learned.ModelRuntime.model_fields.items()
    }
    legacy["schema_version"] = "physicalai.paused-model-runtime/v1"
    legacy["schema"] = legacy.pop("schema_version")
    runtime = canonical(legacy)
    mapping = mapping.model_copy(update={"model_runtime_sha256": digest(runtime)})
    with pytest.raises(ValueError):
        paired_evaluation.aggregate(
            root,
            mapping,
            evidence,
            mapping_sha256="a" * 64,
            evidence_sha256="b" * 64,
            before_root=models["before"],
            after_root=models["after"],
            model_runtime=runtime,
        )


@pytest.mark.parametrize("change", ["runtime-bytes", "image", "profile", "fake-pipeline"])
def test_paired_command_admission_rejects_mismatched_descriptor_and_false_lineage(
    command_pair, change
):
    root, mapping, evidence, models, runtime = command_pair
    if change == "runtime-bytes":
        runtime += b" "
    elif change == "fake-pipeline":
        path = models["after"] / "model.json"
        metadata = read_json(path)
        metadata["training"]["azure_pipeline_job_id"] = metadata["training"]["azure_job_id"]
        path.write_bytes(canonical(metadata))
        native = deepcopy(mapping.evaluation_plan)
        native["policy_after_sha256"] = file_digest(path)
        mapping = mapping.model_copy(update={"evaluation_plan": native})
    else:
        value = json.loads(runtime)
        value["simulator_image" if change == "image" else "control_profile_sha256"] = (
            "unit.azurecr.io/other@sha256:" + "f" * 64 if change == "image" else "f" * 64
        )
        runtime = canonical(value)
        mapping = mapping.model_copy(update={"model_runtime_sha256": digest(runtime)})
    with pytest.raises(ValueError):
        paired_evaluation.aggregate(
            root,
            mapping,
            evidence,
            mapping_sha256="a" * 64,
            evidence_sha256="b" * 64,
            before_root=models["before"],
            after_root=models["after"],
            model_runtime=runtime,
        )


def test_legacy_and_command_model_selection_never_rewrites_the_actual_manifest(command_pair):
    _, _, _, models, encoded = command_pair
    command = batch_learned.parse_model_runtime(json.loads(encoded))
    original = (models["before"] / "model.json").read_bytes()
    model = json.loads(original)
    accepted = batch_learned.validate_runtime_model(
        command, models["before"], scope=Scope(**model["scope"]), model_sha256=digest(original)
    )
    assert accepted == model
    assert (models["before"] / "model.json").read_bytes() == original
    assert accepted["training"]["azure_job_type"] == "command"
    assert not {"azure_pipeline_job_id", "azure_component_job_id"} & accepted["training"].keys()
