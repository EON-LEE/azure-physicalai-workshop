"""CPU-only cohort isolation checks; no trained policy, GPU trial or final quality proof."""

import json

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_simulation_batch import spec as spec

from learning.common import canonical, digest
from simulation import batch_learned, learned_probe, paired_evaluation


def test_default_test_cohort_preserves_the_exact_existing_spec_wire_hash(learned_spec):
    original = learned_spec.model_dump(mode="json", by_alias=True)
    assert "evaluation_split" not in original
    explicit = batch_learned.BatchLearnedSpec.model_validate(
        {**original, "evaluation_split": "test"}
    )
    assert explicit.evaluation_split == learned_spec.evaluation_split == "test"
    assert explicit.model_dump(mode="json", by_alias=True) == original
    assert explicit.sha256 == learned_spec.sha256 == digest(canonical(original))
    validation = batch_learned.BatchLearnedSpec.model_validate(
        {**original, "evaluation_split": "validation"}
    )
    assert validation.sha256 != explicit.sha256
    assert validation.model_dump(mode="json", by_alias=True)["evaluation_split"] == "validation"


@pytest.mark.parametrize(
    "change",
    [
        {"role": "before"},
        {"role": "after"},
        {"pairing_plan_sha256": "e" * 64},
        {"evaluation_split": "train"},
        {"evaluation_split": "integration"},
    ],
)
def test_validation_is_only_an_unpaired_candidate_trial(learned_spec, change):
    value = {
        **learned_spec.model_dump(mode="json", by_alias=True),
        "evaluation_split": "validation",
        **change,
    }
    with pytest.raises(ValueError):
        batch_learned.BatchLearnedSpec.model_validate(value)


@pytest.mark.parametrize(
    "authority",
    [
        {"evaluation_split": "validation", "split": "validation", "seed": seed}
        for seed in range(20001, 20011)
    ],
    indirect=True,
)
def test_original_validation_cases_are_admitted_with_the_same_grant_and_limits(authority):
    spec, documents, scene, profile = authority
    environment, actual, actual_profile, grant = batch_learned.validate_inputs(spec, documents)
    assert actual == scene and actual_profile == profile
    assert environment.document["execution"]["demonstration_split"] == "validation"
    assert 20001 <= actual.seed <= 20010
    assert grant.authorization.controller == "learned"
    assert grant.authorization.model_sha256 == spec.model.manifest.sha256
    assert grant.authorization.purpose == "evaluation"
    assert grant.authorization.max_simulation_steps == 3600
    assert (grant.expires_at - grant.issued_at).total_seconds() == 600


@pytest.mark.parametrize(
    "authority",
    [
        {"evaluation_split": "validation", "split": "validation", "seed": 20000},
        {"evaluation_split": "validation", "split": "validation", "seed": 20011},
        {"evaluation_split": "validation", "split": "validation", "seed": 30001},
        {"evaluation_split": "validation", "split": "test", "seed": 20001},
        {"evaluation_split": "validation", "split": "train", "seed": 10001},
        {"evaluation_split": "validation", "split": "test", "seed": 900002},
        {"evaluation_split": "test", "split": "validation", "seed": 20001},
        {"evaluation_split": "test", "split": "test", "seed": 20001},
    ],
    indirect=True,
)
def test_relabelled_or_out_of_cohort_saved_scenes_cannot_authorize_motion(authority):
    spec, documents, _, _ = authority
    with pytest.raises(ValueError, match="cohort"):
        batch_learned.validate_inputs(spec, documents)


@pytest.mark.parametrize(
    "authority",
    [{"evaluation_split": "validation", "split": "validation", "seed": 20001}],
    indirect=True,
)
@pytest.mark.parametrize("change", ["missing", "split", "seed", "revision"])
def test_validation_must_still_match_the_original_predeclared_frozen_case(authority, change):
    spec, documents, _, _ = authority
    conditions = json.loads(documents["conditions"])
    if change == "missing":
        conditions["cases"] = []
    else:
        conditions["cases"][0][change] = {
            "split": "test",
            "seed": 30001,
            "revision": "f" * 64,
        }[change]
    documents["conditions"] = canonical(conditions)
    grant = json.loads(documents["grant"])
    grant["authorization"]["frozen_plan_sha256"] = digest(documents["conditions"])
    documents["grant"] = canonical(grant)
    value = spec.model_dump(mode="json", by_alias=True)
    value["conditions_canonical_sha256"] = digest(documents["conditions"])
    for name in ("conditions", "grant"):
        value[name].update(sha256=digest(documents[name]), size_bytes=len(documents[name]))
    changed = batch_learned.BatchLearnedSpec.model_validate(value)
    with pytest.raises(ValueError, match="frozen|definition"):
        batch_learned.validate_inputs(changed, documents)


def test_final_pairing_rejects_validation_even_if_an_internal_caller_bypasses_dto_checks(
    learned_spec,
):
    from types import SimpleNamespace

    spec = learned_spec.model_copy(
        update={"evaluation_split": "validation", "role": "before", "pairing_plan_sha256": "a" * 64}
    )
    mapping = SimpleNamespace(
        evaluation_plan={
            "cases": [{"episode_id": "case"}],
            "policy_before_sha256": spec.model.manifest.sha256,
            "scope": {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id},
            "control_profile_sha256": spec.control_profile_sha256,
            "control_profile": {"profile_id": spec.profile_id},
            "criteria_sha256": spec.criteria_canonical_sha256,
            "frozen_plan_sha256": spec.conditions_canonical_sha256,
        },
        runtime={
            "provenance": {
                "code_revision": spec.source_revision,
                "simulator_image_digest": spec.platform.container_image.split("@", 1)[1],
                "robot_asset_sha256": spec.asset_bundle.sha256,
            }
        },
        model_runtime_sha256=spec.model_runtime.sha256,
    )
    assignment = paired_evaluation.Assignment(
        logical_case_id="case", role="before", physical_attempt_id=spec.attempt_id
    )
    model = {
        "checkpoint_files": {
            item.path.removeprefix("checkpoint/"): item.sha256 for item in spec.model.files
        },
        "backbone_manifest_sha256": spec.backbone.manifest.sha256,
    }
    with pytest.raises(ValueError, match="binding"):
        paired_evaluation.bind_attempt(
            mapping, assignment, spec, {"before": model}, mapping_sha256="a" * 64
        )


@pytest.mark.parametrize(
    "authority",
    [{"evaluation_split": "validation", "split": "validation", "seed": 20001}],
    indirect=True,
)
def test_validation_rescore_is_explicit_and_never_final_quality(evidence):
    spec, scene, profile, grant, report, documents = evidence
    report["evaluation_split"] = "validation"
    acceptance = learned_probe.rescore(report, spec, scene, profile, grant)
    assert acceptance["accepted"] is True
    assert acceptance["evaluation_split"] == "validation"
    assert acceptance["learning_quality_proven"] is False
    assert acceptance.get("quality_gate_passed", False) is False
    documents.update({"probe.json": canonical(report), "acceptance.json": canonical(acceptance)})
    receipt, checked = batch_learned.rescore_evidence(spec, documents)
    assert checked == acceptance and receipt == report["capture"]["receipt"]


@pytest.mark.parametrize(
    "authority",
    [{"evaluation_split": "validation", "split": "validation", "seed": 20001}],
    indirect=True,
)
@pytest.mark.parametrize("reported", [None, "test"])
def test_validation_report_cannot_be_unlabelled_or_claim_test(evidence, reported):
    spec, scene, profile, grant, report, _ = evidence
    if reported is not None:
        report["evaluation_split"] = reported
    with pytest.raises(ValueError, match="cohort"):
        learned_probe.rescore(report, spec, scene, profile, grant)


def test_existing_test_report_cannot_be_relabelled_for_validation(evidence):
    spec, scene, profile, grant, report, _ = evidence
    report["evaluation_split"] = "validation"
    with pytest.raises(ValueError, match="cohort"):
        learned_probe.rescore(report, spec, scene, profile, grant)


@pytest.mark.parametrize(
    "authority",
    [{"evaluation_split": "validation", "split": "validation", "seed": 20001}],
    indirect=True,
)
@pytest.mark.parametrize("change", [{"split": "test"}, {"split": "train"}, {"seed": 30001}])
def test_rehashed_raw_capture_cannot_spoof_another_evaluation_cohort(evidence, change):
    spec, scene, profile, grant, report, documents = evidence
    report["evaluation_split"] = "validation"
    manifest = json.loads(documents["raw-manifest.json"])
    manifest["episodes"][0].update(change)
    documents["raw-manifest.json"] = canonical(manifest)
    report["capture"]["receipt"]["manifest_sha256"] = digest(documents["raw-manifest.json"])
    acceptance = learned_probe.rescore(report, spec, scene, profile, grant)
    documents.update({"probe.json": canonical(report), "acceptance.json": canonical(acceptance)})
    with pytest.raises(ValueError, match="capture"):
        batch_learned.rescore_evidence(spec, documents)
