"""CPU mapping/quality tests. Metadata doubles are not actual GPU or learning evidence."""

import importlib
from copy import deepcopy
from dataclasses import asdict
from uuid import UUID

import pytest

from learning.checks.fixtures import PROVENANCE
from learning.common import canonical, digest
from learning.paused.task import PREDICATE_SOURCE
from tests.learning.test_paused_evaluation import plan


def adapter():
    return importlib.import_module("simulation.paired_evaluation")


@pytest.fixture
def mapping_value():
    native = plan()
    runtime = {
        "schema": "physicalai.paused-evaluation-runtime/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "provenance": {
            **asdict(PROVENANCE),
            "source_kind": "isaac_sim",
            "capture_host": "azure_gpu",
            "gpu_model": "CPU metadata double; not hardware evidence",
        },
        "control_profile": native["control_profile"],
        "source_files": {"simulation/control.py": PREDICATE_SOURCE["sha256"]},
        "probe_receipt_sha256": "c" * 64,
        "test_only": False,
    }
    native["runtime_sha256"] = digest(canonical(runtime))
    assignments = []
    for index, case in enumerate(native["cases"]):
        for role in ("before", "after") if index % 2 == 0 else ("after", "before"):
            assignments.append(
                {
                    "logical_case_id": case["episode_id"],
                    "role": role,
                    "physical_attempt_id": str(UUID(int=len(assignments) + 1)),
                }
            )
    return {
        "schema": "physicalai.managed-paired-plan/v1",
        "issued_at_utc": "2026-09-27T00:00:00Z",
        "evaluation_plan": native,
        "runtime": runtime,
        "model_runtime_sha256": "a" * 64,
        "assignments": assignments,
    }


def test_frozen_mapping_keeps_twenty_logical_cases_and_forty_distinct_physical_ids(mapping_value):
    original = deepcopy(mapping_value)
    mapping = adapter().PairingPlan.model_validate(mapping_value)
    assert len(mapping.assignments) == 40
    assert len({item.physical_attempt_id for item in mapping.assignments}) == 40
    assert len({item.logical_case_id for item in mapping.assignments}) == 20
    assert mapping_value == original
    assert all(
        str(item.physical_attempt_id) != item.logical_case_id for item in mapping.assignments
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "physical-duplicate",
        "logical-duplicate",
        "role",
        "seed",
        "profile",
        "criteria",
        "order",
        "same-model",
        "fixture",
    ],
)
def test_invalid_mapping_cannot_relax_native_frozen_comparison(mapping_value, change):
    value = deepcopy(mapping_value)
    if change == "missing":
        value["assignments"].pop()
    elif change == "physical-duplicate":
        value["assignments"][-1]["physical_attempt_id"] = value["assignments"][0][
            "physical_attempt_id"
        ]
    elif change == "logical-duplicate":
        value["assignments"][-1]["logical_case_id"] = value["assignments"][0]["logical_case_id"]
    elif change == "role":
        value["assignments"][0]["role"] = "candidate"
    elif change == "seed":
        value["evaluation_plan"]["cases"][0]["seed"] = 900002
    elif change == "profile":
        value["evaluation_plan"]["control_profile_sha256"] = "0" * 64
    elif change == "criteria":
        value["evaluation_plan"]["criteria_sha256"] = "not-a-hash"
    elif change == "order":
        value["assignments"][0], value["assignments"][1] = (
            value["assignments"][1],
            value["assignments"][0],
        )
    elif change == "same-model":
        value["evaluation_plan"]["policy_after_sha256"] = value["evaluation_plan"][
            "policy_before_sha256"
        ]
    else:
        value["runtime"]["test_only"] = True
    with pytest.raises(ValueError):
        adapter().PairingPlan.model_validate(value)


def test_empty_snapshot_retains_all_slots_and_cannot_pass_quality(mapping_value):
    mapping = adapter().PairingPlan.model_validate(mapping_value)
    result = adapter().summarize(mapping, [], mapping_sha256="b" * 64, evidence_sha256="c" * 64)
    assert result["complete"] is False and result["quality_gate_passed"] is False
    assert result["conclusion"] == "inconclusive"
    assert len(result["attempts"]) == len(result["missing"]) == 40
    assert all(item["status"] == "missing" for item in result["attempts"])
    assert "comparison" not in result


def test_snapshot_rejects_duplicate_foreign_or_unsafe_attempt_inventory(mapping_value):
    expected = mapping_value["assignments"][0]["physical_attempt_id"]
    value = {
        "schema": "physicalai.managed-paired-evidence/v1",
        "mapping_sha256": "b" * 64,
        "snapshot_at_utc": "2026-09-27T01:00:00Z",
        "attempts": [
            {
                "physical_attempt_id": expected,
                "files": {"inputs/spec.json": {"sha256": "c" * 64, "bytes": 128}},
            }
        ],
    }
    adapter().PairingEvidence.model_validate(value)
    for altered in (
        {**value, "attempts": value["attempts"] * 2},
        {
            **value,
            "attempts": [
                {
                    **value["attempts"][0],
                    "files": {
                        "../foreign.json": {"sha256": "c" * 64, "bytes": 128},
                    },
                }
            ],
        },
    ):
        with pytest.raises(ValueError):
            adapter().PairingEvidence.model_validate(altered)


def test_score_projection_never_mutates_or_rewrites_the_original_physical_trials(mapping_value):
    from tests.learning.test_paused_evaluation import trials

    mapping = adapter().PairingPlan.model_validate(mapping_value)
    numeric = {
        (value["episode_id"], value["policy"]): value for value in trials(mapping.evaluation_plan)
    }
    records = []
    for assignment in mapping.assignments:
        trial = deepcopy(numeric[assignment.logical_case_id, assignment.role])
        trial["episode_id"] = str(assignment.physical_attempt_id)
        records.append(
            {
                "physical_attempt_id": str(assignment.physical_attempt_id),
                "logical_case_id": assignment.logical_case_id,
                "role": assignment.role,
                "status": "verified",
                "physical_trial": trial,
                "completion_sha256": "d" * 64,
            }
        )
    originals = deepcopy(records)
    # Only the arithmetic projection is tested here, not the live artifact verifier.
    result = adapter().summarize(
        mapping,
        records,
        mapping_sha256="b" * 64,
        evidence_sha256="c" * 64,
        evidence_verified=True,
    )
    assert records == originals
    assert result["complete"] is True
    assert result["comparison"]["counts"] == {
        "before": {"total": 20, "success": 17},
        "after": {"total": 20, "success": 18},
    }
    assert result["comparison"]["absolute_success_rate_improvement"] == pytest.approx(0.05)
    assert result["quality_gate_passed"] is True
    assert result["schema"] == "physicalai.managed-paired-report/v1"
    assert all(
        record["physical_trial"]["episode_id"] == record["physical_attempt_id"]
        for record in result["attempts"]
    )
    partial = adapter().summarize(
        mapping, records[:-1], mapping_sha256="b" * 64, evidence_sha256="c" * 64
    )
    assert partial["quality_gate_passed"] is False and not partial["complete"]
    assert "comparison" not in partial
    unverified = adapter().summarize(
        mapping, records, mapping_sha256="b" * 64, evidence_sha256="c" * 64
    )
    assert unverified["quality_gate_passed"] is False


def test_shared_native_model_validation_preserves_lineage_and_heldout_denial(tmp_path):
    from learning.common import file_digest, read_json
    from learning.contract import Scope
    from learning.paused import evaluation
    from tests.learning.test_paused_model_artifacts import fixture

    native = plan()
    roots = {role: tmp_path / role for role in ("before", "after")}
    for role, root in roots.items():
        fixture(root)
        model = read_json(root / "model.json")
        model["training"]["episodes"] = [{"episode_id": "training-10001", "seed": 10001}]
        if role == "after":
            model["training"]["parent_model_sha256"] = native["policy_before_sha256"]
        (root / "model.json").write_bytes(canonical(model))
        native[f"policy_{role}_sha256"] = file_digest(root / "model.json")
    assert hasattr(evaluation, "validate_models"), "Reuse the existing native lineage gate."
    checked = evaluation.validate_models(native, roots, scope=Scope(**native["scope"]))
    assert set(checked) == {"before", "after"}
    model = read_json(roots["after"] / "model.json")
    model["training"]["episodes"][0]["seed"] = 30001
    (roots["after"] / "model.json").write_bytes(canonical(model))
    native["policy_after_sha256"] = file_digest(roots["after"] / "model.json")
    with pytest.raises(ValueError, match="leakage"):
        evaluation.validate_models(native, roots, scope=Scope(**native["scope"]))
