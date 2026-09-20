from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from learning.checks.fixtures import LIVE_PROVENANCE, PROVENANCE, SCOPE, frame, overwrite
from learning.common import ContractError, canonical, digest, read_json
from learning.contract import validate_dataset
from learning.evaluation import (
    EpisodeOutcome,
    EvaluationCase,
    EvaluationRecorder,
    GateThresholds,
    build_evaluation_plan,
    evaluate_results,
)

pytest_plugins = ("learning.checks.pytest_fixtures",)


@pytest.fixture
def evaluation(make_dataset, tmp_path):
    def make(*, count=2, fixture=True, fail_episode=None, truncated=False, safety=False):
        dataset = validate_dataset(make_dataset(test_count=count), expected_scope=SCOPE)
        model = {
            "scope": asdict(SCOPE),
            "raw_manifest_sha256": dataset.manifest_sha256,
            "model_name": "franka-act",
            "model_version": "1",
            "training": {
                "test_only": fixture,
                "device": "cpu" if fixture else "cuda",
                "azureml_run_id": None if fixture else "unit-test-ml-attestation",
                "gpu_model": None if fixture else "unit-test-gpu-attestation",
                "episodes": [{"episode_id": "train-episode", "seed": 1}],
            },
            "observation": {"fps": 10, "physics_hz": 60},
        }
        runtime = {
            "azure_resource_id": (
                "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/rg"
                "/providers/Microsoft.Compute/virtualMachines/simulator"
            ),
            "run_id": "unit-test-run",
            "provenance": asdict(PROVENANCE if fixture else LIVE_PROVENANCE),
        }
        cases = [
            EvaluationCase(
                ep.metadata["episode_id"],
                ep.metadata["environment_id"],
                ep.metadata["revision"],
                ep.metadata["seed"],
                bool(ep.metadata["seed"] % 2),
                "rejected" if ep.metadata["seed"] % 2 else "accepted",
                (0.3, 0.1, 0.2),
            )
            for ep in dataset.split("test")
        ]
        model_sha = "a" * 64
        plan = build_evaluation_plan(
            dataset,
            model,
            cases,
            model_sha256=model_sha,
            baseline_revision="e" * 40,
            expected_runtime=runtime,
        )
        root = tmp_path / "evidence"
        recorder = EvaluationRecorder(root, plan=plan, runtime=runtime)
        for controller in ("baseline", "learned"):
            for case in cases:
                failed = controller == "learned" and case.episode_id == fail_episode
                outcome = EpisodeOutcome(
                    controller,
                    case.episode_id,
                    case.environment_id,
                    case.revision,
                    case.seed,
                    "2026-09-20T00:00:00Z",
                    "2026-09-20T00:00:00.2Z",
                    200_000_000,
                    12,
                    2,
                    12,
                    1_200_000_000,
                    (0.9, 0.9, 0.9) if failed else case.expected_destination_position_m,
                    case.expected_destination_id,
                    not (failed and truncated),
                    failed and truncated,
                    "timeout" if failed and truncated else None,
                    ("cartesian_speed_watchdog",) if failed and safety else (),
                    (10.0, 20.0),
                )
                recorder.record(outcome, frame(2).images)
        recorder.finalize()
        return plan, root, model, model_sha

    return make


def evaluate(bundle):
    plan, root, model, model_sha = bundle
    return evaluate_results(
        plan,
        root,
        model,
        expected_scope=SCOPE,
        expected_model_sha256=model_sha,
        expected_plan_sha256=digest(canonical(plan)),
    )


def test_cpu_fixture_success_never_passes_live_learning_quality(evaluation):
    result = evaluate(evaluation())
    assert result["passed"] is False and result["learning_quality_verified"] is False
    assert result["success_counts"] == {"baseline": 2, "learned": 2}
    assert result["total_episode_count"] == 4
    assert result["reasons"] == [
        "no_live_azure_gpu_episodes",
        "no_production_gpu_trained_checkpoint",
        "insufficient_heldout_episodes",
    ]


def test_gate_arithmetic_with_mocked_trusted_gpu_attestation(evaluation):
    result = evaluate(evaluation(count=20, fixture=False))
    assert result["passed"] is True
    assert result["success_counts"] == {"baseline": 20, "learned": 20}
    assert result["latency_p95_ms"] == {"baseline": 20.0, "learned": 20.0}
    assert result["total_episode_count"] == 40
    assert result["failures"] == []


@pytest.mark.parametrize("mode", ["pose", "truncated", "safety"])
def test_final_pose_and_failures_are_recomputed_not_declared_success(evaluation, mode):
    result = evaluate(
        evaluation(
            fail_episode="test-0",
            truncated=mode == "truncated",
            safety=mode == "safety",
        )
    )
    assert result["success_counts"] == {"baseline": 2, "learned": 1}
    assert result["success_rates"]["learned"] == 0.5
    assert len(result["failures"]) == 1
    assert result["failures"][0]["episode_id"] == "test-0"
    assert "learned_policy_regresses_against_paired_baseline" in result["reasons"]
    if mode == "safety":
        assert "safety_violation" in result["reasons"]
    if mode == "truncated":
        assert result["failures"][0]["reason"] == "timeout"


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "duplicate", "seed", "model", "scope", "baseline"]
)
def test_incomplete_or_unbound_evaluation_evidence_rejected(evaluation, kind):
    bundle = evaluation()
    plan, root, _, _ = bundle
    results = read_json(root / "results.json")
    if kind == "missing":
        results["episodes"].pop()
    elif kind == "extra":
        results["episodes"].append(results["episodes"][0])
    elif kind == "duplicate":
        results["episodes"][-1] = results["episodes"][0]
    elif kind == "seed":
        results["episodes"][0]["seed"] = 1
    elif kind == "model":
        results["model_sha256"] = "b" * 64
    elif kind == "scope":
        results["scope"]["owner_id"] = "c" * 64
    else:
        results["baseline_revision"] = "b" * 40
    overwrite(root / "results.json", results)
    with pytest.raises(ContractError):
        evaluate(bundle)
    assert plan["expected_episode_count"] == 2


@pytest.mark.parametrize("kind", ["count", "negative", "nan", "image", "done", "zero-action"])
def test_missing_invalid_or_forged_measurements_fail_closed(evaluation, kind):
    bundle = evaluation()
    _, root, _, _ = bundle
    results = read_json(root / "results.json")
    record = results["episodes"][0]
    if kind == "count":
        record["latencies_ms"].pop()
    elif kind == "negative":
        record["latencies_ms"][0] = -1
    elif kind == "nan":
        path = root / "results.json"
        path.write_text(path.read_text().replace("10.0", "NaN", 1))
    elif kind == "image":
        record["final_images"]["inspection"]["sha256"] = "0" * 64
    elif kind == "done":
        record["terminated"] = False
    else:
        record["control_steps"] = 0
        record["latencies_ms"] = []
    if kind != "nan":
        overwrite(root / "results.json", results)
    with pytest.raises(ContractError):
        evaluate(bundle)


def test_cross_split_seed_leakage_is_rechecked_at_gate(evaluation):
    bundle = evaluation()
    plan, root, model, _ = bundle
    model["training"]["episodes"].append({"episode_id": "different-id", "seed": 100})
    with pytest.raises(ContractError, match="leakage"):
        evaluate(bundle)
    assert root.exists() and plan["cases"][0]["seed"] == 100


def test_thresholds_cannot_disable_episode_floor_or_make_latency_unbounded():
    for change in (
        {"min_episodes": 0},
        {"min_episodes": 1},
        {"max_latency_p95_ms": float("inf")},
        {"min_success_rate": 0},
        {"max_success_regression": 1},
    ):
        with pytest.raises(ContractError):
            replace(GateThresholds(), **change).validate()


def test_incomplete_recorder_has_no_publishable_results(evaluation, tmp_path):
    plan, _, _, _ = evaluation()
    root = tmp_path / "incomplete"
    recorder = EvaluationRecorder(root, plan=plan, runtime=plan["expected_runtime"])
    with pytest.raises(ContractError, match="Incomplete"):
        recorder.finalize()
    assert not (root / "results.json").exists()
