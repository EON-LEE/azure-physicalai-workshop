from copy import deepcopy

import pytest

from apps.api.errors import Problem
from apps.api.learning_ports import JobSpecification
from apps.learning_worker.reports import project_report
from tests.runtime_support import ACTOR
from tests.test_learning_results import evaluated_setup


def sample():
    service, project, baseline, candidate, evaluation = evaluated_setup(10, 10)
    project = project.model_copy(update={"policy_type": "smolvla"})
    run = evaluation.value.model_copy(update={"policy_type": "smolvla"})
    specification = JobSpecification(
        owner_key=ACTOR.owner_key,
        project=project,
        run=run,
        baseline=baseline.model_copy(update={"policy_type": "smolvla"}),
        candidate=candidate.model_copy(update={"policy_type": "smolvla"}),
    )
    native = {
        "schema": "physicalai.smolvla-paired-report/v1",
        "policy_type": "smolvla",
        "scope": {"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key},
        "azure_job_id": run.azure_job_id,
        "azure_component_job_id": f"{run.azure_job_id}-component",
        "specification_sha256": run.specification_sha256,
        "plan_sha256": "9" * 64,
        "runtime_sha256": "8" * 64,
        "control_profile_sha256": "7" * 64,
        "policy_before_sha256": baseline.model_sha256,
        "policy_after_sha256": candidate.model_sha256,
        "conclusion": "not_improved",
        "quality_gate": False,
        "live_gpu_verified": True,
        "total_trial_count": 40,
        "trials": [
            {
                "episode_id": f"test-only-{trial.policy}-{trial.seed}",
                "seed": trial.seed,
                "attempt": trial.attempt,
                "policy": trial.policy,
                "model_sha256": trial.model_sha256,
                "environment_id": trial.environment_id,
                "revision": trial.revision,
                "observed_initial_pose_m": [0.35, 0.25, 0.2],
                "scene_builder_sha256": "6" * 64,
                "final_pose_m": [0.42, -0.22, 0.2],
                "destination_id": "accepted",
                "terminated": True,
                "truncated": False,
                "failure_reason": None if trial.physical_success else "missed_goal",
                "latencies_ms": [10, 20, 30, 40, 50, 60, 70, 80],
                "duration_ms": 20500,
                "safety_violations": [],
                "policy_predict_calls": 20,
                "applied_action_count": 120,
                "reference_route_calls": 0,
                "final_images": {"inspection": "5" * 64, "overview": "4" * 64},
                "physical_success": trial.physical_success,
                "position_error_m": trial.axis_error_m,
            }
            for trial in run.report.trials
        ],
    }
    return specification, native


def test_native_report_projection_retains_failures_actual_latency_pose_and_distinct_plan_hashes():
    specification, native = sample()
    report = project_report(
        ACTOR,
        specification,
        native,
        report_sha256="3" * 64,
        expected_plan_sha256="9" * 64,
    )
    assert report.conclusion == "not_improved"
    assert len(report.trials) == 40
    assert sum(trial.status == "failed" for trial in report.trials) == 20
    assert report.trials[0].inference_p95_ms == 80
    assert report.trials[0].duration_seconds == 20.5
    assert report.trials[0].applied_action_count == 120
    assert report.trials[0].scene_builder_sha256 == "6" * 64
    assert report.trials[0].final_inspection_sha256 == "5" * 64
    assert report.native_plan_sha256 == "9" * 64
    assert report.evaluation_plan_sha256 == specification.project.evaluation_plan.sha256
    assert report.native_plan_sha256 != report.evaluation_plan_sha256


@pytest.mark.parametrize(
    "field", ["scope", "plan_sha256", "azure_job_id", "specification_sha256", "live_gpu_verified"]
)
def test_native_scope_gpu_and_exact_job_proofs_cannot_be_dropped(field):
    specification, native = sample()
    native.pop(field)
    with pytest.raises(Problem):
        project_report(
            ACTOR, specification, native, report_sha256="3" * 64, expected_plan_sha256="9" * 64
        )


def test_only_exact_cancel_reason_is_projected_as_cancellation():
    specification, native = sample()
    failed = next(row for row in native["trials"] if not row["physical_success"])
    original = deepcopy(failed)
    failed["failure_reason"] = "cancelled"
    report = project_report(
        ACTOR, specification, native, report_sha256="3" * 64, expected_plan_sha256="9" * 64
    )
    assert (
        next(t for t in report.trials if t.episode_id == failed["episode_id"]).status == "cancelled"
    )
    failed["failure_reason"] = "text mentioning cancelled but not an actual status"
    report = project_report(
        ACTOR, specification, native, report_sha256="3" * 64, expected_plan_sha256="9" * 64
    )
    assert (
        next(t for t in report.trials if t.episode_id == original["episode_id"]).status == "failed"
    )
