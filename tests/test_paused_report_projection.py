from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_models import CreateProject, LearningProject
from learning.common import canonical, digest
from learning.paused.evaluation import compare_trials
from learning.paused.task import PREDICATE_SOURCE, PREDICATE_VERSION
from tests.learning.test_paused_evaluation import plan, trials
from tests.runtime_support import ACTOR
from tests.test_paused_learning_api import paused_project_payload


def specimen(*, bootstrap=False, before=17, after=18):
    native_plan = plan(bootstrap=bootstrap)
    native_plan["scope"] = {"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key}
    project_value = paused_project_payload()
    project_value.update(
        goal_station_id="rejected",
        control_profile_sha256=native_plan["control_profile_sha256"],
        criteria_sha256=native_plan["criteria_sha256"],
        frozen_plan_sha256=native_plan["frozen_plan_sha256"],
    )
    project_value["evaluation_plan"].update(
        seeds=[case["seed"] for case in native_plan["cases"]],
        cases=[
            {field: case[field] for field in ("seed", "environment_id", "revision")}
            for case in native_plan["cases"]
        ],
    )
    project = LearningProject.create(ACTOR, CreateProject.model_validate(project_value))
    native = compare_trials(
        native_plan, trials(native_plan, before=before, after=after), live_gpu_verified=True
    )
    native["results_sha256"] = "8" * 64
    for row in native["trials"]:
        row["task_evidence"].update(
            predicate_version=PREDICATE_VERSION,
            predicate_source=PREDICATE_SOURCE,
            settled_simulation_seconds=0.3 if row["physical_success"] else 0,
            final_goal_error_m=max(row["position_error_m"]),
            safety_violations=[],
        )
    run = SimpleNamespace(
        id=uuid4(),
        comparison_kind="reference_bootstrap" if bootstrap else "paired_policy",
        azure_job_id="/subscriptions/test/providers/Microsoft.MachineLearningServices/workspaces/test/jobs/evaluation",
        specification_sha256="6" * 64,
    )
    candidate = SimpleNamespace(
        model_sha256=native_plan["candidate_model_sha256"]
        if bootstrap
        else native_plan["policy_after_sha256"],
    )
    baseline = (
        None if bootstrap else SimpleNamespace(model_sha256=native_plan["policy_before_sha256"])
    )
    specification = SimpleNamespace(
        project=project, run=run, candidate=candidate, baseline=baseline
    )
    output = deepcopy(native)
    output.update(
        azure_job_id=run.azure_job_id,
        azure_component_job_id=run.azure_job_id + "-component",
        specification_sha256=run.specification_sha256,
    )
    return specification, native_plan, native, output


def project_verified(specification, native, output, **overrides):
    from apps.learning_worker.paused_reports import project_verified_report

    return project_verified_report(
        ACTOR,
        specification,
        output,
        verified_native=native,
        report_sha256="7" * 64,
        expected_plan_sha256=output["evaluation_plan_sha256"],
        expected_results_sha256="8" * 64,
        **overrides,
    )


def test_projection_requires_independent_rescore_not_live_flag_and_aggregate_json():
    specification, _, _, output = specimen()
    with pytest.raises(Problem) as failure:
        project_verified(specification, None, output)
    assert failure.value.code == "paused_report_unverified"


@pytest.mark.parametrize(
    "change",
    [
        {"quality_gate_passed": False},
        {"live_gpu_verified": False},
        {"results_sha256": None},
        {"total_trial_count": 39},
        {"real_time_admission": True},
        {"criteria_sha256": "0" * 64},
        {"frozen_plan_sha256": "0" * 64},
        {"azure_job_id": "/another/job"},
    ],
)
def test_output_cannot_change_any_verified_aggregate_or_scope(change):
    specification, _, native, output = specimen()
    with pytest.raises(Problem):
        project_verified(specification, native, output | change)


def test_projection_retains_absolute_wall_samples_task_proof_and_all_failed_attempts():
    specification, _, native, output = specimen()
    report = project_verified(specification, native, output)
    assert report.execution_timing == "paused_simulation"
    assert report.real_time_admission is False
    assert len(report.trials) == 40
    assert sum(not trial.physical_success for trial in report.trials) == 5
    assert report.trials[0].wall_duration_ms == 1400
    assert report.trials[0].simulation_duration_ms == 200
    assert report.trials[0].phase_wall_ms["policy"].samples == 2
    assert report.trials[0].phase_wall_ms["policy"].p95 == 260
    assert report.latency_wall_ms["after"].p95 == 260
    assert report.total_wall_duration_ms == 56000
    assert report.total_simulation_duration_ms == 8000
    assert report.native_plan_sha256 != report.evaluation_plan_sha256
    assert report.results_sha256 == "8" * 64
    assert report.criteria_sha256 == specification.project.criteria_sha256
    assert report.trials[0].task_evidence.predicate_source.sha256 == PREDICATE_SOURCE["sha256"]
    assert (
        report.trials[0].task_evidence.grasp_evidence_kind
        == "measured_lift_proximity_finger_gap_no_contact_sensor"
    )
    assert report.absolute_success_rate_improvement == pytest.approx(0.05)
    assert report.quality_gate_passed and report.conclusion == "improved"
    assert len(report.model_dump_json().encode()) < 512 * 1024


def test_static_final_pose_or_omitted_failed_trial_cannot_replace_verified_task_evidence():
    specification, _, native, output = specimen()
    changed = deepcopy(output)
    changed["trials"][0]["task_evidence"]["grasp_verified"] = False
    with pytest.raises(Problem):
        project_verified(specification, native, changed)
    changed = deepcopy(output)
    changed["trials"].pop()
    with pytest.raises(Problem):
        project_verified(specification, native, changed)


def test_reference_bootstrap_is_not_a_before_model_or_an_improvement_claim():
    specification, _, native, output = specimen(bootstrap=True, before=18, after=18)
    report = project_verified(specification, native, output)
    assert report.comparison_kind == "reference_bootstrap"
    assert report.before_model_sha256 is None and report.after_model_sha256 is None
    assert report.candidate_model_sha256 == specification.candidate.model_sha256
    assert report.reference_controller_sha256 == output["reference_controller_sha256"]
    assert report.trials[0].model_sha256 is None
    assert report.trials[0].reference_route_calls == 2
    assert report.trials[0].applied_action_count == 12
    assert report.quality_gate_passed and report.conclusion == "not_improved"


def test_native_plan_checksum_is_canonical_json_not_file_whitespace():
    specification, native_plan, native, output = specimen()
    assert output["evaluation_plan_sha256"] == digest(canonical(native_plan))
    wrong = deepcopy(output)
    wrong["evaluation_plan_sha256"] = "0" * 64
    with pytest.raises(Problem):
        project_verified(specification, native, wrong)


def test_compaction_keeps_sample_count_and_nearest_rank_without_cosmos_overflow():
    specification, _, native, output = specimen()
    native["trials"][0]["heartbeat_gap_ms"] = [1.25] * 100000
    output["trials"][0]["heartbeat_gap_ms"] = [1.25] * 100000
    report = project_verified(specification, native, output)
    summary = report.trials[0].phase_wall_ms["heartbeat"]
    assert summary.samples == 100000 and summary.p50 == summary.p95 == summary.max == 1.25
    assert len(report.model_dump_json().encode()) < 512 * 1024
    assert len(output["trials"][0]["heartbeat_gap_ms"]) == 100000


def test_exact_nearest_rank_quantile_and_empty_nonmodel_phase():
    from apps.learning_worker.paused_reports import summarize_wall

    assert summarize_wall(list(range(1, 21))) == {"samples": 20, "p50": 10, "p95": 19, "max": 20}
    assert summarize_wall([]) == {"samples": 0, "p50": None, "p95": None, "max": None}
