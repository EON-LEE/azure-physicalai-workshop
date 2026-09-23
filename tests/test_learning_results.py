from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_models import (
    EvaluationRun,
    PairedReport,
    PolicyCandidate,
    ReleasePolicy,
    TrialOutcome,
)
from apps.api.learning_service import LearningService, validate_paired_report
from apps.api.models import utcnow
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import ACTOR


def evaluated_setup(before_successes=10, after_successes=20, *, safety=0):
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    service = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, allowed_policy_types=("gr00t_n1_5",)
    )
    project, dataset, train = seed_project_and_dataset(store, request)
    started = service.train(ACTOR, project.value.id, train, project.etag)
    now = utcnow()
    candidate = PolicyCandidate(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="a" * 64,
        project_id=project.value.id,
        dataset_id=dataset.id,
        training_run_id=started.value.id,
        parent_release_id=catalog.record.id,
        policy_type="gr00t_n1_5",
        model_sha256="1" * 64,
        parent_model_sha256=catalog.record.model_sha256,
        processor_sha256="2" * 64,
        manifest_sha256="3" * 64,
        artifact_id=uuid4(),
        optimizer_steps=100,
        azure_job_id=started.value.azure_job_id,
        source_commit="4" * 40,
        model_revision="5" * 40,
        control_profile_id=project.value.control_profile_id,
    )
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(
        update={
            "status": "succeeded",
            "candidate": candidate,
        }
    )
    completed = service.get_job(ACTOR, started.value.id)
    assert completed.value.candidate_id == candidate.id
    trials = tuple(
        TrialOutcome(
            seed=seed,
            environment_id=f"held-out-{seed}",
            revision=f"{seed:064x}",
            attempt=1,
            policy=policy,
            status="succeeded" if index < count else "failed",
            model_sha256=model,
            physical_success=index < count,
            axis_error_m=(0.01, 0.0, 0.0) if index < count else (0.1, 0.0, 0.0),
            duration_seconds=20,
            safety_violations=safety if policy == "after" and index == 0 else 0,
            inference_p95_ms=75,
            applied_action_count=20,
            policy_predict_calls=20,
            reference_route_calls=0,
            recording_id=uuid4(),
            message="Explicit test-only trial",
        )
        for policy, count, model in (
            ("before", before_successes, catalog.record.model_sha256),
            ("after", after_successes, candidate.model_sha256),
        )
        for index, seed in enumerate(project.value.evaluation_plan.seeds)
    )
    report = PairedReport(
        evaluation_plan_sha256=project.value.evaluation_plan.sha256,
        before_model_sha256=catalog.record.model_sha256,
        after_model_sha256=candidate.model_sha256,
        trials=trials,
        conclusion="improved" if after_successes > before_successes else "not_improved",
        quality_gate_passed=after_successes >= 18 and safety == 0,
        report_sha256="6" * 64,
        artifact_id=uuid4(),
    )
    evaluation = EvaluationRun(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="7" * 64,
        project_id=project.value.id,
        status="succeeded",
        backend_job_name="learning-abcd",
        azure_job_id="/subscriptions/test/providers/Microsoft.MachineLearningServices/workspaces/test/jobs/learning-abcd",
        deadline=now,
        approved_cost_usd="10.00",
        specification_sha256="8" * 64,
        candidate_id=candidate.id,
        baseline_release_id=catalog.record.id,
        evaluation_plan_sha256=report.evaluation_plan_sha256,
        report=report,
    )
    stored = store.put_learning(ACTOR.owner_key, evaluation, None)
    return service, project.value, catalog.record, candidate, stored


def test_training_ack_or_missing_verified_new_weights_cannot_produce_candidate_success():
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    service = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, allowed_policy_types=("gr00t_n1_5",)
    )
    project, _, train = seed_project_and_dataset(store, request)
    started = service.train(ACTOR, project.value.id, train, project.etag)
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": "succeeded"})
    with pytest.raises(Problem) as failure:
        service.get_job(ACTOR, train.request_id)
    assert failure.value.code == "candidate_evidence_mismatch"
    assert (
        store.get_learning(ACTOR.owner_key, "training", train.request_id).value.status
        != "succeeded"
    )


@pytest.mark.parametrize(("before", "after", "safety"), [(20, 20, 0), (10, 17, 0), (10, 20, 1)])
def test_no_improvement_insufficient_quality_and_safety_failure_never_publish(
    before, after, safety
):
    service, _, _, candidate, evaluation = evaluated_setup(before, after, safety=safety)
    request = ReleasePolicy(
        request_id=uuid4(),
        candidate_id=candidate.id,
        evaluation_run_id=evaluation.value.id,
        release_approved=True,
    )
    with pytest.raises(Problem) as failure:
        service.release(ACTOR, request, evaluation.etag)
    assert failure.value.code == "release_gate_failed"
    assert service.store.list_learning(ACTOR.owner_key, "release") == []


def test_release_is_separate_review_exact_evaluation_etag_and_idempotent():
    service, _, _, candidate, evaluation = evaluated_setup()
    request = ReleasePolicy(
        request_id=uuid4(),
        candidate_id=candidate.id,
        evaluation_run_id=evaluation.value.id,
        release_approved=True,
    )
    assert service.store.list_learning(ACTOR.owner_key, "release") == []
    with pytest.raises(Problem) as failure:
        service.release(ACTOR, request, "stale")
    assert failure.value.status == 409
    release = service.release(ACTOR, request, evaluation.etag)
    assert release.value.model_sha256 == candidate.model_sha256
    assert release.value.reviewed_by == ACTOR.object_id
    assert service.release(ACTOR, request, "old-response-etag").value.id == release.value.id
    assert len(service.store.list_learning(ACTOR.owner_key, "release")) == 1


def test_evaluation_rejects_selected_successes_missing_conditions_and_changed_model():
    _, project, baseline, candidate, evaluation = evaluated_setup()
    report = evaluation.value.report
    for changed in (
        {"trials": report.trials[:-1]},
        {"after_model_sha256": "f" * 64},
        {"evaluation_plan_sha256": "e" * 64},
        {"trials": (*report.trials[:-1], report.trials[0])},
        {"trials": tuple(item.model_copy(update={"attempt": 2}) for item in report.trials)},
    ):
        with pytest.raises(Problem):
            validate_paired_report(project, baseline, candidate, report.model_copy(update=changed))


def test_retry_attempts_cannot_erase_first_failures_or_omit_an_intermediate_attempt():
    _, project, baseline, candidate, evaluation = evaluated_setup(10, 10)
    report = evaluation.value.report
    failed = next(
        item for item in report.trials if item.policy == "after" and not item.physical_success
    )
    successful_retry = failed.model_copy(
        update={
            "attempt": 3,
            "status": "succeeded",
            "physical_success": True,
            "axis_error_m": (0, 0, 0),
        }
    )
    with pytest.raises(Problem) as failure:
        validate_paired_report(
            project,
            baseline,
            candidate,
            report.model_copy(
                update={
                    "trials": (*report.trials, successful_retry),
                }
            ),
        )
    assert failure.value.code == "evaluation_incomplete"
