from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_models import StartTraining
from apps.api.learning_service import LearningService
from apps.api.models import utcnow
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import ACTOR, OTHER


def setup():
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    learning = LearningService(factory, store, jobs, artifacts, catalog, enabled=True)
    return learning, store, jobs, request


def test_missing_owner_cannot_discover_read_or_submit_another_owners_project():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    for operation in (
        lambda: learning.get(OTHER, "project", project.value.id),
        lambda: learning.train(OTHER, project.value.id, train, project.etag),
    ):
        with pytest.raises(Problem) as failure:
            operation()
        assert failure.value.status == 404
    assert jobs.submissions == []


def test_missing_stale_etag_or_changed_idempotency_input_never_submits_a_paid_job():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    for etag, status in ((None, 428), ("not-current", 409)):
        with pytest.raises(Problem) as failure:
            learning.train(ACTOR, project.value.id, train, etag)
        assert failure.value.status == status
    started = learning.train(ACTOR, project.value.id, train, project.etag)
    changed = StartTraining.model_validate({**train.model_dump(), "optimizer_steps": 10})
    with pytest.raises(Problem) as failure:
        learning.train(ACTOR, project.value.id, changed, project.etag)
    assert failure.value.code == "request_id_reused"
    assert len(jobs.submissions) == 1
    assert started.value.status == "submitted"
    assert started.value.candidate_id is None
    assert started.value.metrics.optimizer_steps is None


def test_concurrent_same_request_and_process_restart_do_not_duplicate_paid_submission():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    with ThreadPoolExecutor(max_workers=4) as threads:
        results = list(
            threads.map(
                lambda _: learning.train(ACTOR, project.value.id, train, project.etag), range(4)
            )
        )
    assert len(jobs.submissions) == 1
    assert {item.value.id for item in results} == {train.request_id}
    restarted = LearningService(
        learning.factory, store, jobs, learning.artifacts, learning.catalog, enabled=True
    )
    assert (
        restarted.train(ACTOR, project.value.id, train, "stale-retry").value.id == train.request_id
    )
    assert len(jobs.submissions) == 1


def test_uncertain_paid_response_is_reconciled_by_name_not_repeated_submit():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    jobs.submit_error = Problem(503, "dependency_unavailable", "Transport interrupted")
    with pytest.raises(Problem) as failure:
        learning.train(ACTOR, project.value.id, train, project.etag)
    assert failure.value.status == 503
    claim = store.get_learning(ACTOR.owner_key, "training", train.request_id)
    assert claim.value.status == "submission_unknown"
    assert claim.value.azure_job_id is None
    jobs.submit_error = None
    observed = learning.get_job(ACTOR, train.request_id)
    assert observed.value.status == "submitted"
    assert observed.value.azure_job_id.endswith(claim.value.backend_job_name)
    assert len(jobs.submissions) == 1
    assert jobs.status_calls == [train.request_id]


def test_cloud_preflight_failure_leaves_no_success_shaped_job():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    jobs.preflight_error = Problem(503, "capacity_unavailable", "No approved AML GPU capacity")
    with pytest.raises(Problem) as failure:
        learning.train(ACTOR, project.value.id, train, project.etag)
    assert failure.value.status == 503
    assert store.get_learning(ACTOR.owner_key, "training", train.request_id) is None
    assert jobs.submissions == []


def test_paid_job_cannot_exceed_cost_steps_or_train_on_held_out_seed():
    learning, store, jobs, request = setup()
    project, dataset, train = seed_project_and_dataset(store, request)
    for changed in ({"maximum_cost_usd": "10.01"}, {"optimizer_steps": 101}):
        with pytest.raises(Problem) as failure:
            learning.train(
                ACTOR,
                project.value.id,
                StartTraining.model_validate({**train.model_dump(), **changed}),
                project.etag,
            )
        assert failure.value.status == 422
    current = store.get_learning(ACTOR.owner_key, "dataset", dataset.id)
    store.put_learning(ACTOR.owner_key, dataset.model_copy(update={"seeds": (200,)}), current.etag)
    with pytest.raises(Problem) as failure:
        learning.train(ACTOR, project.value.id, train, project.etag)
    assert failure.value.code == "held_out_overlap"
    assert jobs.submissions == []


def test_reconciliation_rejects_foreign_owner_or_changed_job_specification():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    started = learning.train(ACTOR, project.value.id, train, project.etag)
    receipt = jobs.receipts[started.value.backend_job_name]
    for changes in ({"owner_key": OTHER.owner_key}, {"specification_sha256": "9" * 64}):
        jobs.receipts[receipt.job_name] = receipt.model_copy(update=changes)
        with pytest.raises(Problem) as failure:
            learning.get_job(ACTOR, started.value.id)
        assert failure.value.code == "learning_receipt_mismatch"
    assert len(jobs.submissions) == 1


def test_deadline_without_an_actual_backend_receipt_blocks_not_succeeds():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    started = learning.train(ACTOR, project.value.id, train, project.etag)
    jobs.receipts.clear()
    current = store.get_learning(ACTOR.owner_key, "training", started.value.id)
    store.put_learning(
        ACTOR.owner_key,
        current.value.model_copy(
            update={
                "status": "submission_unknown",
                "azure_job_id": None,
                "deadline": utcnow() - timedelta(seconds=1),
            }
        ),
        current.etag,
    )
    result = learning.get_job(ACTOR, started.value.id)
    assert result.value.status == "blocked"
    assert result.value.candidate_id is None
    assert len(jobs.submissions) == 1


def test_cancel_ack_is_not_cancelled_and_late_status_cannot_revive_terminal_job():
    learning, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    started = learning.train(ACTOR, project.value.id, train, project.etag)
    request_id = uuid4()
    cancelled = learning.cancel_job(ACTOR, started.value.id, request_id, started.etag)
    assert cancelled.value.status == "cancelling"
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": "cancelled"})
    terminal = learning.get_job(ACTOR, started.value.id)
    assert terminal.value.status == "cancelled"
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": "running"})
    assert learning.get_job(ACTOR, started.value.id).value.status == "cancelled"
    assert (
        learning.cancel_job(ACTOR, started.value.id, request_id, started.etag).value.status
        == "cancelled"
    )
    assert jobs.cancellations == [started.value.id]
