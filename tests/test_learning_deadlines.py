from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_service import LearningService
from tests.learning_api_support import seed_project_and_dataset
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_service import setup


def started_job(monkeypatch, *, status="submitted"):
    service, store, jobs, request = setup()
    project, _, train = seed_project_and_dataset(store, request)
    started = service.train(ACTOR, project.value.id, train, project.etag)
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": status})
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: started.value.deadline)
    return service, store, jobs, started


def restart(service, store, jobs):
    return LearningService(
        service.factory,
        store,
        jobs,
        service.artifacts,
        service.catalog,
        enabled=True,
        allowed_policy_types=service.allowed_policy_types,
    )


@pytest.mark.parametrize("status", ["submitted", "running"])
def test_expired_active_receipt_claims_durable_cancellation_before_one_post(monkeypatch, status):
    service, store, jobs, started = started_job(monkeypatch, status=status)
    original = jobs.cancel

    def cancel(actor, run):
        persisted = store.get_learning(actor.owner_key, run.kind, run.id).value
        assert persisted.cancellation.state == "claimed"
        assert persisted.cancellation.reason == "deadline"
        assert persisted.cancellation.requested_at == run.deadline
        assert persisted.status == "cancelling"
        return original(actor, run)

    jobs.cancel = cancel
    result = service.get_job(ACTOR, started.value.id)
    assert jobs.cancellations == [started.value.id]
    assert result.value.status == "cancelling"
    assert result.value.cancellation.state == "acknowledged"
    assert result.value.candidate_id is None
    assert len(jobs.submissions) == 1


@pytest.mark.parametrize("observed", ["submitted", "running", "cancelling"])
def test_cancel_ack_never_invents_terminal_state_or_repeats_after_restart(monkeypatch, observed):
    service, store, jobs, started = started_job(monkeypatch)
    first = service.get_job(ACTOR, started.value.id)
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": observed})
    reconstructed = restart(service, store, jobs)
    later = reconstructed.get_job(ACTOR, started.value.id)
    assert first.value.status == later.value.status == "cancelling"
    assert later.value.backend_status == observed
    assert jobs.cancellations == [started.value.id]
    assert later.value.deadline == started.value.deadline


def test_concurrent_expiry_reads_reserve_only_one_cancellation(monkeypatch):
    service, _, jobs, started = started_job(monkeypatch, status="running")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: service.get_job(ACTOR, started.value.id), range(8)))
    assert {item.value.status for item in results} == {"cancelling"}
    assert jobs.cancellations == [started.value.id]
    assert len(jobs.submissions) == 1


def test_expired_requeued_receipt_cancels_even_after_prior_running_state(monkeypatch):
    service, _, jobs, started = started_job(monkeypatch, status="running")
    monkeypatch.setattr(
        "apps.api.learning_service.utcnow", lambda: started.value.deadline - timedelta(seconds=1)
    )
    assert service.get_job(ACTOR, started.value.id).value.status == "running"
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"status": "submitted"})
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: started.value.deadline)
    assert service.get_job(ACTOR, started.value.id).value.status == "cancelling"
    assert jobs.cancellations == [started.value.id]


@pytest.mark.parametrize("http_status,state", [(503, "uncertain"), (403, "forbidden")])
def test_failed_cancellation_remains_explicit_and_is_not_reposted(monkeypatch, http_status, state):
    service, store, jobs, started = started_job(monkeypatch)
    calls = []

    def fail(actor, run):
        calls.append(run.id)
        raise Problem(http_status, "test_cancel_error", "The cancellation is not confirmed.")

    jobs.cancel = fail
    with pytest.raises(Problem) as failure:
        service.get_job(ACTOR, started.value.id)
    assert failure.value.status == http_status
    persisted = store.get_learning(ACTOR.owner_key, "training", started.value.id)
    assert persisted.value.cancellation.state == state
    assert persisted.value.error_code == "test_cancel_error"
    observed = restart(service, store, jobs).get_job(ACTOR, started.value.id)
    assert observed.value.status == "cancelling"
    assert observed.value.backend_status == "submitted"
    assert observed.value.error_code == "test_cancel_error"
    assert calls == [started.value.id]
    assert len(jobs.submissions) == 1


def test_crash_after_reservation_cannot_replay_an_uncertain_cancel_post(monkeypatch):
    service, store, jobs, started = started_job(monkeypatch)
    calls = []

    def crash(actor, run):
        calls.append(run.id)
        raise SystemExit("Test-only process loss after persisted cancellation reservation")

    jobs.cancel = crash
    with pytest.raises(SystemExit):
        service.get_job(ACTOR, started.value.id)
    observed = restart(service, store, jobs).get_job(ACTOR, started.value.id)
    assert observed.value.status == "cancelling"
    assert observed.value.cancellation.state == "claimed"
    assert observed.value.backend_status == "submitted"
    assert calls == [started.value.id]


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out"])
def test_expired_actual_terminal_receipt_needs_no_cancel_or_state_relabel(monkeypatch, status):
    service, _, jobs, started = started_job(monkeypatch, status=status)
    result = service.get_job(ACTOR, started.value.id)
    assert result.value.status == status
    assert result.value.backend_status == status
    assert result.value.cancellation is None
    assert jobs.cancellations == []


def test_earlier_operator_deadline_is_persisted_and_enforced(monkeypatch):
    service, _, jobs, started = started_job(monkeypatch)
    cutoff = started.value.deadline - timedelta(seconds=20)
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(update={"job_deadline_utc": cutoff})
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: cutoff)
    result = service.get_job(ACTOR, started.value.id)
    assert result.value.job_deadline_utc == cutoff
    assert result.value.deadline == started.value.deadline
    assert result.value.status == "cancelling"
    assert jobs.cancellations == [started.value.id]


def test_absent_receipt_is_unconfirmed_not_terminal_and_later_receipt_can_be_cancelled(
    monkeypatch,
):
    service, store, jobs, started = started_job(monkeypatch)
    receipt = jobs.receipts.pop(started.value.backend_job_name)
    unknown = started.value.model_copy(
        update={"status": "submission_unknown", "azure_job_id": None}
    )
    store.put_learning(ACTOR.owner_key, unknown, started.etag)
    result = service.get_job(ACTOR, started.value.id)
    assert result.value.status == "submission_unknown"
    assert result.value.error_code == "job_receipt_missing"
    assert jobs.cancellations == []
    jobs.receipts[receipt.job_name] = receipt
    discovered = restart(service, store, jobs).get_job(ACTOR, started.value.id)
    assert discovered.value.status == "cancelling"
    assert jobs.cancellations == [started.value.id]
    assert len(jobs.submissions) == 1


@pytest.mark.parametrize("changed", ["owner", "specification", "deadline"])
def test_unverified_receipt_cannot_authorize_deadline_cancellation(monkeypatch, changed):
    service, _, jobs, started = started_job(monkeypatch)
    receipt = jobs.receipts[started.value.backend_job_name]
    updates = {
        "owner": {"owner_key": OTHER.owner_key},
        "specification": {"specification_sha256": "f" * 64},
        "deadline": {"job_deadline_utc": started.value.deadline + timedelta(seconds=1)},
    }
    jobs.receipts[receipt.job_name] = receipt.model_copy(update=updates[changed])
    with pytest.raises(Problem) as failure:
        service.get_job(ACTOR, started.value.id)
    assert failure.value.code == "learning_receipt_mismatch"
    assert jobs.cancellations == []


def test_manual_cancel_and_later_deadline_share_the_per_job_reservation(monkeypatch):
    service, _, jobs, started = started_job(monkeypatch)
    monkeypatch.setattr(
        "apps.api.learning_service.utcnow", lambda: started.value.deadline - timedelta(seconds=10)
    )
    manual = service.cancel_job(ACTOR, started.value.id, uuid4(), started.etag)
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: started.value.deadline)
    automatic = service.get_job(ACTOR, started.value.id)
    assert manual.value.status == automatic.value.status == "cancelling"
    assert automatic.value.cancellation.reason == "user"
    assert jobs.cancellations == [started.value.id]


def test_foreign_owner_cannot_reconcile_or_cancel_an_expired_job(monkeypatch):
    service, _, jobs, started = started_job(monkeypatch)
    with pytest.raises(Problem) as failure:
        service.get_job(OTHER, started.value.id)
    assert failure.value.status == 404
    assert jobs.cancellations == [] and jobs.status_calls == []


def test_real_cosmos_adapter_preserves_cancellation_and_deadline_once_pinned(monkeypatch):
    from tests.test_learning_cosmos import store as cosmos_store

    service, _, jobs, request = setup()
    service.store = actual = cosmos_store()
    project, _, train = seed_project_and_dataset(actual, request)
    started = service.train(ACTOR, project.value.id, train, project.etag)
    receipt = jobs.receipts[started.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(
        update={"job_deadline_utc": started.value.deadline}
    )
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: started.value.deadline)
    result = service.get_job(ACTOR, started.value.id)
    assert result.value.status == "cancelling"
    assert actual.container.replacements
    for changes in (
        {"job_deadline_utc": None},
        {"job_deadline_utc": started.value.deadline - timedelta(seconds=1)},
        {"cancellation": None},
        {"cancellation": result.value.cancellation.model_copy(update={"request_id": uuid4()})},
    ):
        with pytest.raises(Problem) as failure:
            actual.put_learning(
                ACTOR.owner_key, result.value.model_copy(update=changes), result.etag
            )
        assert failure.value.code == "immutable_learning_record"
    resumed = restart(service, actual, jobs).get_job(ACTOR, started.value.id)
    assert resumed.value.cancellation == result.value.cancellation
    assert jobs.cancellations == [started.value.id]
