import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from apps.api.errors import Problem
from apps.api.learning_models import fingerprint
from apps.api.models import utcnow
from apps.learning_worker.backend import PolicyLearningWorker
from apps.learning_worker.registry import BlobRegistry
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_worker import specification


def utc_text(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def reviewed():
    original, _ = specification()
    spec = original.model_copy(
        update={
            "project": original.project.model_copy(update={"policy_type": "smolvla"}),
            "run": original.run.model_copy(update={"policy_type": "smolvla"}),
            "baseline": original.baseline.model_copy(update={"policy_type": "smolvla"}),
        }
    )
    approval = {
        "expires_at": utc_text(spec.run.deadline + timedelta(hours=1)),
        "maximum_cost_usd": "10.00",
        "gpu_hourly_usd": "1.00",
        "config": {
            "schema": "physicalai.smolvla-azure/v2",
            "job_deadline_utc": utc_text(spec.run.deadline),
            "tenant_id": str(ACTOR.tenant_id),
            "owner_id": ACTOR.owner_key,
            "specification_sha256": spec.run.specification_sha256,
            "policy_type": "smolvla",
            "inputs": {
                "demonstrations": {"sha256": spec.dataset.manifest_sha256},
                "parent_model": {"sha256": spec.baseline.model_sha256},
            },
            "parameters": {"max_steps": spec.run.optimizer_steps, "timeout_seconds": 10},
        },
    }
    return spec, approval


@pytest.mark.parametrize("change", ["missing", "later", "noncanonical", "v1"])
def test_worker_requires_exact_preapproved_absolute_deadline(change):
    spec, approval = reviewed()
    if change == "missing":
        approval["config"].pop("job_deadline_utc")
    elif change == "later":
        approval["config"]["job_deadline_utc"] = utc_text(spec.run.deadline + timedelta(seconds=1))
    elif change == "noncanonical":
        approval["config"]["job_deadline_utc"] = spec.run.deadline.isoformat()
    else:
        approval["config"]["schema"] = "physicalai.smolvla-azure/v1"
    worker = PolicyLearningWorker(SimpleNamespace(approved_plan=lambda *_: approval), None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, spec)
    assert failure.value.code == "worker_deadline_mismatch"


def test_operator_expiry_is_the_earliest_deadline_not_a_renewable_duration():
    spec, approval = reviewed()
    expires = spec.run.deadline - timedelta(seconds=20)
    approval["expires_at"] = utc_text(expires)
    worker = PolicyLearningWorker(SimpleNamespace(approved_plan=lambda *_: approval), None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, spec)
    assert failure.value.code == "worker_deadline_mismatch"
    approval["config"]["job_deadline_utc"] = utc_text(expires)
    assert worker._configuration(ACTOR, spec) == approval["config"]


def test_full_state_approval_counts_only_remaining_optimizer_updates():
    spec, approval = reviewed()
    approval["config"]["parameters"].update(
        max_steps=spec.run.optimizer_steps + 4, resume_mode="full_state"
    )
    approval["config"]["checkpointing"] = {"resume": {"step": 4}}
    worker = PolicyLearningWorker(SimpleNamespace(approved_plan=lambda *_: approval), None, uuid4())
    assert worker._configuration(ACTOR, spec) == approval["config"]


def test_full_state_global_horizon_cannot_be_misreported_as_new_updates():
    spec, approval = reviewed()
    approval["config"]["parameters"]["resume_mode"] = "full_state"
    approval["config"]["checkpointing"] = {"resume": {"step": 4}}
    worker = PolicyLearningWorker(SimpleNamespace(approved_plan=lambda *_: approval), None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, spec)
    assert failure.value.code == "worker_plan_mismatch"


@pytest.mark.parametrize("step", [None, True, "4", 0, -1, 100000])
def test_full_state_remaining_update_budget_requires_a_real_checkpoint_step(step):
    spec, approval = reviewed()
    approval["config"]["parameters"]["resume_mode"] = "full_state"
    approval["config"]["checkpointing"] = {"resume": {"step": step}}
    worker = PolicyLearningWorker(SimpleNamespace(approved_plan=lambda *_: approval), None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, spec)
    assert failure.value.code == "worker_resume_budget"


def test_missing_monitor_enrollment_blocks_before_paid_backend_access():
    spec, approval = reviewed()

    def forbidden(_):
        pytest.fail("A job without durable monitor enrollment must not access AML.")

    worker = PolicyLearningWorker(
        SimpleNamespace(approved_plan=lambda *_: approval),
        None,
        uuid4(),
        allowed_policy_types=("smolvla",),
        sdk_factory=forbidden,
    )
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, spec)
    assert failure.value.code == "job_reconciliation_unenrolled"


class ConditionalBlobs:
    def __init__(self):
        self.items = {}
        self.version = 0
        self.lock = threading.Lock()
        self.writes = []

    def upload_blob(self, name, data, overwrite, *, etag=None, match_condition=None, **kwargs):
        blob = self.get_blob_client(name)
        blob.upload_blob(
            data, overwrite=overwrite, etag=etag, match_condition=match_condition, **kwargs
        )
        return blob

    def get_blob_client(self, name):
        return SimpleNamespace(
            upload_blob=lambda data, **kwargs: self._upload(name, data, **kwargs)
        )

    def _upload(self, name, data, overwrite, *, etag=None, match_condition=None, **kwargs):
        if hasattr(data, "read"):
            data = data.read()
        with self.lock:
            prior = self.items.get(name)
            if prior is not None and not overwrite:
                raise ResourceExistsError(status_code=409)
            if overwrite:
                assert match_condition == MatchConditions.IfNotModified
                if prior is None or prior[1] != etag:
                    raise ResourceModifiedError(status_code=412)
            self.version += 1
            tag = f'"blob-{self.version}"'
            self.items[name] = bytes(data), tag
            self.writes.append((name, overwrite, etag, match_condition))
            return {"etag": tag}

    def download_blob(
        self,
        name,
        *,
        offset=0,
        length=None,
        etag=None,
        match_condition=None,
        max_concurrency=1,
        retry_total=None,
    ):
        with self.lock:
            if name not in self.items:
                raise ResourceNotFoundError(status_code=404)
            data, version = self.items[name]
            if etag is not None:
                assert match_condition == MatchConditions.IfNotModified
                if version != etag:
                    raise ResourceModifiedError(status_code=412)
            body = data[offset:] if length is None else data[offset : offset + length]
            return SimpleNamespace(
                readall=lambda: body,
                chunks=lambda: (body[i : i + 65536] for i in range(0, len(body), 65536)),
                properties=SimpleNamespace(etag=version, size=len(data)),
            )

    def list_blobs(self, *, name_starts_with):
        with self.lock:
            return [
                SimpleNamespace(name=name, size=len(value[0]), etag=value[1])
                for name, value in self.items.items()
                if name.startswith(name_starts_with)
            ]

    def delete_blob(self, name):
        with self.lock:
            if name not in self.items:
                raise ResourceNotFoundError(status_code=404)
            del self.items[name]


class NativeJobs:
    def __init__(self, spec):
        self.spec = spec
        self.azure_status = "Queued"
        self.status_value = "submitted"
        self.missing = False
        self.cancels = []
        self.preflights = 0
        self.cancel_error = None
        self.before_cancel = None

    def preflight(self):
        self.preflights += 1

    def status(self, name):
        assert name == self.spec.run.backend_job_name
        if self.missing:
            raise ResourceNotFoundError(status_code=404)
        return {
            "job_name": name,
            "azure_job_id": (
                "/subscriptions/test/providers/Microsoft.MachineLearningServices"
                f"/workspaces/test/jobs/{name}"
            ),
            "owner_key": ACTOR.owner_key,
            "specification_sha256": self.spec.run.specification_sha256,
            "status": self.status_value,
            "azure_status": self.azure_status,
        }

    def cancel(self, name):
        if self.before_cancel:
            self.before_cancel()
        self.cancels.append(name)
        if self.cancel_error:
            raise self.cancel_error
        return {**self.status(name), "cancellation_requested": True}


def worker_with_registry(*, claimed=True):
    from apps.learning_worker.registry import ReconciliationTarget

    spec, approval = reviewed()
    registry = BlobRegistry.__new__(BlobRegistry)
    registry.container = ConditionalBlobs()
    registry.put(ACTOR, f"projects/{spec.project.id}/train-approval.json", approval)
    client_id = uuid4()
    target = ReconciliationTarget(
        actor_id=ACTOR.object_id,
        job_name=spec.run.backend_job_name,
        specification_sha256=spec.run.specification_sha256,
        job_deadline_utc=spec.run.deadline,
        configuration_sha256=fingerprint(approval["config"]),
    )
    jobs = NativeJobs(spec)
    worker = PolicyLearningWorker(
        registry,
        None,
        client_id,
        allowed_policy_types=("smolvla",),
        sdk_factory=lambda _: (jobs, None),
        reconciliation_enabled=True,
        reconciliation_actor_ids=frozenset({ACTOR.object_id}),
        reconciliation_targets=(target,),
    )
    if claimed:
        registry.claim_job(ACTOR, spec, approval["config"])
    return worker, registry, spec, approval, target, jobs


def test_fresh_tick_enrollment_is_required_and_exactly_bound_before_preflight():
    worker, registry, spec, _, target, jobs = worker_with_registry(claimed=False)
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, spec)
    assert failure.value.code == "job_reconciliation_unenrolled"
    registry.record_heartbeat(ACTOR, target, worker.caller_client_id)
    worker.preflight(ACTOR, spec)
    assert jobs.preflights == 1
    assert registry.job(ACTOR, spec.run.backend_job_name) is None
    assert jobs.cancels == []


@pytest.mark.parametrize("mismatch", ["stale", "actor", "deadline", "config", "identity"])
def test_stale_or_mismatched_enrollment_never_admits_a_paid_job(monkeypatch, mismatch):
    worker, registry, spec, _, target, jobs = worker_with_registry(claimed=False)
    now = utcnow()
    if mismatch == "stale":
        monkeypatch.setattr(
            "apps.learning_worker.registry.utcnow", lambda: now - timedelta(minutes=3)
        )
    if mismatch == "actor":
        worker.reconciliation_actor_ids = frozenset({OTHER.object_id})
    target = target.model_copy(
        update={"job_deadline_utc": target.job_deadline_utc - timedelta(seconds=1)}
        if mismatch == "deadline"
        else {"configuration_sha256": "f" * 64}
        if mismatch == "config"
        else {}
    )
    registry.record_heartbeat(
        ACTOR, target, uuid4() if mismatch == "identity" else worker.caller_client_id
    )
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, spec)
    assert failure.value.code == "job_reconciliation_unenrolled"
    assert jobs.preflights == 0 and jobs.cancels == []


def test_frozen_job_configuration_not_mutable_project_approval_is_used_after_submit():
    worker, registry, spec, approval, _, jobs = worker_with_registry()
    changed = copy.deepcopy(approval)
    changed["config"]["job_deadline_utc"] = utc_text(spec.run.deadline + timedelta(hours=1))
    key = registry.key(ACTOR, f"projects/{spec.project.id}/train-approval.json")
    prior = registry.container.items[key]
    registry.container.items[key] = json.dumps(changed).encode(), prior[1]
    result = worker.status(ACTOR, spec.run)
    assert result.job_deadline_utc == spec.run.deadline
    assert result.azure_status == "Queued"
    assert jobs.cancels == []


def test_worker_cancel_claim_survives_concurrency_and_worker_restart():
    worker, registry, spec, _, _, jobs = worker_with_registry()

    def before_post():
        claim = registry.cancellation(ACTOR, spec, worker._job_configuration(ACTOR, spec))
        assert claim is not None and claim.value.state == "claimed"

    jobs.before_cancel = before_post
    with ThreadPoolExecutor(max_workers=6) as executor:
        receipts = list(executor.map(lambda _: worker.cancel(ACTOR, spec.run), range(6)))
    assert {receipt.status for receipt in receipts} == {"submitted"}
    assert jobs.cancels == [spec.run.backend_job_name]
    restarted = PolicyLearningWorker(
        registry, None, worker.caller_client_id, sdk_factory=lambda _: (jobs, None)
    )
    assert restarted.cancel(ACTOR, spec.run).status == "submitted"
    assert jobs.cancels == [spec.run.backend_job_name]


@pytest.mark.parametrize(
    "failure,state",
    [
        (ServiceRequestError("Test-only lost cancel response"), "uncertain"),
        (
            HttpResponseError(
                response=SimpleNamespace(status_code=403, reason="Forbidden", headers={}),
                message="Test-only forbidden",
            ),
            "forbidden",
        ),
    ],
)
def test_worker_records_uncertain_or_forbidden_cancel_without_retry(failure, state):
    worker, registry, spec, _, _, jobs = worker_with_registry()
    jobs.cancel_error = failure
    with pytest.raises(Problem):
        worker.cancel(ACTOR, spec.run)
    claim = registry.cancellation(ACTOR, spec, worker._job_configuration(ACTOR, spec))
    assert claim.value.state == state
    observed = worker.cancel(ACTOR, spec.run)
    assert observed.status == "submitted"
    assert observed.cancellation_state == state
    assert jobs.cancels == [spec.run.backend_job_name]


def test_missing_receipt_does_not_consume_cancellation_before_late_queued_receipt(monkeypatch):
    worker, registry, spec, _, target, jobs = worker_with_registry()
    monkeypatch.setattr("apps.learning_worker.backend.utcnow", lambda: spec.run.deadline)
    jobs.missing = True
    missing = worker.reconcile_deadline(ACTOR, target)
    assert missing["status"] == "unconfirmed"
    assert registry.cancellation(ACTOR, spec, worker._job_configuration(ACTOR, spec)) is None
    jobs.missing = False
    later = worker.reconcile_deadline(ACTOR, target)
    assert later["status"] == "submitted"
    assert later["cancellation_state"] == "acknowledged"
    assert jobs.cancels == [spec.run.backend_job_name]


@pytest.mark.parametrize(
    "provider,status",
    [
        ("CancelRequested", "cancelling"),
        ("Canceled", "cancelled"),
        ("Failed", "failed"),
    ],
)
def test_actual_provider_cancellation_or_terminal_state_never_needs_another_post(provider, status):
    worker, _, spec, _, _, jobs = worker_with_registry()
    jobs.azure_status, jobs.status_value = provider, status
    result = worker.cancel(ACTOR, spec.run)
    assert result.status == status
    assert jobs.cancels == []


@pytest.mark.parametrize("provider", ["Unknown", "NotResponding", "Paused"])
def test_provider_nonterminal_failed_projection_does_not_prevent_expiry_cancel(
    monkeypatch, provider
):
    worker, _, spec, _, target, jobs = worker_with_registry()
    jobs.azure_status, jobs.status_value = provider, "failed"
    monkeypatch.setattr("apps.learning_worker.backend.utcnow", lambda: spec.run.deadline)
    result = worker.reconcile_deadline(ACTOR, target)
    assert result["status"] == "running"
    assert result["azure_status"] == provider
    assert jobs.cancels == [spec.run.backend_job_name]


def tick_settings(worker, target, *, enabled=True):
    from apps.learning_worker.main import WorkerSettings
    from tests.runtime_support import AUDIENCE, TENANT

    return WorkerSettings(
        tenant_id=TENANT,
        audience=AUDIENCE,
        managed_identity_client_id=worker.caller_client_id,
        registry_account_url="https://testregistry.blob.core.windows.net",
        registry_container="learning",
        capture_account_url="https://testcapture.blob.core.windows.net",
        reconciliation_enabled=enabled,
        reconciliation_actor_ids=frozenset({ACTOR.object_id}) if enabled else frozenset(),
        reconciliation_targets=(target,) if enabled else (),
    )


def test_disabled_tick_has_no_registry_or_azure_access():
    from apps.learning_worker.reconcile import tick

    worker, _, _, _, target, _ = worker_with_registry()

    class NoAccess:
        def __getattr__(self, name):
            pytest.fail("A disabled tick must not access resources.")

    assert tick(tick_settings(worker, target, enabled=False), NoAccess()) == {
        "enabled": False,
        "items": [],
        "training_verified": False,
    }


def test_tick_enrolls_only_the_exact_target_before_submission_without_creating_a_job():
    from apps.learning_worker.reconcile import tick

    worker, registry, spec, _, target, jobs = worker_with_registry(claimed=False)
    result = tick(tick_settings(worker, target), worker)
    assert result["items"][0]["status"] == "awaiting_submission"
    assert registry.heartbeat(ACTOR, target).value.target == target
    assert registry.job(ACTOR, spec.run.backend_job_name) is None
    assert jobs.preflights == 0 and jobs.cancels == []
    worker.preflight(ACTOR, spec)
    assert jobs.preflights == 1


def test_tick_operates_without_api_ui_and_repeated_invocations_do_not_repeat_cancel(monkeypatch):
    from apps.learning_worker.reconcile import tick

    worker, registry, spec, _, target, jobs = worker_with_registry()
    monkeypatch.setattr("apps.learning_worker.backend.utcnow", lambda: spec.run.deadline)
    settings = tick_settings(worker, target)
    for _ in range(2):
        result = tick(settings, worker)
        assert result["items"][0]["status"] == "submitted"
        assert result["items"][0]["cancellation_state"] == "acknowledged"
    assert jobs.cancels == [spec.run.backend_job_name]
    assert len(registry.container.items) == 5


def test_crash_between_enrollment_and_submit_never_automatically_creates_paid_job(monkeypatch):
    worker, registry, spec, _, target, jobs = worker_with_registry(claimed=False)
    registry.record_heartbeat(ACTOR, target, worker.caller_client_id)
    jobs.missing = True

    def crash(config, path, *, deterministic_job_name):
        assert registry.job(ACTOR, deterministic_job_name) == spec
        raise SystemExit("Test-only loss after enrollment/configuration/claim, before paid POST")

    worker.sdk_factory = lambda _: (jobs, crash)
    with pytest.raises(SystemExit):
        worker.submit(ACTOR, spec)
    assert registry.job(ACTOR, spec.run.backend_job_name) == spec
    with pytest.raises(Problem) as failure:
        worker.submit(ACTOR, spec)
    assert failure.value.code == "worker_submission_unconfirmed"
    assert jobs.cancels == []


def test_enrollment_loss_after_plan_generation_blocks_the_paid_post():
    worker, registry, spec, _, target, jobs = worker_with_registry(claimed=False)
    registry.record_heartbeat(ACTOR, target, worker.caller_client_id)

    def lose_monitor(config, path, *, deterministic_job_name):
        registry.container.items.pop(
            registry.key(ACTOR, f"jobs/{target.job_name}/reconciliation.json")
        )
        return "a" * 64

    worker.sdk_factory = lambda _: (jobs, lose_monitor)
    with pytest.raises(Problem) as failure:
        worker.submit(ACTOR, spec)
    assert failure.value.code == "job_reconciliation_unenrolled"
    assert jobs.cancels == []


def test_wrong_owner_and_changed_deadline_cannot_use_an_existing_job_claim():
    worker, _, spec, _, _, jobs = worker_with_registry()
    assert worker.status(OTHER, spec.run) is None
    with pytest.raises(Problem) as failure:
        worker.cancel(
            ACTOR,
            spec.run.model_copy(update={"deadline": spec.run.deadline + timedelta(seconds=1)}),
        )
    assert failure.value.code == "worker_job_mismatch"
    assert jobs.cancels == []


def test_missing_or_duplicate_targets_are_invalid_settings_before_any_client():
    from pydantic import ValidationError

    from apps.learning_worker.main import WorkerSettings

    worker, _, _, _, target, _ = worker_with_registry()
    settings = tick_settings(worker, target).model_dump()
    for override in (
        {"reconciliation_actor_ids": []},
        {"reconciliation_targets": []},
        {"reconciliation_targets": [target, target]},
        {"reconciliation_targets": [target] * 21},
    ):
        with pytest.raises(ValidationError):
            WorkerSettings.model_validate({**settings, **override})


def test_uncertain_tick_cannot_refresh_admission_or_report_reconciliation_success(monkeypatch):
    from apps.learning_worker.reconcile import reconciled, tick

    worker, registry, spec, _, target, jobs = worker_with_registry()
    settings = tick_settings(worker, target)
    original = registry.record_heartbeat(ACTOR, target, worker.caller_client_id)
    monkeypatch.setattr("apps.learning_worker.backend.utcnow", lambda: spec.run.deadline)
    jobs.cancel_error = ServiceRequestError("Test-only cancellation response lost")
    first = tick(settings, worker)["items"][0]
    assert first["status"] == "error" and not reconciled(first)
    second = tick(settings, worker)["items"][0]
    assert second["status"] == "submitted"
    assert second["cancellation_state"] == "uncertain"
    assert not reconciled(second)
    assert registry.heartbeat(ACTOR, target) == original
    assert jobs.cancels == [spec.run.backend_job_name]


def test_production_worker_factory_wires_monitor_settings_to_backend_not_artifacts(monkeypatch):
    from fastapi.testclient import TestClient

    from apps.learning_worker.main import create_worker

    worker, _, _, _, target, _ = worker_with_registry()
    settings = tick_settings(worker, target)
    closed = []
    credential = SimpleNamespace(close=lambda: closed.append("credential"))
    registry = SimpleNamespace(close=lambda: closed.append("registry"))
    monkeypatch.setattr("azure.identity.ManagedIdentityCredential", lambda *, client_id: credential)
    monkeypatch.setattr("apps.learning_worker.registry.BlobRegistry", lambda *_: registry)

    def artifacts(registry, credential, account, container, *, allowed_policy_types):
        return object()

    monkeypatch.setattr("apps.learning_worker.artifacts.VerifiedArtifacts", artifacts)
    with TestClient(create_worker(settings)) as client:
        assert client.get("/healthz").json()["training_verified"] is False
        actual = client.app.state.worker
        assert actual.reconciliation_enabled
        assert actual.reconciliation_targets == (target,)
        assert actual.reconciliation_actor_ids == {ACTOR.object_id}
    assert closed == ["registry", "credential"]
