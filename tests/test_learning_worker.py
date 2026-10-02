from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.errors import Problem
from apps.api.learning_gateway import ManagedLearningGateway
from apps.api.learning_ports import JobSpecification
from apps.api.learning_service import LearningService
from apps.api.models import utcnow
from apps.learning_worker.backend import PolicyLearningWorker
from apps.learning_worker.main import WorkerSettings, create_worker
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import ACTOR, AUDIENCE, TENANT


def specification():
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    project, _, train = seed_project_and_dataset(store, request)
    service = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
        allowed_policy_types=("gr00t_n1_5",),
    )
    service.train(ACTOR, project.value.id, train, project.etag)
    return jobs.submissions[0], jobs


def test_production_worker_denies_model_use_before_clients_or_registry_without_license_admission():
    spec, _ = specification()

    class NoAccess:
        def __getattr__(self, name):
            raise AssertionError("Unapproved learning must not access models or Azure.")

    worker = PolicyLearningWorker(NoAccess(), NoAccess(), uuid4())
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, spec)
    assert failure.value.code == "learning_policy_unapproved"


def test_private_gateway_uses_its_own_managed_identity_and_does_not_retry_paid_posts():
    spec, _ = specification()
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-only-managed-identity-token"
        assert request.headers["x-environment-owner"] == ACTOR.owner_key
        assert request.method == "POST"
        raise httpx.RemoteProtocolError("Lost submit acknowledgement", request=request)

    gateway = ManagedLearningGateway(
        "https://test-worker.azurecontainerapps.io",
        SimpleNamespace(
            get_token=lambda _: SimpleNamespace(token="test-only-managed-identity-token")
        ),
        f"api://{AUDIENCE}/.default",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(Problem) as failure:
        gateway.submit(ACTOR, spec)
    assert failure.value.status == 503
    assert len(calls) == 1
    gateway.close()


def test_private_gateway_preserves_explicit_forbidden_cancellation_without_retry():
    spec, _ = specification()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"code": "private-detail"}})

    gateway = ManagedLearningGateway(
        "https://test-worker.azurecontainerapps.io",
        SimpleNamespace(get_token=lambda _: SimpleNamespace(token="test-only-token")),
        f"api://{AUDIENCE}/.default",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(Problem) as failure:
        gateway.cancel(ACTOR, spec.run)
    assert failure.value.status == 403 and failure.value.code == "worker_forbidden"
    assert len(calls) == 1
    gateway.close()


def test_private_worker_blocks_anonymous_delegated_and_owner_spoofing_before_any_job():
    spec, _ = specification()
    settings = WorkerSettings(
        tenant_id=TENANT,
        audience=AUDIENCE,
        managed_identity_client_id=uuid4(),
        registry_account_url="https://testregistry.blob.core.windows.net",
        registry_container="learning",
        capture_account_url="https://testcapture.blob.core.windows.net",
    )

    class Identity:
        def controller(self, authorization, allowed):
            if authorization != "Bearer test-only-controller":
                raise Problem(401, "authentication_required", "Managed identity required")

    worker = PolicyLearningWorker(None, None, uuid4())
    with TestClient(create_worker(settings, worker, Identity())) as client:
        body = {"actor": ACTOR.model_dump(mode="json"), "payload": spec.model_dump(mode="json")}
        assert client.post("/v1/learning/preflight", json=body).status_code == 401
        assert (
            client.post(
                "/v1/learning/preflight",
                json=body,
                headers={
                    "Authorization": "Bearer test-only-controller",
                    "X-Environment-Owner": "f" * 64,
                },
            ).status_code
            == 403
        )
        denied = client.post(
            "/v1/learning/preflight",
            json=body,
            headers={
                "Authorization": "Bearer test-only-controller",
                "X-Environment-Owner": ACTOR.owner_key,
            },
        )
        assert denied.status_code == 503, denied.text
        assert denied.json()["error"]["code"] == "learning_policy_unapproved"


def test_completed_azure_ack_never_manufactures_candidate_without_artifact_verification():
    spec, jobs = specification()
    raw = jobs.receipts[spec.run.backend_job_name].model_dump()
    raw["status"] = "succeeded"

    class MissingArtifacts:
        def completed_candidate(self, *args):
            raise Problem(503, "candidate_unverified", "No actual output manifest was verified.")

    worker = PolicyLearningWorker(None, MissingArtifacts(), uuid4())
    with pytest.raises(Problem) as failure:
        worker._receipt(ACTOR, spec, raw)
    assert failure.value.code == "candidate_unverified"


def test_worker_specification_rejects_a_different_actor_scope():
    spec, _ = specification()
    altered = JobSpecification.model_validate(
        {
            **spec.model_dump(mode="json"),
            "owner_key": "f" * 64,
        }
    )
    worker = PolicyLearningWorker(None, None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._authorize_specification(ACTOR, altered)
    assert failure.value.status == 403


def test_registered_worker_plan_cannot_exceed_the_original_authorized_runtime():
    spec, _ = specification()
    approval = {
        "expires_at": (utcnow() + timedelta(hours=2)).isoformat(),
        "maximum_cost_usd": "10.00",
        "gpu_hourly_usd": "1.00",
        "config": {
            "tenant_id": str(ACTOR.tenant_id),
            "owner_id": ACTOR.owner_key,
            "specification_sha256": spec.run.specification_sha256,
            "inputs": {
                "demonstrations": {"sha256": spec.dataset.manifest_sha256},
                "parent_model": {"sha256": spec.baseline.model_sha256},
            },
            "parameters": {
                "max_steps": spec.run.optimizer_steps,
                "timeout_seconds": spec.project.budget.training_seconds + 1,
            },
        },
    }
    registry = SimpleNamespace(approved_plan=lambda *_: approval)
    worker = PolicyLearningWorker(registry, None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, spec)
    assert failure.value.code == "worker_time_budget"
