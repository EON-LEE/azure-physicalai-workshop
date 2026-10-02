import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from test_api import api as api
from test_api import headers

from apps.api.learning_gateway import ManagedLearningGateway
from apps.api.learning_models import EvaluationRun
from apps.api.models import utcnow
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_service import setup
from tests.test_paused_report_projection import project_verified, specimen


def test_native_report_download_is_owner_only_and_preserves_exact_file_bytes(api):
    client, _, token = api
    service, store, jobs, _ = setup()
    spec, _, native, output = specimen()
    content = json.dumps(output, indent=4).encode()
    digest = hashlib.sha256(content).hexdigest()
    report = project_verified(spec, native, output).model_copy(update={"report_sha256": digest})
    now = utcnow()
    run = EvaluationRun(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="a" * 64,
        project_id=spec.project.id,
        policy_type="smolvla",
        status="failed",
        backend_job_name=f"learning-{uuid4().hex}",
        azure_job_id=spec.run.azure_job_id,
        deadline=now + timedelta(seconds=60),
        approved_cost_usd="1",
        specification_sha256=spec.run.specification_sha256,
        candidate_id=uuid4(),
        baseline_release_id=spec.project.baseline_release_id,
        evaluation_plan_sha256=spec.project.evaluation_plan.sha256,
        report=report,
        **spec.project.timing_fields(),
    )
    store.put_learning(ACTOR.owner_key, run, None)
    calls = []
    jobs.report_document = lambda actor, job, sha: calls.append((actor, job.id, sha)) or content
    client.app.state.learning = service
    path = f"/api/learning/jobs/{run.id}/report"
    assert client.get(path).status_code == 401
    assert (
        client.get(
            path, headers={"Authorization": f"Bearer {token(oid=str(OTHER.object_id))}"}
        ).status_code
        == 404
    )
    assert calls == []
    response = client.get(path, headers=headers(token))
    assert response.status_code == 200 and response.content == content
    assert response.headers["x-report-sha256"] == digest
    assert response.headers["cache-control"] == "no-store"
    assert calls == [(ACTOR, run.id, digest)]


def test_managed_gateway_download_hash_cannot_be_replaced_in_transit():
    content = b'{"test-only":"verified-native-bytes"}'
    checksum = hashlib.sha256(content).hexdigest()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=content, headers={"Content-Type": "application/json"})

    gateway = ManagedLearningGateway(
        "https://test-worker.internal",
        SimpleNamespace(get_token=lambda _: SimpleNamespace(token="test-only-mi-token")),
        "api://test/.default",
        transport=httpx.MockTransport(handler),
    )
    run = SimpleNamespace(backend_job_name="learning-test-only")
    assert gateway.report_document(ACTOR, run, checksum) == content
    assert requests[0].url.params["report_sha256"] == checksum
    assert requests[0].headers["x-environment-owner"] == ACTOR.owner_key
    from apps.api.errors import Problem

    with pytest.raises(Problem) as failure:
        gateway.report_document(ACTOR, run, "f" * 64)
    assert failure.value.code == "report_digest_mismatch"
    gateway.close()
