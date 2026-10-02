from types import SimpleNamespace
from uuid import UUID, uuid4

from test_api import api as api
from test_api import headers

from apps.api.artifact_models import ArtifactResult
from apps.api.learning_models import CaptureReceipt, CreateDataset, TeachingSession
from apps.api.learning_ports import RuntimeCapture, TeachingRuntimeState
from apps.api.models import DemonstrationResult, Execution, utcnow
from tests.runtime_support import ACTOR, OTHER
from tests.test_artifact_operations import operations
from tests.test_learning_teaching import setup


def async_adapter(service):
    ops, registry = operations()
    calls = []

    def begin(actor, work):
        calls.append(work.id)
        return ops.begin(actor, work)

    service.artifacts = SimpleNamespace(
        artifact_policy=ops.policy,
        begin_artifact=begin,
        artifact_status=ops.recover,
        verify_capture=lambda *_: (_ for _ in ()).throw(AssertionError("No inline verification")),
        seal_dataset=lambda *_: (_ for _ in ()).throw(AssertionError("No inline sealing")),
    )
    return ops, registry, calls


def receipt(project, session):
    case = session.teaching_case
    return CaptureReceipt(
        episode_id=session.command_id,
        artifact_id=uuid4(),
        manifest_sha256="e" * 64,
        frame_count=20,
        source=session.source,
        seed=case.seed,
        task_id=project.task_id,
        control_profile_id=project.control_profile_id,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        split=case.split,
    )


def complete(ops, registry, operation_id, result):
    work = ops.work(ACTOR, operation_id)
    ops.claim(ACTOR, operation_id, uuid4())
    registry.put(
        ACTOR,
        f"artifacts/{result.artifact_id}/index.json",
        {
            "owner_key": ACTOR.owner_key,
            "manifest_sha256": result.manifest_sha256,
            "files": {"manifest.json": result.manifest_sha256},
        },
    )
    registry.put(
        ACTOR,
        f"artifact-operations/{operation_id}/result.json",
        {
            "work_sha256": work.sha256,
            "result": result.model_dump(mode="json"),
            "completed_at": utcnow().isoformat(),
        },
    )
    return ops.recover(ACTOR, operation_id)


def test_capture_poll_enqueues_once_and_remains_uploading_until_actual_manifest_result():
    service, project, started, runtime = setup("reference_controller")
    ops, registry, calls = async_adapter(service)
    expected = receipt(project.value, started.value)
    raw = DemonstrationResult(
        status="uploaded",
        manifest_uri=f"https://test.blob.core.windows.net/data/{expected.episode_id}/manifest.json",
        manifest_sha256=expected.manifest_sha256,
        episode_id=expected.episode_id,
        frame_count=expected.frame_count,
    )
    runtime.state = TeachingRuntimeState(
        session_id=started.value.id,
        lease_id=started.value.lease_id,
        epoch=started.value.epoch,
        command_id=started.value.command_id,
        control_profile_id=project.value.control_profile_id,
        status="succeeded",
        last_sequence=0,
        execution=Execution(command_id=started.value.command_id, status="succeeded"),
        capture=RuntimeCapture(
            capture_id=uuid4(),
            command_id=started.value.command_id,
            epoch=started.value.epoch,
            status="ready",
            receipt=raw,
        ),
    )
    waiting = service.get_teaching(ACTOR, started.value.id)
    assert waiting.value.status == "uploading" and waiting.value.capture is None
    assert waiting.value.verification_status == "queued"
    service.get_teaching(ACTOR, started.value.id)
    assert len(calls) == 1
    complete(
        ops,
        registry,
        waiting.value.artifact_operation_id,
        ArtifactResult(
            artifact_id=expected.artifact_id,
            manifest_sha256=expected.manifest_sha256,
            capture=expected,
        ),
    )
    ready = service.get_teaching(ACTOR, started.value.id)
    assert ready.value.status == "ready"
    assert ready.value.capture == expected
    assert ready.value.capture.source == "reference_controller"


def test_dataset_http_202_is_not_a_ready_dataset_and_status_finalizes_only_verified_result(api):
    client, _, token = api
    service, project, started, _ = setup("reference_controller")
    expected = receipt(project.value, started.value)
    service.store.put_learning(
        ACTOR.owner_key,
        TeachingSession.model_validate(
            {
                **started.value.model_dump(),
                "status": "ready",
                "capture": expected,
            }
        ),
        started.etag,
    )
    ops, registry, calls = async_adapter(service)
    client.app.state.learning = service
    request = CreateDataset(request_id=uuid4(), teaching_session_ids=(started.value.id,))
    response = client.post(
        f"/api/learning/projects/{project.value.id}/datasets",
        json=request.model_dump(mode="json"),
        headers={**headers(token), "If-Match": project.etag},
    )
    assert response.status_code == 202, response.text
    assert response.json()["item"]["kind"] == "artifact_operation"
    assert response.json()["item"]["status"] == "queued"
    assert "work_document" not in response.json()["item"]
    assert service.store.get_learning(ACTOR.owner_key, "dataset", request.request_id) is None
    operation_id = UUID(response.json()["item"]["id"])
    path = f"/api/learning/artifact-operations/{operation_id}"
    assert client.get(path).status_code == 401
    assert (
        client.get(
            path, headers={"Authorization": f"Bearer {token(oid=str(OTHER.object_id))}"}
        ).status_code
        == 404
    )
    assert client.get(path, headers=headers(token)).json()["item"]["status"] == "queued"
    assert len(calls) == 1
    complete(
        ops,
        registry,
        operation_id,
        ArtifactResult(
            artifact_id=request.request_id,
            manifest_sha256="a" * 64,
        ),
    )
    done = client.get(path, headers=headers(token))
    assert done.status_code == 200 and done.json()["item"]["status"] == "ready"
    dataset = service.get(ACTOR, "dataset", request.request_id).value
    assert dataset.reference_controller_count == 1 and dataset.human_teleop_count == 0
    assert dataset.captures == (expected,)
    repeated = client.post(
        f"/api/learning/projects/{project.value.id}/datasets",
        json=request.model_dump(mode="json"),
        headers={**headers(token), "If-Match": "stale-retry"},
    )
    assert repeated.status_code == 201
    assert repeated.json()["item"]["kind"] == "dataset"
    assert len(calls) == 1


def test_private_worker_operation_routes_keep_identity_and_owner_checks_before_queue_write():
    from fastapi.testclient import TestClient

    from apps.api.errors import Problem
    from apps.learning_worker.main import WorkerSettings, create_worker
    from tests.runtime_support import AUDIENCE, TENANT
    from tests.test_artifact_operations import operation_spec

    ops, registry = operations()
    work = operation_spec()
    settings = WorkerSettings(
        tenant_id=TENANT,
        audience=AUDIENCE,
        managed_identity_client_id=uuid4(),
        registry_account_url="https://testregistry.blob.core.windows.net",
        registry_container="artifacts",
        capture_account_url="https://testcapture.blob.core.windows.net",
    )

    class Identity:
        def controller(self, authorization, allowed):
            if authorization != "Bearer test-only-api-mi":
                raise Problem(401, "authentication_required", "App identity is required.")

    worker = SimpleNamespace(artifact_operations=ops)
    body = {"actor": ACTOR.model_dump(mode="json"), "payload": work.model_dump(mode="json")}
    path = f"/v1/learning/artifact-operations/{work.id}"
    with TestClient(create_worker(settings, worker, Identity())) as client:
        assert client.post(path, json=body).status_code == 401
        assert (
            client.post(
                path,
                json=body,
                headers={
                    "Authorization": "Bearer test-only-api-mi",
                    "X-Environment-Owner": OTHER.owner_key,
                },
            ).status_code
            == 403
        )
        assert registry.container.items == {}
        response = client.post(
            path,
            json=body,
            headers={
                "Authorization": "Bearer test-only-api-mi",
                "X-Environment-Owner": ACTOR.owner_key,
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert len(registry.container.items) == 3


def test_concurrent_metadata_read_during_admission_does_not_falsely_stop_pending_operation():
    service, project, started, _ = setup("reference_controller")
    expected = receipt(project.value, started.value)
    service.store.put_learning(
        ACTOR.owner_key,
        TeachingSession.model_validate(
            {
                **started.value.model_dump(),
                "status": "ready",
                "capture": expected,
            }
        ),
        started.etag,
    )
    ops, _, calls = async_adapter(service)
    begin = service.artifacts.begin_artifact

    def concurrent(actor, work):
        observed = service.get_artifact_operation(actor, work.id)
        assert observed.value.status == "queued"
        return begin(actor, work)

    service.artifacts.begin_artifact = concurrent
    request = CreateDataset(request_id=uuid4(), teaching_session_ids=(started.value.id,))
    pending = service.dataset(ACTOR, project.value.id, request, project.etag)
    assert pending.value.status == "queued" and pending.value.result is None
    assert len(calls) == 1 and ops.status(ACTOR, request.request_id).status == "queued"
