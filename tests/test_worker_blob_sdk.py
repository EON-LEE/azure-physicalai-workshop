from datetime import timedelta
from http import HTTPStatus
from uuid import uuid4

import pytest
from azure.core.pipeline.transport import HttpResponse, HttpTransport
from azure.core.utils import case_insensitive_dict
from azure.storage.blob import BlobClient, ContainerClient

from apps.api.errors import Problem
from apps.api.models import utcnow
from apps.learning_worker.registry import (
    BlobRegistry,
    ReconciliationHeartbeat,
    ReconciliationTarget,
)
from tests.runtime_support import ACTOR


class OfflineStream:
    def __init__(self, response):
        self.response = response
        self.content_length = len(response.body())
        self.chunks = iter([response.body()])

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)


class OfflineResponse(HttpResponse):
    def __init__(self, request, status, etag, body=b"", code=None, content_type=None):
        super().__init__(request, None)
        self.status_code = status
        self.reason = HTTPStatus(status).phrase
        self._body = body
        self.headers = case_insensitive_dict(
            {
                "ETag": etag,
                "Last-Modified": "Wed, 23 Sep 2026 12:00:00 GMT",
                "Content-Length": str(len(body)),
                "Content-Type": content_type or ("application/xml" if code else "application/json"),
                "x-ms-blob-type": "BlockBlob",
                "x-ms-request-id": "test-only-offline-request",
                **({"x-ms-error-code": code} if code else {}),
                **(
                    {"Content-Range": f"bytes 0-{len(body) - 1}/{len(body)}"}
                    if status == 206
                    else {}
                ),
            }
        )
        self.content_type = self.headers["Content-Type"]

    def body(self):
        return self._body

    def stream_download(self, pipeline, **kwargs):
        return OfflineStream(self)


class OfflineTransport(HttpTransport):
    """Actual SDK serialization, with every HTTP response supplied locally."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self):
        pass

    def close(self):
        pass

    def __exit__(self, *args):
        self.close()

    def send(self, request, **kwargs):
        self.requests.append(request)
        assert self.responses, "No network or implicit SDK retry is allowed in this test."
        assert "Authorization" not in request.headers
        return OfflineResponse(request, **self.responses.pop(0))


def setup(responses):
    transport = OfflineTransport(responses)
    container = ContainerClient(
        "https://offline.blob.core.windows.net",
        "test-only",
        credential=None,
        transport=transport,
        retry_total=0,
    )
    registry = BlobRegistry.__new__(BlobRegistry)
    registry.container = container
    heartbeat = ReconciliationHeartbeat(
        target=ReconciliationTarget(
            actor_id=ACTOR.object_id,
            job_name=f"learning-{uuid4().hex}",
            specification_sha256="a" * 64,
            configuration_sha256="b" * 64,
            job_deadline_utc=utcnow(),
        ),
        worker_client_id=uuid4(),
        observed_at=utcnow(),
    )
    return registry, container, transport, heartbeat


def test_actual_container_upload_returns_client_after_success_not_an_etag_receipt():
    _, container, transport, _ = setup([{"status": 201, "etag": '"operation-one"'}])
    with container:
        result = container.upload_blob("test-only-proof.json", b"{}")
        assert isinstance(result, BlobClient)
        with pytest.raises(TypeError):
            result["etag"]
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "PUT"
    assert transport.requests[0].headers["If-None-Match"] == "*"


def test_registry_preserves_the_exact_sdk_write_etag_without_later_properties_lookup():
    registry, container, transport, value = setup(
        [
            {"status": 201, "etag": '"operation-one"'},
            {"status": 200, "etag": '"later-unrelated-write"'},
        ]
    )
    with container:
        saved = registry._write_record(ACTOR, "jobs/test-only/reconciliation.json", value, None)
    assert saved.value == value and saved.etag == '"operation-one"'
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "PUT"
    assert transport.requests[0].headers["If-None-Match"] == "*"
    assert len(transport.responses) == 1


def test_actual_sdk_create_conflict_is_409_and_never_retried():
    registry, container, transport, value = setup(
        [
            {"status": 201, "etag": '"operation-one"'},
            {"status": 409, "etag": '"operation-one"', "code": "BlobAlreadyExists"},
        ]
    )
    with container:
        registry._write_record(ACTOR, "jobs/test-only/cancellation.json", value, None)
        with pytest.raises(Problem) as failure:
            registry._write_record(ACTOR, "jobs/test-only/cancellation.json", value, None)
    assert failure.value.code == "registry_revision_conflict"
    assert failure.value.status == 409
    assert len(transport.requests) == 2
    assert all(request.headers["If-None-Match"] == "*" for request in transport.requests)


def test_actual_sdk_replacement_preserves_write_etag_and_rejects_a_stale_if_match():
    registry, container, transport, value = setup(
        [
            {"status": 201, "etag": '"operation-one"'},
            {"status": 201, "etag": '"operation-two"'},
            {"status": 412, "etag": '"operation-two"', "code": "ConditionNotMet"},
        ]
    )
    with container:
        first = registry._write_record(ACTOR, "jobs/test-only/cancellation.json", value, None)
        second = registry._write_record(
            ACTOR, "jobs/test-only/cancellation.json", value, first.etag
        )
        assert second.etag == '"operation-two"'
        with pytest.raises(Problem) as failure:
            registry._write_record(ACTOR, "jobs/test-only/cancellation.json", value, first.etag)
    assert failure.value.code == "registry_revision_conflict"
    assert failure.value.status == 409
    assert len(transport.requests) == 3
    assert transport.requests[1].headers["If-Match"] == '"operation-one"'
    assert transport.requests[2].headers["If-Match"] == '"operation-one"'
    assert "If-None-Match" not in transport.requests[1].headers


def test_existing_heartbeat_reads_through_actual_sdk_download_and_its_matching_etag():
    registry, container, transport, value = setup([])
    transport.responses.append(
        {
            "status": 206,
            "etag": '"existing-heartbeat"',
            "body": value.model_dump_json().encode(),
        }
    )
    with container:
        read = registry._read_record(
            ACTOR, "jobs/test-only/reconciliation.json", ReconciliationHeartbeat
        )
    assert read.value == value and read.etag == '"existing-heartbeat"'
    assert [request.method for request in transport.requests] == ["GET"]


def test_existing_heartbeat_conditionally_updates_without_renewing_target_or_deadline():
    registry, container, transport, value = setup([])
    old = value.model_copy(update={"observed_at": value.observed_at - timedelta(seconds=10)})
    transport.responses.extend(
        [
            {"status": 206, "etag": '"existing-heartbeat"', "body": old.model_dump_json().encode()},
            {"status": 201, "etag": '"updated-heartbeat"'},
        ]
    )
    with container:
        saved = registry.record_heartbeat(ACTOR, old.target, old.worker_client_id)
    assert saved.value.target == old.target
    assert saved.value.worker_client_id == old.worker_client_id
    assert saved.value.observed_at > old.observed_at
    assert saved.etag == '"updated-heartbeat"'
    assert [request.method for request in transport.requests] == ["GET", "PUT"]
    assert transport.requests[1].headers["If-Match"] == '"existing-heartbeat"'
    assert "If-None-Match" not in transport.requests[1].headers


def test_report_inventory_uses_actual_sdk_blob_names_sizes_and_etags(monkeypatch):
    from azure.storage.blob import BlobServiceClient

    from apps.api.learning_models import fingerprint
    from apps.learning_worker.artifacts import VerifiedArtifacts

    key = "tenants/test/owners/test/evidence/results.json"
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://offline.blob.core.windows.net/"
 ContainerName="test-only">
<Prefix>tenants/test/owners/test/evidence/</Prefix><Blobs><Blob><Name>{key}</Name><Properties>
<Last-Modified>Wed, 23 Sep 2026 12:00:00 GMT</Last-Modified><Etag>"evidence-one"</Etag>
<Content-Length>42</Content-Length><BlobType>BlockBlob</BlobType>
</Properties></Blob></Blobs><NextMarker /></EnumerationResults>""".encode()
    transport = OfflineTransport(
        [
            {
                "status": 200,
                "etag": '"listing"',
                "body": body,
                "content_type": "application/xml",
            }
        ]
    )
    client = BlobServiceClient(
        "https://offline.blob.core.windows.net", credential=None, transport=transport, retry_total=0
    )
    monkeypatch.setattr(
        "apps.learning_worker.artifacts.BlobServiceClient", lambda *args, **kwargs: client
    )
    verifier = VerifiedArtifacts(None, None, "https://offline.blob.core.windows.net", "test-only")
    result = verifier._blob_inventory(
        "https://offline.blob.core.windows.net", "test-only", "tenants/test/owners/test/evidence/"
    )
    assert result == fingerprint({key: {"etag": '"evidence-one"', "size": 42}})
    assert len(transport.requests) == 1 and transport.requests[0].method == "GET"
