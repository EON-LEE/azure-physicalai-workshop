"""Serialize real pinned SDK requests through an in-memory HTTP transport, never Azure."""

import json
from urllib.parse import urlsplit

from azure.core.credentials import AccessToken
from azure.core.pipeline.transport import HttpResponse, HttpTransport
from test_simulation_batch import batch_sdk as batch_sdk
from test_simulation_batch import spec as spec
from test_simulation_batch import spec_url

from simulation.batch import BATCH_TOKEN_SCOPE, submit_once


class Response(HttpResponse):
    def __init__(self, request, status, value):
        super().__init__(request, None)
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.reason = "CPU transport fixture"
        self.content_type = "application/json"
        self.payload = json.dumps(value).encode()

    def body(self):
        return self.payload

    def json(self):
        return json.loads(self.payload)

    def stream_download(self, pipeline, **kwargs):
        return iter([self.payload])


class Wire(HttpTransport):
    def __init__(self):
        self.requests = []
        self.job = None
        self.task = None

    def open(self):
        pass

    def close(self):
        pass

    def __exit__(self, *args):
        self.close()

    def send(self, request, **kwargs):
        path = urlsplit(request.url).path
        body = json.loads(request.body) if request.body else None
        self.requests.append((request.method, path, body, dict(request.headers)))
        if request.method == "GET":
            value = self.task if "/tasks/" in path else self.job
            if value is None:
                return Response(
                    request,
                    404,
                    {
                        "code": "NotFound",
                        "message": {"lang": "en-US", "value": "CPU fixture"},
                    },
                )
            return Response(request, 200, value)
        if request.method == "POST" and path.endswith("/jobs"):
            assert body["onAllTasksComplete"] == "noaction"
            self.job = body | {"eTag": '"cpu-etag"'}
            return Response(request, 201, {})
        if request.method == "POST" and path.endswith("/tasks"):
            assert self.job is not None and self.task is None
            self.task = body
            return Response(request, 201, {})
        assert request.method == "PATCH" and self.task is not None
        assert body == {"onAllTasksComplete": "terminatejob"}
        assert request.headers["If-Match"] == '"cpu-etag"'
        self.job.update(body)
        return Response(request, 200, {})


class Credential:
    def get_token(self, *scopes, **kwargs):
        assert scopes == ("https://batch.core.windows.net/.default",)
        return AccessToken("cpu-fixture-not-an-azure-token", 2**31)


def test_actual_sdk_wire_request_shapes_and_reconciliation(spec, batch_sdk):
    wire = Wire()
    with batch_sdk.BatchClient(
        endpoint=spec.platform.account_url,
        credential=Credential(),
        transport=wire,
        retry_total=0,
        credential_scopes=[BATCH_TOKEN_SCOPE],
    ) as client:
        submit_once(client, spec, spec_url(spec), spec.sha256)
        submit_once(client, spec, spec_url(spec), spec.sha256)
    writes = [request for request in wire.requests if request[0] != "GET"]
    assert [(method, path) for method, path, _, _ in writes] == [
        ("POST", "/jobs"),
        ("POST", f"/jobs/{spec.job_id}/tasks"),
        ("PATCH", f"/jobs/{spec.job_id}"),
    ]
    task = writes[1][2]
    assert task["resourceFiles"][0]["identityReference"]["resourceId"] == (
        spec.platform.node_identity_resource_id
    )
    assert task["containerSettings"]["workingDirectory"] == "taskWorkingDirectory"
    assert task["userIdentity"] == {"autoUser": {"scope": "task", "elevationLevel": "nonadmin"}}
    assert task["constraints"]["maxTaskRetryCount"] == 0
