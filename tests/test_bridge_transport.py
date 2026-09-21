from types import SimpleNamespace

import httpx
import pytest

from apps.api import bridge_client
from apps.api.bridge_client import AzureSimulatorBridge
from apps.api.errors import Problem


def bridge(handler):
    credential = SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="test-only-token"))
    return AzureSimulatorBridge(
        "https://sim.example.test",
        credential,
        "test-only-scope",
        transport=httpx.MockTransport(handler),
    )


def test_a_stale_read_connection_is_retried_once_within_the_original_budget(caplog):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.RemoteProtocolError("Test-only idle connection closed.")
        return httpx.Response(200, json={"status": "unavailable", "message": "No active scene."})

    client = bridge(handler)
    try:
        result = client.status("test-owner")
        assert result.message == "No active scene."
        assert len(requests) == 2
        assert all(request.method == "GET" for request in requests)
        assert requests[1].extensions["timeout"]["read"] <= 10
        assert "Retrying one read-only" in caplog.text
    finally:
        client.close()


@pytest.mark.parametrize("path", ["/v1/scene", "/v1/commands", "/v1/commands/example/cancel"])
def test_state_changing_requests_are_never_automatically_repeated(path):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.RemoteProtocolError("The test server might already have handled this POST.")

    client = bridge(handler)
    try:
        with pytest.raises(Problem):
            client._request("POST", path, "test-owner", json={})
        assert len(requests) == 1
    finally:
        client.close()


def test_persistent_read_failure_does_not_loop_or_return_a_success_fallback():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.RemoteProtocolError("Test-only persistent disconnect.")

    client = bridge(handler)
    try:
        with pytest.raises(Problem):
            client.status("test-owner")
        assert len(requests) == 2
    finally:
        client.close()


def test_expired_request_budget_cannot_start_a_retry(monkeypatch):
    clock = iter((0, 11))
    monkeypatch.setattr(bridge_client, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.RemoteProtocolError("Test-only disconnect after the budget.")

    client = bridge(handler)
    try:
        with pytest.raises(Problem):
            client.status("test-owner")
        assert len(requests) == 1
    finally:
        client.close()
