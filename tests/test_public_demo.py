import json
from datetime import timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from runtime_support import ACTOR, PNG, document, service, settings

from apps.api.errors import Problem
from apps.api.main import create_app
from apps.api.models import SaveEnvironment, utcnow
from apps.api.public_demo import PublicDemo
from apps.api.settings import Settings


def test_default_public_page_needs_no_auth_and_makes_no_private_or_paid_calls():
    backend = service()
    backend.store.get_environment = Mock(side_effect=AssertionError("No private data read"))
    backend.store.list_runs = Mock(side_effect=AssertionError("No private history read"))
    backend.bridge.status = Mock(side_effect=AssertionError("No unpublished simulator read"))
    with TestClient(create_app(settings(), backend)) as client:
        response = client.get("/api/demo")
        assert response.status_code == 200
        payload = response.json()
        assert payload["access"] == "public_read_only"
        assert payload["mode"] == "reference"
        assert payload["simulation"]["live_available"] is False
        assert payload["scene"]["data_origin"] == "synthetic_reference_configuration"
        assert payload["capabilities"]["anonymous_control"] is False
        assert payload["agent"]["connectivity"] == "configured"
        assert payload["learning"]["quality_verified"] is False
        encoded = json.dumps(payload)
        assert str(ACTOR.tenant_id) not in encoded
        assert str(ACTOR.object_id) not in encoded
        assert ACTOR.owner_key not in encoded
        assert "auth" not in payload
        assert response.headers["cache-control"] == "public, max-age=1"
        assert (
            client.get(
                "/api/demo", headers={"Authorization": "Bearer not-a-real-token"}
            ).status_code
            == 200
        )
        assert client.get("/api/demo/frame").status_code == 503
        assert client.post("/api/demo", json={"instruction": "move"}).status_code == 405
    assert backend.planner.calls == 0
    assert backend.bridge.dispatches == 0


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/api/environments", None),
        ("GET", "/api/runs", None),
        ("GET", "/api/runtime", None),
        ("POST", "/api/environments", {"document_json": "{}"}),
        (
            "POST",
            "/api/runs",
            {
                "request_id": "12345678-1234-4234-8234-123456789abc",
                "environment_id": "reference-cell",
                "revision": "a" * 64,
                "instruction": "Move",
            },
        ),
    ],
)
def test_public_viewing_does_not_disable_operator_authentication(method, path, payload):
    with TestClient(create_app(settings(), service())) as client:
        response = client.request(method, path, json=payload)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"


def prepared_publication():
    backend = service()
    doc = document()
    doc["display_name"] = "Operator private title must not be published"
    record = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))
    backend.activate(ACTOR, record.environment_id, record.revision)
    configuration = settings().model_copy(
        update={
            "public_demo_publish_live": True,
            "public_demo_owner_id": ACTOR.object_id,
            "public_demo_environment_id": record.environment_id,
            "public_demo_revision": record.revision,
        }
    )
    return backend, record, configuration


def test_explicit_reference_publication_has_real_frames_but_no_private_fields():
    backend, record, configuration = prepared_publication()
    with TestClient(create_app(configuration, backend)) as client:
        snapshot = client.get("/api/demo").json()
        assert snapshot["mode"] == "live"
        assert snapshot["simulation"]["frame_url"] == "/api/demo/frame"
        assert record.display_name not in json.dumps(snapshot)
        frame = client.get("/api/demo/frame")
        assert frame.status_code == 200
        assert frame.content == PNG
        assert frame.headers["x-frame-id"]
        assert frame.headers["x-physics-steps"] == "42"
        assert backend.planner.calls == backend.bridge.dispatches == 0


def test_changed_revision_revokes_even_a_cached_public_frame():
    backend, record, configuration = prepared_publication()
    public = PublicDemo(configuration, backend)
    assert public.frame("overview")[0] == PNG
    edited = document()
    edited["display_name"] = "Another private revision"
    backend.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(edited),
            expected_revision=record.revision,
        ),
    )
    with pytest.raises(Problem) as failure:
        public.frame("overview")
    assert failure.value.code == "public_live_unavailable"
    assert record.environment_id not in failure.value.message
    with pytest.raises(Problem):
        public.frame("overview")


def test_custom_customer_layout_is_not_published_even_when_revision_is_pinned():
    backend, record, configuration = prepared_publication()
    custom = document()
    custom["stations"][0]["position_m"][0] = -0.3
    updated = backend.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(custom),
            expected_revision=record.revision,
        ),
    )
    configuration = configuration.model_copy(update={"public_demo_revision": updated.revision})
    public = PublicDemo(configuration, backend)
    assert public.snapshot()["mode"] == "reference"
    with pytest.raises(Problem):
        public.frame("overview")


def test_missing_publication_configuration_is_a_startup_error():
    values = settings().model_dump()
    values["public_demo_publish_live"] = True
    with pytest.raises(ValidationError, match="explicit owner"):
        Settings.model_validate(values)


def test_connectivity_proof_does_not_publish_response_identifiers_or_claim_inspection():
    backend = service()
    backend.agent_probe_verified_at = utcnow() - timedelta(hours=2)
    snapshot = PublicDemo(settings(), backend).snapshot()
    assert snapshot["agent"]["connectivity"] == "verified"
    assert snapshot["agent"]["verification_scope"] == "connectivity_only"
    assert snapshot["agent"]["verified_at"]
    assert "response_id" not in json.dumps(snapshot)


def test_unavailable_frame_never_returns_a_previous_or_placeholder_image():
    backend, record, configuration = prepared_publication()
    backend.bridge.captured_at = utcnow() - timedelta(seconds=5)
    with TestClient(create_app(configuration, backend)) as client:
        response = client.get("/api/demo/frame")
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert response.headers["cache-control"] == "no-store"


def test_public_snapshot_returns_copies_not_shared_mutable_data():
    public = PublicDemo(settings(), service())
    first = public.snapshot()
    first["scene"]["stations"][0]["position_m"][0] = 1000
    assert (
        public.snapshot()["scene"]["stations"][0]["position_m"][0]
        == document()["stations"][0]["position_m"][0]
    )
