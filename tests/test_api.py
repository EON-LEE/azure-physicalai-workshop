import json
from datetime import timedelta
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from runtime_support import ACTOR, AUDIENCE, TENANT, document, service, settings

from apps.api.auth import EntraTokens
from apps.api.main import create_app
from apps.api.models import utcnow


@pytest.fixture
def api():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    identity = EntraTokens(TENANT, AUDIENCE, key_resolver=lambda token: key.public_key())
    backend = service()

    def token(**overrides):
        now = utcnow()
        claims = {
            "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
            "aud": str(AUDIENCE),
            "tid": str(TENANT),
            "oid": str(ACTOR.object_id),
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "scp": "access_as_user",
        }
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256")

    with TestClient(create_app(settings(), backend, identity)) as client:
        yield client, backend, token


def headers(token):
    return {"Authorization": f"Bearer {token()}"}


def test_bootstrap_exposes_only_public_auth_configuration(api):
    client, _, _ = api
    payload = client.get("/api/config").json()
    assert set(payload) == {"api_version", "deployment", "auth"}
    assert set(payload["auth"]) == {"tenant_id", "client_id", "scope"}
    assert payload["deployment"] == "azure"
    assert client.get("/healthz").json()["status"] == "process_running"


def test_authentication_is_required_even_with_spoofed_easy_auth_headers(api):
    client, _, _ = api
    response = client.get("/api/environments", headers={"X-MS-CLIENT-PRINCIPAL": "spoof"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"aud": "wrong"}, 401),
        ({"iss": "https://example.test"}, 401),
        ({"exp": utcnow() - timedelta(minutes=1)}, 401),
        ({"scp": "read_other_api"}, 403),
        ({"tid": str(uuid4())}, 403),
    ],
)
def test_invalid_identity_or_scope_fails(api, overrides, status):
    client, _, token = api
    response = client.get(
        "/api/environments", headers={"Authorization": f"Bearer {token(**overrides)}"}
    )
    assert response.status_code == status


def test_save_conflict_activate_plan_approve_and_read_evidence(api):
    client, backend, token = api
    auth = headers(token)
    saved = client.post(
        "/api/environments",
        headers=auth,
        json={
            "document_json": json.dumps(document()),
            "expected_revision": None,
        },
    )
    assert saved.status_code == 201
    environment = saved.json()
    duplicate = client.post(
        "/api/environments",
        headers=auth,
        json={
            "document_json": json.dumps(document()),
            "expected_revision": None,
        },
    )
    assert duplicate.status_code == 409
    activated = client.post(
        f"/api/environments/{environment['environment_id']}/activate",
        headers=auth,
        json={"revision": environment["revision"]},
    )
    assert activated.status_code == 202
    frame = client.get(
        f"/api/environments/{environment['environment_id']}/frame",
        headers=auth,
        params={"revision": environment["revision"], "camera": "overview"},
    )
    assert frame.status_code == 200
    assert frame.content.startswith(b"\x89PNG")
    assert frame.headers["x-frame-id"]
    assert frame.headers["cache-control"] == "no-store"
    started = client.post(
        "/api/runs",
        headers=auth,
        json={
            "request_id": str(uuid4()),
            "environment_id": environment["environment_id"],
            "revision": environment["revision"],
            "instruction": "Inspect this part.",
        },
    )
    assert started.status_code == 201, started.text
    run = started.json()
    assert run["status"] == "awaiting_approval"
    assert backend.bridge.dispatches == 0
    assert client.get(f"/api/runs/{run['id']}/observation", headers=auth).content == frame.content
    approved = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=auth,
        json={"plan_response_id": run["plan"]["model_response_id"]},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "running"
    assert backend.bridge.dispatches == 1


def test_raw_duplicate_json_and_schema_errors_are_visible(api):
    client, _, token = api
    auth = headers(token)
    response = client.post(
        "/api/environments",
        headers=auth,
        json={"document_json": '{"environment_id":"a","environment_id":"b"}'},
    )
    assert response.status_code == 422
    assert "Duplicate" in response.json()["error"]["message"]
    invalid = document()
    invalid["stations"][0]["position_m"][0] = 100
    response = client.post(
        "/api/environments", headers=auth, json={"document_json": json.dumps(invalid)}
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["code"] == "station_bounds"


def test_body_limit_and_unknown_api_do_not_render_successful_html(api):
    client, _, token = api
    response = client.post(
        "/api/environments", headers=headers(token), content=b"x" * (1024 * 1024 + 1)
    )
    assert response.status_code == 413
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/").status_code == 503


def test_template_endpoint_and_security_headers(api):
    client, _, token = api
    response = client.get("/api/environment-templates", headers=headers(token))
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    assert "unsafe-eval" not in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_application_tokens_are_only_accepted_for_the_allowlisted_controller(api):
    _, _, token = api
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    identity = EntraTokens(TENANT, AUDIENCE, key_resolver=lambda value: key.public_key())
    now = utcnow()
    claims = {
        "iss": identity.issuer,
        "aud": str(AUDIENCE),
        "tid": str(TENANT),
        "oid": str(ACTOR.object_id),
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    encoded = jwt.encode(claims, key, algorithm="RS256")
    identity.controller(f"Bearer {encoded}", {ACTOR.object_id})
    from apps.api.errors import Problem

    with pytest.raises(Problem):
        identity.controller(f"Bearer {encoded}", {uuid4()})
    with pytest.raises(Problem):
        identity.user(f"Bearer {encoded}")
