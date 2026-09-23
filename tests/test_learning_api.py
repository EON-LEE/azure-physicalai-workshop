from uuid import uuid4

import pytest
from test_api import api as api
from test_api import headers

from apps.api.learning_service import LearningService
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import OTHER


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/learning/capabilities"),
        ("GET", "/api/learning/projects"),
        ("GET", f"/api/learning/projects/{uuid4()}"),
        ("POST", "/api/learning/projects"),
        ("GET", f"/api/learning/jobs/{uuid4()}"),
        ("POST", f"/api/learning/jobs/{uuid4()}/cancel"),
        ("POST", f"/api/teaching-sessions/{uuid4()}/jog"),
        ("POST", "/api/policy-releases"),
    ],
)
def test_learning_routes_authenticate_before_any_disabled_or_resource_disclosure(api, method, path):
    client, _, _ = api
    response = client.request(
        method, path, json={}, headers={"X-MS-CLIENT-PRINCIPAL": "not-an-identity"}
    )
    assert response.status_code == 401, response.text
    assert response.json()["error"]["code"] == "authentication_required"


def test_learning_capability_is_explicitly_off_without_breaking_existing_runtime(api):
    client, _, token = api
    response = client.get("/api/learning/capabilities", headers=headers(token))
    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert response.json()["status"] == "disabled"
    assert response.json()["policy_types"] == ["gr00t_n1_5"]
    assert response.json()["training_verified"] is False
    assert client.get("/api/runtime", headers=headers(token)).status_code == 200


def test_disabled_learning_has_no_successful_placeholder_job(api):
    client, backend, token = api
    response = client.get(f"/api/learning/jobs/{uuid4()}", headers=headers(token))
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "learning_disabled"
    assert backend.store.items == {}


def test_public_learning_is_curated_not_anonymous_history_or_write_access(api):
    client, backend, _ = api
    response = client.get("/api/demo/learning")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "api_version": "public-learning-v1",
        "status": "not_published",
        "publication": None,
    }
    assert response.headers["cache-control"] == "no-store"
    assert client.post("/api/demo/learning", json={}).status_code == 405
    assert backend.store.items == {}


def test_http_owner_etag_and_paid_request_retries_are_enforced_before_dispatch(api):
    client, _, token = api
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    client.app.state.learning = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
    )
    project, _, train = seed_project_and_dataset(store, request)
    resource = f"/api/learning/projects/{project.value.id}"
    auth = headers(token)
    response = client.get(resource, headers=auth)
    assert response.status_code == 200
    assert response.headers["etag"] == response.json()["etag"] == project.etag
    assert "owner_key" not in response.json()["item"]
    other = {"Authorization": f"Bearer {token(oid=str(OTHER.object_id))}"}
    assert client.get(resource, headers=other).status_code == 404
    path = f"{resource}/train"
    body = train.model_dump(mode="json")
    assert client.post(path, json=body, headers=auth).status_code == 428
    assert client.post(path, json=body, headers={**auth, "If-Match": "old"}).status_code == 409
    assert (
        client.post(path, json=body, headers={**other, "If-Match": project.etag}).status_code == 404
    )
    assert jobs.submissions == []
    submitted = client.post(path, json=body, headers={**auth, "If-Match": project.etag})
    assert submitted.status_code == 202, submitted.text
    assert submitted.json()["item"]["status"] == "submitted"
    assert submitted.json()["item"]["azure_job_id"]
    assert submitted.json()["item"]["candidate_id"] is None
    repeated = client.post(path, json=body, headers={**auth, "If-Match": "stale-after-response"})
    assert repeated.status_code == 202
    assert len(jobs.submissions) == 1
    changed = client.post(path, json={**body, "optimizer_steps": 99}, headers=auth)
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "request_id_reused"


def test_http_cannot_smuggle_authority_or_model_paths_with_explicit_paid_approval(api):
    client, _, token = api
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    client.app.state.learning = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True
    )
    project, _, train = seed_project_and_dataset(store, request)
    path = f"/api/learning/projects/{project.value.id}/train"
    auth = {**headers(token), "If-Match": project.etag}
    for changes in (
        {"paid_approved": False},
        {"paid_approved": 1},
        {"owner_key": OTHER.owner_key},
        {"model_uri": "https://unapproved.test/model"},
        {"policy_type": "act_auxiliary"},
        {"maximum_cost_usd": "100.00"},
    ):
        response = client.post(
            path, json={**train.model_dump(mode="json"), **changes}, headers=auth
        )
        assert response.status_code == 422, response.text
    assert jobs.submissions == []
