from uuid import uuid4

from test_api import api as api
from test_api import headers

from apps.api.learning_models import CreateProject
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_teaching_cases import activate_case, varied_context


def test_authenticated_case_selection_is_only_a_case_id_and_returns_the_original_pins(api):
    client, _, token = api
    service, raw, cases, _ = varied_context()
    project = service.create_project(ACTOR, CreateProject.model_validate(raw))
    client.app.state.learning = service
    activate_case(service, cases[2])
    path = f"/api/learning/projects/{project.value.id}/teaching-sessions"
    body = {
        "request_id": str(uuid4()),
        "case_id": cases[2]["case_id"],
        "source": "human_teleop",
        "motion_approved": True,
    }
    assert client.post(path, json=body).status_code == 401
    assert (
        client.post(
            path,
            json=body,
            headers={
                "Authorization": f"Bearer {token(oid=str(OTHER.object_id))}",
                "If-Match": project.etag,
            },
        ).status_code
        == 404
    )
    auth = {**headers(token), "If-Match": project.etag}
    denied = client.post(path, json={**body, "seed": 10001}, headers=auth)
    assert denied.status_code == 422
    assert service.runtime.starts == []
    response = client.post(path, json=body, headers=auth)
    assert response.status_code == 202, response.text
    assert response.json()["item"]["teaching_case"] == cases[2]
    assert response.headers["etag"] == response.json()["etag"]
    assert service.runtime.starts[0].split == "validation"
    duplicate = client.post(path, json=body, headers=auth)
    assert duplicate.status_code == 202
    assert len(service.runtime.starts) == 1
    changed = client.post(path, json={**body, "case_id": cases[0]["case_id"]}, headers=auth)
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "request_id_reused"
