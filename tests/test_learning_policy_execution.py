import json
from uuid import uuid4

from test_api import api as api
from test_api import headers

from apps.api.models import Execution, SaveEnvironment
from tests.runtime_support import ACTOR, document


def test_released_policy_request_never_silently_uses_reference_dispatch(api):
    client, backend, token = api
    saved = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(document())))
    backend.activate(ACTOR, saved.environment_id, saved.revision)
    response = client.post(
        "/api/runs",
        headers=headers(token),
        json={
            "request_id": str(uuid4()),
            "environment_id": saved.environment_id,
            "revision": saved.revision,
            "instruction": "검사하고 승인된 정책으로 이동합니다.",
            "policy_release_id": str(uuid4()),
        },
    )
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "learning_disabled"
    assert backend.bridge.dispatches == 0
    assert backend.store.list_runs(ACTOR.owner_key) == []


def test_execution_allows_missing_applied_sha_until_an_action_is_really_applied():
    payload = {
        "command_id": str(uuid4()),
        "status": "queued",
        "policy_runtime": {
            "execution_mode": "learned",
            "policy_release_id": str(uuid4()),
            "applied_model_sha": None,
            "control_profile_id": "franka-position-hold-10hz-v1",
            "policy_predict_calls": 0,
            "applied_action_count": 0,
            "reference_route_calls": 0,
        },
    }
    parsed = Execution.model_validate(payload)
    assert parsed.policy_runtime.applied_model_sha is None
    assert parsed.policy_runtime.applied_action_count == 0
