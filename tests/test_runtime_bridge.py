from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from runtime_support import ACTOR, OTHER
from test_teaching_runtime import jog
from test_teaching_runtime import teaching as teaching

from apps.api.errors import Problem
from simulation.http import BridgeSettings, create_bridge_app


def client_for(core):
    class Identity:
        def controller(self, authorization, principals):
            if authorization != "Bearer cpu-test-controller":
                raise Problem(401, "unauthorized", "A controller identity is required.")

    settings = BridgeSettings(
        entra_tenant_id=ACTOR.tenant_id,
        entra_api_client_id=uuid4(),
        bridge_allowed_principal_ids=[uuid4()],
        sim_tls_cert_file="test-only.pem",
        sim_tls_key_file="test-only-key.pem",
    )
    return TestClient(create_bridge_app(core, settings, Identity()))


def auth(owner=ACTOR.owner_key):
    return {"Authorization": "Bearer cpu-test-controller", "X-Environment-Owner": owner}


def test_teaching_bridge_cannot_accept_anonymous_or_unscoped_motion(teaching):
    core, request, _ = teaching
    with client_for(core) as client:
        assert client.post("/v1/teaching", json=request.model_dump(mode="json")).status_code == 401
        assert (
            client.post(
                "/v1/teaching",
                json=request.model_dump(mode="json"),
                headers={"Authorization": "Bearer cpu-test-controller"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/teaching",
                json=request.model_dump(mode="json"),
                headers=auth(OTHER.owner_key),
            ).status_code
            == 409
        )
    assert core.active_command is None


def test_bridge_keeps_session_owner_isolation_and_rejects_raw_joints(teaching):
    core, request, clock = teaching
    with client_for(core) as client:
        response = client.post("/v1/teaching", json=request.model_dump(mode="json"), headers=auth())
        assert response.status_code == 202
        assert response.json()["demonstrator_kind"] == "human_teleop"
        core.next_action()
        core.begin_motion(request.command_id)
        path = f"/v1/teaching/{request.session_id}"
        assert client.get(path, headers=auth(OTHER.owner_key)).status_code == 404
        assert client.get(path, headers=auth()).json()["status"] == "running"
        payload = jog(request, clock).model_dump(mode="json")
        assert (
            client.post(
                path + "/input",
                json=payload | {"joint_positions": [0] * 9},
                headers=auth(),
            ).status_code
            == 422
        )
        assert client.post(path + "/input", json=payload, headers=auth()).status_code == 200
        assert client.post(path + "/input", json=payload).status_code == 401
        assert (
            client.get(
                f"/v1/commands/{request.command_id}/capture",
                headers=auth(OTHER.owner_key),
            ).status_code
            == 404
        )


@pytest.mark.parametrize("path", ["/v1/policy/commands", "/v1/commands"])
def test_no_joint_array_actuator_is_exposed_at_either_command_endpoint(teaching, path):
    core, _, _ = teaching
    with client_for(core) as client:
        assert (
            client.post(path, json={"joint_positions": [0] * 9}, headers=auth()).status_code == 422
        )
    assert core.active_command is None
