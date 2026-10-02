import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_api import api as api
from test_api import headers

from apps.api.errors import Problem
from apps.api.models import (
    Decision,
    Execution,
    PolicyRuntime,
    ReleasedPolicyBinding,
    SaveEnvironment,
    StartRun,
    utcnow,
)
from apps.api.service import content_hash
from tests.runtime_support import ACTOR, OTHER, document

TASK = "manufacturing-part-placement-v1"
INSTRUCTION = (
    "Pick up the synthetic part from the source platform and place it in the quarantine tray."
)


def setup_skill(api):
    client, backend, token = api
    content = document()
    content["scene"]["seed"] = 10001
    content["scene"]["template_id"] = "inspection-cell-learning-v1"
    saved = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(content)))
    backend.activate(ACTOR, saved.environment_id, saved.revision)
    released = SimpleNamespace(
        value=ReleasedPolicyBinding(
            policy_release_id=uuid4(),
            policy_type="smolvla",
            model_sha256="a" * 64,
            processor_sha256="b" * 64,
            manifest_sha256="a" * 64,
            control_profile_id="franka-position-hold-10hz-v1",
            task_id=TASK,
            goal_station_id="rejected",
            instruction=INSTRUCTION,
        )
    )

    class Authorizer:
        def resolve_for_run(self, actor, release_id, environment_id, revision):
            if actor != ACTOR or release_id != released.value.policy_release_id:
                raise Problem(404, "policy_release_missing", "No reviewed owner policy.")
            if (environment_id, revision) != (saved.environment_id, saved.revision):
                raise Problem(409, "policy_scene_mismatch", "Release belongs to another scene.")
            return released.value

    def inspect(instruction, observation):
        backend.planner.calls += 1
        return Decision(
            classification="accepted",
            object_id=observation.object_id,
            summary="Test-only normal part; no defect classification should be invented.",
        ), "test-only-real-shaped-normal-inspection"

    learned = []

    def dispatch(owner, command, policy):
        learned.append((owner, command, policy))
        result = Execution(
            command_id=command.command_id,
            status="queued",
            policy_runtime=PolicyRuntime(
                policy_type=policy.policy_type,
                policy_release_id=policy.policy_release_id,
                control_profile_id=policy.control_profile_id,
                applied_model_sha=None,
                policy_predict_calls=0,
                applied_action_count=0,
                reference_route_calls=0,
            ),
        )
        backend.bridge.results[(owner, command.command_id)] = result
        return result

    backend.policies = Authorizer()
    backend.planner.inspect = inspect
    backend.bridge.dispatch_policy = dispatch
    body = {
        "request_id": str(uuid4()),
        "environment_id": saved.environment_id,
        "revision": saved.revision,
        "instruction": INSTRUCTION,
        "execution_mode": "released_skill",
        "policy_release_id": str(released.value.policy_release_id),
    }
    return client, backend, token, released, saved, body, learned


def test_normal_part_can_plan_approved_quarantine_skill_without_cv_inspection(api):
    client, backend, token, released, _, body, learned = setup_skill(api)
    response = client.post("/api/runs", json=body, headers=headers(token))
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["status"] == "awaiting_approval"
    assert backend.planner.calls == 0
    assert learned == [] and backend.bridge.dispatches == 0
    assert run["execution_mode"] == "released_skill"
    assert run["plan"]["kind"] == "released_skill"
    assert run["plan"]["target_station_id"] == released.value.goal_station_id == "rejected"
    assert run["plan"]["task_id"] == TASK and run["plan"]["instruction"] == INSTRUCTION
    assert "classification" not in run["plan"] and "model_response_id" not in run["plan"]
    assert run["plan"]["observation_id"]
    assert run["plan"]["expires_at"]
    approved = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": run["plan"]["skill_plan_id"]},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "running"
    assert len(learned) == 1 and learned[0][1].target_station_id == "rejected"
    assert backend.planner.calls == 0 and backend.bridge.dispatches == 0
    again = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": run["plan"]["skill_plan_id"]},
    )
    assert again.status_code == 200 and len(learned) == 1


def test_omitted_skill_instruction_uses_the_exact_server_release_task(api):
    client, backend, token, _, _, body, _ = setup_skill(api)
    body.pop("instruction")
    response = client.post("/api/runs", json=body, headers=headers(token))
    assert response.status_code == 201
    run = response.json()
    assert run["instruction"] == run["plan"]["instruction"] == INSTRUCTION
    assert backend.planner.calls == 0
    image = client.get(f"/api/runs/{run['id']}/observation", headers=headers(token))
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")


@pytest.mark.parametrize("target", ["rejected", "accepted"])
def test_skill_result_requires_actual_learned_evidence_at_its_reviewed_goal(api, target):
    client, backend, token, released, saved, body, learned = setup_skill(api)
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": run["plan"]["skill_plan_id"]},
    ).raise_for_status()
    command_id = UUID(run["id"])
    position = next(
        item["position_m"] for item in saved.document["stations"] if item["id"] == target
    )
    backend.bridge.results[(ACTOR.owner_key, command_id)] = Execution(
        command_id=command_id,
        status="succeeded",
        final_position=position,
        completed_at=utcnow(),
        policy_runtime=PolicyRuntime(
            policy_type=released.value.policy_type,
            policy_release_id=released.value.policy_release_id,
            control_profile_id=released.value.control_profile_id,
            applied_model_sha=released.value.model_sha256,
            policy_predict_calls=2,
            applied_action_count=12,
            reference_route_calls=0,
        ),
    )
    observed = client.get(f"/api/runs/{run['id']}", headers=headers(token)).json()
    if target == "rejected":
        assert observed["status"] == "succeeded"
        assert observed["execution"]["final_position"] == position
        assert observed["execution"]["policy_runtime"]["applied_model_sha"] == "a" * 64
    else:
        assert observed["status"] == "failed"
        assert observed["error"]["code"] == "wrong_destination"
    assert "classification" not in observed["plan"]
    assert backend.planner.calls == 0 and backend.bridge.dispatches == 0 and len(learned) == 1


def test_inspection_request_keeps_legacy_fingerprint(api):
    _, _, _, _, saved, _, _ = setup_skill(api)
    legacy = {
        "request_id": str(uuid4()),
        "environment_id": saved.environment_id,
        "revision": saved.revision,
        "instruction": "Inspect the actual part.",
    }
    for mode in ({}, {"execution_mode": "inspection"}):
        parsed = StartRun.model_validate({**legacy, **mode})
        assert content_hash(parsed.fingerprint_document()) == content_hash(legacy)


@pytest.mark.parametrize(
    "override",
    [
        {"execution_mode": "inspection"},
        {"policy_release_id": None},
        {"task_id": "caller-override"},
        {"goal_station_id": "accepted"},
        {"model_sha256": "c" * 64},
    ],
)
def test_mode_mixing_and_client_task_goal_model_overrides_are_rejected(api, override):
    client, backend, token, _, _, body, learned = setup_skill(api)
    response = client.post("/api/runs", json={**body, **override}, headers=headers(token))
    assert response.status_code == 422
    assert backend.planner.calls == 0 and learned == []
    assert backend.store.list_runs(ACTOR.owner_key) == []


def test_wrong_instruction_or_unknown_release_cannot_become_a_skill_plan(api):
    client, backend, token, _, _, body, learned = setup_skill(api)
    mismatch = client.post(
        "/api/runs",
        json={**body, "instruction": "Inspect and classify as defective."},
        headers=headers(token),
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "policy_instruction_mismatch"
    missing = client.post(
        "/api/runs", json={**body, "policy_release_id": str(uuid4())}, headers=headers(token)
    )
    assert missing.status_code == 404
    assert backend.planner.calls == 0 and learned == []


def test_unknown_released_goal_is_rejected_before_capture_or_planning(api):
    client, backend, token, released, _, body, learned = setup_skill(api)
    released.value = released.value.model_copy(update={"goal_station_id": "not-in-saved-scene"})
    result = client.post("/api/runs", json=body, headers=headers(token))
    assert result.status_code == 409 and result.json()["error"]["code"] == "policy_goal_mismatch"
    assert backend.planner.calls == 0 and learned == []


@pytest.mark.parametrize("approval", ["missing", "both", "wrong-kind", "wrong-skill-id"])
def test_approval_is_exactly_one_plan_reference_of_the_right_kind(api, approval):
    client, _, token, _, _, body, learned = setup_skill(api)
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    bodies = {
        "missing": {},
        "both": {
            "skill_plan_id": run["plan"]["skill_plan_id"],
            "plan_response_id": "not-a-foundry-id",
        },
        "wrong-kind": {"plan_response_id": run["plan"]["skill_plan_id"]},
        "wrong-skill-id": {"skill_plan_id": str(uuid4())},
    }
    result = client.post(
        f"/api/runs/{run['id']}/approve", json=bodies[approval], headers=headers(token)
    )
    assert result.status_code == (422 if approval in ("missing", "both") else 409)
    assert learned == []


@pytest.mark.parametrize(
    "changed",
    [
        {"goal_station_id": "accepted"},
        {"task_id": "another-task"},
        {"instruction": "Different released instruction."},
        {"model_sha256": "c" * 64},
        {"processor_sha256": "c" * 64},
        {"policy_type": "gr00t_n1_7"},
    ],
)
def test_skill_approval_rechecks_immutable_release_binding(api, changed):
    client, _, token, released, _, body, learned = setup_skill(api)
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    released.value = released.value.model_copy(update=changed)
    response = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": run["plan"]["skill_plan_id"]},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "policy_release_changed"
    assert learned == []


@pytest.mark.parametrize("changed", ["goal", "task", "model", "observation", "expired", "scene"])
def test_skill_plan_tampering_stale_observation_and_expiry_do_not_dispatch(api, changed):
    client, backend, token, _, _, body, learned = setup_skill(api)
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    stored = backend.store.get_run(ACTOR.owner_key, UUID(run["id"]))
    updates = {
        "goal": {"target_station_id": "accepted"},
        "task": {"task_id": "another-task"},
        "model": {"model_sha256": "f" * 64},
        "observation": {"observation_id": uuid4()},
        "expired": {"expires_at": utcnow() - timedelta(seconds=1)},
    }
    if changed == "scene":
        backend.bridge.epoch = uuid4()
    else:
        altered = stored.value.model_copy(
            update={"plan": stored.value.plan.model_copy(update=updates[changed])}
        )
        backend.store.put_run(ACTOR.owner_key, altered, stored.etag)
    response = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": run["plan"]["skill_plan_id"]},
    )
    assert response.status_code == 409
    assert learned == [] and backend.planner.calls == 0


def test_another_owner_cannot_read_or_approve_a_released_skill_plan(api):
    client, _, token, _, _, body, learned = setup_skill(api)
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    other = {"Authorization": f"Bearer {token(oid=str(OTHER.object_id))}"}
    assert client.get(f"/api/runs/{run['id']}", headers=other).status_code == 404
    assert (
        client.post(
            f"/api/runs/{run['id']}/approve",
            headers=other,
            json={"skill_plan_id": run["plan"]["skill_plan_id"]},
        ).status_code
        == 404
    )
    assert learned == []


def test_legacy_inspection_still_uses_foundry_truth_and_accept_station(api):
    client, backend, token, _, _, body, learned = setup_skill(api)
    body.pop("policy_release_id")
    body.pop("execution_mode")
    body["instruction"] = "Inspect the part without overriding classification."
    run = client.post("/api/runs", json=body, headers=headers(token)).json()
    assert backend.planner.calls == 1
    assert run["execution_mode"] == "inspection" and run["plan"]["kind"] == "inspection"
    assert run["plan"]["classification"] == "accepted"
    assert run["plan"]["target_station_id"] == "accepted"
    assert run["plan"]["model_response_id"] == "test-only-real-shaped-normal-inspection"
    wrong = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"skill_plan_id": str(uuid4())},
    )
    assert wrong.status_code == 409
    approved = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=headers(token),
        json={"plan_response_id": run["plan"]["model_response_id"]},
    )
    assert approved.status_code == 200
    assert backend.bridge.dispatches == 1 and learned == []
