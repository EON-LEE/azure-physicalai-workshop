import json
from uuid import uuid4

import pytest
from test_api import api as api
from test_api import headers

from apps.api.models import Execution, PolicyRuntime, ReleasedPolicyBinding, SaveEnvironment, utcnow
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
            "execution_mode": "released_skill",
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
            "policy_type": "smolvla",
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


def test_approved_policy_is_pinned_in_plan_and_only_the_learned_dispatch_is_used(api):
    client, backend, token = api
    saved = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(document())))
    backend.activate(ACTOR, saved.environment_id, saved.revision)
    binding = ReleasedPolicyBinding(
        policy_release_id=uuid4(),
        policy_type="gr00t_n1_7",
        model_sha256="a" * 64,
        processor_sha256="b" * 64,
        manifest_sha256="a" * 64,
        control_profile_id="franka-position-hold-10hz-v1",
        task_id="part-kitting-v1",
        goal_station_id="rejected",
        instruction="승인된 격리 트레이로 부품을 옮깁니다.",
    )

    class Authorizer:
        def resolve_for_run(self, actor, release_id, environment_id, revision):
            assert actor == ACTOR and release_id == binding.policy_release_id
            assert (environment_id, revision) == (saved.environment_id, saved.revision)
            return binding

    learned = []

    def dispatch(owner, command, policy):
        learned.append((owner, command, policy))
        return Execution(
            command_id=command.command_id,
            status="queued",
            policy_runtime=PolicyRuntime(
                policy_type=policy.policy_type,
                policy_release_id=policy.policy_release_id,
                applied_model_sha=None,
                control_profile_id=policy.control_profile_id,
                policy_predict_calls=0,
                applied_action_count=0,
                reference_route_calls=0,
            ),
        )

    backend.policies = Authorizer()
    backend.bridge.dispatch_policy = dispatch
    auth = headers(token)
    started = client.post(
        "/api/runs",
        headers=auth,
        json={
            "request_id": str(uuid4()),
            "environment_id": saved.environment_id,
            "revision": saved.revision,
            "execution_mode": "released_skill",
            "policy_release_id": str(binding.policy_release_id),
        },
    )
    assert started.status_code == 201
    run = started.json()
    assert run["policy"]["model_sha256"] == binding.model_sha256
    assert learned == []
    approved = client.post(
        f"/api/runs/{run['id']}/approve",
        headers=auth,
        json={
            "skill_plan_id": run["plan"]["skill_plan_id"],
        },
    )
    assert approved.status_code == 200, approved.text
    assert len(learned) == 1 and backend.bridge.dispatches == 0
    assert approved.json()["status"] == "running"
    assert approved.json()["execution"]["policy_runtime"]["applied_model_sha"] is None


@pytest.mark.parametrize("model_sha", [None, "c" * 64])
def test_command_success_without_matching_actual_model_application_is_not_success(model_sha):
    from apps.api.errors import Problem
    from apps.api.models import RunRecord
    from apps.api.service import FactoryService

    now = utcnow()
    binding = ReleasedPolicyBinding(
        policy_release_id=uuid4(),
        policy_type="gr00t_n1_7",
        model_sha256="a" * 64,
        processor_sha256="b" * 64,
        manifest_sha256="a" * 64,
        control_profile_id="franka-position-hold-10hz-v1",
        task_id="part-kitting-v1",
        goal_station_id="accepted",
        instruction="부품을 놓습니다.",
    )
    run = RunRecord(
        id=uuid4(),
        environment_id="reference-cell",
        revision="a" * 64,
        instruction="검사",
        status="running",
        created_at=now,
        updated_at=now,
        request_fingerprint="b" * 64,
        environment_document={},
        policy=binding,
    )
    runtime = PolicyRuntime(
        policy_type=binding.policy_type,
        policy_release_id=binding.policy_release_id,
        applied_model_sha=model_sha,
        control_profile_id=binding.control_profile_id,
        policy_predict_calls=1 if model_sha else 0,
        applied_action_count=1 if model_sha else 0,
        reference_route_calls=0,
    )
    with pytest.raises(Problem) as failure:
        FactoryService._apply_execution(
            run,
            Execution(
                command_id=run.id,
                status="succeeded",
                policy_runtime=runtime,
                completed_at=now,
                final_position=(0, 0, 0),
            ),
        )
    assert failure.value.code == "learned_execution_unverified"
    assert run.status == "running"
