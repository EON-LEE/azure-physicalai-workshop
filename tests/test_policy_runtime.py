"""CPU learned dispatch tests, with an explicitly injected non-GPU provider."""

from datetime import timedelta
from uuid import uuid4

import pytest
from runtime_support import ACTOR, OTHER
from test_teaching_runtime import teaching as teaching

from apps.api.errors import Problem
from apps.api.models import MotionCommand
from simulation.runtime_contracts import CONTROL_PROFILE_ID, PolicyCommand, PolicyRuntime


class Provider:
    def authorize(self, owner, request, profile):
        if owner != ACTOR.owner_key:
            raise Problem(404, "policy_missing", "No policy release for this owner.")

    def create(self, owner, request, profile):
        raise RuntimeError("This CPU provider has no GPU model.")


def request_for(teaching):
    core, teaching_request, clock = teaching
    fields = teaching_request.model_dump(
        include={
            "command_id",
            "environment_id",
            "revision",
            "epoch",
            "state_revision",
            "observation_id",
            "object_id",
            "target_station_id",
        }
    )
    request = PolicyCommand(
        policy_type="smolvla",
        command=MotionCommand(**fields, deadline=clock[0] + timedelta(seconds=30)),
        policy_release_id=uuid4(),
        model_sha256="b" * 64,
        control_profile_id=CONTROL_PROFILE_ID,
        task=teaching_request.task,
    )
    core.tenant_id = str(ACTOR.tenant_id)
    core.policy_provider = Provider()
    return core, request, clock


def test_missing_policy_provider_is_explicit_and_never_routes_reference(teaching):
    core, request, _ = request_for(teaching)
    core.policy_provider = None
    with pytest.raises(Problem, match="policy"):
        core.dispatch_policy(ACTOR.owner_key, request)
    assert core.next_action() is None
    assert core.active_command is None


def test_learned_dispatch_has_immutable_release_fingerprint_and_zero_application_evidence(teaching):
    core, request, _ = request_for(teaching)
    execution = core.dispatch_policy(ACTOR.owner_key, request)
    assert execution.status == "queued"
    assert execution.policy_runtime.applied_model_sha is None
    assert execution.policy_runtime.applied_action_count == 0
    assert core.dispatch_policy(ACTOR.owner_key, request) == execution
    assert core.next_action().request == request
    assert core.next_action() is None
    with pytest.raises(Problem, match="reused"):
        core.dispatch_policy(
            ACTOR.owner_key, request.model_copy(update={"policy_release_id": uuid4()})
        )
    with pytest.raises(Problem, match="reused"):
        core.dispatch(ACTOR.owner_key, request.command)


def test_learned_context_excludes_evaluator_state_and_stops_on_cancellation(teaching):
    core, request, _ = request_for(teaching)
    core.dispatch_policy(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command.command_id)
    binding = core.binding(request.command.command_id)
    context = core.policy_context(binding)
    assert context.approved and context.active
    assert context.scope.owner_id == ACTOR.owner_key
    assert context.destination_id == request.task.goal_id
    assert not hasattr(context, "seed")
    assert not hasattr(context, "object_position")
    core.cancel(ACTOR.owner_key, request.command.command_id)
    assert not core.policy_context(binding).active
    called = []
    with pytest.raises(RuntimeError, match="active"):
        core.apply_guarded(binding, lambda: called.append(True))
    assert not called


@pytest.mark.parametrize("seconds", [31, 300])
def test_learned_tasks_cannot_borrow_the_longer_teaching_session_budget(teaching, seconds):
    core, request, clock = request_for(teaching)
    request = request.model_copy(
        update={
            "command": request.command.model_copy(
                update={"deadline": clock[0] + timedelta(seconds=seconds)}
            ),
        }
    )
    with pytest.raises(Problem, match="deadline"):
        core.dispatch_policy(ACTOR.owner_key, request)
    assert core.active_command is None


def test_policy_runtime_metrics_survive_physical_finish_but_reject_late_binding(teaching):
    core, request, _ = request_for(teaching)
    core.dispatch_policy(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command.command_id)
    binding = core.binding(request.command.command_id)
    metrics = PolicyRuntime(
        policy_type=request.policy_type,
        policy_release_id=request.policy_release_id,
        applied_model_sha=request.model_sha256,
        policy_predict_calls=1,
        applied_action_count=6,
    )
    assert core.publish_policy_metrics(binding, metrics)
    core.finish("succeeded", (0.22, -0.38, 0.2), binding=binding)
    result = core.command(ACTOR.owner_key, request.command.command_id)
    assert result.policy_runtime == metrics
    assert not core.publish_policy_metrics(binding, metrics)
    assert core.command(ACTOR.owner_key, request.command.command_id) == result
    with pytest.raises(Problem):
        core.command(OTHER.owner_key, request.command.command_id)
