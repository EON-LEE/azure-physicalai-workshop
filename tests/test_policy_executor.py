"""CPU policy/actuator doubles. No result here is GR00T or Isaac GPU evidence."""

from dataclasses import replace
from uuid import uuid4

import pytest
from runtime_support import ACTOR, PNG

from learning.contract import CameraSample, ControlProfile, Scope
from learning.inference import (
    ControlContext,
    GuardedPolicyAdapter,
    PolicyObservation,
    SafetyLimits,
)
from simulation.policy_executor import PolicyExecutor

JOINTS = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02)


class CountingPolicy:
    fps, physics_hz, chunk_size, n_action_steps = 10, 60, 16, 1
    model_sha256 = "b" * 64
    scope = Scope(str(ACTOR.tenant_id), ACTOR.owner_key)
    metadata = {"policy_type": "smolvla"}

    def __init__(self):
        self.predict_calls = 0
        self.resets = 0
        self.after_predict = None

    def reset(self):
        self.resets += 1

    def predict_chunk(self, observation):
        self.predict_calls += 1
        if self.after_predict:
            self.after_predict()
        return ((0.001,) + JOINTS[1:],) * 16


@pytest.fixture
def policy():
    clock = [1_000_000_000]
    model = CountingPolicy()
    contexts = [
        ControlContext(
            model.scope,
            "cell",
            "a" * 64,
            "episode-1",
            "epoch-1",
            "command-1",
            "rejected",
            True,
            True,
            clock[0] + 30_000_000_000,
        )
    ]
    adapter = GuardedPolicyAdapter(model, limits=SafetyLimits(), clock_ns=lambda: clock[0])
    executor = PolicyExecutor(
        adapter,
        profile=ControlProfile("c" * 64),
        release_id=uuid4(),
        policy_type="smolvla",
        expected_model_sha256=model.model_sha256,
        context=lambda: contexts[0],
        clock_ns=lambda: clock[0],
        apply_guard=lambda apply: apply(),
    )
    observation = PolicyObservation(
        model.scope,
        "cell",
        "a" * 64,
        "episode-1",
        "2026-09-23T00:00:00Z",
        clock[0],
        0,
        JOINTS,
        {name: CameraSample(PNG, 1, 0, clock[0]) for name in ("inspection", "overview")},
    )
    return executor, adapter, model, contexts, clock, observation


def test_real_policy_targets_are_the_only_values_submitted_to_the_actuator(policy):
    executor, _, model, _, clock, observation = policy
    executor.start()
    command = executor.predict(observation)
    applied = []
    for step in range(1, 7):
        executor.apply(command, physics_step=step, actuator=lambda targets: applied.append(targets))
        clock[0] += 10_000_000
    assert applied == [command.targets] * 6
    assert applied[0] == (0.001,) + JOINTS[1:]
    stats = executor.metrics()
    assert stats.applied_model_sha == model.model_sha256
    assert stats.policy_predict_calls == model.predict_calls == 1
    assert stats.applied_action_count == 6
    assert stats.reference_route_calls == 0


@pytest.mark.parametrize("field", ["active", "epoch", "command_id", "scope", "destination_id"])
def test_cancel_or_changed_binding_during_prediction_cannot_apply_late_results(policy, field):
    executor, _, model, contexts, _, observation = policy
    executor.start()
    changed = {
        "active": False,
        "epoch": "epoch-2",
        "command_id": "command-2",
        "scope": Scope(str(ACTOR.tenant_id), "f" * 64),
        "destination_id": "accepted",
    }
    model.after_predict = lambda: contexts.__setitem__(
        0, replace(contexts[0], **{field: changed[field]})
    )
    with pytest.raises((RuntimeError, ValueError)):
        executor.predict(observation)
    assert executor.metrics().applied_action_count == 0
    assert executor.metrics().applied_model_sha is None


@pytest.mark.parametrize("failure", ["model", "expiry", "cancel", "reset", "physics_step"])
def test_queued_policy_action_is_rechecked_immediately_before_application(policy, failure):
    executor, _, model, contexts, clock, observation = policy
    executor.start()
    command = executor.predict(observation)
    if failure == "model":
        model.model_sha256 = "d" * 64
    elif failure == "expiry":
        clock[0] = command.expires_at_monotonic_ns
    elif failure == "cancel":
        contexts[0] = replace(contexts[0], active=False)
    elif failure == "reset":
        executor.stop()
    applied = []
    with pytest.raises((RuntimeError, ValueError)):
        executor.apply(
            command,
            physics_step=2 if failure == "physics_step" else 1,
            actuator=lambda targets: applied.append(targets),
        )
    assert not applied
    assert executor.metrics().applied_model_sha is None


def test_inference_latency_overrun_is_failure_not_reference_or_clipped_success(policy):
    executor, _, model, _, clock, observation = policy
    executor.start()
    model.after_predict = lambda: clock.__setitem__(0, clock[0] + 81_000_000)
    with pytest.raises(ValueError, match="latency"):
        executor.predict(observation)
    assert executor.metrics().reference_route_calls == 0
    assert executor.metrics().applied_action_count == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "e" * 64),
        ("hold_steps", 1),
        ("physics_step", 1),
        ("targets", (float("nan"),) + JOINTS[1:]),
    ],
)
def test_guarded_port_response_still_requires_independent_actuator_validation(policy, field, value):
    executor, adapter, _, _, _, observation = policy
    executor.start()
    original = adapter.step
    adapter.step = lambda obs, context: replace(original(obs, context), **{field: value})
    with pytest.raises((ValueError, RuntimeError)):
        executor.predict(observation)
    assert executor.metrics().applied_action_count == 0


def test_failed_actuator_submission_never_claims_a_model_was_applied(policy):
    executor, _, _, _, _, observation = policy
    executor.start()
    command = executor.predict(observation)

    def broken_actuator(targets):
        raise RuntimeError("Fixture actuator rejected the action")

    with pytest.raises(RuntimeError, match="actuator"):
        executor.apply(command, physics_step=1, actuator=broken_actuator)
    assert executor.metrics().applied_model_sha is None
    assert executor.metrics().policy_predict_calls == 1
