from __future__ import annotations

from dataclasses import replace

import pytest

from learning.checks.fixtures import JOINTS, SCOPE, frame
from learning.common import ContractError
from learning.inference import (
    ControlContext,
    GuardedPolicyAdapter,
    PolicyObservation,
    SafetyLimits,
)


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now


class Policy:
    fps = 10
    physics_hz = 60
    chunk_size = 3
    n_action_steps = 2
    model_sha256 = "b" * 64
    scope = SCOPE

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.chunk = (JOINTS,) * self.chunk_size
        self.advance_clock = None

    def reset(self):
        self.resets += 1

    def predict_chunk(self, observation):
        assert len(observation.joint_positions) == 9
        self.calls += 1
        if self.advance_clock:
            self.advance_clock()
        return self.chunk


def observation(index=0):
    sample = frame(index)
    return PolicyObservation(
        SCOPE,
        "customer-line",
        "f" * 64,
        "episode",
        sample.captured_at_utc,
        sample.monotonic_ns,
        sample.physics_step,
        sample.joint_positions,
        sample.images,
    )


@pytest.fixture
def controlled():
    clock, policy = Clock(), Policy()
    context = ControlContext(
        SCOPE,
        "customer-line",
        "f" * 64,
        "episode",
        "epoch",
        "command",
        "accepted",
        approved=True,
        active=True,
        deadline_monotonic_ns=10_000_000_000,
    )
    adapter = GuardedPolicyAdapter(policy, limits=SafetyLimits(max_chunk_steps=2), clock_ns=clock)
    adapter.reset(context)
    return adapter, context, policy, clock


def test_chunk_execution_keeps_exact_nine_joint_control_cadence(controlled):
    adapter, context, policy, clock = controlled
    for index in range(3):
        clock.now = observation(index).monotonic_ns
        command = adapter.step(observation(index), context)
        assert command.targets == JOINTS
        assert command.physics_step == index * 6
        assert command.hold_steps == 6
        assert command.expires_at_monotonic_ns == clock.now + 100_000_000
        assert command.model_sha256 == policy.model_sha256
    assert policy.calls == 2


@pytest.mark.parametrize(
    "change",
    [
        {"approved": False},
        {"active": False},
        {"environment_id": "other"},
        {"revision": "e" * 64},
        {"episode_id": "other"},
        {"epoch": "other"},
        {"command_id": "other"},
        {"destination_id": "other"},
        {"deadline_monotonic_ns": 1},
        {"scope": replace(SCOPE, owner_id="c" * 64)},
    ],
)
def test_guard_changes_invalidate_queued_actions(controlled, change):
    adapter, context, _, _ = controlled
    with pytest.raises(ContractError):
        adapter.step(observation(), replace(context, **change))
    with pytest.raises(ContractError, match="approved reset"):
        adapter.step(observation(), context)


@pytest.mark.parametrize(
    "change",
    [
        {"scope": replace(SCOPE, owner_id="d" * 64)},
        {"environment_id": "other"},
        {"episode_id": "other"},
        {"revision": "0" * 64},
        {"monotonic_ns": 0},
        {"physics_step": -1},
        {"joint_positions": (0.0,) * 7},
        {"images": {}},
    ],
)
def test_observation_identity_and_dimensions_fail_closed(controlled, change):
    adapter, context, _, _ = controlled
    with pytest.raises(ContractError):
        adapter.step(replace(observation(), **change), context)


@pytest.mark.parametrize("change", ["nan", "shape", "bounds", "slew", "future-chunk"])
def test_invalid_predictions_are_rejected_not_clipped(controlled, change):
    adapter, context, policy, _ = controlled
    actions = list(policy.chunk)
    if change == "nan":
        actions[0] = (float("nan"), *JOINTS[1:])
    elif change == "shape":
        actions[0] = JOINTS[:7]
    elif change == "bounds":
        actions[0] = (*JOINTS[:7], 0.2, 0.2)
    elif change == "slew":
        actions[0] = (0.5, *JOINTS[1:])
    else:
        actions[2] = (5.0, *JOINTS[1:])
    policy.chunk = actions
    with pytest.raises(ContractError):
        adapter.step(observation(), context)
    assert adapter.faulted is True


def test_inference_latency_overrun_requires_reset(controlled):
    adapter, context, policy, clock = controlled
    policy.advance_clock = lambda: setattr(clock, "now", clock.now + 90_000_000)
    with pytest.raises(ContractError, match="latency"):
        adapter.step(observation(), context)
    assert adapter.faulted is True


def test_command_expiring_during_inference_returns_no_action(controlled):
    adapter, context, policy, clock = controlled
    context = replace(context, deadline_monotonic_ns=clock.now + 10_000_000)
    adapter.reset(context)
    policy.advance_clock = lambda: setattr(clock, "now", clock.now + 20_000_000)
    with pytest.raises(ContractError, match="deadline"):
        adapter.step(observation(), context)


def test_stale_observation_and_queue_cannot_be_replayed(controlled):
    adapter, context, _, clock = controlled
    adapter.step(observation(), context)
    clock.now = 1_300_000_000
    with pytest.raises(ContractError, match="expired"):
        adapter.step(observation(1), context)


def test_stop_and_new_episode_flush_chunk(controlled):
    adapter, context, policy, clock = controlled
    adapter.step(observation(), context)
    adapter.stop()
    with pytest.raises(ContractError, match="approved reset"):
        adapter.step(observation(), context)
    next_context = replace(context, episode_id="next")
    clock.now = 1_000_000_000
    adapter.reset(next_context)
    adapter.step(replace(observation(), episode_id="next"), next_context)
    assert policy.calls == 2 and policy.resets == 3


def test_skipped_tick_rejects_chunk_continuation(controlled):
    adapter, context, _, clock = controlled
    adapter.step(observation(), context)
    clock.now = observation(2).monotonic_ns
    with pytest.raises(ContractError, match="control tick"):
        adapter.step(observation(2), context)


def test_repeated_camera_frame_is_rejected(controlled):
    adapter, context, _, clock = controlled
    adapter.step(observation(), context)
    clock.now = observation(1).monotonic_ns
    new = observation(1)
    image = replace(new.images["inspection"], rendering_frame=0)
    with pytest.raises(ContractError, match="Repeated"):
        adapter.step(replace(new, images={**new.images, "inspection": image}), context)


def test_unapproved_chunk_horizon_and_limits_rejected():
    with pytest.raises(ContractError, match="chunk horizon"):
        GuardedPolicyAdapter(Policy())
    with pytest.raises(ContractError, match="ceiling"):
        GuardedPolicyAdapter(
            Policy(), limits=SafetyLimits(max_chunk_steps=2, max_joint_velocity=(3.0,) * 9)
        )
    with pytest.raises(ContractError, match="one control interval"):
        GuardedPolicyAdapter(
            Policy(), limits=SafetyLimits(max_chunk_steps=2, max_inference_latency_ms=100)
        )
