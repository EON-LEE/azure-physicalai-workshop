from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import JOINTS, SCOPE
from learning.common import ContractError
from learning.contract import DemonstrationSource
from tests.learning.test_paused_contract import context, observation, profile


class Policy:
    policy_type = "smolvla"
    execution_timing = "paused_simulation"
    real_time_admission = False
    scope = SCOPE
    model_sha256 = "c" * 64
    chunk_size, n_action_steps = 50, 1
    fps, physics_hz = 10, 60
    task = DemonstrationSource(
        kind="reference_controller",
        task_id="manufacturing-part-placement-v1",
        instruction="Move the part to the approved tray.",
        goal_id="rejected",
    )

    def __init__(self, clock):
        self.profile = profile()
        self.clock, self.calls, self.resets = clock, 0, 0
        self.actions = (JOINTS,) * 50
        self.delay_ns = 250_000_000
        self.entered = self.release = None

    def reset(self):
        self.resets += 1

    def predict_chunk(self, value, ctx):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(5)
        self.clock.now += self.delay_ns
        return self.actions


def adapter():
    from learning.paused.inference import PausedGuardedPolicyAdapter

    clock = SimpleNamespace(now=2_000_000_000)
    policy = Policy(clock)
    result = PausedGuardedPolicyAdapter(policy, profile=profile(), clock_ns=lambda: clock.now)
    result.reset(context())
    return result, policy, clock


def test_paused_adapter_returns_one_bounded_nine_joint_command_without_latency_division():
    value, policy, _ = adapter()
    command = value.step(observation(), context())
    assert command.targets == JOINTS and command.hold_steps == 6
    assert command.inference_latency_ms == 250.0
    assert command.expires_at_monotonic_ns == 4_250_000_000
    assert command.freeze_id == observation().freeze_id
    assert command.observation_sha256 == observation().sha256
    assert command.context_sha256 == context().sha256
    assert command.real_time_admission is False
    assert value.predict_calls == policy.calls == 1


def test_original_operation_budget_includes_worker_queue_delay():
    value, _, clock = adapter()
    clock.now += 500_000_000
    command = value.step(observation(), context())
    assert command.inference_latency_ms == 750.0


@pytest.mark.parametrize("change", ["late", "nan", "shape", "bounds", "tracking", "future-chunk"])
def test_invalid_or_expired_prediction_faults_without_clipping_or_fallback(change):
    value, policy, _ = adapter()
    if change == "late":
        policy.delay_ns = 2_000_000_000
    elif change == "shape":
        policy.actions = (JOINTS[:6],) * 50
    elif change == "nan":
        policy.actions = ((float("nan"), *JOINTS[1:]),) * 50
    elif change == "bounds":
        policy.actions = ((*JOINTS[:7], 0.2, 0.2),) * 50
    elif change == "tracking":
        policy.actions = ((0.051, *JOINTS[1:]),) * 50
    else:
        policy.actions = (JOINTS,) * 49 + ((9.0, *JOINTS[1:]),)
    with pytest.raises(ContractError):
        value.step(observation(), context())
    with pytest.raises(ContractError, match="reset|fault"):
        value.step(observation(), context())
    assert policy.calls == 1


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("wall_deadline_ns", 600_000_000_000),
        ("model_sha256", "0" * 64),
        ("command_id", "other"),
        ("destination_id", "unapproved-goal"),
        ("scope", replace(SCOPE, owner_id="b" * 64)),
    ],
)
def test_stable_episode_authority_cannot_change_after_reset(field, bad):
    value, policy, _ = adapter()
    with pytest.raises(ContractError):
        value.step(observation(), replace(context(), **{field: bad}))
    assert policy.calls == 0


def test_cancel_invalidates_inflight_response_without_blocking_or_resetting_the_model():
    value, policy, _ = adapter()
    policy.entered, policy.release = Event(), Event()
    failures = []

    def predict():
        try:
            value.step(observation(), context())
        except ContractError as exc:
            failures.append(str(exc))

    worker = Thread(target=predict)
    worker.start()
    assert policy.entered.wait(2)
    with pytest.raises(ContractError, match="in.flight"):
        value.reset(context())
    value.stop()
    assert policy.resets == 1
    policy.release.set()
    worker.join(2)
    assert not worker.is_alive() and failures
    assert "cancel" in failures[0]
    assert policy.calls == 1


def test_a_freeze_cannot_be_predicted_twice():
    value, policy, _ = adapter()
    value.step(observation(), context())
    with pytest.raises(ContractError, match="freeze|cadence"):
        value.step(observation(), context())
    assert policy.calls == 1


def test_real_time_policy_cannot_enter_paused_adapter():
    from learning.paused.inference import PausedGuardedPolicyAdapter

    policy = Policy(SimpleNamespace(now=2_000_000_000))
    policy.execution_timing = "real_time"
    with pytest.raises(ContractError, match="paused"):
        PausedGuardedPolicyAdapter(policy, profile=profile())


def test_paused_policy_cannot_be_admitted_by_old_real_time_adapter():
    from learning.inference import GuardedPolicyAdapter

    with pytest.raises(ContractError, match="real.time"):
        GuardedPolicyAdapter(Policy(SimpleNamespace(now=2_000_000_000)))


def test_reset_cannot_change_the_model_goal_even_with_new_episode_authority():
    from learning.paused.inference import PausedGuardedPolicyAdapter

    clock = SimpleNamespace(now=2_000_000_000)
    policy = Policy(clock)
    value = PausedGuardedPolicyAdapter(policy, profile=profile(), clock_ns=lambda: clock.now)
    with pytest.raises(ContractError, match="goal"):
        value.reset(replace(context(), destination_id="another-goal"))
    assert policy.resets == policy.calls == 0
