"""CPU-only freeze and timing tests. These are not GPU or real-time admission evidence."""

from dataclasses import replace
from uuid import uuid4

import pytest

from learning.contract import AppliedControl
from simulation.paused_control import FrozenPhysicsState, PausedEpisode

JOINTS = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02)


def state():
    return FrozenPhysicsState(
        epoch=uuid4(),
        physics_step=100,
        world_time=100 / 60,
        joint_positions=JOINTS,
        joint_velocities=(0.0,) * 9,
        object_position=(0.35, 0.25, 0.2),
        object_orientation=(1.0, 0.0, 0.0, 0.0),
        object_linear_velocity=(0.0, 0.0, 0.0),
        object_angular_velocity=(0.0, 0.0, 0.0),
    )


def episode(*, wall_seconds=600, steps=1800):
    clock = [1_000_000_000]
    active = [True]
    initial = state()
    runtime = PausedEpisode(
        initial,
        wall_deadline_ns=clock[0] + wall_seconds * 1_000_000_000,
        max_simulation_steps=steps,
        authorized=lambda: active[0],
        clock_ns=lambda: clock[0],
    )
    return runtime, initial, clock, active


def ready(runtime, current, clock, *, observe_ms=300, predict_ms=500):
    freeze = runtime.begin_observation(current)
    clock[0] += observe_ms * 1_000_000
    runtime.observation_ready(freeze, current)
    clock[0] += predict_ms * 1_000_000
    runtime.policy_ready(freeze, current, JOINTS)
    return freeze


def tick(runtime, current, clock, *, wall_ms=100):
    runtime.before_tick(current)
    clock[0] += wall_ms * 1_000_000
    current = replace(
        current, physics_step=current.physics_step + 1, world_time=current.world_time + 1 / 60
    )
    done = runtime.after_tick(
        current,
        AppliedControl(
            current.physics_step,
            clock[0],
            JOINTS,
            (0.0,) * 9,
            (1.0,) * 7 + (0.0, 0.0),
        ),
    )
    return current, done


def test_camera_and_policy_wait_do_not_advance_simulation_or_remint_a_freeze():
    runtime, initial, clock, _ = episode()
    freeze = runtime.begin_observation(initial)
    for _ in range(6):
        clock[0] += 50_000_000
        runtime.waiting(initial)
    runtime.observation_ready(freeze, initial)
    for _ in range(10):
        clock[0] += 50_000_000
        runtime.waiting(initial)
    runtime.policy_ready(freeze, initial, JOINTS)
    assert runtime.metrics()["simulation_steps"] == 0
    assert runtime.metrics()["real_time_admission"] is False
    assert runtime.metrics()["execution_timing"] == "paused_simulation"


def test_one_action_executes_exactly_six_ticks_despite_longer_wall_duration():
    runtime, initial, clock, _ = episode()
    ready(runtime, initial, clock)
    current = initial
    for index in range(6):
        current, complete = tick(runtime, current, clock)
        assert complete is (index == 5)
    with pytest.raises(RuntimeError):
        runtime.before_tick(current)
    metrics = runtime.metrics()
    assert metrics["simulation_steps"] == 6
    assert metrics["simulation_elapsed_seconds"] == pytest.approx(0.1)
    assert metrics["wall_elapsed_ms"] == 1400
    interval = metrics["intervals"][0]
    assert interval["observation_wall_ms"] == 300
    assert interval["policy_wall_ms"] == 500
    assert interval["hold_wall_ms"] == 600
    assert interval["interval_wall_ms"] == 1400
    assert interval["simulated_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", uuid4()),
        ("physics_step", 101),
        ("world_time", 3.0),
        ("joint_positions", (0.001,) + JOINTS[1:]),
        ("joint_velocities", (0.001,) * 9),
        ("object_position", (0.36, 0.25, 0.2)),
        ("object_linear_velocity", (0.01, 0.0, 0.0)),
    ],
)
def test_any_actual_physics_state_change_while_pending_invalidates_the_action(field, value):
    runtime, initial, _, _ = episode()
    runtime.begin_observation(initial)
    with pytest.raises(RuntimeError, match="frozen"):
        runtime.waiting(replace(initial, **{field: value}))
    with pytest.raises(RuntimeError):
        runtime.before_tick(initial)


@pytest.mark.parametrize("phase", ["observation", "policy"])
def test_each_pending_operation_has_an_absolute_two_second_deadline(phase):
    runtime, initial, clock, _ = episode()
    freeze = runtime.begin_observation(initial)
    if phase == "policy":
        runtime.observation_ready(freeze, initial)
    clock[0] += 2_000_000_001
    with pytest.raises(RuntimeError, match="deadline"):
        runtime.waiting(initial)
    assert runtime.metrics()["simulation_steps"] == 0


def test_stage_transitions_cannot_renew_the_original_five_second_interval():
    runtime, initial, clock, _ = episode()
    ready(runtime, initial, clock, observe_ms=1900, predict_ms=1900)
    current = initial
    for _ in range(5):
        current, _ = tick(runtime, current, clock, wall_ms=200)
    runtime.before_tick(current)
    clock[0] += 201_000_000
    current = replace(
        current, physics_step=current.physics_step + 1, world_time=current.world_time + 1 / 60
    )
    with pytest.raises(RuntimeError, match="deadline"):
        runtime.after_tick(
            current,
            AppliedControl(current.physics_step, clock[0], JOINTS, (0.0,) * 9, (0.0,) * 9),
        )
    assert runtime.metrics()["simulation_steps"] == 6
    assert not runtime.metrics()["intervals"]


@pytest.mark.parametrize("stop", ["cancel", "authority", "wall"])
def test_late_policy_response_loses_to_cancel_authority_or_wall_deadline(stop):
    runtime, initial, clock, active = episode(wall_seconds=1)
    freeze = runtime.begin_observation(initial)
    runtime.observation_ready(freeze, initial)
    if stop == "cancel":
        runtime.cancel()
    elif stop == "authority":
        active[0] = False
    else:
        clock[0] += 1_000_000_000
    with pytest.raises(RuntimeError):
        runtime.policy_ready(freeze, initial, JOINTS)
    assert runtime.metrics()["simulation_steps"] == 0


def test_policy_output_for_another_freeze_is_rejected_without_applied_ticks():
    runtime, initial, _, _ = episode()
    freeze = runtime.begin_observation(initial)
    runtime.observation_ready(freeze, initial)
    with pytest.raises(RuntimeError, match="freeze"):
        runtime.policy_ready(uuid4(), initial, JOINTS)
    assert runtime.metrics()["simulation_steps"] == 0


def test_cancel_between_physics_ticks_preserves_partial_evidence_without_padding():
    runtime, initial, clock, _ = episode()
    ready(runtime, initial, clock)
    current, _ = tick(runtime, initial, clock)
    runtime.cancel()
    with pytest.raises(RuntimeError):
        runtime.before_tick(current)
    assert runtime.metrics()["simulation_steps"] == 1
    assert not runtime.metrics()["intervals"]


def test_simulation_budget_cannot_be_extended_by_remaining_wall_budget():
    runtime, initial, clock, _ = episode(steps=6)
    ready(runtime, initial, clock)
    current = initial
    for _ in range(6):
        current, _ = tick(runtime, current, clock)
    with pytest.raises(RuntimeError, match="simulation"):
        runtime.begin_observation(current)
    assert runtime.metrics()["wall_elapsed_ms"] < 600_000


def test_synchronous_sdk_overrun_is_a_failure_not_an_unreported_heartbeat_gap():
    runtime, initial, clock, _ = episode()
    ready(runtime, initial, clock)
    runtime.before_tick(initial)
    clock[0] += 2_001_000_000
    advanced = replace(initial, physics_step=101, world_time=initial.world_time + 1 / 60)
    with pytest.raises(RuntimeError, match="heartbeat"):
        runtime.after_tick(
            advanced,
            AppliedControl(101, clock[0], JOINTS, (0.0,) * 9, (0.0,) * 9),
        )
    assert runtime.metrics()["simulation_steps"] == 1


def test_frozen_snapshot_cannot_share_mutable_vectors_with_pending_callbacks():
    original = state()
    mutable = list(original.joint_positions)
    unsafe = replace(original, joint_positions=mutable)
    with pytest.raises(ValueError, match="immutable"):
        PausedEpisode(
            unsafe,
            wall_deadline_ns=601_000_000_000,
            max_simulation_steps=1800,
            authorized=lambda: True,
            clock_ns=lambda: 1_000_000_000,
        )


def test_parent_frozen_two_second_hold_cap_intersects_the_original_whole_interval():
    runtime, initial, clock, _ = episode()
    ready(runtime, initial, clock, observe_ms=100, predict_ms=100)
    clock[0] += 2_000_000_001
    with pytest.raises(RuntimeError, match="deadline"):
        runtime.before_tick(initial)
    assert runtime.metrics()["simulation_steps"] == 0


def test_delayed_driver_start_cannot_renew_or_hide_original_admission_wall_time():
    initial = state()
    runtime = PausedEpisode(
        initial,
        started_ns=1_000_000_000,
        wall_deadline_ns=601_000_000_000,
        max_simulation_steps=1800,
        authorized=lambda: True,
        clock_ns=lambda: 1_500_000_000,
    )
    assert runtime.metrics()["wall_elapsed_ms"] == 500
    assert runtime.started_ns == 1_000_000_000
