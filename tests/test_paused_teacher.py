"""CPU expert progression tests, not a physical demonstration or model-quality claim."""

from dataclasses import replace

import pytest
from test_paused_control import state
from test_teaching_runtime import teaching as teaching

from simulation.paused_teacher import PausedReferenceTeacher


def teacher(teaching):
    core, _, _ = teaching
    clock = [1_000_000_000]
    initial = state()
    runtime = PausedReferenceTeacher(
        core.spec,
        initial=initial,
        tcp=(0.35, 0.25, 0.38),
        target_station_id="rejected",
        wall_deadline_ns=601_000_000_000,
        clock_ns=lambda: clock[0],
    )
    return runtime, initial, clock


def test_repeated_wall_waits_cannot_advance_expert_dwell_or_targets(teaching):
    runtime, initial, clock = teacher(teaching)
    first = runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08)
    for _ in range(100):
        clock[0] += 10_000_000
        assert runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08) == first
    assert runtime.route.elapsed == 0
    assert runtime.route.index == 0
    assert runtime.route.settled == 0


def test_grasp_dwell_advances_only_after_six_actual_ticks_with_measured_contact(teaching):
    runtime, initial, clock = teacher(teaching)
    runtime.route.index = 3
    runtime.route.target = runtime.route.current.position
    runtime.target(initial, tcp=runtime.route.current.position, finger_gap=0.05)
    for _ in range(10):
        clock[0] += 100_000_000
        runtime.target(initial, tcp=runtime.route.current.position, finger_gap=0.05)
    assert runtime.route.settled == 0
    completed = replace(
        initial, physics_step=initial.physics_step + 6, world_time=initial.world_time + 0.1
    )
    runtime.target(completed, tcp=runtime.route.current.position, finger_gap=0.05)
    assert runtime.route.settled == pytest.approx(0.1)
    assert runtime.route.index == 3


def test_skipped_repeated_or_unbound_physics_cannot_become_elapsed_expert_time(teaching):
    runtime, initial, _ = teacher(teaching)
    runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08)
    skipped = replace(
        initial, physics_step=initial.physics_step + 12, world_time=initial.world_time + 0.2
    )
    with pytest.raises(RuntimeError, match="six"):
        runtime.target(skipped, tcp=(0.35, 0.25, 0.38), finger_gap=0.08)


def test_expert_wall_deadline_remains_hard_even_when_no_simulation_step_occurs(teaching):
    runtime, initial, clock = teacher(teaching)
    runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08)
    clock[0] = runtime.wall_deadline_ns
    with pytest.raises(RuntimeError, match="wall"):
        runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08)
    assert runtime.route.elapsed == 0


def test_completed_expert_never_issues_another_target_while_the_world_is_frozen(teaching):
    runtime, initial, _ = teacher(teaching)
    runtime.route.index = len(runtime.route.points)
    assert runtime.target(initial, tcp=(0.35, 0.25, 0.38), finger_gap=0.08) is None
