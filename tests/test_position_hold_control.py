"""CPU checks of actual interval evidence and watchdog decisions, not GPU proof."""

from dataclasses import replace

import pytest
from test_capture_lifecycle import sample

from learning.contract import AppliedControl, ControlProfile
from simulation.control import (
    HoldCapture,
    TaskWatchdog,
    check_measured_motion,
    validate_position_target,
)

JOINTS = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02)


class Recorder:
    def __init__(self):
        self.frames = []
        self.sealed = False
        self.invalid = None

    def append(self, frame):
        self.frames.append(frame)

    def seal(self):
        self.sealed = True

    def invalidate(self, message):
        self.invalid = message


def apply_ticks(capture, observation, count=6):
    for tick in range(1, count + 1):
        capture.applied(
            AppliedControl(
                observation.physics_step + tick,
                observation.monotonic_ns + tick * 16_666_667,
                observation.commanded_joint_targets,
                (0.0,) * 9,
                (1.0,) * 7 + (0.0, 0.0),
            )
        )


def test_only_actual_six_tick_intervals_can_be_sealed_without_terminal_padding():
    recorder = Recorder()
    capture = HoldCapture(recorder, ControlProfile("a" * 64))
    first, second = sample(0), sample(6)
    capture.begin(first)
    apply_ticks(capture, first)
    assert not recorder.frames
    capture.begin(second)
    assert len(recorder.frames) == 1
    apply_ticks(capture, second)
    capture.finish(truncated=False)
    assert recorder.sealed
    assert len(recorder.frames) == 2
    assert [f.physics_step for f in recorder.frames] == [0, 6]
    assert [c.physics_step for c in recorder.frames[-1].applied_controls] == list(range(7, 13))
    assert recorder.frames[-1].terminated
    assert not recorder.frames[-1].truncated
    assert all(
        c.commanded_joint_velocities == (0.0,) * 9
        for f in recorder.frames
        for c in f.applied_controls
    )


@pytest.mark.parametrize("ticks", [0, 1, 5])
def test_partial_interval_cancellation_invalidates_without_fabricating_ticks(ticks):
    recorder = Recorder()
    capture = HoldCapture(recorder, ControlProfile("a" * 64))
    frame = sample(0)
    capture.begin(frame)
    apply_ticks(capture, frame, count=ticks)
    with pytest.raises(RuntimeError, match="partial"):
        capture.finish(truncated=True)
    assert recorder.invalid
    assert not recorder.sealed
    assert not recorder.frames


@pytest.mark.parametrize("bad", ["step", "velocity", "target", "gravity", "timestamp"])
def test_actual_actuation_evidence_is_checked_before_recording(bad):
    recorder = Recorder()
    capture = HoldCapture(recorder, ControlProfile("a" * 64))
    frame = sample(0)
    capture.begin(frame)
    control = AppliedControl(1, frame.monotonic_ns + 1, JOINTS, (0.0,) * 9, (1.0,) * 7 + (0.0, 0.0))
    changes = {
        "step": {"physics_step": 2},
        "velocity": {"commanded_joint_velocities": (0.01,) * 9},
        "target": {"commanded_joint_targets": (0.01,) + JOINTS[1:]},
        "gravity": {"gravity_efforts": (1.0,) * 9},
        "timestamp": {"monotonic_ns": frame.monotonic_ns},
    }
    with pytest.raises(ValueError):
        capture.applied(replace(control, **changes[bad]))
    assert not recorder.frames


def test_position_hold_rejects_unsafe_predictions_instead_of_clipping():
    assert validate_position_target(JOINTS, JOINTS) == JOINTS
    with pytest.raises(ValueError, match="tracking|slew"):
        validate_position_target(JOINTS, (0.051,) + JOINTS[1:])
    with pytest.raises(ValueError, match="tracking|slew"):
        validate_position_target(JOINTS, JOINTS[:7] + (0.025, 0.02))
    with pytest.raises(ValueError):
        validate_position_target(JOINTS, (float("nan"),) + JOINTS[1:])


def test_measured_cartesian_speed_limit_is_shared_and_never_widened():
    assert (
        check_measured_motion((0.3, 0, 0.4), (0.3, 0, 0.402), JOINTS, dt=1 / 60, speed_limit=0.2)
        < 0.2
    )
    with pytest.raises(RuntimeError, match="speed"):
        check_measured_motion((0.3, 0, 0.4), (0.3, 0, 0.404), JOINTS, dt=1 / 60, speed_limit=0.25)


def test_terminal_success_requires_measured_grasp_release_goal_and_settling():
    initial, goal = (0.35, 0.25, 0.2), (0.22, -0.38, 0.2)
    watchdog = TaskWatchdog(initial, goal)
    opened = JOINTS[:7] + (0.04, 0.04)
    for _ in range(60):
        assert not watchdog.observe(tcp=(0.22, -0.38, 0.35), part=goal, joints=opened)
    lifted = (0.35, 0.25, 0.27)
    for _ in range(6):
        assert not watchdog.observe(tcp=lifted, part=lifted, joints=JOINTS)
    assert watchdog.grasp_verified
    for _ in range(60):
        assert not watchdog.observe(tcp=goal, part=goal, joints=JOINTS)
    for _ in range(30):
        completed = watchdog.observe(tcp=(0.22, -0.38, 0.35), part=goal, joints=opened)
    assert completed


def test_goal_proximity_without_low_measured_part_speed_is_not_success():
    watchdog = TaskWatchdog((0.35, 0.25, 0.2), (0.22, -0.38, 0.2))
    lifted = (0.35, 0.25, 0.27)
    for _ in range(6):
        watchdog.observe(tcp=lifted, part=lifted, joints=JOINTS)
    opened = JOINTS[:7] + (0.04, 0.04)
    for tick in range(60):
        part = (0.22 + (0.02 if tick % 2 else -0.02), -0.38, 0.2)
        assert not watchdog.observe(tcp=(0.22, -0.38, 0.35), part=part, joints=opened)
