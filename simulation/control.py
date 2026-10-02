"""Physics-independent checks for the reviewed Isaac position-hold servo."""

from __future__ import annotations

from dataclasses import replace
from math import dist, hypot, isfinite

from learning.common import require, vector
from learning.contract import (
    AppliedControl,
    ControlProfile,
    FrameSample,
    bounded_joints,
    validate_joint_tracking,
)
from simulation.motion import arm_gravity_efforts


class HoldCapture:
    def __init__(self, recorder, profile: ControlProfile) -> None:
        profile.validate()
        self.recorder, self.profile = recorder, profile
        self.current: FrameSample | None = None
        self.completed: FrameSample | None = None
        self.controls: list[AppliedControl] = []
        self.finished = False

    def begin(self, sample: FrameSample) -> None:
        require(not self.finished and self.current is None, "A held interval is already active")
        require(not sample.terminated and not sample.truncated, "Premature terminal sample")
        if self.completed is not None:
            require(
                sample.physics_step - self.completed.physics_step == self.profile.hold_steps
                and sample.monotonic_ns >= self.completed.applied_controls[-1].monotonic_ns,
                "Observation skipped an actual held interval",
            )
            self.recorder.append(self.completed)
            self.completed = None
        self.current = sample
        self.controls = []

    def applied(self, control: AppliedControl) -> None:
        require(self.current is not None and not self.finished, "No observed interval to apply")
        previous_ns = self.controls[-1].monotonic_ns if self.controls else self.current.monotonic_ns
        require(
            control.physics_step == self.current.physics_step + len(self.controls) + 1
            and control.monotonic_ns > previous_ns,
            "Missing, repeated or nonmonotonic actual actuation tick",
        )
        require(
            bounded_joints(control.commanded_joint_targets, "applied targets")
            == self.current.commanded_joint_targets,
            "The applied target changed inside a held interval",
        )
        require(
            vector(control.commanded_joint_velocities, 9, "applied velocities") == (0.0,) * 9,
            "The position-hold servo requires zero velocity targets",
        )
        require(
            arm_gravity_efforts(control.gravity_efforts) == control.gravity_efforts,
            "Gravity efforts must be actual arm-only compensation",
        )
        self.controls.append(control)
        if len(self.controls) == self.profile.hold_steps:
            self.completed = replace(self.current, applied_controls=tuple(self.controls))
            self.current = None

    def finish(self, *, truncated: bool) -> None:
        if self.current is not None:
            self.recorder.invalidate("Capture interrupted a partial six-tick hold interval.")
            raise RuntimeError("Cannot publish a partial six-tick hold interval.")
        if self.completed is None or self.finished:
            self.recorder.invalidate("Capture has no completed interval to seal.")
            raise RuntimeError("Capture has no actual terminal interval.")
        self.recorder.append(replace(self.completed, terminated=not truncated, truncated=truncated))
        self.recorder.seal()
        self.finished = True


def validate_position_target(measured, targets, previous=None) -> tuple[float, ...]:
    selected = bounded_joints(targets, "position targets")
    validate_joint_tracking(selected, measured, previous, fps=10)
    return selected


def check_measured_motion(previous, current, joints, *, dt: float, speed_limit: float) -> float:
    bounded_joints(joints, "measured joints")
    point = vector(current, 3, "measured end effector")
    require(isfinite(dt) and 0 < dt <= 0.1, "Invalid measured control interval")
    require(isfinite(speed_limit) and speed_limit > 0, "Invalid Cartesian speed limit")
    if not (0.15 <= hypot(*point[:2]) <= 0.75 and 0.08 <= point[2] <= 0.6):
        raise RuntimeError("Measured end effector left the approved cell workspace.")
    speed = (
        0.0 if previous is None else dist(vector(previous, 3, "previous end effector"), point) / dt
    )
    if speed > min(0.2, speed_limit) + 1e-9:
        raise RuntimeError(f"Cartesian-speed watchdog exceeded: {speed:.3f} m/s.")
    return speed


class TaskWatchdog:
    def __init__(self, initial_part, goal) -> None:
        self.initial_part = vector(initial_part, 3, "initial part position")
        self.goal = vector(goal, 3, "measured task goal")
        self.previous_part = self.initial_part
        self.grasp_verified = False
        self.grasp_seconds = 0.0
        self.settled_seconds = 0.0
        self.goal_error_m = dist(self.initial_part, self.goal)

    def observe(self, *, tcp, part, joints, dt=1 / 60) -> bool:
        point = vector(part, 3, "measured part position")
        effector = vector(tcp, 3, "measured end effector")
        measured = bounded_joints(joints, "measured joints")
        require(isfinite(dt) and 0 < dt <= 0.1, "Invalid task measurement interval")
        finger_gap = sum(measured[7:])
        grasp = (
            point[2] >= self.initial_part[2] + 0.05
            and dist(effector, point) <= 0.09
            and 0.005 <= finger_gap <= 0.055
        )
        self.grasp_seconds = self.grasp_seconds + dt if grasp else 0.0
        self.grasp_verified = self.grasp_verified or self.grasp_seconds >= 0.05
        speed = dist(point, self.previous_part) / dt
        self.previous_part = point
        self.goal_error_m = dist(point, self.goal)
        placed = (
            self.grasp_verified
            and all(abs(a - b) <= 0.04 for a, b in zip(point, self.goal, strict=True))
            and finger_gap >= 0.07
            and dist(effector, point) >= 0.05
            and speed <= 0.02
        )
        self.settled_seconds = self.settled_seconds + dt if placed else 0.0
        return self.settled_seconds >= 0.3
