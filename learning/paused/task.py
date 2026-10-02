"""Evaluator-only rescoring of the existing runtime TaskWatchdog; no contact sensor claim."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, hypot

from learning.common import integer, require, sha256, token, utc, vector
from learning.contract import bounded_joints
from learning.paused.contract import PausedControlProfile

PREDICATE_VERSION = "physicalai.measured-grasp-transport/v1"
PREDICATE_SOURCE = {
    "commit": "49ec269fe9d2503031e696aaaafcbf63f8631ec6",
    "path": "simulation/control.py",
    "git_blob": "5e88bf92fbc28b5097f061d83b05f5bc6d5a1a27",
    "sha256": "8e8666f1f9a0113f253047124a1b18d65362b6671a3a859984db35e8a65af300",
    "symbol": "TaskWatchdog",
}


@dataclass(frozen=True)
class TaskState:
    captured_at_utc: str
    monotonic_ns: int
    physics_step: int
    object_position_m: tuple[float, float, float]
    object_linear_velocity_m_s: tuple[float, float, float]
    tcp_position_m: tuple[float, float, float]
    joint_positions: tuple[float, ...]
    command_id: str
    epoch: str
    policy_predict_calls: int
    applied_action_count: int
    reference_route_calls: int
    applied_model_sha256: str | None

    def validate(self) -> None:
        utc(self.captured_at_utc)
        integer(self.monotonic_ns, "actual task measurement time", 1)
        integer(self.physics_step, "actual physics step")
        for name in ("object_position_m", "object_linear_velocity_m_s", "tcp_position_m"):
            vector(getattr(self, name), 3, name)
        bounded_joints(self.joint_positions, "actual task joints")
        for name in ("command_id", "epoch"):
            token(getattr(self, name), name)
        for name in ("policy_predict_calls", "applied_action_count", "reference_route_calls"):
            integer(getattr(self, name), name)
        if self.applied_model_sha256 is not None:
            sha256(self.applied_model_sha256, "actual actuator model")


class MeasuredTaskPredicate:
    """Independent numeric reference; golden checks compare the pinned runtime implementation."""

    def __init__(self, initial_part, goal) -> None:
        self.initial_part = vector(initial_part, 3, "initial part position")
        self.goal = vector(goal, 3, "task goal")
        self.previous_part = self.initial_part
        self.grasp_verified = False
        self.grasp_seconds = 0.0
        self.settled_seconds = 0.0
        self.goal_error_m = dist(self.initial_part, self.goal)

    def observe(self, *, tcp, part, joints) -> bool:
        point = vector(part, 3, "measured object")
        effector = vector(tcp, 3, "measured TCP")
        measured = bounded_joints(joints, "actual joints")
        gap = sum(measured[7:])
        grasp = (
            point[2] >= self.initial_part[2] + 0.05
            and dist(effector, point) <= 0.09
            and 0.005 <= gap <= 0.055
        )
        self.grasp_seconds = self.grasp_seconds + 1 / 60 if grasp else 0.0
        self.grasp_verified = self.grasp_verified or self.grasp_seconds >= 0.05
        speed = dist(point, self.previous_part) / (1 / 60)
        self.previous_part = point
        self.goal_error_m = dist(point, self.goal)
        placed = (
            self.grasp_verified
            and all(abs(a - b) <= 0.04 for a, b in zip(point, self.goal, strict=True))
            and gap >= 0.07
            and dist(effector, point) >= 0.05
            and speed <= 0.02
        )
        self.settled_seconds = self.settled_seconds + 1 / 60 if placed else 0.0
        return self.settled_seconds >= 0.3


def evaluate_task_states(
    states: list[TaskState],
    *,
    initial,
    goal,
    profile: PausedControlProfile | None = None,
) -> dict:
    max_steps = 1800
    if profile is not None:
        profile.validate()
        max_steps = profile.max_simulation_steps
    require(
        isinstance(states, list) and 2 <= len(states) <= max_steps + 1,
        "Missing actual per-tick task trace",
    )
    predicate = MeasuredTaskPredicate(initial, goal)
    first = states[0]
    first.validate()
    require(
        all(abs(a - b) <= 0.001 for a, b in zip(first.object_position_m, initial, strict=True)),
        "Task trace does not start at the frozen case pose",
    )
    previous, maximum_speed, settled, violations = first, 0.0, False, []
    for state in states[1:]:
        state.validate()
        require(
            state.physics_step == previous.physics_step + 1
            and state.monotonic_ns > previous.monotonic_ns
            and utc(state.captured_at_utc) > utc(previous.captured_at_utc)
            and state.command_id == first.command_id
            and state.epoch == first.epoch
            and state.applied_action_count == previous.applied_action_count + 1
            and state.policy_predict_calls >= previous.policy_predict_calls
            and state.reference_route_calls >= previous.reference_route_calls,
            "Skipped task state, reminted clock or changed actuator authority",
        )
        speed = dist(state.tcp_position_m, previous.tcp_position_m) * 60
        maximum_speed = max(maximum_speed, speed)
        if speed > 0.2 + 1e-9:
            violations.append(f"tcp_speed_at_physics_step_{state.physics_step}")
        if not (
            0.15 <= hypot(*state.tcp_position_m[:2]) <= 0.75
            and 0.08 <= state.tcp_position_m[2] <= 0.6
        ):
            violations.append(f"tcp_workspace_at_physics_step_{state.physics_step}")
        settled = predicate.observe(
            tcp=state.tcp_position_m, part=state.object_position_m, joints=state.joint_positions
        )
        previous = state
    return {
        "predicate_version": PREDICATE_VERSION,
        "predicate_source": PREDICATE_SOURCE,
        "grasp_evidence_kind": "measured_lift_proximity_finger_gap_no_contact_sensor",
        "grasp_verified": predicate.grasp_verified,
        "settled": settled,
        "settled_simulation_seconds": predicate.settled_seconds,
        "final_goal_error_m": predicate.goal_error_m,
        "maximum_tcp_speed_m_s": maximum_speed,
        "safety_violations": violations,
    }
