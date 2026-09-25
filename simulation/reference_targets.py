"""Bound reference-expert trajectories before submission; never repair learned policy outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import dist

from learning.common import finite, vector
from learning.contract import (
    DEFAULT_JOINT_VELOCITY_LIMITS,
    JOINT_NAMES,
    JOINT_UNITS,
    bounded_joints,
)
from simulation.control import check_measured_motion, validate_position_target
from simulation.motion import move_toward

REFERENCE_LIMIT_FRACTION = 0.9
CONTROL_DT = 0.1
MAX_FK_CANDIDATES = 8
MAX_REFERENCE_TRACE_INTERVALS = 300


@dataclass(frozen=True)
class ReferenceTargetPlan:
    targets: tuple[float, ...]
    path_fraction: float
    joint_fraction: float
    fk_candidate_evaluations: int
    predicted_tcp_step_m: float
    limiting_joint_indices: tuple[int, ...]


def plan_reference_targets(
    measured: tuple[float, ...],
    requested: tuple[float, ...],
    previous: tuple[float, ...] | None,
    *,
    forward_kinematics: Callable[[tuple[float, ...]], tuple[float, float, float]],
    max_cartesian_speed_m_s: float,
) -> ReferenceTargetPlan:
    measured = bounded_joints(measured, "reference measured joints")
    requested = bounded_joints(requested, "reference requested joints")
    previous = (
        bounded_joints(previous, "reference previous issued targets")
        if previous is not None
        else None
    )
    speed = finite(max_cartesian_speed_m_s, "reference Cartesian speed")
    if not 0 < speed <= 0.2:
        raise ValueError("Reference planning requires the unchanged bounded Cartesian speed.")

    # The gripper already has a pressure target; arm retiming must not release it.
    start = measured[:7] + requested[7:]
    lower, upper = 0.0, 1.0
    limiting_joints: set[int] = set()
    limits = tuple(
        value * CONTROL_DT * REFERENCE_LIMIT_FRACTION for value in DEFAULT_JOINT_VELOCITY_LIMITS
    )
    for reference in (measured, previous):
        if reference is None:
            continue
        for index, (origin, goal, center, limit) in enumerate(
            zip(start, requested, reference, limits, strict=True)
        ):
            delta = goal - origin
            if delta == 0:
                if abs(origin - center) > limit:
                    raise ValueError("No feasible reference tracking/slew path intersection.")
                continue
            ends = ((center - limit - origin) / delta, (center + limit - origin) / delta)
            if max(ends) < upper:
                limiting_joints = {index}
            elif max(ends) == upper and upper < 1:
                limiting_joints.add(index)
            lower, upper = max(lower, min(ends)), min(upper, max(ends))
    if lower > upper:
        raise ValueError("No feasible reference tracking/slew path intersection.")
    distance = dist(start, requested)
    if distance and upper <= 1e-12:
        raise ValueError("Requested reference motion has no feasible forward progress.")

    origin_tcp = vector(forward_kinematics(measured), 3, "reference measured FK")
    check_measured_motion(None, origin_tcp, measured, dt=CONTROL_DT, speed_limit=speed)
    fraction = upper
    for candidate in range(1, MAX_FK_CANDIDATES + 1):
        targets = move_toward(start, requested, distance * fraction) if distance else requested
        predicted = vector(forward_kinematics(targets), 3, "reference planned FK")
        displacement = dist(origin_tcp, predicted)
        if displacement <= speed * CONTROL_DT:
            targets = validate_position_target(measured, targets, previous)
            check_measured_motion(origin_tcp, predicted, targets, dt=CONTROL_DT, speed_limit=speed)
            return ReferenceTargetPlan(
                targets, fraction, upper, candidate, displacement, tuple(sorted(limiting_joints))
            )
        reduced = max(lower, fraction / 2)
        if reduced >= fraction:
            break
        fraction = reduced
    raise ValueError("No feasible reference Cartesian step within the bounded FK search.")


def reference_tracking_violations(
    measured: tuple[float, ...],
    requested: tuple[float, ...],
    previous: tuple[float, ...] | None,
) -> list[dict]:
    measured = vector(measured, 9, "reference measured joints")
    requested = vector(requested, 9, "reference requested joints")
    previous = vector(previous, 9, "reference previous targets") if previous is not None else None
    failures = []
    for constraint, reference in (("tracking", measured), ("slew", previous)):
        if reference is None:
            continue
        for index, (target, center, velocity) in enumerate(
            zip(requested, reference, DEFAULT_JOINT_VELOCITY_LIMITS, strict=True)
        ):
            limit = velocity * CONTROL_DT
            if abs(target - center) > limit + 1e-8:
                failures.append(
                    {
                        "constraint": constraint,
                        "joint_index": index,
                        "joint_name": JOINT_NAMES[index],
                        "unit": JOINT_UNITS[index],
                        "measured": measured[index],
                        "requested": target,
                        "previous_issued": previous[index] if previous is not None else None,
                        "delta": target - center,
                        "limit_per_interval": limit,
                    }
                )
    return failures
