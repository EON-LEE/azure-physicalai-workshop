"""Bound reference-expert trajectories before submission; never repair learned policy outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import dist
from pathlib import Path

from learning.common import file_digest, finite, require, vector
from learning.contract import (
    DEFAULT_JOINT_VELOCITY_LIMITS,
    JOINT_LOWER,
    JOINT_NAMES,
    JOINT_UNITS,
    JOINT_UPPER,
    bounded_joints,
)
from simulation.control import check_measured_motion, validate_position_target
from simulation.motion import move_toward

REFERENCE_LIMIT_FRACTION = 0.9
CONTROL_DT = 0.1
MAX_FK_CANDIDATES = 8
MAX_REFERENCE_TRACE_INTERVALS = 300
PLANNING_STEP_LIMITS = tuple(
    value * CONTROL_DT * REFERENCE_LIMIT_FRACTION for value in DEFAULT_JOINT_VELOCITY_LIMITS
)
GRASP_FRAME_VERSION = "franka-default-inner-pad-centroid/v1"
GRASP_ASSET_SHA256 = "72956d2a7f0313d7effcff46c6b43ec616af8d2e1dd2055e4a152ad09767308e"
GRASP_SOURCE_CALIBRATION_SHA256 = "8990211d9cbd4c19c99b0c496cf64a624acefc2b55dcbf406b7bef25e03f3a68"
GRASP_CONTACT_PHASES = frozenset({"lower-to-part", "grasp", "lower-to-destination", "release"})
TCP_TO_PAD_CENTROID_M = (0.000002636002657491832, 0.0, 0.002904602840903575)
# Verified inner-triangle area centroid plus the authored finger origin, relative
# to right_gripper (hand z=0.1 m, opposite hand x/y axes), not the whole-finger box.
GRASP_ASSET_FILES = (
    (
        "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "33718d4409ffaee8140ec9ff2c17e8fc11de7b22bde992e9f0b85d0711a921e2",
    ),
    (
        "Isaac/Robots/FrankaRobotics/FrankaPanda/Props/panda_leftfinger.usd",
        "f44bb0ac9d905a243b43263d72eb344b0e88a42dc07469eafb47874ec75227d0",
    ),
    (
        "Isaac/Robots/FrankaRobotics/FrankaPanda/Props/panda_rightfinger.usd",
        "18d03137c2403524127f9a6e7db96780c1b663042563cba02cb14fa632e81850",
    ),
    (
        "Isaac/Robots/FrankaRobotics/FrankaPanda/configuration/franka_robot_schema.usd",
        "5ccff6da4abdb74deb90c4007df5edb3c5bcddc3b6eeed38682838eb55e28074",
    ),
)


def reference_tcp_target(contact_point, orientation) -> tuple[float, float, float]:
    point = vector(contact_point, 3, "desired reference contact point")
    offset = _oriented_pad_offset(orientation)
    return tuple(value - delta for value, delta in zip(point, offset, strict=True))


def reference_contact_point(tcp_point, orientation) -> tuple[float, float, float]:
    point = vector(tcp_point, 3, "measured reference TCP")
    offset = _oriented_pad_offset(orientation)
    return tuple(value + delta for value, delta in zip(point, offset, strict=True))


def _oriented_pad_offset(orientation) -> tuple[float, float, float]:
    w, x, y, z = vector(orientation, 4, "reference TCP orientation")
    require(abs(w * w + x * x + y * y + z * z - 1) <= 1e-6, "A unit TCP orientation is required")
    a, b, c = TCP_TO_PAD_CENTROID_M
    tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
    return (
        a + w * tx + y * tz - z * ty,
        b + w * ty + z * tx - x * tz,
        c + w * tz + x * ty - y * tx,
    )


def verify_grasp_calibration_asset(
    root: Path, asset: Path, archive_sha256: str, variants: dict[str, str]
) -> dict:
    return {
        **verify_paused_franka_asset(root, asset, archive_sha256, variants),
        "calibration_id": GRASP_FRAME_VERSION,
        "tcp_frame": "right_gripper",
        "contact_frame": "inner_pad_centroid",
        "tcp_to_pad_centroid_m": TCP_TO_PAD_CENTROID_M,
        "source_calibration_sha256": GRASP_SOURCE_CALIBRATION_SHA256,
    }


def verify_paused_franka_asset(
    root: Path, asset: Path, archive_sha256: str, variants: dict[str, str]
) -> dict:
    require(
        archive_sha256 == GRASP_ASSET_SHA256, "Reference grasp calibration asset archive mismatch"
    )
    require(
        variants == {"Mesh": "Performance", "Gripper": "Default"},
        "Reference grasp calibration requires the verified asset variants",
    )
    require(
        root.is_absolute()
        and root.is_dir()
        and not root.is_symlink()
        and asset.is_absolute()
        and not asset.is_symlink(),
        "Reference grasp calibration requires the verified local asset cache",
    )
    marker = root / ".complete"
    require(
        marker.is_file()
        and not marker.is_symlink()
        and marker.stat().st_size <= 128
        and marker.read_text(encoding="ascii") == GRASP_ASSET_SHA256,
        "Reference grasp calibration asset cache marker mismatch",
    )
    require(
        asset.resolve() == (root / GRASP_ASSET_FILES[0][0]).resolve(),
        "Reference grasp calibration root asset mismatch",
    )
    observed = {}
    for relative, checksum in GRASP_ASSET_FILES:
        path = root / relative
        require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve())
            and path.stat().st_size <= 1024**2,
            "Reference grasp calibration geometry is missing or unbounded",
        )
        require(
            file_digest(path) == checksum, "Reference grasp calibration geometry checksum mismatch"
        )
        observed[relative] = checksum
    return {
        "archive_sha256": GRASP_ASSET_SHA256,
        "variants": dict(variants),
        "verified_geometry_sha256": observed,
    }


@dataclass(frozen=True)
class ReferenceTargetPlan:
    targets: tuple[float, ...]
    path_fraction: float
    joint_fraction: float
    fk_candidate_evaluations: int
    predicted_tcp_step_m: float
    limiting_joint_indices: tuple[int, ...]


def _path_fraction_bounds(start, requested, measured, previous, limits):
    lower, upper = 0.0, 1.0
    limiting_joints: set[int] = set()
    for reference in (measured, previous):
        if reference is None:
            continue
        for index, (origin, goal, center, limit) in enumerate(
            zip(start, requested, reference, limits, strict=True)
        ):
            delta = goal - origin
            if delta == 0:
                if abs(origin - center) > limit + 1e-12:
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
    return lower, upper, tuple(sorted(limiting_joints))


def reference_gripper_targets(
    measured: tuple[float, ...], previous: tuple[float, ...] | None, *, closed: bool
) -> tuple[float, ...]:
    measured = bounded_joints(measured, "reference measured joints")
    previous = (
        bounded_joints(previous, "reference previous issued targets")
        if previous is not None
        else measured
    )
    if type(closed) is not bool:
        raise ValueError("A reference gripper command must explicitly open or close.")
    start = previous[7:]
    requested = (0.0, 0.0) if closed else (0.04, 0.04)
    feasible = []
    for actual, issued, goal, limit, joint_lower, joint_upper in zip(
        measured[7:],
        start,
        requested,
        PLANNING_STEP_LIMITS[7:],
        JOINT_LOWER[7:],
        JOINT_UPPER[7:],
        strict=True,
    ):
        lower = max(joint_lower, actual - limit, issued - limit)
        upper = min(joint_upper, actual + limit, issued + limit)
        require(lower <= upper, "No feasible reference gripper tracking/slew box intersection.")
        feasible.append(min(upper, max(lower, goal)))
    # Like GripperRamp, integrate from the issued target; keep a feasible
    # pressure hold without requiring a measured-to-goal ray to enter the box.
    targets = move_toward(start, tuple(feasible), 0.025 * CONTROL_DT)
    _path_fraction_bounds(targets, targets, measured[7:], start, PLANNING_STEP_LIMITS[7:])
    return targets


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
    lower, upper, limiting_joints = _path_fraction_bounds(
        start, requested, measured, previous, PLANNING_STEP_LIMITS
    )
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
                targets, fraction, upper, candidate, displacement, limiting_joints
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
