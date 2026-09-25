"""Measured-state waypoint progression and bounded commands for the reference arm."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, dist, isfinite, sin, sqrt

Point = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def arm_gravity_efforts(gravity: tuple[float, ...]) -> tuple[float, ...]:
    limits = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    if len(gravity) != 9 or not all(isfinite(value) for value in gravity):
        raise ValueError("Finite gravity efforts for all nine Franka joints are required.")
    if any(abs(value) > limit for value, limit in zip(gravity[:7], limits, strict=True)):
        raise ValueError("Gravity compensation exceeds the reference arm's effort limits.")
    return (*gravity[:7], 0.0, 0.0)


def move_toward(
    start: tuple[float, ...], target: tuple[float, ...], maximum: float
) -> tuple[float, ...]:
    if len(start) != len(target) or maximum <= 0 or not isfinite(maximum):
        raise ValueError("A bounded displacement and matching vectors are required.")
    if not all(isfinite(value) for value in (*start, *target)):
        raise ValueError("Motion vectors must be finite.")
    distance = dist(start, target)
    fraction = min(1.0, maximum / distance) if distance else 1.0
    return tuple(a + (b - a) * fraction for a, b in zip(start, target, strict=True))


def rotate_toward(start: Quaternion, target: Quaternion, maximum_radians: float) -> Quaternion:
    def unit(values):
        norm = sqrt(sum(value * value for value in values))
        if not isfinite(norm) or norm < 1e-9:
            raise ValueError("A finite unit orientation is required.")
        return tuple(value / norm for value in values)

    left, right = unit(start), unit(target)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if dot < 0:
        right, dot = tuple(-value for value in right), -dot
    theta = acos(min(1.0, max(-1.0, dot)))
    if theta < 1e-6:
        return right
    weight = min(1.0, maximum_radians / (2 * theta))
    return tuple(
        (sin((1 - weight) * theta) * a + sin(weight * theta) * b) / sin(theta)
        for a, b in zip(left, right, strict=True)
    )


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: Point
    closed: bool
    settle_seconds: float = 0.12


class GripperRamp:
    def __init__(self, position: tuple[float, float], dt: float = 1 / 60) -> None:
        if len(position) != 2 or not 0 < dt <= 0.1:
            raise ValueError("The reference gripper requires two fingers and a bounded timestep.")
        self.position = position
        self.dt = dt

    def next_command(self, closed: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
        previous = self.position
        # Keep accumulating the drive target after contact so the fingers retain grip force.
        self.position = move_toward(
            previous, (0.0, 0.0) if closed else (0.04, 0.04), 0.025 * self.dt
        )
        velocity = tuple(
            (new - old) / self.dt for old, new in zip(previous, self.position, strict=True)
        )
        return self.position, velocity


class InspectionRoute:
    def __init__(
        self,
        start: Point,
        part: Point,
        inspection: Point,
        destination: Point,
        speed: float,
        dt: float = 1 / 60,
    ) -> None:
        self._initialize(start, speed, dt)
        lift = max(0.38, part[2] + 0.14, inspection[2] + 0.14, destination[2] + 0.14)
        self.points = (
            Waypoint("lift-clear", (start[0], start[1], max(lift, start[2])), False),
            Waypoint("approach-part", (part[0], part[1], lift), False),
            Waypoint("lower-to-part", part, False),
            Waypoint("grasp", part, True, 0.65),
            Waypoint("lift-part", (part[0], part[1], lift), True),
            Waypoint("to-inspection", (inspection[0], inspection[1], lift), True),
            Waypoint("inspect", inspection, True, 0.4),
            Waypoint("lift-inspected-part", (inspection[0], inspection[1], lift), True),
            Waypoint("to-destination", (destination[0], destination[1], lift), True),
            Waypoint("lower-to-destination", destination, True),
            Waypoint("release", destination, False, 0.6),
            Waypoint("retreat", (destination[0], destination[1], lift), False, 0.2),
        )

    def _initialize(self, start: Point, speed: float, dt: float) -> None:
        if not 0 < speed <= 0.25 or not 0 < dt <= 0.1:
            raise ValueError("Reference route limits are invalid.")
        self.index = 0
        self.target = start
        self.speed = speed
        self.dt = dt
        self.settled = 0.0
        self.elapsed = 0.0

    @property
    def done(self) -> bool:
        return self.index == len(self.points)

    @property
    def current(self) -> Waypoint:
        if self.done:
            raise ValueError("The reference route is complete.")
        return self.points[self.index]

    def next_target(
        self, measured: Point, finger_gap: float, part_position: Point
    ) -> tuple[Point, bool]:
        if not isfinite(finger_gap) or finger_gap < -1e-5:
            raise ValueError("Measured gripper opening must be finite and nonnegative.")
        waypoint = self.current
        self.elapsed += self.dt
        if self.elapsed > 120:
            raise RuntimeError("Reference route exceeded its simulation-time limit.")
        self.target = move_toward(self.target, waypoint.position, self.speed * self.dt)
        grip_ready = (waypoint.name != "grasp" or finger_gap <= 0.055) and (
            waypoint.name != "release" or finger_gap >= 0.07
        )
        arrived = dist(measured, waypoint.position) < 0.012
        if waypoint.name == "release":
            # Opening the fingers can displace the wrist through contact with the platform.
            arrived = dist(part_position, waypoint.position) < 0.04
        if dist(self.target, waypoint.position) < 1e-6 and arrived and grip_ready:
            self.settled += self.dt
            if self.settled >= waypoint.settle_seconds:
                self.index += 1
                self.settled = 0.0
        else:
            self.settled = 0.0
        return self.target, waypoint.closed


class PickPlaceRoute(InspectionRoute):
    """Task-specific tool/payload clearance; not whole-arm collision certification."""

    def __init__(
        self,
        start: Point,
        part: Point,
        destination: Point,
        stations: tuple[Point, ...],
        speed: float,
        dt: float = 0.1,
    ) -> None:
        self._initialize(start, speed, dt)
        if not stations or any(
            len(point) != 3 or not all(isfinite(value) for value in point)
            for point in (start, part, destination, *stations)
        ):
            raise ValueError("Finite frozen workcell positions are required for the task route.")
        # Match the unchanged 5 cm part and platform cuboids built by IsaacWorkcell.load.
        platform_top = max(point[2] - 0.045 + 0.04 / 2 for point in stations)
        half_part, clearance, required_lift = 0.025, 0.025, 0.05
        lift = max(
            platform_top + half_part + clearance,
            part[2] + required_lift + clearance,
            destination[2] + required_lift + clearance,
        )
        if not 0.08 <= lift <= 0.6:
            raise ValueError("The task clearance leaves the existing measured workspace.")
        initial_lift = (
            (Waypoint("lift-clear", (start[0], start[1], lift), False),) if start[2] < lift else ()
        )
        self.points = (
            *initial_lift,
            Waypoint("approach-part", (part[0], part[1], lift), False),
            Waypoint("lower-to-part", part, False),
            Waypoint("grasp", part, True, 0.65),
            Waypoint("lift-part", (part[0], part[1], lift), True),
            Waypoint("to-destination", (destination[0], destination[1], lift), True),
            Waypoint("lower-to-destination", destination, True),
            Waypoint("release", destination, False, 0.6),
            Waypoint("retreat", (destination[0], destination[1], lift), False, 0.2),
        )
