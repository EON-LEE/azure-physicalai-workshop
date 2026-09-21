from math import dist, isclose

import pytest

from simulation.motion import (
    GripperRamp,
    InspectionRoute,
    arm_gravity_efforts,
    move_toward,
    rotate_toward,
)


def test_each_cartesian_goal_displacement_respects_the_declared_speed():
    route = InspectionRoute((0.3, 0, 0.5), (-0.5, 0, 0.2), (0, 0.4, 0.2), (0.5, 0.4, 0.2), 0.13)
    current = route.target
    steps = 0
    while not route.done:
        previous = route.target
        current, _ = route.next_target(current, 0.04 if route.current.closed else 0.08, current)
        assert dist(previous, current) <= route.speed * route.dt + 1e-10
        steps += 1
    assert steps > 100
    assert route.points[-1].position == current


def test_phase_does_not_advance_when_the_measured_arm_has_not_reached_its_goal():
    route = InspectionRoute((0, 0, 0.5), (0.4, 0, 0.2), (0.3, 0.2, 0.2), (0.3, -0.2, 0.2), 0.13)
    for _ in range(180):
        route.next_target((9, 9, 9), 0.08, (0.4, 0, 0.2))
    assert route.index == 0


def test_quaternion_steps_are_normalized_and_choose_the_short_arc():
    orientation = (1.0, 0.0, 0.0, 0.0)
    for _ in range(600):
        orientation = rotate_toward(orientation, (0, 0, 1, 0), 0.01)
        assert isclose(sum(x * x for x in orientation), 1, abs_tol=1e-8)
    assert isclose(abs(orientation[2]), 1, abs_tol=1e-6)
    assert rotate_toward(orientation, tuple(-x for x in orientation), 0.01) == orientation


def test_route_waits_for_measured_grip_contact_before_lifting():
    route = InspectionRoute((0.4, 0, 0.38), (0.4, 0, 0.2), (0.5, 0.1, 0.2), (0.4, -0.2, 0.2), 0.13)
    route.index = 3
    route.target = route.current.position
    for _ in range(100):
        route.next_target(route.target, 0.08, route.target)
    assert route.index == 3
    for _ in range(40):
        route.next_target(route.target, 0.05, route.target)
    assert route.index == 4


def test_gripper_retains_a_closed_drive_target_after_contact_and_releases_gradually():
    gripper = GripperRamp((0.04, 0.04))
    previous = gripper.position
    for closed in [True] * 180 + [False] * 180:
        position, velocity = gripper.next_command(closed)
        assert dist(position, previous) <= 0.025 * gripper.dt + 1e-10
        assert dist(velocity, (0, 0)) <= 0.025 + 1e-10
        previous = position
        if closed:
            last_closed = position
    assert last_closed == (0, 0)
    assert gripper.position == (0.04, 0.04)


def test_gravity_feed_forward_is_arm_only_and_respects_physical_effort_limits():
    gravity = (0, -7, 0, 19, 0, 1.2, 0, 0.005, -0.005)
    assert arm_gravity_efforts(gravity) == (*gravity[:7], 0, 0)
    for invalid in ((0,) * 8, (float("nan"),) * 9, (88,) + (0,) * 8):
        with pytest.raises(ValueError):
            arm_gravity_efforts(invalid)


def test_release_requires_open_fingers_and_the_measured_part_at_destination():
    route = InspectionRoute((0.4, 0, 0.38), (0.4, 0, 0.2), (0.5, 0.1, 0.2), (0.4, -0.2, 0.2), 0.13)
    route.index = 10
    route.target = route.current.position
    wrist = (0.38, -0.22, 0.2)
    for _ in range(60):
        route.next_target(wrist, 0.08, (0.4, 0, 0.2))
    assert route.index == 10
    for _ in range(60):
        route.next_target(wrist, 0.05, (0.4, -0.2, 0.2))
    assert route.index == 10
    for _ in range(36):
        route.next_target(wrist, 0.08, (0.4, -0.2, 0.2))
    assert route.index == 11


@pytest.mark.parametrize("maximum", [0, -1, float("nan"), float("inf")])
def test_invalid_motion_bounds_are_explicit_errors(maximum):
    with pytest.raises(ValueError):
        move_toward((0, 0, 0), (1, 1, 1), maximum)
