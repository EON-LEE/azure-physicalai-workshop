"""CPU route geometry and task-selection checks, not whole-arm collision/GPU proof."""

from dataclasses import replace
from math import dist

import pytest
from test_paused_dispatch import paused_core as paused_core
from test_paused_runtime import running as running

from simulation.paused_runtime import PausedReferenceRuntime

TASK_ID = "manufacturing-part-placement-v1"
INSTRUCTION = (
    "Pick up the synthetic part from the source platform and place it in the quarantine tray."
)
# Actual lower-to-destination snapshot from report
# 45bf208a7e83580bfb681c08fdc0e9289a33f9297b7164ef7677dc74698c4466.
PLACEMENT_PART = (0.21740186214447021, -0.3954625725746155, 0.20188230276107788)
PLACEMENT_TCP = (0.2176568370070878, -0.3953806701619986, 0.20496888123480964)
PLACEMENT_CONTACT = (0.21760740629357592, -0.39568847048330624, 0.2020810550318224)
PLACEMENT_FINGERS = (0.02488286979496479, 0.025328975170850754)


@pytest.fixture
def pick_place(running):
    _, core, request, hardware, recorder, _ = running
    request.task.task_id = TASK_ID
    request.task.instruction = INSTRUCTION
    runtime = PausedReferenceRuntime(core, request, hardware, recorder)
    return runtime, core, request, hardware, recorder


def test_exact_approved_pick_place_task_does_not_visit_an_unrequested_inspection_station(
    pick_place,
):
    runtime, core, _, _, _ = pick_place
    route = runtime.teacher.route
    names = [point.name for point in route.points]
    assert names == [
        "approach-part",
        "lower-to-part",
        "grasp",
        "lift-part",
        "to-destination",
        "lower-to-destination",
        "release",
        "retreat",
    ]
    assert route.points[-3].position == core.spec.station("rejected").position
    assert route.points[1].position == runtime.teacher.initial.object_position
    assert route.points[0].position[:2] == runtime.teacher.initial.object_position[:2]


def test_direct_pregrasp_and_transfer_clear_known_platforms_and_carried_cube(pick_place):
    runtime, core, _, _, _ = pick_place
    route = runtime.teacher.route
    part = runtime.teacher.initial.object_position
    destination = core.spec.station("rejected").position
    highest_platform_top = max(
        station.position[2] - 0.045 + 0.04 / 2 for station in core.spec.stations
    )
    expected = max(
        part[2] + 0.05 + 0.025, destination[2] + 0.05 + 0.025, highest_platform_top + 0.025 + 0.025
    )
    assert route.points[0].position[2] == expected
    assert expected == pytest.approx(0.275)
    assert route.points[3].position[2] - 0.025 >= highest_platform_top + 0.025
    assert route.points[4].position[2] == expected
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        height = route.target[2] + fraction * (route.points[0].position[2] - route.target[2])
        assert height >= highest_platform_top + 0.05


def test_direct_route_keeps_velocity_and_nominal_simulation_budget_without_claiming_actuator_speed(
    pick_place,
):
    runtime, _, _, _, _ = pick_place
    route = runtime.teacher.route
    measured = route.target
    ticks = 0
    while not route.done:
        previous = route.target
        measured, _ = route.next_target(
            measured,
            0.05 if route.current.closed else 0.08,
            measured,
            grasp_verified=route.index > runtime.teacher.lift_index,
        )
        assert dist(previous, measured) <= 0.1 * 0.1 + 1e-12
        ticks += 6
        assert ticks <= 1800
    assert route.elapsed < 30
    assert ticks > 0


def test_generic_paused_reference_tasks_keep_the_old_inspection_route(running):
    runtime, _, _, _, _, _ = running
    assert "inspect" in [point.name for point in runtime.teacher.route.points]
    assert len(runtime.teacher.route.points) == 12


def test_canonical_task_id_with_a_different_instruction_cannot_select_the_direct_task_route(
    running,
):
    _, core, request, hardware, recorder, _ = running
    request.task.task_id = TASK_ID
    request.task.instruction = "Inspect the part before placing it."
    with pytest.raises(ValueError, match="task"):
        PausedReferenceRuntime(core, request, hardware, recorder)


def test_direct_route_cannot_leave_lift_without_measured_part_grasp(pick_place):
    runtime, _, _, hardware, _ = pick_place
    teacher = runtime.teacher
    route = teacher.route
    route.index = next(
        index for index, point in enumerate(route.points) if point.name == "to-destination"
    )
    current = replace(
        teacher.initial,
        physics_step=teacher.initial.physics_step + 6,
        world_time=teacher.initial.world_time + 0.1,
    )
    with pytest.raises(RuntimeError, match="part grasp"):
        teacher.target(current, tcp=route.current.position, finger_gap=0.000001)
    assert not teacher.grasp_verified
    assert not hardware.actions


def lowering_teacher(pick_place, *, part=PLACEMENT_PART, verified=True):
    runtime, _, _, _, _ = pick_place
    teacher = runtime.teacher
    route = teacher.route
    route.index = next(
        index
        for index, waypoint in enumerate(route.points)
        if waypoint.name == "lower-to-destination"
    )
    route.target = route.current.position
    teacher.grasp_verified = verified
    previous = replace(
        teacher.initial,
        physics_step=3344,
        world_time=55.73333624005318,
        object_position=part,
        joint_positions=teacher.initial.joint_positions[:7] + PLACEMENT_FINGERS,
    )
    teacher.previous = previous
    return teacher, previous


def lowering_tick(teacher, previous, *, tcp=PLACEMENT_TCP, contact=PLACEMENT_CONTACT, gap=None):
    current = replace(
        previous, physics_step=previous.physics_step + 6, world_time=previous.world_time + 0.1
    )
    return current, teacher.target(
        current,
        tcp=tcp,
        route_point=contact,
        finger_gap=sum(PLACEMENT_FINGERS) if gap is None else gap,
    )


def test_observed_supported_placement_advances_to_release_after_existing_dwell(pick_place):
    teacher, previous = lowering_teacher(pick_place)
    goal = teacher.route.current.position
    assert dist(PLACEMENT_CONTACT, goal) > 0.012
    assert all(abs(a - b) <= 0.04 for a, b in zip(PLACEMENT_PART[:2], goal[:2], strict=True))
    assert abs(PLACEMENT_PART[2] - goal[2]) < 0.012
    # Repeating the observed pose over later six-tick boundaries is a CPU
    # stationary fixture, not a claim that the failed attempt reached these ticks.
    current, issued = lowering_tick(teacher, previous)
    assert issued == (goal, True)
    assert teacher.route.current.name == "lower-to-destination"
    current, issued = lowering_tick(teacher, current)
    assert issued == (goal, True)
    assert teacher.route.current.name == "release"
    for _ in range(10):
        assert (
            teacher.target(
                current,
                tcp=PLACEMENT_TCP,
                route_point=PLACEMENT_CONTACT,
                finger_gap=sum(PLACEMENT_FINGERS),
            )
            == issued
        )
    assert teacher.route.current.name == "release" and not teacher.route.done
    _, issued = lowering_tick(teacher, current)
    assert issued == (goal, False)
    assert teacher.route.current.name == "release"


@pytest.mark.parametrize(
    "part",
    [
        (0.260001, -0.38, 0.2),
        (0.22, -0.420001, 0.2),
        (0.22, -0.38, 0.212001),
        (0.22, -0.38, 0.187999),
    ],
)
def test_lowering_cannot_open_for_wrong_object_xy_or_height_even_when_tool_arrives(
    pick_place, part
):
    teacher, current = lowering_teacher(pick_place, part=part)
    goal = teacher.route.current.position
    for _ in range(3):
        current, issued = lowering_tick(teacher, current, tcp=goal, contact=goal)
        assert issued[1] is True
    assert teacher.route.current.name == "lower-to-destination"
    assert teacher.route.settled == 0


@pytest.mark.parametrize("gap", [0.0, 0.0049, 0.0551, 0.08])
def test_lowering_requires_a_still_held_gripper_not_only_historical_grasp(pick_place, gap):
    teacher, current = lowering_teacher(pick_place)
    goal = teacher.route.current.position
    for _ in range(3):
        current, issued = lowering_tick(teacher, current, tcp=goal, contact=goal, gap=gap)
        assert issued[1] is True
    assert teacher.route.current.name == "lower-to-destination"


def test_lowering_requires_prior_measured_grasp_and_the_commanded_target_at_goal(pick_place):
    teacher, current = lowering_teacher(pick_place, verified=False)
    with pytest.raises(RuntimeError, match="part grasp"):
        lowering_tick(teacher, current)
    assert teacher.route.current.name == "lower-to-destination"
    teacher.grasp_verified = True
    goal = teacher.route.current.position
    teacher.route.target = (goal[0], goal[1], goal[2] + 0.05)
    for _ in range(2):
        current, issued = lowering_tick(teacher, teacher.previous)
        assert issued[1] is True
    assert dist(teacher.route.target, goal) > 1e-6
    assert teacher.route.current.name == "lower-to-destination"
    assert teacher.route.settled == 0


def test_lowering_uses_per_axis_object_xy_and_the_existing_stricter_vertical_tolerance(pick_place):
    part = (0.2599, -0.4199, 0.2119)
    teacher, current = lowering_teacher(pick_place, part=part)
    assert dist(part[:2], teacher.route.current.position[:2]) > 0.04
    for _ in range(2):
        current, issued = lowering_tick(teacher, current)
        assert issued[1] is True
    assert teacher.route.current.name == "release"
    assert not teacher.route.done


def test_legacy_inspection_lowering_keeps_the_original_tool_arrival_condition(running):
    runtime, _, _, _, _, _ = running
    route = runtime.teacher.route
    route.index = next(
        index
        for index, waypoint in enumerate(route.points)
        if waypoint.name == "lower-to-destination"
    )
    route.target = route.current.position
    for _ in range(3):
        route.next_target(PLACEMENT_CONTACT, sum(PLACEMENT_FINGERS), PLACEMENT_PART)
    assert route.current.name == "lower-to-destination"


def test_route_done_cannot_substitute_for_measured_release_retreat_and_settled_goal(pick_place):
    runtime, _, _, hardware, _ = pick_place
    runtime.teacher.route.index = len(runtime.teacher.route.points)
    runtime.advance()
    with pytest.raises(RuntimeError, match="without measured task success"):
        runtime.advance()
    assert not runtime.succeeded and not hardware.actions
