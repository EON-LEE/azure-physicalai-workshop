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
        measured, _ = route.next_target(measured, 0.05 if route.current.closed else 0.08, measured)
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
