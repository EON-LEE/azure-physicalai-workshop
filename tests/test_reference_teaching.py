"""CPU expert/intent checks; real grasp/placement remains an independent GPU gate."""

from datetime import timedelta
from math import dist

import pytest
from test_teaching_runtime import teaching as teaching

from simulation.reference_teaching import ReferenceTeacher


def expert(teaching):
    core, request, clock = teaching
    request = request.model_copy(
        update={
            "demonstrator_kind": "reference_controller",
            "session_expires_at": clock[0] + timedelta(seconds=30),
        }
    )
    return ReferenceTeacher(
        request, core.spec, tcp=(0.3, 0, 0.5), part=core.spec.part_position
    ), clock


def test_scripted_expert_never_claims_to_be_customer_input(teaching):
    core, request, _ = teaching
    with pytest.raises(ValueError, match="reference_controller"):
        ReferenceTeacher(request, core.spec, tcp=(0.3, 0, 0.5), part=core.spec.part_position)


def test_expert_emits_only_short_granted_bounded_cartesian_and_gripper_intents(teaching):
    teacher, clock = expert(teaching)
    previous = 0
    for _ in range(10):
        intent = teacher.next_input(
            now=clock[0],
            tcp=(0.3, 0, 0.5),
            finger_gap=0.08,
            part=teacher.route.points[2].position,
        )
        assert dist(intent.delta_xyz_m, (0, 0, 0)) <= 0.01 + 1e-12
        assert intent.sequence == previous + 1
        assert intent.deadman and intent.grant_id
        assert (intent.expires_at - clock[0]).total_seconds() <= 0.25
        assert intent.expires_at <= intent.grant_expires_at
        assert "joint_positions" not in intent.model_dump()
        previous = intent.sequence
        clock[0] += timedelta(milliseconds=100)


def test_expert_cannot_progress_a_grasp_on_a_scripted_timer_alone(teaching):
    teacher, clock = expert(teaching)
    teacher.route.index = 3
    teacher.route.target = teacher.route.current.position
    for _ in range(20):
        intent = teacher.next_input(
            now=clock[0],
            tcp=teacher.route.current.position,
            finger_gap=0.08,
            part=teacher.route.current.position,
        )
        assert intent.gripper == "close"
        assert teacher.route.index == 3
        clock[0] += timedelta(milliseconds=100)


def test_expert_keeps_the_original_thirty_second_collection_deadline(teaching):
    teacher, clock = expert(teaching)
    clock[0] += timedelta(seconds=31)
    with pytest.raises(ValueError, match="deadline"):
        teacher.next_input(
            now=clock[0],
            tcp=(0.3, 0, 0.5),
            finger_gap=0.08,
            part=teacher.route.points[2].position,
        )


def test_completed_expert_stops_issuing_intents_instead_of_replaying_a_route(teaching):
    teacher, clock = expert(teaching)
    teacher.route.index = len(teacher.route.points)
    assert teacher.done
    assert (
        teacher.next_input(
            now=clock[0],
            tcp=teacher.route.points[-1].position,
            finger_gap=0.08,
            part=teacher.route.points[-2].position,
        )
        is None
    )


def test_expert_rejects_a_lift_without_the_actual_part_in_the_gripper(teaching):
    teacher, clock = expert(teaching)
    teacher.route.index = 5
    with pytest.raises(ValueError, match="grasp|lift"):
        teacher.next_input(
            now=clock[0],
            tcp=teacher.route.points[4].position,
            finger_gap=0,
            part=teacher.route.points[2].position,
        )


def test_expert_plan_can_finish_within_thirty_seconds_under_ideal_tracking(teaching):
    teacher, clock = expert(teaching)
    tcp, part, gap = (0.3, 0, 0.5), teacher.initial_part, 0.08
    started = clock[0]
    while not teacher.done:
        intent = teacher.next_input(now=clock[0], tcp=tcp, finger_gap=gap, part=part)
        tcp = tuple(a + b for a, b in zip(tcp, intent.delta_xyz_m, strict=True))
        gap = 0.05 if intent.gripper == "close" else 0.08
        if intent.gripper == "close":
            part = tcp
        clock[0] += timedelta(milliseconds=100)
    assert (clock[0] - started).total_seconds() < 30
    assert teacher.grasp_verified
