"""CPU pressure-target regressions using observed joints; no contact-force/GPU proof."""

import sys
from math import dist
from types import ModuleType, SimpleNamespace

import pytest
from test_isaac_control_actuation import Action, Array
from test_isaac_control_actuation import hardware as hardware
from test_isaac_control_actuation import paused_hardware as paused_hardware
from test_paused_dispatch import paused_core as paused_core
from test_teaching_runtime import teaching as teaching

from simulation.control import validate_position_target
from simulation.reference_targets import reference_gripper_targets

LIFT_START = (
    0.16833630204200745,
    -0.16870388388633728,
    0.43576911091804504,
    -2.6266863346099854,
    0.11297493427991867,
    2.470061779022217,
    1.2951865196228027,
    0.024999694898724556,
    0.02504146285355091,
)
OBSERVED_ISSUED = (
    0.17093753814697266,
    -0.1681964248418808,
    0.4330010414123535,
    -2.626713275909424,
    0.11231426149606705,
    2.4704749584198,
    1.297432780265808,
    0.023233404066660326,
    0.023272221011245046,
)


def test_paused_reference_accumulates_a_bounded_pressure_target_instead_of_resetting_to_measured(
    paused_hardware,
):
    cell, _, _ = paused_hardware
    cell.robot.joints = Array(LIFT_START)
    cell.issued_targets = OBSERVED_ISSUED
    cell.controller.forward = lambda **kwargs: Action(Array(LIFT_START[:7]), joint_indices=range(7))
    previous = cell.issued_targets
    first = cell.paused_reference_targets(
        (0.3515619933605194, 0.23659035563468933, 0.19999991357326508),
        True,
        phase="grasp",
        control_tick=213,
    )
    assert first[7] < OBSERVED_ISSUED[7] and first[8] < OBSERVED_ISSUED[8]
    assert first[:7] == LIFT_START[:7]
    # Synthetic immobile fingers represent a bounded contact fixture; these
    # repeated future measurements were not recorded by the actual attempt.
    cell.robot.apply_action = lambda action: cell.robot.actions.append(action)
    for interval in range(4):
        targets = (
            first
            if interval == 0
            else cell.paused_reference_targets(
                (0.3515619933605194, 0.23659035563468933, 0.19999991357326508),
                True,
                phase="grasp",
                control_tick=214 + interval,
            )
        )
        validate_position_target(LIFT_START, targets, previous)
        assert dist(previous[7:], targets[7:]) <= 0.0025 + 1e-12
        assert all(
            0 < q - target <= 0.0036 + 1e-12
            for q, target in zip(LIFT_START[7:], targets[7:], strict=True)
        )
        for _ in range(6):
            applied = cell.apply_paused_tick(targets)
            assert applied.commanded_joint_targets == targets
            assert applied.commanded_joint_velocities == (0.0,) * 9
        previous = targets
    saturated = cell.paused_reference_targets(
        (0.35, 0.236, 0.2),
        True,
        phase="lift-part",
        control_tick=219,
    )
    assert saturated[7:] == previous[7:]
    assert len(cell.robot.actions) == 24
    assert cell.robot.joints == list(LIFT_START)


def test_reference_pressure_release_starts_from_the_previously_issued_target(paused_hardware):
    cell, _, _ = paused_hardware
    cell.robot.joints = Array(LIFT_START)
    cell.issued_targets = LIFT_START[:7] + (LIFT_START[7] - 0.0035, LIFT_START[8] - 0.0035)
    previous = cell.issued_targets
    cell.controller.forward = lambda **kwargs: Action(Array(LIFT_START[:7]), joint_indices=range(7))
    opened = cell.paused_reference_targets(
        (0.35, 0.236, 0.2),
        False,
        phase="release",
        control_tick=220,
    )
    assert all(new > old for new, old in zip(opened[7:], previous[7:], strict=True))
    assert dist(opened[7:], previous[7:]) <= 0.0025 + 1e-12
    validate_position_target(LIFT_START, opened, previous)


def test_reference_gripper_planning_cannot_hide_an_infeasible_measured_slew_intersection(
    paused_hardware,
):
    cell, _, _ = paused_hardware
    cell.robot.joints = Array(LIFT_START)
    cell.issued_targets = LIFT_START[:7] + (0.0, 0.0)
    with pytest.raises(ValueError, match="feasible|slew"):
        cell.paused_reference_targets(
            (0.35, 0.236, 0.2),
            True,
            phase="grasp",
            control_tick=213,
        )
    assert not cell.robot.actions


def test_small_observed_finger_jitter_keeps_accumulated_pressure_inside_the_new_tracking_envelope():
    previous = LIFT_START[:7] + (LIFT_START[7] - 0.0036, LIFT_START[8] - 0.0036)
    # Both values are the actual six-tick completion measurements from the lift-start row.
    measured = LIFT_START[:7] + (0.025001080706715584, 0.02504284493625164)
    fingers = reference_gripper_targets(measured, previous, closed=True)
    validate_position_target(measured, measured[:7] + fingers, previous)
    assert all(
        0 < q - target <= 0.0036 + 1e-12 for q, target in zip(measured[7:], fingers, strict=True)
    )
    assert dist(fingers, previous[7:]) <= 0.0025


def test_new_paused_command_does_not_inherit_old_gripper_pressure(paused_hardware):
    cell, core, request = paused_hardware
    cell.robot.joints = Array(LIFT_START)
    cell.issued_targets = OBSERVED_ISSUED
    cell.stop()
    cell.prepare_paused_reference(request, core)
    assert cell.issued_targets is None
    cell.controller.forward = lambda **kwargs: Action(Array(LIFT_START[:7]), joint_indices=range(7))
    first = cell.paused_reference_targets(
        (0.35, 0.236, 0.2),
        True,
        phase="grasp",
        control_tick=0,
    )
    assert dist(first[7:], LIFT_START[7:]) <= 0.0025 + 1e-12


def test_cancelled_command_cannot_apply_the_planned_gripper_pressure(paused_hardware):
    cell, core, request = paused_hardware
    cell.robot.joints = Array(LIFT_START)
    cell.issued_targets = OBSERVED_ISSUED
    targets = cell.paused_reference_targets(
        (0.35, 0.236, 0.2),
        True,
        phase="grasp",
        control_tick=213,
    )
    core.cancel(cell.control_binding.owner, request.command_id)
    with pytest.raises(RuntimeError, match="authority"):
        cell.apply_paused_tick(targets)
    assert not cell.robot.actions


def test_actual_loaded_drive_and_finger_geometry_diagnostics_only_read_native_state(
    paused_hardware,
    monkeypatch,
):
    cell, core, request = paused_hardware
    reads = []
    kp, kd, max_effort = tuple(range(1, 10)), tuple(range(11, 20)), tuple(range(21, 30))
    cell.robot.get_articulation_controller = lambda: SimpleNamespace(
        get_gains=lambda: (kp, kd), get_max_efforts=lambda: max_effort
    )

    def forbidden_new_view(*args, **kwargs):
        raise AssertionError("Diagnostics must not construct or initialize a physical view.")

    monkeypatch.setattr(
        sys.modules["isaacsim.core.prims"], "SingleRigidPrim", forbidden_new_view, raising=False
    )
    stage_module = ModuleType("isaacsim.core.experimental.utils.stage")
    xform_module = ModuleType("isaacsim.core.experimental.utils.xform")
    stage_module.get_current_stage = lambda *, backend: SimpleNamespace(
        GetPrimAtPath=lambda path: SimpleNamespace(path=path, backend=backend)
    )

    def world_pose(prim, *, device):
        assert prim.backend == "fabric" and device == "cpu"
        reads.append(prim.path)
        return SimpleNamespace(numpy=lambda: (0.35, 0.25, 0.2)), SimpleNamespace(
            numpy=lambda: (1.0, 0.0, 0.0, 0.0)
        )

    xform_module.get_world_pose = world_pose
    monkeypatch.setitem(sys.modules, stage_module.__name__, stage_module)
    monkeypatch.setitem(sys.modules, xform_module.__name__, xform_module)
    collision_api = object()
    root = SimpleNamespace(HasAPI=lambda api: False)
    collider = SimpleNamespace(
        HasAPI=lambda api: api is collision_api,
        GetPath=lambda: "/World/Robot/panda_leftfinger/collisions/pad",
    )
    bbox = SimpleNamespace(
        ComputeRelativeBound=lambda prim, ancestor: SimpleNamespace(
            ComputeAlignedRange=lambda: SimpleNamespace(
                GetMin=lambda: (-0.01, -0.005, 0.0),
                GetMax=lambda: (0.01, 0.005, 0.045),
            )
        )
    )
    cell.robot.prim_path = "/World/Robot"
    cell.world.stage = SimpleNamespace(GetPrimAtPath=lambda path: root)
    pxr = sys.modules["pxr"]
    monkeypatch.setattr(
        pxr,
        "Usd",
        SimpleNamespace(
            TimeCode=SimpleNamespace(Default=lambda: None), PrimRange=lambda prim: (root, collider)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pxr, "UsdPhysics", SimpleNamespace(CollisionAPI=collision_api), raising=False
    )
    monkeypatch.setattr(pxr.UsdGeom, "Tokens", SimpleNamespace(default_="default"), raising=False)
    monkeypatch.setattr(pxr.UsdGeom, "BBoxCache", lambda *args: bbox, raising=False)
    cell.prepare_paused_reference(request, core)
    asset = cell.reference_target_evidence()["gripper_asset"]
    assert asset["drives"]["status"] == "available"
    assert asset["drives"]["stiffness"] == kp
    assert asset["drives"]["damping"] == kd
    assert asset["drives"]["max_effort"] == max_effort
    assert asset["links"]["panda_leftfinger"]["collision_extents"]["status"] == "available"
    targets = cell.paused_reference_targets((0.35, 0.25, 0.3), True, phase="grasp", control_tick=0)
    for _ in range(6):
        cell.apply_paused_tick(targets)
    row = cell.reference_target_evidence()["intervals"][0]
    assert row["last_part_position"] == (0.35, 0.25, 0.2)
    assert row["last_finger_world_poses"]["panda_leftfinger"]["position_m"] == (0.35, 0.25, 0.2)
    assert set(reads) == {"/World/Robot/panda_leftfinger", "/World/Robot/panda_rightfinger"}
    assert row["last_finger_world_poses"]["panda_leftfinger"]["backend"] == "fabric_hierarchy"
    assert "contact_force" not in row


def test_unavailable_optional_asset_diagnostics_never_invent_drive_or_contact_values(
    paused_hardware,
):
    cell, core, request = paused_hardware
    cell.prepare_paused_reference(request, core)
    asset = cell.reference_target_evidence()["gripper_asset"]
    assert asset["drives"]["status"] == "unavailable"
    assert asset["drives"]["error"]
    assert "stiffness" not in asset["drives"] and "max_effort" not in asset["drives"]
    assert asset["links"]["panda_leftfinger"]["collision_extents"]["status"] == "unavailable"
    targets = cell.paused_reference_targets((0.35, 0.25, 0.3), True, phase="grasp", control_tick=0)
    assert cell.apply_paused_tick(targets).commanded_joint_targets == targets
