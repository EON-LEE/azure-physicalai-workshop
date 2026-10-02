"""CPU drive doubles using actual reported baseline arrays; not a force/stability proof."""

import sys
from copy import deepcopy
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from runtime_support import ACTOR
from test_capture_lifecycle import binding as capture_binding
from test_isaac_control_actuation import Array
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_teaching_runtime import teaching as teaching

from simulation.paused_gripper_servo import DriveReadback, PausedGripperServo
from simulation.reference_targets import GRASP_ASSET_SHA256

ORIGINAL = DriveReadback(
    (22918.3125,) * 7 + (400.0, 0.0),
    (4583.66259765625,) * 7 + (80.0, 0.0),
    (87.0,) * 4 + (12.0,) * 3 + (7.199999809265137, 0.0),
)
AUTHORED = {
    "archive_sha256": GRASP_ASSET_SHA256,
    "joint_names": tuple(f"panda_joint{i}" for i in range(1, 8))
    + ("panda_finger_joint1", "panda_finger_joint2"),
    "driven_joint_index": 7,
    "driven_joint_type": "PhysicsPrismaticJoint",
    "drive_type": "force",
    "authored_stiffness": 400.0,
    "authored_damping": 80.0,
    "authored_max_force": 7.199999809265137,
    "passive_joint_index": 8,
    "passive_has_drive": False,
    "mimic_axis": "rotX",
    "mimic_gearing": -1.0,
    "mimic_reference": "panda_finger_joint1",
}


@pytest.fixture
def servo():
    state = [ORIGINAL]
    writes = []

    def write(value):
        writes.append(value)
        gains = list(state[0].stiffness)
        gains[7] = value
        state[0] = replace(state[0], stiffness=tuple(gains))

    controller = PausedGripperServo(read=lambda: state[0], write_stiffness=write)
    return controller, state, writes, capture_binding()


def test_only_the_owned_actuated_finger_stiffness_changes_and_restores(servo):
    controller, state, writes, owner = servo
    controller.apply(owner, AUTHORED)
    assert state[0].stiffness == ORIGINAL.stiffness[:7] + (2000.0, 0.0)
    assert state[0].damping == ORIGINAL.damping
    assert state[0].max_effort == ORIGINAL.max_effort
    controller.verify(owner)
    controller.apply(owner, AUTHORED)
    assert writes == [2000.0]
    evidence = controller.evidence()
    assert evidence["before"]["stiffness"][7] == 400
    assert evidence["after"]["stiffness"][7] == 2000
    assert evidence["contact_forces_measured"] is False
    controller.restore()
    controller.restore()
    assert state[0] == ORIGINAL
    assert writes == [2000.0, 400.0]
    assert controller.evidence()["status"] == "restored"
    controller.apply(replace(owner, command_id=uuid4()), AUTHORED)
    assert writes == [2000.0, 400.0, 2000.0]


@pytest.mark.parametrize("change", ["already-calibrated", "kd", "force", "passive", "nan"])
def test_unowned_or_unexpected_live_parameters_fail_before_any_gain_write(servo, change):
    controller, state, writes, owner = servo
    if change == "already-calibrated":
        state[0] = replace(ORIGINAL, stiffness=ORIGINAL.stiffness[:7] + (2000.0, 0.0))
    elif change == "kd":
        state[0] = replace(ORIGINAL, damping=ORIGINAL.damping[:7] + (81.0, 0.0))
    elif change == "force":
        state[0] = replace(ORIGINAL, max_effort=ORIGINAL.max_effort[:7] + (8.0, 0.0))
    elif change == "passive":
        state[0] = replace(ORIGINAL, stiffness=ORIGINAL.stiffness[:8] + (400.0,))
    else:
        state[0] = replace(ORIGINAL, stiffness=(float("nan"),) + ORIGINAL.stiffness[1:])
    with pytest.raises(ValueError):
        controller.apply(owner, AUTHORED)
    assert writes == []


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("archive_sha256", "f" * 64),
        ("drive_type", "acceleration"),
        ("driven_joint_index", 8),
        ("passive_has_drive", True),
        ("mimic_reference", "panda_joint1"),
        ("mimic_gearing", 1.0),
        ("authored_max_force", 8.0),
        ("authored_damping", 90.0),
    ],
)
def test_unexpected_asset_or_mimic_metadata_never_gets_repaired(servo, field, bad):
    controller, _, writes, owner = servo
    with pytest.raises(ValueError):
        controller.apply(owner, {**AUTHORED, field: bad})
    assert writes == []


def test_an_ignored_native_setter_is_not_a_successful_calibration(servo):
    controller, _, _, owner = servo
    controller.write_stiffness = lambda value: None
    with pytest.raises(RuntimeError, match="readback"):
        controller.apply(owner, AUTHORED)
    assert controller.evidence()["after"]["stiffness"][7] == 400
    assert controller.evidence()["status"] == "failed"
    controller.restore()
    assert controller.evidence()["status"] == "restored"


def test_other_joint_mutation_or_changed_owner_never_becomes_an_idempotent_apply(servo):
    controller, state, writes, owner = servo
    controller.apply(owner, AUTHORED)
    with pytest.raises(ValueError, match="owner"):
        controller.apply(replace(owner, command_id=uuid4()), AUTHORED)
    assert writes == [2000]
    state[0] = replace(state[0], damping=(1.0,) + state[0].damping[1:])
    with pytest.raises(RuntimeError, match="changed"):
        controller.verify(owner)
    with pytest.raises(RuntimeError, match="restore"):
        controller.restore()
    assert writes == [2000]
    assert controller.evidence()["restore_error"]


def test_evidence_is_a_snapshot_not_mutable_actuation_authority(servo):
    controller, _, _, owner = servo
    original = deepcopy(AUTHORED)
    controller.apply(owner, original)
    original["drive_type"] = "acceleration"
    evidence = controller.evidence()
    evidence["authored"]["drive_type"] = "acceleration"
    assert controller.evidence()["authored"]["drive_type"] == "force"


@pytest.mark.parametrize("controller", ["reference_controller", "learned"])
def test_real_adapter_uses_only_public_native_per_joint_setter_in_both_paused_modes(
    hardware,
    paused_core,
    monkeypatch,
    controller,
):
    cell, original_core, _, _ = hardware
    cell._read_paused_drive_snapshot = type(cell)._read_paused_drive_snapshot.__get__(cell)
    cell._write_paused_finger_stiffness = type(cell)._write_paused_finger_stiffness.__get__(cell)
    core, _, request = paused_core
    core.clock_ns = original_core.clock_ns
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command_id)
    cell.spec, cell.scene_epoch = core.spec, core.epoch
    live, writes = [ORIGINAL], []
    cell.robot.get_articulation_controller = lambda: SimpleNamespace(
        get_gains=lambda: (Array(live[0].stiffness), Array(live[0].damping)),
        get_max_efforts=lambda: Array(live[0].max_effort),
    )

    def setter(*, kps, indices, joint_indices, save_to_usd):
        assert tuple(indices) == (0,) and tuple(joint_indices) == (7,)
        assert save_to_usd is False
        assert len(kps) == len(kps[0]) == 1
        writes.append(kps[0][0])
        live[0] = replace(live[0], stiffness=live[0].stiffness[:7] + (kps[0][0], 0.0))

    cell.dynamics.set_gains = setter
    cell.dynamics.is_physics_handle_valid = lambda: True
    manager = sys.modules["isaacsim.core.simulation_manager"].SimulationManager
    monkeypatch.setattr(manager, "get_physics_sim_view", lambda: object(), raising=False)
    timeline = ModuleType("omni.timeline")
    timeline.get_timeline_interface = lambda: SimpleNamespace(is_stopped=lambda: False)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline)
    monkeypatch.setattr(
        cell, "_read_paused_gripper_asset", lambda: deepcopy(AUTHORED), raising=False
    )
    if controller == "learned":
        request = request.model_copy(update={"controller": "learned", "model_sha256": "b" * 64})
        cell.prepare_paused_learned(request, core)
        assert cell.controller is None
    else:
        cell.prepare_paused_reference(request, core)
    assert writes == [2000.0]
    assert live[0].stiffness[7] == 2000
    assert live[0].max_effort == ORIGINAL.max_effort
    assert live[0].damping == ORIGINAL.damping
    assert not cell.robot.actions and cell.world.current_time_step_index == 0
    cell.stop()
    assert writes == [2000.0, 400.0]
    assert live[0] == ORIGINAL
    assert cell.paused_gripper_servo_evidence()["status"] == "restored"


def test_active_owned_baseline_reversion_is_not_silently_treated_as_our_restore(servo):
    controller, state, writes, owner = servo
    controller.apply(owner, AUTHORED)
    state[0] = ORIGINAL
    with pytest.raises(RuntimeError, match="restore"):
        controller.restore()
    assert writes == [2000]


@pytest.mark.parametrize("cause", ["stopped-timeline", "missing-view", "invalid-handle"])
def test_native_gain_setter_cannot_fall_through_to_usd_authoring(hardware, monkeypatch, cause):
    cell, _, _, _ = hardware
    cell._write_paused_finger_stiffness = type(cell)._write_paused_finger_stiffness.__get__(cell)
    writes = []
    cell.dynamics.set_gains = lambda **kwargs: writes.append(kwargs)
    cell.dynamics.is_physics_handle_valid = lambda: cause != "invalid-handle"
    manager = sys.modules["isaacsim.core.simulation_manager"].SimulationManager
    monkeypatch.setattr(
        manager,
        "get_physics_sim_view",
        lambda: None if cause == "missing-view" else object(),
        raising=False,
    )
    timeline = ModuleType("omni.timeline")
    timeline.get_timeline_interface = lambda: SimpleNamespace(
        is_stopped=lambda: cause == "stopped-timeline"
    )
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline)
    with pytest.raises(ValueError, match="USD gain fallback"):
        cell._write_paused_finger_stiffness(2000)
    assert writes == [] and not cell.robot.actions
