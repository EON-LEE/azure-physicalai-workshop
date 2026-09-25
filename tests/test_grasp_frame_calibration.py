"""Asset-bound CPU frame tests; a millimetre calibration is not a proven grasp fix."""

import hashlib
import math
import sys
from types import SimpleNamespace

import pytest
from test_isaac_control_actuation import hardware as hardware
from test_isaac_control_actuation import paused_hardware as paused_hardware
from test_paused_dispatch import paused_core as paused_core
from test_paused_runtime import running as running
from test_teaching_runtime import teaching as teaching

from simulation import reference_targets
from simulation.paused_teacher import PICK_PLACE_INSTRUCTION, PICK_PLACE_TASK_ID
from simulation.reference_targets import (
    GRASP_ASSET_SHA256,
    TCP_TO_PAD_CENTROID_M,
    reference_contact_point,
    reference_tcp_target,
    verify_grasp_calibration_asset,
)


@pytest.fixture
def bound_asset(tmp_path, monkeypatch):
    root = tmp_path / GRASP_ASSET_SHA256
    root.mkdir()
    files = []
    for name, _ in reference_targets.GRASP_ASSET_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("CPU calibration fixture, not NVIDIA asset bytes: " + name).encode()
        path.write_bytes(payload)
        files.append((name, hashlib.sha256(payload).hexdigest()))
    monkeypatch.setattr(reference_targets, "GRASP_ASSET_FILES", tuple(files))
    (root / ".complete").write_text(GRASP_ASSET_SHA256, encoding="ascii")
    monkeypatch.setenv("FRANKA_ASSET_SHA256", GRASP_ASSET_SHA256)
    monkeypatch.setenv("FRANKA_ASSET_ROOT", str(root))
    monkeypatch.setenv("FRANKA_USD_PATH", str(root / files[0][0]))
    return root, root / files[0][0], {"Mesh": "Performance", "Gripper": "Default"}


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ],
)
def test_contact_command_and_measurement_are_inverse_for_down_and_rotated_orientations(orientation):
    contact = (0.35, 0.25, 0.2)
    tcp = reference_tcp_target(contact, orientation)
    assert reference_contact_point(tcp, orientation) == pytest.approx(contact, abs=1e-12)
    assert math.dist(tcp, contact) == pytest.approx(math.dist((0, 0, 0), TCP_TO_PAD_CENTROID_M))
    if orientation == (0.0, 0.0, 1.0, 0.0):
        assert tcp[2] == pytest.approx(contact[2] + 0.002904602840903575)
    if orientation == (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0):
        assert tcp[0] == pytest.approx(contact[0] - 0.002904602840903575)
        assert tcp[2] == pytest.approx(contact[2] + TCP_TO_PAD_CENTROID_M[0])


@pytest.mark.parametrize("bad", [(0, 0, 0, 0), (2, 0, 0, 0), (float("nan"), 0, 0, 1)])
def test_invalid_orientation_never_becomes_a_default_frame(bad):
    with pytest.raises(ValueError):
        reference_tcp_target((0.35, 0.25, 0.2), bad)


def test_asset_identity_binds_archive_marker_variants_and_all_calibrated_geometry_files(
    bound_asset,
):
    root, asset, variants = bound_asset
    binding = verify_grasp_calibration_asset(root, asset, GRASP_ASSET_SHA256, variants)
    assert binding["archive_sha256"] == GRASP_ASSET_SHA256
    assert binding["calibration_id"] == "franka-default-inner-pad-centroid/v1"
    assert binding["tcp_to_pad_centroid_m"] == TCP_TO_PAD_CENTROID_M
    assert binding["variants"] == variants


@pytest.mark.parametrize("changed", ["archive", "marker", "geometry", "variant"])
def test_each_asset_identity_mismatch_fails_without_a_calibration_fallback(bound_asset, changed):
    root, asset, variants = bound_asset
    checksum = "f" * 64 if changed == "archive" else GRASP_ASSET_SHA256
    if changed == "variant":
        variants = {**variants, "Gripper": "Robotiq"}
    if changed == "marker":
        (root / ".complete").write_text("f" * 64, encoding="ascii")
    if changed == "geometry":
        asset.write_bytes(b"changed calibration geometry")
    with pytest.raises(ValueError):
        verify_grasp_calibration_asset(root, asset, checksum, variants)


@pytest.fixture
def calibrated_hardware(paused_hardware, bound_asset, monkeypatch):
    cell, core, request = paused_hardware
    _, _, variants = bound_asset
    cell.robot.prim_path = "/World/Robot"
    sets = SimpleNamespace(
        GetVariantSet=lambda name: SimpleNamespace(GetVariantSelection=lambda: variants[name])
    )
    cell.world.stage = SimpleNamespace(
        GetPrimAtPath=lambda path: SimpleNamespace(GetVariantSets=lambda: sets)
    )
    request.task.task_id = PICK_PLACE_TASK_ID
    request.task.instruction = PICK_PLACE_INSTRUCTION
    cell.prepare_paused_reference(request, core)
    return cell, core, request


def test_only_canonical_teacher_contact_targets_change_not_transit_or_issued_joint_labels(
    calibrated_hardware,
):
    cell, _, _ = calibrated_hardware
    cell.orientation_target = (0.0, 0.0, 1.0, 0.0)
    calls = []
    forward = cell.controller.forward

    def observed_forward(**kwargs):
        calls.append(tuple(kwargs["target_end_effector_position"]))
        return forward(**kwargs)

    cell.controller.forward = observed_forward
    point = (0.35, 0.25, 0.2)
    for tick, phase in enumerate(
        (
            "lower-to-part",
            "grasp",
            "lower-to-destination",
            "release",
            "approach-part",
            "lift-part",
            "to-destination",
            "retreat",
        )
    ):
        targets = cell.paused_reference_targets(
            point, phase not in {"release", "retreat"}, phase=phase, control_tick=tick
        )
        expected = (
            (point[0] + TCP_TO_PAD_CENTROID_M[0], point[1], point[2] + TCP_TO_PAD_CENTROID_M[2])
            if phase in reference_targets.GRASP_CONTACT_PHASES
            else point
        )
        assert all(abs(a - b) < 1e-12 for a, b in zip(calls[-1], expected, strict=True))
        if tick:
            assert math.dist(calls[-2], calls[-1]) <= 0.003
        diagnostic = cell.reference_target_diagnostic
        assert diagnostic["reference_target_position"] == point
        assert diagnostic["cartesian_target"] == calls[-1]
        assert diagnostic["measured_tcp_frame"] == "right_gripper"
        assert diagnostic["reference_target_frame"] == (
            "inner_pad_centroid"
            if phase in reference_targets.GRASP_CONTACT_PHASES
            else "right_gripper"
        )
        for _ in range(6):
            applied = cell.apply_paused_tick(targets)
            assert applied.commanded_joint_targets == targets
            assert applied.commanded_joint_velocities == (0.0,) * 9
    assert len(cell.robot.actions) == 48
    assert cell.reference_target_evidence()["gripper_asset"]["drives"]["status"] == "unavailable"


def test_measured_contact_reference_uses_actual_not_desired_orientation(
    calibrated_hardware, monkeypatch
):
    cell, _, _ = calibrated_hardware
    cell.orientation_target = (1.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(
        sys.modules["simulation.isaac_adapter"],
        "rot_matrix_to_quat",
        lambda rotation: (0.0, 0.0, 1.0, 0.0),
    )
    raw_tcp = (0.35, 0.25, 0.20290460284090358)
    contact = cell.paused_reference_route_point(raw_tcp, "grasp")
    assert abs(contact[2] - 0.2) < 1e-12
    assert cell.paused_reference_route_point(raw_tcp, "lift-part") == raw_tcp


def test_wrong_asset_fails_before_reference_calibration_can_generate_motion(
    calibrated_hardware, monkeypatch
):
    cell, core, request = calibrated_hardware
    monkeypatch.setenv("FRANKA_ASSET_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="asset|archive"):
        cell.prepare_paused_reference(request, core)
    assert not cell.robot.actions


def test_route_progress_uses_inverse_contact_measurement_but_grasp_proof_uses_raw_tcp(running):
    runtime, _, _, hardware, _, _ = running
    runtime.teacher.route.index = 3
    runtime.teacher.route.target = runtime.teacher.route.current.position
    runtime.teacher.route.settled = 0.6
    goal = runtime.teacher.route.current.position
    raw = (goal[0], goal[1], goal[2] + 0.014)
    contact = (goal[0], goal[1], raw[2] - TCP_TO_PAD_CENTROID_M[2])
    hardware._measured_tcp = lambda: raw
    hardware.paused_reference_route_point = lambda tcp, phase: contact
    for _ in range(10):
        runtime.advance()
    assert runtime.teacher.route.current.name == "lift-part"
    assert not hardware.goal


def test_calibrated_arrival_never_replaces_the_raw_tcp_grasp_distance_guard(running):
    from dataclasses import replace

    runtime, _, _, _, _, _ = running
    teacher = runtime.teacher
    teacher.route.index = teacher.lift_index + 1
    part = teacher.initial.object_position
    lifted = (part[0], part[1], part[2] + 0.06)
    state = replace(
        teacher.initial,
        physics_step=teacher.initial.physics_step + 6,
        world_time=teacher.initial.world_time + 0.1,
        object_position=lifted,
    )
    with pytest.raises(RuntimeError, match="part grasp"):
        teacher.target(
            state,
            tcp=(lifted[0], lifted[1], lifted[2] + 0.091),
            route_point=(lifted[0], lifted[1], lifted[2] + 0.088),
            finger_gap=0.05,
        )
    assert not teacher.grasp_verified


def test_noncanonical_reference_targets_stay_in_the_original_tcp_frame(
    paused_hardware, monkeypatch
):
    cell, _, _ = paused_hardware
    module = sys.modules["simulation.isaac_adapter"]

    def forbidden_calibration(*args, **kwargs):
        raise AssertionError("Unselected reference tasks must not use contact calibration")

    monkeypatch.setattr(module, "reference_tcp_target", forbidden_calibration)
    point = (0.35, 0.25, 0.2)
    cell.paused_reference_targets(point, True, phase="grasp", control_tick=0)
    assert cell.reference_target_diagnostic["cartesian_target"] == point
    assert cell.reference_target_diagnostic["reference_target_frame"] == "right_gripper"
