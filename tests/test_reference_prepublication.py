"""CPU construction-order doubles; no claim about actual Isaac startup or rendering speed."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from runtime_support import ACTOR, PNG
from test_isaac_control_actuation import Array
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_teaching_runtime import teaching as teaching

from simulation.paused_runtime import PausedReferenceRuntime


@pytest.fixture
def prepared_scene(hardware, paused_core, monkeypatch):
    cell, original_core, _, clock = hardware
    core, _, request = paused_core
    core.clock_ns = original_core.clock_ns
    core.tenant_id = str(ACTOR.tenant_id)
    core.ready = False
    cell.spec, cell.scene_epoch, cell.paused_scene_core = core.spec, core.epoch, core
    cell.part.get_world_pose = lambda: (Array(cell.spec.part_position), Array([1, 0, 0, 0]))
    permit = core.paused_authorizer.authorize(ACTOR.owner_key, request, core.paused_profile)
    events = []
    adapter = sys.modules["simulation.isaac_adapter"]
    native_rmp, native_kinematics = adapter.RMPFlowController, adapter.LulaKinematicsSolver

    def slow_rmp(**kwargs):
        events.append(("rmp_init", cell.world.current_time_step_index, clock[1]))
        assert cell.controller is None and cell.control_done
        assert not cell.robot.actions
        clock[1] += 2_500_000_000
        return native_rmp(**kwargs)

    def slow_kinematics(**kwargs):
        events.append(("kinematics_init", cell.world.current_time_step_index, clock[1]))
        assert not cell.robot.actions
        clock[1] += 700_000_000
        return native_kinematics(**kwargs)

    monkeypatch.setattr(adapter, "RMPFlowController", slow_rmp)
    monkeypatch.setattr(adapter, "LulaKinematicsSolver", slow_kinematics)

    def geometry():
        events.append(("geometry", cell.world.current_time_step_index, clock[1]))
        clock[1] += 400_000_000
        return {"drives": {"status": "unavailable"}, "links": {}}

    monkeypatch.setattr(cell, "_read_gripper_asset_evidence", geometry)
    monkeypatch.setattr(adapter.time, "monotonic_ns", lambda: clock[1])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[1] / 1e9)

    class Camera:
        def __init__(self):
            self.frame = {
                "rendering_frame": {
                    "referenceTimeNumerator": 0,
                    "referenceTimeDenominator": 1_000_000_000,
                },
                "rendering_time": 0.0,
                "rgb": PNG,
            }

        def get_current_frame(self):
            return self.frame

        def get_resolution(self):
            return (320, 320)

        def get_frequency(self):
            return -1

    cell.cameras = {name: Camera() for name in ("inspection", "overview")}
    monkeypatch.setattr(cell, "_encode_published_rgb", lambda frame: frame["rgb"])

    def render():
        clock[1] += 1_000_000
        events.append(("render", cell.world.current_time_step_index, clock[1]))
        for camera in cell.cameras.values():
            camera.frame.update(
                rendering_time=cell.world.current_time,
                rendering_frame={
                    "referenceTimeNumerator": round(cell.world.current_time * 1e9),
                    "referenceTimeDenominator": 1_000_000_000,
                },
            )

    cell.world.render = render
    return cell, core, request, permit, clock, events


def dispatch(core, request):
    core.ready = True
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    assert core.begin_motion(request.command_id)


def test_slow_reference_construction_precedes_original_last_warmup_publication(prepared_scene):
    cell, core, request, permit, clock, events = prepared_scene
    state = cell.frozen_physics_state(core.epoch)
    cell.prepare_paused_reference_components(core, permit)
    assert cell.frozen_physics_state(core.epoch) == state
    assert not cell.robot.actions and cell.controller is None
    assert cell._prepared_reference.controller.forward_calls == 0
    assert cell._read_paused_drive_snapshot().stiffness[7] == 400
    assert core.active_command is None
    assert [name for name, _, _ in events] == ["kinematics_init", "rmp_init", "geometry"]
    cell.prime_control_profile()
    publication = cell.paused_publications.publication
    assert publication.frozen_state.physics_step == 60
    assert len(cell.robot.actions) == 60
    assert len(cell.control_warmup_timings) == 60
    assert publication.freeze_established_ns > events[-2][2]
    dispatch(core, request)
    runtime = PausedReferenceRuntime(core, request, cell)
    runtime.advance()
    runtime.advance()
    assert runtime.episode.phase == "applying"
    assert cell.world.current_time_step_index == 60
    assert runtime.episode.metrics()["simulation_steps"] == 0
    assert runtime.episode.metrics()["wall_elapsed_ms"] < 2000
    assert cell.controller.forward_calls == 1
    assert cell._read_paused_drive_snapshot().stiffness[7] == 2000
    assert sum(name == "rmp_init" for name, _, _ in events) == 1
    assert sum(name == "kinematics_init" for name, _, _ in events) == 1
    assert runtime.observation.monotonic_ns == publication.published_ns
    assert (
        runtime.observation.images["inspection"].monotonic_ns
        == dict(publication.images)["inspection"].monotonic_ns
    )
    evidence = cell.reference_preparation_evidence()
    assert evidence["components"]["completed_ns"] < publication.freeze_established_ns
    assert evidence["components"]["duration_ms"] == 3600
    assert evidence["command_start"]["cache_reused"] is True
    assert evidence["first_observation"]["publication_id"] == str(publication.publication_id)
    assert 0 <= evidence["first_observation"]["publication_age_ns"] < 2_000_000_000


@pytest.mark.parametrize(
    "change", ["epoch", "robot", "world", "profile", "task", "owner", "tenant"]
)
def test_cached_reference_components_cannot_cross_their_exact_loaded_scene_binding(
    prepared_scene, change
):
    cell, core, request, permit, _, _ = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    if change == "epoch":
        core.epoch = uuid4()
    elif change == "robot":
        from copy import copy

        cell.robot = copy(cell.robot)
    elif change == "world":
        from copy import copy

        cell.world = copy(cell.world)
    elif change == "profile":
        core.paused_profile = replace(core.paused_profile, servo_profile_sha256="f" * 64)
    elif change == "owner":
        core.owner = "f" * 64
    elif change == "tenant":
        core.tenant_id = str(uuid4())
    else:
        request = request.model_copy(
            update={
                "task": request.task.model_copy(update={"instruction": "Another task"}),
            }
        )
    with pytest.raises(RuntimeError, match="prepar|cache|binding"):
        cell.prepare_paused_reference(request, core)
    assert not cell.robot.actions and cell.controller is None


def test_no_component_cache_can_extend_the_original_publication_age(prepared_scene):
    cell, core, request, permit, clock, _ = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    cell.prime_control_profile()
    publication = cell.paused_publications.publication
    clock[1] = publication.published_ns + 2_000_000_001
    dispatch(core, request)
    runtime = PausedReferenceRuntime(core, request, cell)
    runtime.advance()
    with pytest.raises(RuntimeError, match="future-dated or expired"):
        runtime.advance()
    assert cell.world.current_time_step_index == 60
    assert cell.paused_publications.publication == publication
    assert (
        cell.reference_preparation_evidence()["first_observation"]["publication_age_ns"]
        > 2_000_000_000
    )


def test_nonreference_preparation_cannot_create_an_rmp_controller(prepared_scene):
    cell, core, _, permit, _, events = prepared_scene
    learned = permit.model_copy(
        update={
            "controller": "learned",
            "authorization_kind": "evaluation_grant",
            "policy_type": "smolvla",
            "model_sha256": "b" * 64,
        }
    )
    with pytest.raises(ValueError, match="reference"):
        cell.prepare_paused_reference_components(core, learned)
    assert events == [] and not cell.robot.actions


def test_stop_invalidates_unconsumed_reference_components(prepared_scene):
    cell, core, request, permit, _, _ = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    cell.stop()
    with pytest.raises(RuntimeError, match="prepar|cache|binding"):
        cell.prepare_paused_reference(request, core)
    assert not cell.robot.actions


def test_prepared_reference_cache_is_consumed_once_and_live_gains_are_not_cached(
    prepared_scene, monkeypatch
):
    cell, core, request, permit, _, events = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    cell.prime_control_profile()
    dispatch(core, request)
    monkeypatch.setattr(
        cell,
        "_read_gripper_drive_evidence",
        lambda: {"status": "available", "stiffness": cell._read_paused_drive_snapshot().stiffness},
    )
    cell.prepare_paused_reference(request, core)
    assert cell.reference_target_evidence()["gripper_asset"]["drives"]["stiffness"][7] == 2000
    assert cell.reference_preparation_evidence()["components"]["drive_readback_before_warmup"] == {
        "status": "unavailable"
    }
    with pytest.raises(RuntimeError, match="consumed"):
        cell.prepare_paused_reference(request, core)
    assert sum(name == "rmp_init" for name, _, _ in events) == 1
    assert len(cell.robot.actions) == 60


def test_component_preparation_rejects_an_sdk_constructor_that_moves_the_scene(
    prepared_scene, monkeypatch
):
    cell, core, _, permit, _, _ = prepared_scene
    original = sys.modules["simulation.isaac_adapter"].RMPFlowController

    def moves_scene(**kwargs):
        value = original(**kwargs)
        cell.robot.joints[0] += 0.001
        return value

    monkeypatch.setattr(sys.modules["simulation.isaac_adapter"], "RMPFlowController", moves_scene)
    with pytest.raises(RuntimeError, match="changed the physical"):
        cell.prepare_paused_reference_components(core, permit)
    assert cell.controller is None and not cell.robot.actions


def test_loading_or_resetting_the_robot_drops_unconsumed_reference_components(
    prepared_scene, monkeypatch
):
    cell, core, _, permit, _, _ = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    monkeypatch.setattr(cell, "_validate_asset_bundle", lambda: None)
    monkeypatch.setattr(
        sys.modules["simulation.isaac_adapter"], "can_reset_in_place", lambda *a: True
    )
    cell.world.reset = lambda *, soft: None
    monkeypatch.setattr(cell, "_prepare_episode", lambda: None)
    cell.load(cell.spec)
    assert cell._prepared_reference is None
    assert cell.reference_preparation_evidence() == {}


def test_cancel_before_prepared_component_consumption_cannot_issue_an_action(prepared_scene):
    cell, core, request, permit, _, _ = prepared_scene
    cell.prepare_paused_reference_components(core, permit)
    cell.prime_control_profile()
    dispatch(core, request)
    core.cancel(ACTOR.owner_key, request.command_id)
    with pytest.raises(RuntimeError, match="no longer active"):
        PausedReferenceRuntime(core, request, cell)
    assert len(cell.robot.actions) == cell.world.current_time_step_index == 60


def test_operator_load_orders_reference_preparation_before_warmup(paused_core, tmp_path):
    from simulation.run_isaac import SimulatorRuntime

    core, environment, request = paused_core
    permit = core.paused_authorizer.authorize(ACTOR.owner_key, request, core.paused_profile)
    calls = []
    hardware = SimpleNamespace(
        world=None,
        scene_epoch=None,
        load=lambda spec: calls.append("load"),
        prepare_paused_reference_components=lambda protocol, authority: calls.append(
            "reference_prepare"
        ),
        prime_control_profile=lambda on_tick=None: calls.append("warmup_and_original_publication"),
    )
    runtime = SimulatorRuntime(
        core, hardware, heartbeat=tmp_path / "heartbeat", reference_preparation=permit
    )
    core.activate(ACTOR.owner_key, environment)
    runtime.tick()
    assert calls == ["load", "reference_prepare", "warmup_and_original_publication"]


def test_normal_runtime_load_never_prepares_a_reference_controller_implicitly(
    paused_core, tmp_path
):
    from simulation.run_isaac import SimulatorRuntime

    core, environment, _ = paused_core
    calls = []

    def forbidden(*args):
        raise AssertionError("Learned/default runtime must not preconstruct RMPflow")

    hardware = SimpleNamespace(
        world=None,
        scene_epoch=None,
        load=lambda spec: calls.append("load"),
        prepare_paused_reference_components=forbidden,
        prime_control_profile=lambda on_tick=None: calls.append("warmup"),
    )
    runtime = SimulatorRuntime(core, hardware, heartbeat=tmp_path / "heartbeat")
    core.activate(ACTOR.owner_key, environment)
    runtime.tick()
    assert calls == ["load", "warmup"]
