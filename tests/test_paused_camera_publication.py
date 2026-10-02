"""CPU callback/clock doubles; they do not prove actual RTX publication timing."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from runtime_support import ACTOR, PNG
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_teaching_runtime import teaching as teaching

from learning.common import canonical
from simulation.camera_observation import observation_barrier


@pytest.fixture
def frozen_cameras(hardware, paused_core, monkeypatch):
    cell, _, _, clock = hardware
    core, _, _ = paused_core
    core.clock_ns = lambda: clock[1]
    core.tenant_id = str(ACTOR.tenant_id)
    cell.spec, cell.scene_epoch = core.spec, core.epoch
    cell.world.current_time = 1.3333334028720856
    cell.world.current_time_step_index = 80
    native = {"time": 80 / 60, "steps": 80, "fabric_time": 80 / 60}
    cell.test_settings_values["/rtx/hydra/supportMultiTickRate"] = True
    manager = ModuleType("isaacsim.core.simulation_manager")
    manager.SimulationManager = SimpleNamespace(
        get_simulation_time=lambda: native["time"],
        get_num_physics_steps=lambda: native["steps"],
    )
    stage = ModuleType("isaacsim.core.experimental.utils.stage")
    attribute = SimpleNamespace(Get=lambda: native["fabric_time"])
    prim = SimpleNamespace(GetAttribute=lambda name: attribute)
    stage.get_current_stage = lambda *, backend: SimpleNamespace(GetPrimAtPath=lambda path: prim)
    monkeypatch.setitem(sys.modules, manager.__name__, manager)
    monkeypatch.setitem(sys.modules, stage.__name__, stage)

    class Camera:
        def __init__(self):
            self.frame = {
                "rendering_time": 1.316666666,
                "rendering_frame": {
                    "referenceTimeNumerator": 1_316_666_666,
                    "referenceTimeDenominator": 1_000_000_000,
                },
                "rgb": b"stale physics-step-79 pixels",
            }

        def get_current_frame(self):
            return self.frame

    cell.cameras = {name: Camera() for name in ("inspection", "overview")}
    controls = {"publish_on": 3, "renders": 0, "render_ns": 10_000_000, "after_render": None}

    def render():
        controls["renders"] += 1
        clock[1] += controls["render_ns"]
        if controls["renders"] == controls["publish_on"]:
            for sensor in cell.cameras.values():
                sensor.frame.update(
                    rendering_time=1.333333333,
                    rendering_frame={
                        "referenceTimeNumerator": 1_333_333_333,
                        "referenceTimeDenominator": 1_000_000_000,
                    },
                    rgb=PNG,
                )
        if controls["after_render"] is not None:
            controls["after_render"]()

    cell.world.render = render
    monkeypatch.setattr(cell, "_encode_published_rgb", lambda frame: frame["rgb"])
    # Exercise the current production barrier rather than the shared adapter fixture's stub.
    monkeypatch.setattr(
        sys.modules["simulation.isaac_adapter"], "observation_barrier", observation_barrier
    )
    return cell, core, clock, native, controls


@pytest.mark.parametrize("publish_on", [1, 3, 8])
def test_paused_publication_drains_callbacks_until_exact_current_native_frame_without_physics(
    frozen_cameras, publish_on
):
    cell, core, _, native, controls = frozen_cameras
    controls["publish_on"] = publish_on
    initial = cell.frozen_physics_state(core.epoch)
    publication = cell._paused_publication(core)
    cell.paused_publications.record(publication)
    assert controls["renders"] == publish_on
    assert cell.frozen_physics_state(core.epoch) == initial
    assert not cell.robot.actions
    for image in dict(publication.images).values():
        assert image.png == PNG
        assert image.physics_step == 80
        assert image.simulation_time_numerator == 1_333_333_333
        assert image.simulation_time_denominator == 1_000_000_000
    assert publication.freeze_established_ns <= publication.joint_sample_ns
    assert publication.joint_sample_ns <= dict(publication.images)["inspection"].monotonic_ns
    evidence = cell.paused_camera_evidence()
    assert evidence["render_calls"] == publish_on
    assert len(evidence["attempts"]) == publish_on
    assert (
        evidence["attempts"][0]["before"]["cameras"]["inspection"]["rendering_time"] == 1.316666666
    )
    for attempt in evidence["attempts"]:
        for boundary in ("before", "after"):
            observed = attempt[boundary]
            assert observed["world_physics_index"] == 80
            assert observed["native_clocks"]["physics_steps"] == native["steps"]
            assert observed["native_clocks"]["simulation_time"] == native["time"]
            assert observed["native_clocks"]["external_simulation_time"] == native["fabric_time"]


def test_eight_stale_callbacks_fail_with_original_camera_metadata_and_no_added_physics(
    frozen_cameras,
):
    cell, core, _, _, controls = frozen_cameras
    controls["publish_on"] = 9
    with pytest.raises(ValueError, match="not synchronized"):
        cell._paused_publication(core)
    assert controls["renders"] == 8
    evidence = cell.paused_camera_evidence()
    assert len(evidence["attempts"]) == 8
    assert len(canonical(evidence)) < 32 * 1024
    assert evidence["failure"]
    for sensor in cell.cameras.values():
        assert sensor.frame["rendering_time"] == 1.316666666
        assert sensor.frame["rgb"] == b"stale physics-step-79 pixels"
    assert cell.world.current_time_step_index == 80
    assert cell.paused_publications.publication is None


def test_paused_callback_flush_intersects_original_deadline_without_renewal(frozen_cameras):
    cell, core, clock, _, controls = frozen_cameras
    controls.update(publish_on=3, render_ns=700_000_000)
    deadline = clock[1] + 1_100_000_000
    with pytest.raises(RuntimeError, match="budget|deadline"):
        cell._paused_publication(core, deadline_ns=deadline)
    assert controls["renders"] == 2
    assert cell.paused_camera_evidence()["deadline_ns"] == deadline
    assert cell.world.current_time_step_index == 80


@pytest.mark.parametrize("drift", ["time", "steps", "fabric_time"])
def test_native_clock_disagreement_is_not_fixed_by_relabelling_or_additional_renders(
    frozen_cameras, drift
):
    cell, core, _, native, controls = frozen_cameras
    native[drift] -= 1 if drift == "steps" else 1 / 60
    with pytest.raises(RuntimeError, match="clock"):
        cell._paused_publication(core)
    assert controls["renders"] == 0
    evidence = cell.paused_camera_evidence()["before"]["native_clocks"]
    assert evidence["simulation_time"] == native["time"]
    assert evidence["physics_steps"] == native["steps"]
    assert evidence["external_simulation_time"] == native["fabric_time"]


def test_nonadvancing_render_checks_the_full_frozen_scene_after_each_callback(frozen_cameras):
    cell, core, _, _, controls = frozen_cameras
    controls["after_render"] = lambda: cell.robot.joints.__setitem__(0, 0.001)
    with pytest.raises(RuntimeError, match="frozen"):
        cell._paused_publication(core)
    assert controls["renders"] == 1
    assert cell.paused_publications.publication is None


def test_unavailable_clock_diagnostic_does_not_mask_the_original_stale_camera_failure(
    frozen_cameras,
):
    cell, core, _, native, controls = frozen_cameras
    native["fabric_time"] = None
    controls["publish_on"] = 9
    with pytest.raises(ValueError, match="not synchronized"):
        cell._paused_publication(core)
    evidence = cell.paused_camera_evidence()
    assert controls["renders"] == 8
    observed = evidence["before"]["native_clocks"]
    assert observed["status"] == "unavailable"
    assert observed["simulation_time"] == 80 / 60
    assert "external_simulation_time" not in observed
    assert observed["fabric_error"]
    assert "not synchronized" in evidence["failure"]


def test_original_two_second_wall_budget_is_not_extended_to_drain_all_callbacks(frozen_cameras):
    cell, core, _, _, controls = frozen_cameras
    controls.update(publish_on=4, render_ns=700_000_000)
    with pytest.raises(RuntimeError, match="budget"):
        cell._paused_publication(core)
    assert controls["renders"] == 3
    assert cell.world.current_time_step_index == 80


def test_repeated_or_mismatched_native_frames_cannot_become_a_synchronized_rgb_pair(
    frozen_cameras,
):
    cell, core, _, _, controls = frozen_cameras
    controls["publish_on"] = 1

    def mismatched_pair():
        cell.cameras["overview"].frame["rendering_frame"] = {
            "referenceTimeNumerator": 1_332_333_333,
            "referenceTimeDenominator": 1_000_000_000,
        }
        cell.cameras["overview"].frame["rendering_time"] = 1.332333333

    controls["after_render"] = mismatched_pair
    with pytest.raises(ValueError, match="not synchronized"):
        cell._paused_publication(core)
    assert controls["renders"] == 8
    assert cell.paused_publications.publication is None


def test_realtime_barrier_still_stops_after_two_callbacks(frozen_cameras):
    cell, _, clock, _, controls = frozen_cameras
    with pytest.raises(ValueError, match="not synchronized"):
        observation_barrier(
            cell.world,
            cell.cameras,
            dt=1 / 60,
            physics_step=80,
            clock_ns=lambda: clock[1],
            deadline_ns=clock[1] + 100_000_000,
        )
    assert controls["renders"] == 2
