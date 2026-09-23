"""CPU sensor-ordering doubles; actual camera timing must still pass on the GPU."""

from fractions import Fraction
from types import SimpleNamespace

import pytest

from simulation.camera_observation import observation_barrier, render_identity, sensor_launch_config


def test_real_isaac6_fabric_identity_is_not_fabricated_as_a_local_counter():
    assert render_identity(42) == (42, 1)
    assert render_identity(
        {
            "referenceTimeNumerator": 2_000_000_000,
            "referenceTimeDenominator": 1_000_000_000,
        }
    ) == (2_000_000_000, 1_000_000_000)
    for malformed in (None, True, {}, {"referenceTimeNumerator": 1, "referenceTimeDenominator": 0}):
        with pytest.raises(ValueError):
            render_identity(malformed)


def camera(time):
    frame = {
        "rendering_time": time,
        "rendering_frame": {
            "referenceTimeNumerator": int(time * 1000),
            "referenceTimeDenominator": 1000,
        },
    }
    return SimpleNamespace(frame=frame, get_current_frame=lambda: dict(frame))


def test_render_barrier_observes_both_cameras_without_advancing_any_physics():
    clock = [1_000_000_000]
    cameras = {name: camera(0.9) for name in ("overview", "inspection")}
    world = SimpleNamespace(current_time=1.0, current_time_step_index=60)
    renders = []

    def render():
        renders.append(True)
        clock[0] += 10_000_000
        for sensor in cameras.values():
            sensor.frame.update(
                rendering_time=1.0,
                rendering_frame={
                    "referenceTimeNumerator": 1000,
                    "referenceTimeDenominator": 1000,
                },
            )

    world.render = render
    stamp = observation_barrier(
        world,
        cameras,
        dt=1 / 60,
        physics_step=60,
        clock_ns=lambda: clock[0],
        deadline_ns=1_100_000_000,
    )
    assert stamp == clock[0]
    assert len(renders) == 1
    assert world.current_time == 1.0 and world.current_time_step_index == 60


def test_barrier_never_accepts_a_render_call_that_advanced_physics():
    cameras = {name: camera(1.0) for name in ("overview", "inspection")}
    world = SimpleNamespace(current_time=1.0, current_time_step_index=60)

    def render():
        world.current_time_step_index += 1
        world.current_time += 1 / 60

    world.render = render
    with pytest.raises(RuntimeError, match="physics"):
        observation_barrier(
            world, cameras, dt=1 / 60, physics_step=60, clock_ns=lambda: 1, deadline_ns=100
        )


def test_barrier_keeps_strict_time_match_and_reports_both_actual_sensor_states():
    cameras = {"overview": camera(1.0), "inspection": camera(0.9)}
    world = SimpleNamespace(current_time=1.0, current_time_step_index=60, render=lambda: None)
    with pytest.raises(ValueError) as error:
        observation_barrier(
            world, cameras, dt=1 / 60, physics_step=60, clock_ns=lambda: 1, deadline_ns=100
        )
    text = str(error.value)
    assert "overview" in text and "inspection" in text and "0.9" in text
    assert "physics_step" in text and "world_time" in text


def test_barrier_cannot_spend_beyond_the_existing_control_budget():
    clock = [1_000_000_000]
    cameras = {name: camera(1.0) for name in ("overview", "inspection")}
    world = SimpleNamespace(current_time=1.0, current_time_step_index=60)
    world.render = lambda: clock.__setitem__(0, 1_101_000_000)
    with pytest.raises(RuntimeError, match="budget"):
        observation_barrier(
            world,
            cameras,
            dt=1 / 60,
            physics_step=60,
            clock_ns=lambda: clock[0],
            deadline_ns=1_100_000_000,
        )


@pytest.mark.parametrize("publish_on_sixth_tick", [False, True])
def test_nonadvancing_render_cannot_publish_unrendered_physics_but_sixth_tick_can(
    publish_on_sixth_tick,
):
    cameras = {name: camera(0.483333333) for name in ("overview", "inspection")}
    for sensor in cameras.values():
        sensor.frame["rendering_frame"] = {
            "referenceTimeNumerator": 483333333,
            "referenceTimeDenominator": 1_000_000_000,
        }
    clock = [1_000_000_000]

    class World:
        current_time = 0.483333333
        current_time_step_index = 29

        def render(self):
            clock[0] += 1_000_000

        def step(self, *, render):
            self.current_time_step_index += 1
            self.current_time += 1 / 60
            clock[0] += 1_000_000
            if render:
                for sensor in cameras.values():
                    sensor.frame.update(
                        rendering_time=self.current_time,
                        rendering_frame={
                            "referenceTimeNumerator": round(self.current_time * 1_000_000_000),
                            "referenceTimeDenominator": 1_000_000_000,
                        },
                    )

    world = World()
    observation_barrier(
        world,
        cameras,
        dt=1 / 60,
        physics_step=27,
        clock_ns=lambda: clock[0],
        deadline_ns=1_100_000_000,
    )
    for tick in range(6):
        world.step(render=publish_on_sixth_tick and tick == 5)
    assert world.current_time_step_index == 35
    if publish_on_sixth_tick:
        observation_barrier(
            world,
            cameras,
            dt=1 / 60,
            physics_step=33,
            clock_ns=lambda: clock[0],
            deadline_ns=1_100_000_000,
        )
    else:
        with pytest.raises(ValueError, match="not synchronized"):
            observation_barrier(
                world,
                cameras,
                dt=1 / 60,
                physics_step=33,
                clock_ns=lambda: clock[0],
                deadline_ns=1_100_000_000,
            )
    assert world.current_time_step_index == 35


def test_already_published_fresh_current_frames_avoid_gpu_redraw_and_keep_actual_timestamp():
    cameras = {name: camera(1.0) for name in ("overview", "inspection")}

    def redundant_render():
        raise AssertionError("Already published current frames must not trigger another GPU render")

    world = SimpleNamespace(current_time=1.0, current_time_step_index=60, render=redundant_render)
    stamp = observation_barrier(
        world,
        cameras,
        dt=1 / 60,
        physics_step=60,
        clock_ns=lambda: 1_000_000_000,
        deadline_ns=1_100_000_000,
        published_ns=950_000_000,
        previous_identities={name: (900, 1000) for name in cameras},
    )
    assert stamp == 950_000_000


def test_a_repeated_native_identity_is_not_accepted_as_an_already_fresh_frame():
    cameras = {name: camera(1.0) for name in ("overview", "inspection")}
    renders = []
    world = SimpleNamespace(
        current_time=1.0, current_time_step_index=60, render=lambda: renders.append(True)
    )
    with pytest.raises(ValueError, match="synchronized|fresh"):
        observation_barrier(
            world,
            cameras,
            dt=1 / 60,
            physics_step=60,
            clock_ns=lambda: 1_000_000_000,
            deadline_ns=1_100_000_000,
            published_ns=950_000_000,
            previous_identities={name: (1000, 1000) for name in cameras},
        )
    assert len(renders) <= 2


def test_real_numeric_sdk_time_keeps_exact_tolerance_and_explains_freshness():
    cameras = {name: camera(1.0) for name in ("overview", "inspection")}
    for sensor in cameras.values():
        sensor.frame["rendering_time"] = Fraction(1, 1)
    calls = []
    world = SimpleNamespace(
        current_time=1.0, current_time_step_index=60, render=lambda: calls.append(True)
    )
    diagnostics = {}
    stamp = observation_barrier(
        world,
        cameras,
        dt=1 / 60,
        physics_step=60,
        clock_ns=lambda: 1_000_000_000,
        deadline_ns=1_100_000_000,
        published_ns=950_000_000,
        previous_identities={name: (900, 1000) for name in cameras},
        diagnostics=diagnostics,
    )
    assert stamp == 950_000_000 and not calls
    assert diagnostics["reused_published_frame"] is True
    assert diagnostics["published_age_ns"] == 50_000_000
    assert (
        diagnostics["before"]["cameras"]["inspection"]["rendering_time_type"]
        == "fractions.Fraction"
    )


def test_control_launch_disables_only_unused_viewport_and_uses_real_rtx_sensor_rendering():
    config = sensor_launch_config()
    assert config["headless"] is True
    assert config["disable_viewport_updates"] is True
    assert config["renderer"] == "RaytracedLighting"
    assert (config["width"], config["height"]) == (320, 320)
