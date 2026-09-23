"""CPU scheduling-contract tests; no PhysX or GPU performance is inferred here."""

from types import SimpleNamespace

import pytest

from simulation.camera_observation import sensor_launch_config
from simulation.physics_scheduling import (
    PHYSICS_THREAD_SETTING,
    physics_scheduling_readback,
    require_control_scheduling,
)


class Settings:
    def __init__(self, value):
        self.value = value
        self.reads = []

    def get(self, key):
        self.reads.append(key)
        assert key == PHYSICS_THREAD_SETTING
        return self.value


def context(*, device="cpu", gpu_dynamics=False):
    return SimpleNamespace(device=device, is_gpu_dynamics_enabled=lambda: gpu_dynamics)


def test_only_control_launch_requests_main_thread_physx_without_other_physics_changes():
    config = sensor_launch_config()
    assert config["extra_args"] == ["--/persistent/physics/numThreads=0"]
    assert set(config) == {
        "headless",
        "width",
        "height",
        "renderer",
        "disable_viewport_updates",
        "extra_args",
    }


def test_scheduling_evidence_reads_actual_thread_count_and_cpu_backend_without_mutation():
    settings = Settings(0)
    readback = physics_scheduling_readback(settings, phase="scene_ready", context=context())
    require_control_scheduling(readback)
    assert readback == {
        "schema": "physicalai.cpu-physics-scheduling/v1",
        "phase": "scene_ready",
        "setting": PHYSICS_THREAD_SETTING,
        "requested_num_threads": 0,
        "observed_num_threads": 0,
        "physics_device": "cpu",
        "gpu_dynamics_enabled": False,
    }
    assert settings.reads == [PHYSICS_THREAD_SETTING]
    assert settings.value == 0


@pytest.mark.parametrize("observed", [8, 1, None, True, False, 0.0, "0"])
def test_missing_ignored_or_wrongly_typed_thread_setting_never_defaults_to_zero(observed):
    settings = Settings(observed)
    readback = physics_scheduling_readback(settings, phase="scene_ready", context=context())
    assert readback["observed_num_threads"] is observed
    with pytest.raises(RuntimeError, match="numThreads"):
        require_control_scheduling(readback)
    assert settings.value is observed


@pytest.mark.parametrize(
    "physics",
    [
        context(device="cuda:0"),
        context(gpu_dynamics=True),
        context(device=None),
        context(gpu_dynamics=None),
    ],
)
def test_cpu_only_optimization_rejects_gpu_or_unknown_backend_instead_of_switching_it(physics):
    readback = physics_scheduling_readback(Settings(0), phase="scene_ready", context=physics)
    with pytest.raises(RuntimeError, match="CPU"):
        require_control_scheduling(readback)


def test_startup_thread_readback_does_not_invent_a_physics_scene_before_world_exists():
    readback = physics_scheduling_readback(Settings(0), phase="application_ready")
    assert readback["physics_device"] is None
    assert readback["gpu_dynamics_enabled"] is None
    require_control_scheduling(readback)
    with pytest.raises(RuntimeError, match="CPU"):
        require_control_scheduling(readback | {"phase": "scene_ready"})
