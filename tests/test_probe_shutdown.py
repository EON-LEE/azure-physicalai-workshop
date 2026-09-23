"""CPU lifecycle tests: no Isaac/GPU or Blob operations are performed here."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from runtime_support import ACTOR, PNG, document, service

from apps.api.errors import Problem
from apps.api.models import SaveEnvironment
from simulation import probe_control
from simulation.core import LoadScene, StartTeaching
from simulation.runtime_contracts import CaptureReceipt


@pytest.fixture
def probe(monkeypatch, tmp_path):
    doc = document()
    doc["execution"].update(record_demonstration=True, demonstration_split="test")
    environment = service().save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))
    record = tmp_path / "environment.json"
    record.write_text(environment.model_dump_json())
    args = probe_control.arguments(
        [
            "--environment-record",
            str(record),
            "--owner",
            ACTOR.owner_key,
            "--tenant-id",
            str(ACTOR.tenant_id),
            "--mode",
            "teaching",
            "--output",
            str(tmp_path / "probe.json"),
            "--confirm-isolated-simulator",
        ]
    )
    monkeypatch.setenv("ENTRA_TENANT_ID", str(ACTOR.tenant_id))
    monkeypatch.setattr(probe_control, "initialize_probe_assets", lambda: None, raising=False)
    events = []
    at_close = []
    real_fsync = os.fsync

    def sync(handle):
        events.append("fsync")
        real_fsync(handle)

    monkeypatch.setattr(probe_control.os, "fsync", sync)

    class Application:
        def is_running(self):
            return True

        def close(self):
            events.append("close")
            at_close.append(json.loads(args.output.read_text()) if args.output.exists() else None)
            raise SystemExit(0)

    hardware = SimpleNamespace(
        control_mode="reference",
        warmup_steps=0,
        control_timings=[{"completed_step": (index + 1) * 6} for index in range(100)],
        initial_state_evidence=lambda: {
            "observed_initial_pose_m": (0.35, 0.25, 0.2),
            "scene_builder_sha256": "b" * 64,
        },
        physics_scheduling={
            "scene_ready": {
                "schema": "physicalai.cpu-physics-scheduling/v1",
                "phase": "scene_ready",
                "setting": "/persistent/physics/numThreads",
                "requested_num_threads": 0,
                "observed_num_threads": 0,
                "physics_device": "cpu",
                "gpu_dynamics_enabled": False,
            }
        },
    )
    module = ModuleType("simulation.isaac_adapter")
    module.IsaacWorkcell = lambda: hardware
    monkeypatch.setitem(sys.modules, "simulation.isaac_adapter", module)
    monkeypatch.setattr(probe_control, "create_simulation_app", lambda **kwargs: Application())
    return args, events, at_close, module


@pytest.mark.parametrize(
    "error",
    [
        Problem(503, "gpu_probe_failed", "CPU fixture simulates a pre-receipt GPU failure."),
        RuntimeError("CPU fixture runtime failure"),
        AttributeError("CPU fixture SDK interface failure"),
    ],
)
def test_failure_receipt_is_durable_before_close_and_system_exit_cannot_mask_error(
    probe, monkeypatch, error
):
    args, events, at_close, _ = probe

    class Runtime:
        def __init__(self, core, hardware, **kwargs):
            pass

        def tick(self):
            raise error

        def close(self):
            events.append("runtime_close")

    monkeypatch.setattr(probe_control, "SimulatorRuntime", Runtime)
    with pytest.raises(type(error), match="CPU fixture"):
        probe_control.run_probe(args)
    assert at_close[0] is not None
    assert at_close[0]["failure_type"] == type(error).__name__
    assert at_close[0]["probe_completed"] is False
    assert at_close[0]["model_weights_loaded"] is False
    assert at_close[0]["physics_scheduling"]["scene_ready"]["observed_num_threads"] == 0
    assert events.index("fsync") < events.index("close")


def test_hardware_initialization_failure_is_recorded_and_closes_the_created_application(
    probe, monkeypatch
):
    args, events, at_close, module = probe

    def failed_hardware():
        raise TypeError("CPU fixture hardware initialization failed")

    monkeypatch.setattr(module, "IsaacWorkcell", failed_hardware)
    with pytest.raises(TypeError, match="hardware initialization"):
        probe_control.run_probe(args)
    assert at_close[0]["failure_type"] == "TypeError"
    assert at_close[0]["physical_status"] == "not_started"
    assert events.index("fsync") < events.index("close")


def test_completed_receipt_is_synced_before_system_exit_and_returns_actual_evidence(
    probe, monkeypatch
):
    args, events, at_close, _ = probe

    class Runtime:
        def __init__(self, core, hardware, *, heartbeat, **kwargs):
            self.core, self.heartbeat = core, heartbeat

        def tick(self):
            action = self.core.next_action()
            if isinstance(action, LoadScene):
                for camera in ("overview", "inspection"):
                    self.core.publish_frame(
                        camera, PNG, (0.35, 0.25, 0.2), 1, epoch=self.core.epoch
                    )
            elif isinstance(action, StartTeaching):
                command_id = action.request.command_id
                self.core.begin_motion(command_id)
                capture = self.core.begin_capture(command_id)
                self.core.finish("cancelled", (0.35, 0.25, 0.2))
                self.core.publish_capture(
                    capture,
                    "ready",
                    receipt=CaptureReceipt(
                        "https://test.blob.core.windows.net/demos/manifest.json",
                        "c" * 64,
                        str(command_id),
                        100,
                    ),
                )
            self.heartbeat.write_text("1.0")

        def close(self):
            events.append("runtime_close")

    monkeypatch.setattr(probe_control, "SimulatorRuntime", Runtime)
    result = probe_control.run_probe(args)
    assert at_close[0] == json.loads(json.dumps(result))
    assert result["probe_completed"] is True
    assert result["physical_task_success"] is False
    assert result["physics_scheduling"]["scene_ready"]["physics_device"] == "cpu"
    assert events.index("fsync") < events.index("close")


def test_unexpected_process_exit_is_recorded_as_incomplete_not_success(probe, monkeypatch):
    args, _, at_close, _ = probe

    class Runtime:
        def __init__(self, core, hardware, **kwargs):
            pass

        def tick(self):
            raise SystemExit(0)

        def close(self):
            pass

    monkeypatch.setattr(probe_control, "SimulatorRuntime", Runtime)
    with pytest.raises(SystemExit) as stopped:
        probe_control.run_probe(args)
    assert stopped.value.code == 1
    assert at_close[0]["failure_type"] == "SystemExit"
    assert at_close[0]["probe_completed"] is False


def test_atomic_receipt_publication_never_overwrites_another_attempt(tmp_path):
    path = tmp_path / "receipt.json"
    receipt = {
        "schema": "physicalai.operator-attempt-failure/v1",
        "physical_status": "not_started",
        "probe_completed": False,
    }
    probe_control._persist_receipt(path, receipt)
    with pytest.raises(FileExistsError):
        probe_control._persist_receipt(path, receipt | {"probe_completed": True})
    assert json.loads(path.read_text()) == receipt
    assert list(tmp_path.glob("*.new")) == []
