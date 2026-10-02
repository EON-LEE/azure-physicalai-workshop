"""CPU orchestration doubles test failure evidence only, never learned physical quality."""

import json
import sys
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_simulation_batch import spec as spec

from apps.api.models import EnvironmentRecord, RunError
from learning.paused.task import TaskState
from simulation import learned_probe
from simulation.paused_contracts import PausedRuntimeMetrics

PRIMARY = "Policy IPC peer disconnected"
CAMERA = "Final frozen cameras did not advance within eight non-physics renders"
PREFIX = "PHYSICALAI_LEARNED_DIAGNOSTIC "


@pytest.fixture
def failed_probe(evidence, tmp_path, monkeypatch, capsys):
    from simulation import capture_status, core, paused_deployment, probe_control, run_isaac

    spec, scene, profile, grant, old_report, documents = evidence
    environment = EnvironmentRecord.model_validate_json(documents["inputs/environment.json"])
    initial = TaskState(**old_report["task_states"][0])
    terminal = replace(initial, monotonic_ns=initial.monotonic_ns + 1, policy_predict_calls=1)
    metrics = PausedRuntimeMetrics(
        profile_id=profile.profile_id,
        control_profile_sha256=profile.sha256,
        controller="learned",
        phase="stopped",
        policy_predict_calls=1,
    )
    result = SimpleNamespace(
        status="failed",
        error=RunError(code="simulation_failed", message=PRIMARY),
        simulation_runtime=metrics,
    )
    capture = SimpleNamespace(
        status="invalid",
        model_dump=lambda **kwargs: {"status": "invalid", "error": "No completed interval."},
    )
    events, logged = [], []
    protocol = SimpleNamespace(
        ready=True,
        error=None,
        epoch=UUID(initial.epoch),
        state_revision=60,
        captures={(grant.authorization.owner, spec.attempt_id): capture},
        monotonic_deadlines={(grant.authorization.owner, spec.attempt_id): 601_000_000_000},
        clock_ns=lambda: terminal.monotonic_ns,
        clock_utc=lambda: grant.issued_at,
        activate=lambda *a: None,
        observe=lambda *a: SimpleNamespace(observation_id=uuid4(), object_id="part"),
        dispatch_simulation_episode=lambda *a: events.append("dispatch"),
        command=lambda *a: result,
    )

    def final_cameras(*args, **kwargs):
        events.append("final_cameras")
        logged.append(capsys.readouterr().err)
        raise ValueError(CAMERA)

    hardware = SimpleNamespace(
        paused_publications=SimpleNamespace(
            publication=SimpleNamespace(private_evidence=lambda: {"original": True})
        ),
        _paused_publication=final_cameras,
        paused_gripper_servo_evidence=lambda: {"status": "restored"},
        paused_camera_evidence=lambda: {"render_calls": 8, "failure": CAMERA},
    )

    class Trace:
        def __init__(self, *args):
            self.states = [initial]

        def recording(self, *, end_ns):
            return {
                "task_states": [asdict(initial)],
                "heartbeat_ns": [initial.monotonic_ns],
                "episode_budget": old_report["episode_budget"],
            }

        def terminal_state(self):
            events.append("terminal_observation")
            return terminal

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            events.append("runtime_close")

        def tick(self):
            raise AssertionError("A terminal episode cannot advance physics for diagnostics.")

    application = SimpleNamespace(
        is_running=lambda: True, close=lambda: events.append("application_close")
    )
    monkeypatch.setattr(
        learned_probe, "load_inputs", lambda *a: (environment, scene, profile, grant)
    )
    monkeypatch.setattr(learned_probe, "LearnedTrace", Trace)
    monkeypatch.setattr(learned_probe, "measured_runtime_type", lambda: Runtime)
    monkeypatch.setattr(probe_control, "initialize_probe_assets", lambda: None)
    monkeypatch.setattr(run_isaac, "create_simulation_app", lambda **k: application)
    monkeypatch.setattr(core, "SimulationCore", lambda *a, **k: protocol)
    monkeypatch.setattr(
        capture_status, "CaptureStatusStore", lambda *a: SimpleNamespace(get=lambda: None)
    )
    monkeypatch.setattr(
        paused_deployment.InstalledPausedPolicyProvider, "load", lambda *a: object()
    )
    module = ModuleType("simulation.isaac_adapter")
    module.IsaacWorkcell = lambda: hardware
    monkeypatch.setitem(sys.modules, "simulation.isaac_adapter", module)
    output = tmp_path / "probe.json"
    return SimpleNamespace(
        call=lambda: learned_probe.run(
            spec,
            paths={},
            model_root=tmp_path / "model",
            socket_path=tmp_path / "model.sock",
            output=output,
        ),
        spec=spec,
        output=output,
        events=events,
        logged=logged,
        runtime=Runtime,
        application=application,
        primary=result.error.model_dump(mode="json"),
        result=result,
    )


def test_primary_and_actual_zero_action_counters_survive_secondary_final_camera_failure(
    failed_probe,
):
    report = failed_probe.call()
    saved = json.loads(failed_probe.output.read_bytes())
    assert saved == json.loads(json.dumps(report))
    assert saved["error"] == failed_probe.primary
    assert saved["physical_status"] == "failed"
    assert saved["metrics"]["policy_predict_calls"] == 1
    assert saved["metrics"]["applied_action_count"] == 0
    assert saved["metrics"]["applied_model_sha256"] is None
    diagnostics = saved["diagnostics"]
    assert diagnostics["schema"] == "physicalai.paused-learned-diagnostics/v1"
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["initial_task_state"]["policy_predict_calls"] == 0
    assert diagnostics["terminal_task_state"]["policy_predict_calls"] == 1
    assert (
        diagnostics["terminal_task_state"]["physics_step"]
        == diagnostics["initial_task_state"]["physics_step"]
    )
    assert len(saved["task_states"]) == 1
    assert saved["final_images"] is None and "trial" not in saved
    assert saved.get("native_acceptance") is None
    assert diagnostics["secondary_errors"] == [
        {"phase": "final_frozen_cameras", "type": "ValueError", "message": CAMERA}
    ]
    assert failed_probe.events.index("terminal_observation") < failed_probe.events.index(
        "final_cameras"
    )
    lines = [line for line in failed_probe.logged[0].splitlines() if line.startswith(PREFIX)]
    assert len(lines) == 1 and len(lines[0].encode()) < 16 * 1024
    snapshot = json.loads(lines[0][len(PREFIX) :])
    assert snapshot["command_error"] == failed_probe.primary
    assert snapshot["metrics"]["policy_predict_calls"] == 1
    assert snapshot["terminal_task_state"]["applied_action_count"] == 0
    assert "png_base64" not in lines[0]


def test_runtime_close_failure_is_secondary_and_does_not_prevent_the_receipt(
    failed_probe, monkeypatch
):
    def broken_close(self):
        raise OSError("capture upload unavailable")

    monkeypatch.setattr(failed_probe.runtime, "close", broken_close)
    report = failed_probe.call()
    assert report["error"] == failed_probe.primary
    assert [item["phase"] for item in report["diagnostics"]["secondary_errors"]] == [
        "final_frozen_cameras",
        "runtime_close",
    ]
    assert json.loads(failed_probe.output.read_bytes())["error"] == failed_probe.primary
    assert "application_close" in failed_probe.events


def test_receipt_io_failure_leaves_original_terminal_snapshot_in_existing_log(
    failed_probe, monkeypatch, capsys
):
    from simulation import probe_control

    def broken_receipt(*args):
        raise OSError("receipt disk full")

    monkeypatch.setattr(probe_control, "_persist_receipt", broken_receipt)
    report = failed_probe.call()
    assert report["error"] == failed_probe.primary
    assert report["diagnostics"]["secondary_errors"][-1]["phase"] == "receipt_publication"
    assert not failed_probe.output.exists()
    assert PRIMARY in failed_probe.logged[0]
    assert "receipt disk full" in capsys.readouterr().err


def test_missing_terminal_measurement_stays_unknown_not_zero(failed_probe, monkeypatch):
    def unavailable(self):
        raise ValueError("epoch changed before terminal readback")

    monkeypatch.setattr(learned_probe.LearnedTrace, "terminal_state", unavailable)
    report = failed_probe.call()
    assert report["diagnostics"]["terminal_task_state"] is None
    assert report["diagnostics"]["secondary_errors"][0]["phase"] == "terminal_observation"
    assert report["error"] == failed_probe.primary
    assert len(report["task_states"]) == 1


def test_application_teardown_cannot_replace_persisted_primary(failed_probe, capsys):
    def broken_close():
        raise OSError("application teardown unavailable")

    failed_probe.application.close = broken_close
    report = failed_probe.call()
    assert report["error"] == failed_probe.primary
    assert json.loads(failed_probe.output.read_bytes())["error"] == failed_probe.primary
    assert report["diagnostics"]["secondary_errors"][-1]["phase"] == "application_close"
    assert "application teardown unavailable" in capsys.readouterr().err
