"""CPU CLI/config checks: this module does not initialize an actual simulator."""

import json
import subprocess
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
from runtime_support import ACTOR
from test_paused_dispatch import paused_core as paused_core

from apps.api.models import utcnow
from simulation.paused_probe import arguments


def flags(tmp_path):
    return [
        "--environment-record",
        str(tmp_path / "saved.json"),
        "--operator-grant",
        str(tmp_path / "grant.json"),
        "--grant-sha256",
        "a" * 64,
        "--criteria",
        str(tmp_path / "criteria.json"),
        "--criteria-sha256",
        "b" * 64,
        "--conditions",
        str(tmp_path / "conditions.json"),
        "--conditions-sha256",
        "c" * 64,
        "--mode",
        "reference-task",
        "--output",
        str(tmp_path / "receipt.json"),
    ]


def test_paused_operator_cli_requires_isolation_and_all_pinned_authority_inputs(tmp_path):
    with pytest.raises(SystemExit) as error:
        arguments(flags(tmp_path))
    assert error.value.code == 2
    request = arguments(flags(tmp_path) + ["--confirm-isolated-simulator"])
    assert request.mode == "reference-task"
    assert request.grant_sha256 == "a" * 64


def test_unavailable_learned_provider_is_not_a_scripted_cli_fallback(tmp_path):
    args = flags(tmp_path)
    args[args.index("reference-task")] = "learned"
    with pytest.raises(SystemExit):
        arguments(args + ["--confirm-isolated-simulator"])


def test_paused_cli_import_and_help_never_load_gpu_or_model_frameworks():
    code = (
        "import sys; import simulation.paused_probe; "
        "assert not {'isaacsim','carb','omni','pxr','torch','lerobot'} & set(sys.modules); "
        "simulation.paused_probe.arguments(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NON_REALTIME_SIMULATION" in result.stdout


def test_private_reference_diagnostics_are_saved_before_teardown_on_failure(
    paused_core, monkeypatch, tmp_path
):
    from simulation import paused_probe as probe

    _, environment, _ = paused_core
    args = arguments(flags(tmp_path) + ["--confirm-isolated-simulator"])
    monkeypatch.setattr(probe, "read_json", lambda path: environment.model_dump(mode="json"))
    authorization = SimpleNamespace(owner=ACTOR.owner_key)
    authority = SimpleNamespace(
        grant=SimpleNamespace(
            authorization=authorization,
            tenant_id=ACTOR.tenant_id,
            expires_at=utcnow() + timedelta(seconds=60),
        )
    )
    monkeypatch.setattr(probe.OperatorPausedAuthority, "load", lambda *a, **kw: authority)
    monkeypatch.setattr(probe, "initialize_probe_assets", lambda: None)
    monkeypatch.setattr(
        probe, "create_simulation_app", lambda **kw: SimpleNamespace(is_running=lambda: True)
    )
    monkeypatch.setattr(
        probe, "CaptureStatusStore", lambda path: SimpleNamespace(get=lambda *a: None)
    )
    monkeypatch.setattr(
        probe,
        "SimulationCore",
        lambda *a, **kw: SimpleNamespace(ready=False, activate=lambda *a: None),
    )
    diagnostic = {"latest": {"failure": "tracking limit", "control_tick": 2}, "intervals": []}
    module = ModuleType("simulation.isaac_adapter")
    module.IsaacWorkcell = lambda: SimpleNamespace(reference_target_evidence=lambda: diagnostic)
    monkeypatch.setitem(sys.modules, "simulation.isaac_adapter", module)
    persisted = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def tick(self):
            raise ValueError("Target exceeds the declared 10Hz joint tracking limit")

        def close(self):
            persisted.append(json.loads(args.output.read_text()))

    monkeypatch.setattr(probe, "SimulatorRuntime", Runtime)
    monkeypatch.setattr(probe, "_close_application", lambda *args: None)
    with pytest.raises(ValueError, match="tracking limit"):
        probe.run(args)
    assert persisted[0]["reference_target_evidence"] == diagnostic
    assert persisted[0]["failure"] == "Target exceeds the declared 10Hz joint tracking limit"
