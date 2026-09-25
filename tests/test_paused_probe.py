"""CPU CLI/config checks: this module does not initialize an actual simulator."""

import subprocess
import sys

import pytest

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
