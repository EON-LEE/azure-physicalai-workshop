import subprocess
import sys
from uuid import uuid4

import pytest
from runtime_support import ACTOR

from simulation.probe_control import arguments


def probe_arguments(tmp_path):
    return [
        "--environment-record",
        str(tmp_path / "approved-test-environment.json"),
        "--owner",
        ACTOR.owner_key,
        "--tenant-id",
        str(uuid4()),
        "--mode",
        "policy-fixture",
        "--output",
        str(tmp_path / "probe.json"),
    ]


def test_gpu_probe_never_defaults_to_execution_without_isolation_acknowledgement(tmp_path):
    with pytest.raises(SystemExit) as error:
        arguments(probe_arguments(tmp_path))
    assert error.value.code == 2


@pytest.mark.parametrize("count", ["0", "99", "151", "100000"])
def test_gpu_probe_cannot_request_unbounded_control_intervals(tmp_path, count):
    with pytest.raises(SystemExit) as error:
        arguments(
            probe_arguments(tmp_path)
            + [
                "--confirm-isolated-simulator",
                "--intervals",
                count,
            ]
        )
    assert error.value.code == 2


def test_probe_help_is_cpu_safe_and_explicit_about_not_proving_model_quality():
    result = subprocess.run(
        [sys.executable, "-m", "simulation.probe_control", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "never a model or learning-quality gate" in " ".join(result.stdout.split())
