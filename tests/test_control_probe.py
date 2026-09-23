import subprocess
import sys
from uuid import uuid4

import pytest
from runtime_support import ACTOR

from apps.api.models import Execution, RunError
from simulation.probe_control import arguments, outcome_fields
from simulation.runtime_contracts import CaptureStatus


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


def test_full_reference_collection_requires_an_explicit_task_without_changing_case(tmp_path):
    values = probe_arguments(tmp_path)
    values[values.index("policy-fixture")] = "collect-reference"
    parsed = arguments(
        values
        + [
            "--confirm-isolated-simulator",
            "--task-id",
            "place-part",
            "--instruction",
            "Place the part into the tray.",
            "--goal-id",
            "rejected",
        ]
    )
    assert parsed.mode == "collect-reference"
    assert parsed.task_id == "place-part"


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out"])
def test_failed_reference_attempt_remains_an_explicit_failure_even_without_dataset(status):
    command = uuid4()
    result = Execution(
        command_id=command,
        status=status,
        error=RunError(code=status, message="CPU fixture failure; not a real trial"),
    )
    capture = CaptureStatus(
        capture_id=uuid4(),
        command_id=command,
        epoch=uuid4(),
        status="invalid",
        message="Interrupted partial interval",
    )
    receipt = outcome_fields("collect-reference", result, capture)
    assert receipt["physical_status"] == status
    assert receipt["physical_task_success"] is False
    assert receipt["error"]["code"] == status
    assert receipt["capture"]["status"] == "invalid"
    assert receipt["learning_quality_proven"] is False


def test_a_mechanics_fixture_never_becomes_a_task_or_model_learning_success():
    result = Execution(command_id=uuid4(), status="succeeded")
    receipt = outcome_fields("policy-fixture", result, None)
    assert not receipt["physical_task_success"]
    assert not receipt["model_weights_loaded"]
    assert not receipt["production_ready"]
