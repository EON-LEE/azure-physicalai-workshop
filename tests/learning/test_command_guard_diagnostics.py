"""Real pinned Unix IPC/guard tracebacks with CPU policies; not GPU or quality evidence."""

import io
import json
import os
import time
from dataclasses import asdict, replace
from threading import Thread

import pytest

from learning.checks.fixtures import JOINTS, SCOPE
from learning.common import ContractError, canonical
from learning.contract import JOINT_LOWER, JOINT_UPPER, bounded_joints
from learning.paused import command_model
from learning.paused.ipc import SocketChunkPolicy
from tests.learning.test_paused_contract import context, observation, profile
from tests.learning.test_paused_inference import Policy

PREFIX = "PHYSICALAI_COMMAND_DIAGNOSTIC "
SCHEMA = "physicalai.paused-command-diagnostic/v1"


@pytest.fixture
def rejected(tmp_path, tmp_path_factory, monkeypatch):
    sockets = tmp_path_factory.mktemp("guard")

    def execute(actions, *, stream=None):
        directory = tmp_path / str(time.monotonic_ns())
        directory.mkdir(mode=0o700)
        path = sockets / (directory.name + ".sock")
        binding = directory / "binding.json"
        binding.write_bytes(
            canonical(
                {
                    "execution_timing": "paused_simulation",
                    "real_time_admission": False,
                    "scope": asdict(SCOPE),
                    "control_profile": asdict(profile()),
                    "criteria_sha256": "d" * 64,
                    "frozen_plan_sha256": "e" * 64,
                }
            )
        )
        log = io.StringIO() if stream is None else stream
        model_calls, outcome = [], {}

        class CpuPolicy:
            policy_type, execution_timing, real_time_admission = (
                "smolvla",
                "paused_simulation",
                False,
            )
            model_sha256, scope, task = "c" * 64, SCOPE, Policy.task
            secret = "CREDENTIAL_SENTINEL_DO_NOT_LOG"

            def __init__(self):
                self.profile = profile()

            def reset(self):
                pass

            def predict_chunk(self, obs, ctx):
                model_calls.append((obs, ctx))
                return actions

        monkeypatch.setattr(
            command_model, "LocalCommandPausedSmolVLAPolicy", lambda *a, **k: CpuPolicy()
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "command_model",
                "--model-root",
                str(directory),
                "--backbone-root",
                str(directory),
                "--binding",
                str(binding),
                "--model-sha256",
                "c" * 64,
                "--socket-path",
                str(path),
            ],
        )
        monkeypatch.setattr("sys.stderr", log)

        def server():
            try:
                command_model.main()
            except SystemExit as exc:
                outcome["exit"] = exc

        worker = Thread(target=server, daemon=True)
        worker.start()
        deadline = time.monotonic() + 2
        while not path.exists() and time.monotonic() < deadline and worker.is_alive():
            time.sleep(0.001)
        assert path.exists(), outcome
        original = observation()
        delta = time.monotonic_ns() - original.monotonic_ns
        obs = replace(
            original,
            **{
                name: getattr(original, name) + delta
                for name in ("monotonic_ns", "observation_started_ns", "joint_sample_ns")
            },
            images={
                name: replace(image, monotonic_ns=image.monotonic_ns + delta)
                for name, image in original.images.items()
            },
        )
        original_context = context()
        ctx = replace(
            original_context,
            observation_sha256=obs.sha256,
            **{
                name: getattr(original_context, name) + delta
                for name in (
                    "episode_started_ns",
                    "wall_deadline_ns",
                    "interval_started_ns",
                    "interval_deadline_ns",
                    "operation_started_ns",
                    "operation_deadline_ns",
                )
            },
        )
        client = SocketChunkPolicy(
            path, scope=SCOPE, model_sha256="c" * 64, profile=profile(), task=Policy.task
        )
        with pytest.raises(ContractError, match="disconnected"):
            client.predict_chunk(obs, ctx)
        worker.join(2)
        assert not worker.is_alive()
        assert isinstance(outcome["exit"].__cause__, ContractError)
        assert outcome["exit"].code
        assert len(model_calls) == client.predict_calls == 1
        outcome.update(log=log, observation=obs, context=ctx)
        return outcome

    if os.name != "posix":
        pytest.skip("Deployment IPC requires Linux SO_PEERCRED.")
    return execute


def actions(index=37, value=0.045):
    result = [list(JOINTS) for _ in range(50)]
    result[index][8] = value
    return result


def diagnostic(outcome):
    lines = outcome["log"].getvalue().splitlines()
    records = [line for line in lines if line.startswith(PREFIX)]
    assert len(records) == 1, (
        "The CLI must emit one structured diagnostic before its original exit."
    )
    assert len((records[0] + "\n").encode()) <= 16 * 1024
    value = json.loads(records[0][len(PREFIX) :])
    assert value["schema"] == SCHEMA and value["diagnostic_only"] is True
    assert "png_base64" not in records[0] and "CREDENTIAL_SENTINEL" not in records[0]
    return value


@pytest.mark.parametrize("bad", [-0.003, 0.045])
def test_cli_preserves_exact_guard_vector_and_original_validated_bindings(rejected, bad):
    original = actions(value=bad)
    outcome = rejected(original)
    value = diagnostic(outcome)
    assert value["kind"] == "joint_guard_rejected"
    assert value["joint_guard"] == {
        "label": "paused Smol action",
        "joint": "panda_finger_joint2",
        "joint_index": 8,
        "value": bad,
        "targets": original[37],
        "low": JOINT_LOWER[8],
        "high": JOINT_UPPER[8],
        "tolerance": 1e-6,
        "horizon_index": 37,
        "horizon_index_status": "unique_identity",
    }
    assert value["request"]["sequence"] == 0
    assert value["request"]["model_sha256"] == "c" * 64
    assert value["request"]["observation_sha256"] == outcome["observation"].sha256
    assert value["request"]["context_sha256"] == outcome["context"].sha256
    assert value["request"]["freeze_id"] == outcome["observation"].freeze_id
    assert value["context"] == asdict(outcome["context"])
    timing = value["inference"]
    assert timing["latency_ms"] == (timing["ended_ns"] - timing["started_ns"]) / 1e6
    assert "panda_finger_joint2 outside reference limits" in str(outcome["exit"])
    with pytest.raises(ContractError, match="panda_finger_joint2"):
        bounded_joints(value["joint_guard"]["targets"], "paused Smol action")


def test_repeated_original_row_identity_is_unknown_not_a_guessed_horizon(rejected):
    original = actions(index=4)
    original[19] = original[4]
    value = diagnostic(rejected(original))
    assert value["joint_guard"]["horizon_index"] is None
    assert value["joint_guard"]["horizon_index_status"] == "unknown"
    assert value["joint_guard"]["value"] == 0.045


@pytest.mark.parametrize("change", ["nan", "infinity", "row_shape", "horizon", "numeric_type"])
def test_other_contract_failures_do_not_manufacture_joint_guard_data(rejected, change):
    original = actions()
    if change == "row_shape":
        original[37].pop()
    elif change == "horizon":
        original.pop()
    else:
        original[37][8] = {"nan": float("nan"), "infinity": float("inf"), "numeric_type": "bad"}[
            change
        ]
    value = diagnostic(rejected(original))
    assert value["kind"] == "unavailable"
    assert not {"joint_guard", "request", "context", "inference"} & value.keys()


def test_unrelated_same_named_frames_are_not_joint_guard_evidence():
    def serve():
        def make_response():
            bounded_joints(actions()[37], "paused Smol action")

        make_response()

    try:
        serve()
    except ContractError as exc:
        value = command_model.guard_diagnostic(exc)
    assert value["kind"] == "unavailable"
    assert "joint_guard" not in value


def test_traceback_walk_is_bounded_even_with_real_inner_guard_frames(rejected):
    original = rejected(actions())["exit"].__cause__

    def propagate(level):
        if level:
            return propagate(level - 1)
        raise original

    try:
        propagate(80)
    except ContractError as exc:
        value = command_model.guard_diagnostic(exc)
    assert value["kind"] == "unavailable"
    assert value["reason"] == "traceback_limit"


def test_oversized_diagnostic_is_explicit_unknown_not_truncated_joint_evidence(
    rejected, monkeypatch
):
    outcome = rejected(actions())
    actual = diagnostic(outcome)
    monkeypatch.setattr(
        command_model, "guard_diagnostic", lambda exc: {**actual, "extra": "x" * 16384}
    )
    log = io.StringIO()
    assert command_model.emit_guard_diagnostic(outcome["exit"].__cause__, stream=log)
    bounded = diagnostic({"log": log})
    assert bounded["kind"] == "unavailable" and bounded["reason"] == "diagnostic_size_limit"
    assert "joint_guard" not in bounded


@pytest.mark.parametrize("failure", [OSError("disk full"), ValueError("closed stream")])
def test_diagnostic_io_failure_cannot_replace_original_guard_exit(rejected, failure):
    class BrokenStream:
        def write(self, value):
            raise failure

        def flush(self):
            raise AssertionError("No second I/O attempt after the write failed.")

    outcome = rejected(actions(), stream=BrokenStream())
    assert "panda_finger_joint2 outside reference limits" in str(outcome["exit"])
    assert "structured diagnostic unavailable" in str(outcome["exit"])
    assert isinstance(outcome["exit"].__cause__, ContractError)


@pytest.mark.parametrize("operation", ["flush", "partial_write"])
def test_incomplete_diagnostic_output_is_not_reported_as_logged(rejected, operation):
    class IncompleteStream:
        def write(self, line):
            return len(line) if operation == "flush" else len(line) - 1

        def flush(self):
            raise OSError("flush failed")

    outcome = rejected(actions(), stream=IncompleteStream())
    assert "structured diagnostic unavailable" in str(outcome["exit"])
    assert "panda_finger_joint2 outside reference limits" in str(outcome["exit"])


def test_logging_keeps_native_prediction_load_and_guard_code_objects_unchanged():
    from learning import contract
    from learning.paused import ipc
    from learning.paused.model import LocalPausedSmolVLAPolicy

    policy = command_model.LocalCommandPausedSmolVLAPolicy
    assert not {"predict_chunk", "reset", "_load"} & policy.__dict__.keys()
    assert policy.predict_chunk is LocalPausedSmolVLAPolicy.predict_chunk
    assert policy._load is LocalPausedSmolVLAPolicy._load
    assert command_model.serve is ipc.serve
    assert command_model.make_response is ipc.make_response
    assert command_model.bounded_joints is contract.bounded_joints
