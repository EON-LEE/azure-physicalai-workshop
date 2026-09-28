"""CPU-only native-process doubles; diagnostics do not admit incomplete trials."""

import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_simulation_batch import spec as spec

from learning.common import canonical
from simulation import batch_learned, batch_task


@pytest.fixture
def native_run(evidence, tmp_path, monkeypatch):
    from learning.smolvla import artifacts

    spec, scene, profile, grant, complete, documents = evidence
    directory = tmp_path / "native"
    directory.mkdir()
    work = tmp_path / "task"
    work.mkdir()
    monkeypatch.setenv("AZ_BATCH_TASK_WORKING_DIR", str(work))
    monkeypatch.setattr(batch_task, "configure_native_environment", lambda *a: None)
    monkeypatch.setattr(batch_task, "gpu_preflight", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(batch_learned, "load_inputs", lambda *a: (None, scene, profile, grant))
    monkeypatch.setattr(batch_learned, "read_model_runtime", lambda *a: SimpleNamespace())
    monkeypatch.setattr(
        batch_learned, "verify_model_runtime", lambda *a, **k: {"cpu_fixture": True}
    )
    monkeypatch.setattr(batch_learned, "download_bundle", lambda *a, **k: None)
    monkeypatch.setattr(batch_learned, "model_command", lambda **k: ["cpu-process-double"])
    monkeypatch.setattr(artifacts, "validate_backbone", lambda *a, **k: None)
    metadata = {
        "backbone_manifest_sha256": spec.backbone.manifest.sha256,
        "task": grant.authorization.task.model_dump(),
        "control_profile": asdict(profile),
        "criteria_sha256": spec.criteria_canonical_sha256,
        "frozen_plan_sha256": spec.conditions_canonical_sha256,
    }
    monkeypatch.setattr(batch_learned, "validate_runtime_model", lambda *a, **k: metadata)
    store = SimpleNamespace(download=lambda blob, path: path.write_bytes(b'{"cpu_fixture":true}'))
    original_read = Path.read_bytes
    original_digest = batch_learned.file_digest
    manifest = Path("/data/demonstrations") / spec.owner_id / str(spec.attempt_id) / "manifest.json"
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: documents["raw-manifest.json"] if path == manifest else original_read(path),
    )
    monkeypatch.setattr(
        batch_learned,
        "file_digest",
        lambda path: (
            complete["capture"]["receipt"]["manifest_sha256"]
            if path == manifest
            else original_digest(path)
        ),
    )

    def run(report, *, server_exit=1, log_bytes=b"", process_error=None, exit_code=1):
        @contextmanager
        def process(*args, **kwargs):
            yield SimpleNamespace(pid=123, poll=lambda: server_exit)

        def bounded(command, *, log, **kwargs):
            if report is not None:
                (directory / "probe.json").write_bytes(canonical(report))
            log.write(log_bytes)
            log.flush()
            if process_error is not None:
                raise process_error
            return subprocess.CompletedProcess(command, exit_code)

        monkeypatch.setattr(batch_learned, "model_process", process)
        monkeypatch.setattr(batch_task, "run_bounded", bounded)
        return batch_learned.run_native(spec, {}, directory, time.monotonic() + 20, store=store)

    failed = {
        **complete,
        "physical_status": "failed",
        "error": {
            "code": "simulation_failed",
            "message": "Policy IPC peer disconnected",
            "retryable": False,
        },
        "metrics": {
            **complete["metrics"],
            "policy_predict_calls": 1,
            "applied_action_count": 0,
            "simulation_steps": 0,
            "applied_model_sha256": None,
        },
        "task_states": complete["task_states"][:1],
        "final_images": None,
        "capture": {"status": "invalid", "receipt": None},
        "diagnostics": {
            "schema": "physicalai.paused-learned-diagnostics/v1",
            "diagnostic_only": True,
            "secondary_errors": [
                {
                    "phase": "final_frozen_cameras",
                    "type": "ValueError",
                    "message": "Camera unavailable",
                }
            ],
        },
    }
    failed.pop("trial")
    return SimpleNamespace(
        run=run,
        failed=failed,
        complete=complete,
        directory=directory,
        spec=spec,
        grant=grant,
        profile=profile,
    )


def test_server_exit_does_not_mask_persisted_primary_error_or_admit_zero_action_trial(native_run):
    verdict = native_run.run(native_run.failed)
    assert verdict["accepted"] is False and verdict["outcome"] == "incomplete"
    assert verdict["failure"] == "Policy IPC peer disconnected"
    diagnostics = verdict["diagnostics"]
    assert diagnostics["command_error"] == native_run.failed["error"]
    assert diagnostics["model_server_exit_code"] == 1
    assert diagnostics["metrics"]["policy_predict_calls"] == 1
    assert diagnostics["metrics"]["applied_action_count"] == 0
    assert diagnostics["secondary_errors"][0]["phase"] == "final_frozen_cameras"
    assert (native_run.directory / "probe.json").read_bytes() == canonical(native_run.failed)
    assert not (native_run.directory / "raw-manifest.json").exists()


def test_native_timeout_keeps_saved_command_result_and_reports_timeout_separately(native_run):
    verdict = native_run.run(
        native_run.failed,
        server_exit=None,
        process_error=subprocess.TimeoutExpired("CPU process double", 20),
    )
    assert verdict["failure"] == "Policy IPC peer disconnected"
    assert verdict["outcome"] == "incomplete" and verdict["accepted"] is False
    assert verdict["diagnostics"]["secondary_errors"][-1]["phase"] == "probe_process"


def test_missing_report_and_model_exit_remain_explicit_unknown(native_run):
    verdict = native_run.run(None)
    assert verdict["outcome"] == "incomplete" and verdict["accepted"] is False
    assert verdict["diagnostics"]["command_error"] is None
    assert verdict["diagnostics"]["metrics"] is None
    assert verdict["diagnostics"]["model_server_exit_code"] == 1


def test_legacy_complete_successful_trial_verdict_has_no_diagnostic_shape_change(native_run):
    verdict = native_run.run(native_run.complete, server_exit=None, exit_code=0)
    assert verdict == {
        "accepted": True,
        "outcome": "accepted",
        "native_acceptance": "accepted",
        "controller": "learned",
        "model_sha256": native_run.spec.model.manifest.sha256,
        "probe_exit_code": 0,
        "physical_status": "succeeded",
        "capture_status": "ready",
        "raw_manifest": native_run.complete["capture"]["receipt"],
    }


@pytest.fixture
def guard_record(native_run):
    from learning.checks.fixtures import JOINTS
    from learning.contract import JOINT_LOWER, JOINT_UPPER, DemonstrationSource, Scope
    from learning.paused.ipc import BINDINGS, make_request
    from tests.learning.test_paused_contract import context, observation

    spec, grant, profile = native_run.spec, native_run.grant, native_run.profile
    scope = Scope(str(spec.platform.tenant_id), spec.owner_id)
    obs = replace(
        observation(),
        scope=scope,
        environment_id=grant.authorization.environment_id,
        revision=grant.authorization.revision,
        episode_id=str(spec.attempt_id),
        control_profile_sha256=profile.sha256,
    )
    ctx = replace(
        context(),
        scope=scope,
        environment_id=obs.environment_id,
        revision=obs.revision,
        episode_id=obs.episode_id,
        command_id=obs.episode_id,
        model_sha256=spec.model.manifest.sha256,
        control_profile_sha256=profile.sha256,
        observation_sha256=obs.sha256,
        simulation_step_deadline=obs.physics_step + profile.max_simulation_steps,
    )
    request = make_request(
        obs,
        ctx,
        model_sha256=spec.model.manifest.sha256,
        profile=profile,
        task=DemonstrationSource(
            "learned",
            **grant.authorization.task.model_dump(),
            source_policy_sha256=spec.model.manifest.sha256,
        ),
        request_id="cpu-diagnostic-request",
        sequence=0,
        now_ns=2_000_000_000,
    )
    return {
        "schema": "physicalai.paused-command-diagnostic/v1",
        "diagnostic_only": True,
        "kind": "joint_guard_rejected",
        "message": "paused Smol action: panda_finger_joint2 outside reference limits",
        "request": {key: request[key] for key in BINDINGS},
        "context": asdict(ctx),
        "joint_guard": {
            "label": "paused Smol action",
            "joint": "panda_finger_joint2",
            "joint_index": 8,
            "targets": [*JOINTS[:8], 0.045],
            "value": 0.045,
            "low": JOINT_LOWER[8],
            "high": JOINT_UPPER[8],
            "tolerance": 1e-6,
            "horizon_index": 49,
            "horizon_index_status": "unique_identity",
        },
        "inference": {
            "started_ns": 2_000_000_001,
            "ended_ns": 2_010_000_001,
            "latency_ms": 10.0,
        },
    }


def model_line(record):
    return b"PHYSICALAI_COMMAND_DIAGNOSTIC " + json.dumps(record).encode() + b"\n"


def test_original_bound_model_guard_is_primary_but_still_not_scorable(native_run, guard_record):
    verdict = native_run.run(native_run.failed, log_bytes=model_line(guard_record))
    assert verdict["failure"] == guard_record["message"]
    assert verdict["diagnostics"]["model_diagnostic"] == guard_record
    assert verdict["diagnostics"]["command_error"] == native_run.failed["error"]
    assert verdict["outcome"] == "incomplete" and verdict["accepted"] is False
    assert not (native_run.directory / "raw-manifest.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "task",
        "owner",
        "attempt",
        "context_hash",
        "extra_context",
        "horizon",
        "nonfinite",
        "bound_type",
        "oversize",
        "duplicate",
    ],
)
def test_foreign_or_malformed_log_cannot_replace_real_command_error(
    native_run, guard_record, change
):
    if change == "model":
        guard_record["request"]["model_sha256"] = "f" * 64
    elif change == "task":
        guard_record["request"]["task_sha256"] = "f" * 64
    elif change == "owner":
        guard_record["context"]["scope"]["owner_id"] = "f" * 64
    elif change == "attempt":
        guard_record["context"]["command_id"] = "different-attempt"
    elif change == "context_hash":
        guard_record["request"]["context_sha256"] = "f" * 64
    elif change == "extra_context":
        guard_record["context"]["unexpected"] = "not allowlisted"
    elif change == "horizon":
        guard_record["joint_guard"]["horizon_index_status"] = "unknown"
    elif change == "nonfinite":
        guard_record["joint_guard"]["value"] = float("nan")
    elif change == "bound_type":
        guard_record["joint_guard"]["low"] = False
    elif change == "oversize":
        guard_record["extra"] = "x" * 16_384
    log = model_line(guard_record)
    if change == "duplicate":
        log *= 2
    verdict = native_run.run(native_run.failed, log_bytes=log)
    assert verdict["failure"] == native_run.failed["error"]["message"]
    assert verdict["diagnostics"]["model_diagnostic"]["kind"] == "unavailable"
    assert verdict["outcome"] == "incomplete"


@pytest.mark.parametrize("filename", ["acceptance.json", "acceptance.log"])
def test_secondary_acceptance_write_failure_preserves_command_error(
    native_run, monkeypatch, filename
):
    original_bytes, original_text = Path.write_bytes, Path.write_text

    def write_bytes(path, payload):
        if path.name == filename:
            raise OSError("acceptance publication disk full")
        return original_bytes(path, payload)

    def write_text(path, payload, *args, **kwargs):
        if path.name == filename:
            raise OSError("acceptance publication disk full")
        return original_text(path, payload, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    monkeypatch.setattr(Path, "write_text", write_text)
    verdict = native_run.run(native_run.failed, server_exit=None)
    assert verdict["failure"] == native_run.failed["error"]["message"]
    assert verdict["diagnostics"]["secondary_errors"][-1]["phase"] == "acceptance_publication"
    assert verdict["accepted"] is False


@pytest.mark.parametrize("failure", ["server_exit", "timeout"])
def test_diagnostics_do_not_create_native_evidence_past_legacy_process_failure_barrier(
    native_run, failure
):
    verdict = native_run.run(
        native_run.complete,
        server_exit=1 if failure == "server_exit" else None,
        process_error=(
            subprocess.TimeoutExpired("CPU process double", 20) if failure == "timeout" else None
        ),
        exit_code=0,
    )
    assert verdict["outcome"] == "incomplete" and verdict["accepted"] is False
    assert not (native_run.directory / "raw-manifest.json").exists()
    assert not (native_run.directory / "acceptance.json").exists()
