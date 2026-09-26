"""CPU durable-attempt tests; no simulator, GPU, Azure job or Blob service is used."""

import json
import os
import sys

import pytest
from test_simulation_batch import spec as spec

from learning.common import canonical
from simulation.batch_task import run_episode, validate_gpu_evidence


class Store:
    def __init__(self):
        self.claimed = False
        self.writes = []
        self.completion = None

    def claim(self, value):
        if self.claimed:
            return False
        self.claimed = True
        self.writes.append("claim.json")
        return True

    def download(self, reference, path):
        path.write_bytes(b"{}")

    def verify_asset_reference(self):
        pass

    def publish(self, name, path):
        from learning.common import file_digest

        self.writes.append(name)
        return {"path": name, "sha256": file_digest(path), "bytes": path.stat().st_size}

    def finish(self, value):
        self.writes.append("completion.json")
        self.completion = value


def gpu():
    from simulation.batch_task import RAY_TRACING_EXTENSIONS

    return {
        "schema": "physicalai.batch-gpu-preflight/v1",
        "gpu_count": 1,
        "gpu_name": "NVIDIA A10-24Q",
        "driver_version": "570.237",
        "vulkan_devices": [
            {
                "name": "NVIDIA A10-24Q",
                "vendor_id": 0x10DE,
                "ray_tracing_extensions": sorted(RAY_TRACING_EXTENSIONS),
            }
        ],
        "simulation_app_started": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_name": "NVIDIA A100"},
        {"gpu_count": 2},
        {"driver_version": "535.0"},
        {"vulkan_devices": []},
        {"simulation_app_started": True},
    ],
)
def test_missing_real_rtx_grid_capabilities_fail_closed(change):
    with pytest.raises(ValueError):
        validate_gpu_evidence(gpu() | change)


def prepare(spec, tmp_path):
    path = tmp_path / "spec.json"
    path.write_bytes(canonical(spec.model_dump(mode="json", by_alias=True)))
    return path


def test_preempted_batch_internal_requeue_cannot_start_the_physics_episode_again(spec, tmp_path):
    store = Store()
    spec_path = prepare(spec, tmp_path)
    calls = []

    def interrupted(*args):
        calls.append(True)
        raise KeyboardInterrupt("CPU fixture abrupt node loss")

    with pytest.raises(KeyboardInterrupt):
        run_episode(spec, spec_path, store, directory=tmp_path / "first", runner=interrupted)
    assert store.claimed and store.completion is None
    replay = run_episode(spec, spec_path, store, directory=tmp_path / "requeue", runner=interrupted)
    assert replay["outcome"] == "incomplete" and replay["accepted"] is False
    assert len(calls) == 1
    assert "completion.json" not in store.writes


def test_native_failed_or_partial_episode_preserves_proof_but_never_reports_success(spec, tmp_path):
    store = Store()
    spec_path = prepare(spec, tmp_path)

    def failed(spec, paths, directory, deadline):
        (directory / "probe.json").write_text(
            json.dumps(
                {
                    "physical_status": "timed_out",
                    "physical_task_success": False,
                    "capture": {"status": "invalid", "receipt": None},
                }
            )
        )
        (directory / "probe.log").write_text("CPU fixture native timeout; not GPU evidence")
        (directory / "acceptance.json").write_text('{"accepted":false}')
        return {"accepted": False, "outcome": "failed", "native_acceptance": "rejected"}

    result = run_episode(spec, spec_path, store, directory=tmp_path / "attempt", runner=failed)
    assert result["accepted"] is False
    assert "probe.json" in store.writes and "probe.log" in store.writes
    assert store.writes[-1] == "completion.json"
    assert store.completion["attempt_id"] == str(spec.attempt_id)


def test_no_terminal_native_receipt_is_incomplete_not_a_batch_exit_zero_success(spec, tmp_path):
    store = Store()
    spec_path = prepare(spec, tmp_path)

    def no_proof(*args):
        return {"accepted": False, "outcome": "incomplete", "native_acceptance": "missing"}

    result = run_episode(spec, spec_path, store, directory=tmp_path / "attempt", runner=no_proof)
    assert result["outcome"] == "incomplete" and result["accepted"] is False
    assert store.writes[-1] == "completion.json"


def test_failed_artifact_upload_never_publishes_a_terminal_manifest(spec, tmp_path):
    store = Store()
    path = prepare(spec, tmp_path)

    def rejected(*args):
        return {"accepted": False, "outcome": "incomplete", "native_acceptance": "missing"}

    def fail_upload(*args):
        raise OSError("CPU fixture storage upload failed")

    store.publish = fail_upload
    with pytest.raises(OSError):
        run_episode(spec, path, store, directory=tmp_path / "attempt", runner=rejected)
    assert store.claimed
    assert store.completion is None and "completion.json" not in store.writes


def test_invalid_approved_input_stops_before_any_native_runner(spec, tmp_path):
    store = Store()

    def changed_input(*args):
        raise ValueError("Approved input checksum changed.")

    store.download = changed_input
    calls = []
    result = run_episode(
        spec,
        prepare(spec, tmp_path),
        store,
        directory=tmp_path / "attempt",
        runner=lambda *args: calls.append(True),
    )
    assert calls == [] and result["outcome"] == "incomplete"
    assert "checksum" in result["failure"]


def test_cpu_import_boundary_does_not_load_optional_batch_gpu_or_model_packages():
    import subprocess

    code = """
import importlib.abc, sys
blocked = ('azure.batch', 'isaacsim', 'omni', 'torch')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('CPU import crossed optional/GPU boundary: ' + fullname)
sys.meta_path.insert(0, Boundary())
import simulation.batch
import simulation.batch_task
assert not any(name in sys.modules for name in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


def test_success_is_not_published_until_uploaded_raw_manifest_is_verified(spec, tmp_path):
    store = Store()
    path = prepare(spec, tmp_path)
    verified = []

    def verify(receipt, local_manifest):
        verified.append(receipt)
        raise ValueError(
            "The private uploaded raw manifest differs from the validated local bytes."
        )

    store.verify_raw_manifest = verify

    def accepted(spec, paths, directory, deadline):
        (directory / "raw-manifest.json").write_text('{"CPU-fixture":true}')
        return {
            "accepted": True,
            "outcome": "accepted",
            "native_acceptance": "accepted",
            "raw_manifest": {"episode_id": str(spec.attempt_id), "manifest_sha256": "b" * 64},
        }

    result = run_episode(spec, path, store, directory=tmp_path / "attempt", runner=accepted)
    assert len(verified) == 1
    assert result["accepted"] is False and result["outcome"] == "incomplete"
    assert "uploaded raw manifest differs" in result["failure"]


def test_owned_subprocess_group_timeout_stops_wrapper_and_its_child(tmp_path):
    from simulation import batch_task

    assert hasattr(batch_task, "run_bounded"), (
        "Native timeouts must terminate the owned process group."
    )
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)']);"
        f"open({str(pid_file)!r},'w').write(str(p.pid));"
        "time.sleep(120)"
    )
    import subprocess

    with (tmp_path / "process.log").open("wb") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            batch_task.run_bounded([sys.executable, "-c", code], timeout=1, cwd=tmp_path, log=log)
    child = int(pid_file.read_text())
    # SIGKILL delivery is asynchronous; pidfd readiness proves the child has exited.
    import select

    try:
        descriptor = os.pidfd_open(child)
    except ProcessLookupError:
        descriptor = None
    if descriptor is not None:
        try:
            poll = select.poll()
            poll.register(descriptor, select.POLLIN)
            assert poll.poll(2000), "Owned child survived the timeout."
        finally:
            os.close(descriptor)
    assert child != os.getpid()
