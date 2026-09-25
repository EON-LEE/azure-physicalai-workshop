from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.models import utcnow
from apps.learning_worker.registry import BlobRegistry
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_worker import specification
from tests.test_worker_deadlines import ConditionalBlobs


def operation_spec():
    from apps.api.artifact_models import ArtifactWork

    spec, _ = specification()
    now = utcnow()
    return ArtifactWork(
        id=uuid4(),
        actor=ACTOR,
        operation="dataset",
        project=spec.project,
        target_id=uuid4(),
        created_at=now,
        deadline=now + timedelta(seconds=600),
        max_bytes=20 * 1024**3,
        max_files=100000,
        captures=spec.dataset.captures,
    )


def operations(*, enabled=True):
    from apps.learning_worker.artifact_operations import ArtifactOperations

    registry = BlobRegistry.__new__(BlobRegistry)
    registry.container = ConditionalBlobs()
    result = ArtifactOperations(
        registry,
        enabled=enabled,
        actor_ids=frozenset({ACTOR.object_id}),
        maximum_seconds=1800,
        capture_bytes=4 * 1024**3,
        dataset_bytes=20 * 1024**3,
    )
    return result, registry


def test_disabled_or_unlisted_artifact_work_cannot_claim_any_queue_item():
    ops, registry = operations(enabled=False)
    work = operation_spec()
    with pytest.raises(Problem) as error:
        ops.begin(ACTOR, work)
    assert error.value.code == "artifact_operations_disabled"
    assert registry.container.items == {}
    ops.enabled = True
    with pytest.raises(Problem):
        ops.begin(OTHER, work)
    assert registry.container.items == {}


def test_begin_is_a_durable_metadata_claim_not_inline_validation_or_model_work():
    ops, registry = operations()
    work = operation_spec()
    first = ops.begin(ACTOR, work)
    assert first.status == "queued" and first.result is None
    second = ops.begin(ACTOR, work)
    assert first == second
    assert len(registry.container.items) == 3
    assert ops.status(ACTOR, work.id).status == "queued"
    assert ops.status(OTHER, work.id) is None


def test_reusing_operation_id_cannot_change_frozen_inputs_or_budgets():
    ops, _ = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    with pytest.raises(Problem) as failure:
        ops.begin(ACTOR, work.model_copy(update={"max_bytes": work.max_bytes - 1}))
    assert failure.value.status == 409
    assert ops.status(ACTOR, work.id).deadline == work.deadline


def test_only_one_claimant_can_start_heavy_processing_and_restart_is_uncertain(monkeypatch):
    ops, _ = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    first = ops.claim(ACTOR, work.id, uuid4())
    assert first is not None and first.status == "running"
    assert ops.claim(ACTOR, work.id, uuid4()) is None
    monkeypatch.setattr(
        "apps.learning_worker.artifact_operations.utcnow",
        lambda: first.heartbeat_at + timedelta(seconds=31),
    )
    assert ops.recover(ACTOR, work.id).status == "uncertain"
    assert ops.claim(ACTOR, work.id, uuid4()) is None


def test_queued_time_counts_and_expired_work_never_starts(monkeypatch):
    ops, _ = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    monkeypatch.setattr("apps.learning_worker.artifact_operations.utcnow", lambda: work.deadline)
    assert ops.claim(ACTOR, work.id, uuid4()) is None
    assert ops.status(ACTOR, work.id).status == "timed_out"


def test_complete_result_requires_matching_verified_manifest_before_ready():
    from apps.api.artifact_models import ArtifactResult

    ops, registry = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    claim = ops.claim(ACTOR, work.id, uuid4())
    result = ArtifactResult(artifact_id=work.target_id, manifest_sha256="a" * 64)
    registry.put(
        ACTOR,
        f"artifact-operations/{work.id}/result.json",
        {
            "work_sha256": work.sha256,
            "result": result.model_dump(mode="json"),
            "completed_at": utcnow().isoformat(),
        },
    )
    with pytest.raises(Problem):
        ops.recover(ACTOR, work.id)
    registry.put(
        ACTOR,
        f"artifacts/{work.target_id}/index.json",
        {
            "owner_key": ACTOR.owner_key,
            "manifest_sha256": "a" * 64,
            "files": {"manifest.json": "a" * 64},
        },
    )
    ready = ops.recover(ACTOR, work.id)
    assert ready.status == "ready" and ready.result == result
    assert ready.claim_id == claim.claim_id


def test_disk_and_network_budget_fail_before_a_host_is_filled(monkeypatch):
    from apps.learning_worker.artifact_operations import ArtifactBudget

    monkeypatch.setattr(
        "apps.learning_worker.artifact_operations.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1024),
    )
    budget = ArtifactBudget(utcnow() + timedelta(seconds=60), max_bytes=100, max_files=2)
    with pytest.raises(Problem) as failure:
        budget.require_disk(500, ".")
    assert failure.value.code == "artifact_disk_budget"
    budget.consume(80)
    with pytest.raises(Problem) as failure:
        budget.consume(21)
    assert failure.value.code == "artifact_byte_budget"


def test_resident_runner_stops_its_exact_process_group_before_persisting_uncertainty(monkeypatch):
    from apps.learning_worker.artifact_runner import ArtifactRunner

    ops, _ = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    events = []

    class Process:
        pid = 123456789
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert self.returncode is not None
            return self.returncode

    process = Process()
    runner = ArtifactRunner(ops, ACTOR.tenant_id)

    def spawn(args, **kwargs):
        events.append("start")
        assert args[1:3] == ["-m", "apps.learning_worker.artifact_task"]
        assert kwargs["start_new_session"] is True
        runner.stopping.set()
        return process

    def stop(pid, signal):
        assert pid == process.pid
        events.append("kill")
        process.returncode = -15

    finish = ops.finish

    def persist(*args, **kwargs):
        assert process.poll() is not None
        events.append("persist")
        return finish(*args, **kwargs)

    runner.process_factory = spawn
    monkeypatch.setattr("apps.learning_worker.artifact_runner.os.killpg", stop)
    monkeypatch.setattr(ops, "finish", persist)
    runner.process_one(ACTOR, work.id)
    assert events == ["start", "kill", "persist"]
    assert ops.status(ACTOR, work.id).status == "uncertain"
    assert ops.pending(ACTOR) == []


def test_pending_queue_is_owner_scoped_and_never_lists_customer_history():
    ops, registry = operations()
    work = operation_spec()
    ops.begin(ACTOR, work)
    registry.put(ACTOR, "jobs/not-artifact/history.json", {"private": "unrelated"})
    assert ops.pending(ACTOR) == [work.id]
    assert ops.pending(OTHER) == []


def test_real_cpu_subprocess_expires_before_timeout_state_is_recorded():
    import subprocess
    import sys

    from apps.learning_worker.artifact_runner import ArtifactRunner

    ops, _ = operations()
    original = operation_spec()
    work = original.model_copy(update={"deadline": original.created_at + timedelta(seconds=2)})
    ops.begin(ACTOR, work)
    processes = []

    def spawn(args, **kwargs):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    runner = ArtifactRunner(ops, ACTOR.tenant_id, process_factory=spawn)
    try:
        runner.process_one(ACTOR, work.id)
        assert len(processes) == 1 and processes[0].poll() is not None
        assert ops.status(ACTOR, work.id).status == "timed_out"
        assert ops.pending(ACTOR) == []
    finally:
        for process in processes:
            if process.poll() is None:
                runner.terminate(process)
