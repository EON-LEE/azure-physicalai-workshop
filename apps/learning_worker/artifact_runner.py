"""One resident, bounded artifact processor, independent of HTTP and deadline cancellation."""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from apps.api.errors import Problem
from apps.api.models import Principal, utcnow
from apps.learning_worker.artifact_operations import TERMINAL

log = logging.getLogger(__name__)


class ArtifactRunner:
    def __init__(self, operations, tenant_id, *, process_factory=subprocess.Popen):
        self.operations = operations
        self.tenant_id = tenant_id
        self.process_factory = process_factory
        self.stopping = threading.Event()
        self.thread = None
        self.process = None

    @staticmethod
    def terminate(process):
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def process_one(self, actor, operation_id):
        state = self.operations.recover(actor, operation_id)
        if state is None:
            return
        if state.status in TERMINAL:
            self.operations.dequeue(actor, operation_id)
            return
        claim_id = uuid4()
        state = self.operations.claim(actor, operation_id, claim_id)
        if state is None:
            return
        remaining = max(0, (state.deadline - utcnow()).total_seconds())
        end = time.monotonic() + remaining
        try:
            process = self.process_factory(
                [
                    sys.executable,
                    "-m",
                    "apps.learning_worker.artifact_task",
                    "--operation-id",
                    str(operation_id),
                    "--actor-id",
                    str(actor.object_id),
                    "--claim-id",
                    str(claim_id),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.operations.finish(
                actor,
                operation_id,
                claim_id,
                status="failed",
                error_code="artifact_process_start",
                message="The bounded artifact subprocess could not start.",
            )
            raise Problem(
                503, "artifact_process_start", "Artifact processor could not start."
            ) from exc
        self.process = process
        last_heartbeat = time.monotonic()
        kill_lock = threading.Lock()

        def stop_child():
            with kill_lock:
                self.terminate(process)

        watchdog = threading.Timer(max(0, end - time.monotonic()), stop_child)
        watchdog.daemon = True
        watchdog.start()
        try:
            while process.poll() is None:
                if self.stopping.wait(0.5) or utcnow() >= state.deadline or time.monotonic() >= end:
                    stop_child()
                    stopped = self.stopping.is_set()
                    self.operations.finish(
                        actor,
                        operation_id,
                        claim_id,
                        status="uncertain" if stopped else "timed_out",
                        error_code="artifact_worker_stopped" if stopped else "artifact_deadline",
                        message="Processor stopped; no unverified result was promoted or replayed.",
                    )
                    return
                if time.monotonic() - last_heartbeat >= 5:
                    if not self.operations.heartbeat(actor, operation_id, claim_id):
                        stop_child()
                        return
                    last_heartbeat = time.monotonic()
            final = self.operations.recover(actor, operation_id)
            if final.status != "ready":
                failure = self.operations.registry.get(
                    actor, f"artifact-operations/{operation_id}/failure.json"
                )
                if failure is not None and not isinstance(failure, dict):
                    raise Problem(503, "artifact_failure_metadata", "Failure metadata is invalid.")
                self.operations.finish(
                    actor,
                    operation_id,
                    claim_id,
                    status="timed_out" if utcnow() >= state.deadline else "failed",
                    error_code="artifact_deadline"
                    if utcnow() >= state.deadline
                    else failure.get("error_code", "artifact_processing_failed")
                    if failure
                    else "artifact_processing_failed",
                    message="No verified manifest was committed; the processor was not replayed.",
                )
        except (Problem, OSError):
            stop_child()
            self.operations.finish(
                actor,
                operation_id,
                claim_id,
                status="uncertain",
                error_code="artifact_control_lost",
                message="Processor control or heartbeat failed; no unverified success.",
            )
            raise
        finally:
            watchdog.cancel()
            self.process = None
            final = self.operations.status(actor, operation_id)
            if final is not None and final.status in TERMINAL:
                self.operations.dequeue(actor, operation_id)

    def run(self):
        while not self.stopping.is_set():
            for actor_id in self.operations.actor_ids:
                actor = Principal(tenant_id=self.tenant_id, object_id=actor_id)
                try:
                    for operation_id in self.operations.pending(actor):
                        if self.stopping.is_set():
                            return
                        self.process_one(actor, operation_id)
                except (Problem, OSError) as exc:
                    log.error(
                        "Artifact runner failed: %s", getattr(exc, "code", type(exc).__name__)
                    )
            self.stopping.wait(2)

    def start(self):
        if self.operations.enabled:
            self.thread = threading.Thread(target=self.run, name="artifact-runner", daemon=True)
            self.thread.start()

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(12)
            if self.thread.is_alive():
                raise RuntimeError("Artifact processor did not stop within its shutdown budget.")
