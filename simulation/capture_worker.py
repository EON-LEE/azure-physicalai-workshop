"""Bounded off-thread persistence for immutable, main-thread physics observations."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Protocol

from azure.core.exceptions import AzureError

from apps.api.models import DemonstrationResult
from learning.contract import FrameSample
from simulation.runtime_contracts import (
    CaptureBinding,
    CaptureReceipt,
    CaptureStatus,
    CaptureUpdate,
)

log = logging.getLogger(__name__)


class CaptureBackend(Protocol):
    def append(self, sample: FrameSample) -> None: ...

    def finalize_and_upload(self, on_uploading: Callable[[], None]) -> CaptureReceipt: ...


class CaptureWorker:
    def __init__(
        self,
        binding: CaptureBinding,
        factory: Callable[[], CaptureBackend],
        *,
        max_pending_frames: int = 64,
        max_pending_bytes: int = 64 * 1024 * 1024,
        terminal_sink: Callable[[CaptureUpdate], None] | None = None,
    ) -> None:
        if not 1 <= max_pending_frames <= 600 or not 1024 <= max_pending_bytes <= 256 * 1024**2:
            raise ValueError("Capture queue budgets are outside the bounded runtime limits.")
        self.binding = binding
        self.factory = factory
        self.terminal_sink = terminal_sink
        self.max_pending_frames = max_pending_frames
        self.max_pending_bytes = max_pending_bytes
        self.condition = threading.Condition()
        self.queue: deque[tuple[FrameSample, int]] = deque()
        self.pending_frames = 0
        self.pending_bytes = 0
        self.terminal_received = False
        self.prepared = threading.Event()
        self.upload_started = threading.Event()
        self.state = CaptureStatus(
            capture_id=binding.capture_id,
            command_id=binding.command_id,
            epoch=binding.epoch,
            status="recording",
        )
        self.thread = threading.Thread(
            target=self._run, name=f"capture-{binding.capture_id}", daemon=True
        )
        self.thread.start()

    def append(self, sample: FrameSample) -> None:
        # Include bounded non-image metadata and the six-tick actuator evidence in the budget.
        size = sum(len(image.png) for image in sample.images.values()) + 16384
        with self.condition:
            if self.state.status != "recording" or self.terminal_received:
                raise RuntimeError(f"Capture is {self.state.status}; no more samples are accepted.")
            if (
                self.pending_frames >= self.max_pending_frames
                or self.pending_bytes + size > self.max_pending_bytes
            ):
                self.invalidate("Capture backlog exceeded its bounded memory budget.")
                raise RuntimeError(self.state.message)
            self.pending_frames += 1
            self.pending_bytes += size
            self.terminal_received = sample.terminated or sample.truncated
            self.queue.append((deepcopy(sample), size))
            self.condition.notify()

    def seal(self) -> None:
        with self.condition:
            if self.state.status != "recording":
                raise RuntimeError(f"Capture is {self.state.status}; it cannot be sealed.")
            if not self.terminal_received:
                self.invalidate("Capture has no actual terminal interval.")
                raise RuntimeError(self.state.message)
            self.state = self.state.model_copy(update={"status": "finalizing"})
            self.condition.notify()

    def snapshot(self) -> CaptureUpdate:
        with self.condition:
            return CaptureUpdate(self.binding, self.state.model_copy(deep=True))

    def invalidate(self, message: str) -> None:
        with self.condition:
            if self.state.status in {"ready", "invalid"}:
                return
            self.state = self.state.model_copy(
                update={"status": "invalid", "receipt": None, "message": message}
            )
            self.queue.clear()
            self.condition.notify_all()

    def close(self, timeout: float = 0.0) -> None:
        self.invalidate("Capture worker stopped before publication.")
        self.thread.join(timeout)

    def _uploading(self) -> None:
        with self.condition:
            if self.state.status == "invalid":
                raise RuntimeError("Capture invalidated before manifest publication.")
            self.state = self.state.model_copy(update={"status": "uploading"})
            self.upload_started.set()

    def _run(self) -> None:
        try:
            backend = self.factory()
            self.prepared.set()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.queue or self.state.status != "recording")
                    if self.state.status == "invalid":
                        return
                    if not self.queue:
                        break
                    frame, size = self.queue.popleft()
                backend.append(frame)
                with self.condition:
                    self.pending_frames -= 1
                    self.pending_bytes -= size
            receipt = backend.finalize_and_upload(self._uploading)
            result = DemonstrationResult(status="uploaded", **asdict(receipt))
            if result.episode_id != self.binding.command_id:
                raise ValueError("Published capture does not match its bound command.")
            with self.condition:
                if self.state.status == "invalid":
                    return
                terminal = self.state.model_copy(update={"status": "ready", "receipt": result})
            if self.terminal_sink is not None:
                self.terminal_sink(CaptureUpdate(self.binding, terminal))
            with self.condition:
                self.state = terminal
        except (ValueError, RuntimeError, TypeError, OSError, AzureError):
            log.exception("Capture worker failed; no ready dataset will be reported")
            self.invalidate("Capture persistence or validation failed; no dataset was published.")
        finally:
            if self.snapshot().state.status not in {"ready", "invalid"}:
                log.error("Capture worker exited without a terminal publication state")
                self.invalidate("Capture worker exited unexpectedly before publication.")
            if self.terminal_sink is not None and self.snapshot().state.status == "invalid":
                try:
                    self.terminal_sink(self.snapshot())
                except (ValueError, RuntimeError, TypeError, OSError):
                    log.exception("Failed capture could not be persisted to its private journal")
