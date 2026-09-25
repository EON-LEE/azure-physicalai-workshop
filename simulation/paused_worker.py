"""One bounded policy request outside the Isaac thread, with explicit stale-result fencing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from copy import deepcopy
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar("T")
log = logging.getLogger(__name__)


class PausedPolicyWorker(Generic[T]):
    def __init__(self, *, clock_ns: Callable[[], int]) -> None:
        self.clock_ns = clock_ns
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.freeze_id: UUID | None = None
        self.deadline_ns = 0
        self.result: T | None = None
        self.error: Exception | None = None
        self.finished = False
        self.cancelled = False

    def submit(self, freeze_id: UUID, predict: Callable[[], T], *, deadline_ns: int) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError(
                    "A policy request is still pending; no second worker is allowed."
                )
            if self.freeze_id is not None and not self.cancelled:
                raise RuntimeError("The previous pending result has not been consumed.")
            now = self.clock_ns()
            if not isinstance(freeze_id, UUID) or not 0 < deadline_ns - now <= 2_000_000_000:
                raise ValueError(
                    "A frozen policy request needs its original bounded wall deadline."
                )
            self.freeze_id, self.deadline_ns = freeze_id, deadline_ns
            self.result, self.error = None, None
            self.finished = self.cancelled = False
            self.thread = threading.Thread(
                target=self._run, args=(predict,), name=f"paused-policy-{freeze_id}", daemon=True
            )
            self.thread.start()

    def poll(self, freeze_id: UUID) -> T | None:
        with self.lock:
            if self.cancelled or freeze_id != self.freeze_id:
                self.cancel()
                raise RuntimeError("The policy result belongs to an invalidated freeze.")
            if self.clock_ns() >= self.deadline_ns:
                self.cancel()
                raise RuntimeError("The original policy request wall deadline expired.")
            if not self.finished:
                return None
            self.freeze_id = None
            if self.error is not None:
                raise RuntimeError(
                    f"The pending policy request failed: {self.error}"
                ) from self.error
            if self.result is None:
                raise RuntimeError("The policy worker exited without a valid result.")
            result, self.result = self.result, None
            return result

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            self.result = None

    def _run(self, predict: Callable[[], T]) -> None:
        try:
            result = predict()
            if result is None:
                raise ValueError("A policy prediction cannot be an empty success result.")
            with self.lock:
                if not self.cancelled:
                    self.result = deepcopy(result)
        except (ValueError, RuntimeError, TypeError, OSError) as exc:
            log.exception("Paused policy request failed without actuator fallback")
            with self.lock:
                self.error = exc
        finally:
            with self.lock:
                self.finished = True
                if self.result is None and self.error is None and not self.cancelled:
                    log.error("Paused policy worker terminated unexpectedly without a result")
                    self.error = RuntimeError("Policy worker terminated without a result.")
