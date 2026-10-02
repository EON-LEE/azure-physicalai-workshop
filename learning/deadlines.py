from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from learning.common import ContractError, require, utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def validate_deadline(value: object) -> datetime:
    deadline = utc(value)
    require(
        deadline.isoformat().replace("+00:00", "Z") == value,
        "Job deadline must use canonical UTC ending Z",
    )
    return deadline


class JobDeadline:
    """An immutable UTC cutoff; a wall-clock rollback cannot extend this process's budget."""

    def __init__(
        self,
        value: str,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.value = value
        self.expires_at = validate_deadline(value)
        self._clock = clock or utcnow
        self._monotonic = monotonic or time.monotonic
        self._cutoff = self._monotonic() + (self.expires_at - self._clock()).total_seconds()

    def remaining_seconds(self) -> float:
        return min(
            (self.expires_at - self._clock()).total_seconds(),
            self._cutoff - self._monotonic(),
        )

    def check(self) -> float:
        remaining = self.remaining_seconds()
        require(remaining > 0, "Approved job deadline expired; no workload admission or success")
        return remaining


def run_bounded(command: Sequence[str], deadline: JobDeadline) -> None:
    """Supervise a Linux component and its descendants, including blocking CUDA/file phases."""
    require(os.name == "posix", "Bounded production components require Linux process groups")
    deadline.check()
    with subprocess.Popen(list(command), start_new_session=True) as process:
        try:
            while True:
                try:
                    remaining = deadline.check()
                except ContractError:
                    # Do not reap the group leader before killing its descendants: its PID
                    # must remain reserved while it is used as the process-group identifier.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
                try:
                    code = process.wait(timeout=min(remaining, 1.0))
                except subprocess.TimeoutExpired:
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, command)
                deadline.check()
                return
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
