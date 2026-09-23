"""CPU IPC concurrency fixtures, not a GPU model or scheduler-timing proof."""

import threading
from uuid import uuid4

import pytest

from simulation.paused_worker import PausedPolicyWorker


def test_pending_model_does_not_block_main_thread_heartbeat_or_allow_a_second_request():
    now = [1_000_000_000]
    worker = PausedPolicyWorker(clock_ns=lambda: now[0])
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    freeze = uuid4()

    def model():
        assert threading.get_ident() != caller
        entered.set()
        assert release.wait(5)
        return ("actual-returned-targets",)

    try:
        worker.submit(freeze, model, deadline_ns=now[0] + 2_000_000_000)
        assert entered.wait(1)
        for _ in range(10):
            now[0] += 50_000_000
            assert worker.poll(freeze) is None
        with pytest.raises(RuntimeError, match="pending"):
            worker.submit(uuid4(), model, deadline_ns=now[0] + 1_000_000_000)
        release.set()
        worker.thread.join(1)
        assert worker.poll(freeze) == ("actual-returned-targets",)
        with pytest.raises(RuntimeError):
            worker.poll(freeze)
    finally:
        release.set()
        worker.cancel()


@pytest.mark.parametrize("cause", ["cancel", "timeout", "freeze_changed"])
def test_late_worker_result_cannot_reauthorize_cancelled_expired_or_different_freeze(cause):
    now = [1_000_000_000]
    worker = PausedPolicyWorker(clock_ns=lambda: now[0])
    release = threading.Event()
    freeze = uuid4()

    def model():
        assert release.wait(5)
        return (0.0,) * 9

    try:
        worker.submit(freeze, model, deadline_ns=now[0] + 2_000_000_000)
        if cause == "cancel":
            worker.cancel()
        elif cause == "timeout":
            now[0] += 2_000_000_001
        release.set()
        worker.thread.join(1)
        with pytest.raises(RuntimeError):
            worker.poll(uuid4() if cause == "freeze_changed" else freeze)
    finally:
        release.set()
        worker.cancel()


def test_model_errors_are_propagated_not_replaced_with_reference_actions(caplog):
    worker = PausedPolicyWorker(clock_ns=lambda: 1_000_000_000)
    freeze = uuid4()

    def broken_model():
        raise OSError("CPU fixture IPC disconnected")

    worker.submit(freeze, broken_model, deadline_ns=3_000_000_000)
    worker.thread.join(1)
    with pytest.raises(RuntimeError, match="policy"):
        worker.poll(freeze)
    assert "CPU fixture IPC disconnected" in caplog.text


def test_cancelled_blocking_worker_keeps_its_slot_until_it_really_exits():
    worker = PausedPolicyWorker(clock_ns=lambda: 1_000_000_000)
    release = threading.Event()
    entered = threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return 1

    try:
        worker.submit(uuid4(), blocked, deadline_ns=3_000_000_000)
        assert entered.wait(1)
        worker.cancel()
        with pytest.raises(RuntimeError, match="pending"):
            worker.submit(uuid4(), lambda: 2, deadline_ns=3_000_000_000)
        release.set()
        worker.thread.join(1)
        current = uuid4()
        worker.submit(current, lambda: 3, deadline_ns=3_000_000_000)
        worker.thread.join(1)
        assert worker.poll(current) == 3
    finally:
        release.set()
        worker.cancel()
