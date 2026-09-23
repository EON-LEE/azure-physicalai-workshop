"""CPU concurrency/protocol tests, not evidence of Isaac/GPU timing or physics."""

import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from runtime_support import ACTOR, OTHER, PNG
from test_demonstration_wiring import active_capture

from apps.api.errors import Problem
from learning.contract import CameraSample, FrameSample
from simulation.capture_worker import CaptureWorker
from simulation.run_isaac import SimulatorRuntime
from simulation.runtime_contracts import CaptureBinding, CaptureReceipt


def binding():
    return CaptureBinding(ACTOR.owner_key, "cell", "a" * 64, uuid4(), uuid4(), uuid4())


def sample(index, *, terminal=False):
    mono = 1_000_000_000 + index * 16_666_667
    joints = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02)
    return FrameSample(
        captured_at_utc=(datetime(2026, 9, 23, tzinfo=UTC) + timedelta(seconds=index / 60))
        .isoformat()
        .replace("+00:00", "Z"),
        monotonic_ns=mono,
        physics_step=index,
        joint_positions=joints,
        commanded_joint_targets=joints,
        images={name: CameraSample(PNG, index, index, mono) for name in ("inspection", "overview")},
        terminated=terminal,
    )


class Backend:
    def __init__(self, context, *, block_append=None, block_upload=None):
        self.context = context
        self.thread = threading.get_ident()
        self.frames = []
        self.block_append = block_append
        self.block_upload = block_upload
        self.appending = threading.Event()
        self.uploading = threading.Event()

    def append(self, frame):
        assert threading.get_ident() == self.thread
        self.appending.set()
        if self.block_append:
            assert self.block_append.wait(5), "Test failed to release writer"
        self.frames.append(frame)

    def finalize_and_upload(self, on_uploading):
        assert threading.get_ident() == self.thread
        assert self.frames[-1].terminated or self.frames[-1].truncated
        on_uploading()
        self.uploading.set()
        if self.block_upload:
            assert self.block_upload.wait(5), "Test failed to release upload"
        return CaptureReceipt(
            f"https://test.blob.core.windows.net/demonstrations/{self.context.command_id}/manifest.json",
            "b" * 64,
            str(self.context.command_id),
            len(self.frames),
        )


def test_writer_creation_append_and_upload_never_block_the_physics_thread():
    context = binding()
    release_factory, entered_factory = threading.Event(), threading.Event()
    release_upload = threading.Event()
    backends = []

    def factory():
        entered_factory.set()
        assert release_factory.wait(5), "Capture constructor blocked the test"
        backend = Backend(context, block_upload=release_upload)
        backends.append(backend)
        return backend

    worker = CaptureWorker(context, factory)
    try:
        assert entered_factory.wait(2)
        worker.append(sample(0))
        worker.append(sample(1, terminal=True))
        worker.seal()
        assert worker.snapshot().state.status == "finalizing"
        release_factory.set()
        worker_thread = worker.thread
        assert worker_thread is not threading.current_thread()
        assert worker_thread.is_alive()
        # A blocked uploader must leave the caller free to advance physics/heartbeat.
        assert worker.upload_started.wait(2)
        assert worker.snapshot().state.status == "uploading"
        release_upload.set()
        worker.thread.join(2)
        result = worker.snapshot()
        assert result.binding == context
        assert result.state.status == "ready"
        assert result.state.receipt.frame_count == 2
        assert backends[0].thread != threading.get_ident()
    finally:
        release_factory.set()
        release_upload.set()
        worker.close(2)


@pytest.mark.parametrize("limit", ["frames", "bytes"])
def test_queue_exhaustion_invalidates_instead_of_dropping_or_waiting(limit):
    context = binding()
    release, entered = threading.Event(), threading.Event()
    published = []

    def factory():
        entered.set()
        assert release.wait(5)
        published.append(Backend(context))
        return published[-1]

    worker = CaptureWorker(
        context,
        factory,
        max_pending_frames=1 if limit == "frames" else 10,
        max_pending_bytes=1024 if limit == "bytes" else 1024 * 1024,
    )
    try:
        assert entered.wait(2)
        if limit == "frames":
            worker.append(sample(0))
        frame = sample(1)
        if limit == "bytes":
            frame = replace(
                frame, images={"inspection": CameraSample(b"x" * 1025, 1, 1, 1)}
            )
        with pytest.raises(RuntimeError, match="backlog"):
            worker.append(frame)
        assert worker.snapshot().state.status == "invalid"
        with pytest.raises(RuntimeError, match="invalid"):
            worker.seal()
        release.set()
        worker.thread.join(2)
        assert not published[0].uploading.is_set()
    finally:
        release.set()
        worker.close(2)


def test_writer_failure_is_explicit_and_can_never_publish_a_ready_capture(caplog):
    context = binding()
    release = threading.Event()

    class Broken(Backend):
        def append(self, frame):
            assert release.wait(5)
            raise OSError("Fixture disk is full")

    worker = CaptureWorker(context, lambda: Broken(context))
    try:
        worker.append(sample(0))
        worker.append(sample(1, terminal=True))
        worker.seal()
        release.set()
        worker.thread.join(2)
        result = worker.snapshot().state
        assert result.status == "invalid"
        assert result.receipt is None
        assert "Capture" in result.message
        assert "Fixture disk is full" in caplog.text
    finally:
        release.set()
        worker.close(2)


def test_capture_is_bound_to_physical_result_not_current_active_command(monkeypatch):
    core, command = active_capture(monkeypatch)
    assert core.begin_motion(command.command_id)
    context = core.begin_capture(command.command_id)
    core.finish("succeeded", (0.22, -0.38, 0.2))
    completed = core.command(ACTOR.owner_key, command.command_id)
    assert completed.status == "succeeded"
    assert completed.demonstration is None
    assert core.capture(ACTOR.owner_key, command.command_id).status == "recording"
    with pytest.raises(Problem, match="not found"):
        core.capture(OTHER.owner_key, command.command_id)
    receipt = CaptureReceipt(
        "https://test.blob.core.windows.net/demonstrations/manifest.json",
        "b" * 64,
        str(command.command_id),
        2,
    )
    assert core.publish_capture(context, "finalizing")
    assert core.publish_capture(context, "uploading")
    assert core.publish_capture(context, "ready", receipt=receipt)
    result = core.command(ACTOR.owner_key, command.command_id)
    assert result.status == completed.status
    assert result.completed_at == completed.completed_at
    assert result.final_position == completed.final_position
    assert result.demonstration.manifest_sha256 == receipt.manifest_sha256
    assert core.active_command is None


@pytest.mark.parametrize("changed", ["owner", "epoch", "command_id", "capture_id"])
def test_late_or_misbound_capture_completion_is_rejected(monkeypatch, changed):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    context = core.begin_capture(command.command_id)
    core.finish("cancelled", None, "Operator stopped motion")
    stale = replace(context, **{changed: OTHER.owner_key if changed == "owner" else uuid4()})
    assert not core.publish_capture(stale, "invalid", message="Old callback")
    assert core.capture(ACTOR.owner_key, command.command_id).status == "recording"
    assert core.command(ACTOR.owner_key, command.command_id).status == "cancelled"
    assert core.active_command is None


def test_reactivation_fences_upload_completion_without_poisoning_new_scene(monkeypatch):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    context = core.begin_capture(command.command_id)
    core.finish("succeeded", (0.22, -0.38, 0.2))
    core.activate(ACTOR.owner_key, core.environment)
    assert not core.publish_capture(context, "invalid", message="Late old upload")
    assert core.capture(ACTOR.owner_key, command.command_id).status == "invalid"
    assert core.error is None
    assert core.next_action().epoch == core.epoch


def test_main_loop_physics_and_actual_heartbeat_continue_during_upload(monkeypatch, tmp_path):
    from simulation.core import StartMotion

    core, command = active_capture(monkeypatch)
    core.pending.append(StartMotion(command, command.target_station_id))
    release = threading.Event()
    clock = [10.0]

    class World:
        def play(self):
            pass

    class Hardware:
        def __init__(self):
            self.world = World()
            self.recording = None
            self.target = None
            self.steps = 0
            self.complete = False
            self.started = False

        def start(self, target, recording):
            self.target, self.recording = target, recording
            self.recording.append(sample(0))
            self.started = True

        def advance(self):
            self.steps += 1
            completed, self.complete = self.complete, False
            return completed

        def stop(self):
            pass

        def position(self):
            return (0.22, -0.38, 0.2)

        def motion_phase(self):
            return "idle"

        def capture(self, camera):
            return None

        def finish_recording(self, truncated):
            recorder, self.recording = self.recording, None
            recorder.append(sample(1, terminal=True))
            recorder.seal()

    hardware = Hardware()
    heartbeat = tmp_path / "heartbeat"
    runtime = SimulatorRuntime(
        core,
        hardware,
        heartbeat=heartbeat,
        clock=lambda: clock[0],
        capture_factory=lambda context: Backend(context, block_upload=release),
    )
    try:
        runtime.tick()
        assert runtime.capture_worker.prepared.wait(2)
        runtime.tick()
        assert hardware.started
        hardware.complete = True
        runtime.tick()
        assert runtime.capture_worker.upload_started.wait(2)
        result = core.command(ACTOR.owner_key, command.command_id)
        assert result.status == "succeeded"
        assert result.demonstration is None
        steps = hardware.steps
        clock[0] += 1.1
        runtime.tick()
        assert float(heartbeat.read_text()) == clock[0]
        assert hardware.steps > steps
        assert core.command(ACTOR.owner_key, command.command_id).completed_at == result.completed_at
        release.set()
        runtime.capture_worker.thread.join(2)
        runtime.tick()
        assert core.capture(ACTOR.owner_key, command.command_id).status == "ready"
    finally:
        release.set()
        runtime.close()


def test_cancel_before_main_thread_start_is_physically_terminal(monkeypatch, tmp_path):
    core, command = active_capture(monkeypatch)
    core.cancel(ACTOR.owner_key, command.command_id)
    hardware = SimpleNamespace(
        world=None,
        recording=None,
        stop=lambda: None,
        position=lambda: (0.35, 0.25, 0.2),
    )
    runtime = SimulatorRuntime(core, hardware, heartbeat=tmp_path / "heartbeat")
    runtime.tick()
    assert core.command(ACTOR.owner_key, command.command_id).status == "cancelled"
    assert core.active_command is None


@pytest.mark.parametrize("stop", ["cancelled", "timed_out"])
def test_completion_race_cannot_overrule_cancel_or_original_deadline(monkeypatch, stop):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    if stop == "cancelled":
        core.cancel(ACTOR.owner_key, command.command_id)
    else:
        core.deadlines[core.active_command] = datetime(2020, 1, 1, tzinfo=UTC)
    core.finish("succeeded", (0.22, -0.38, 0.2), binding=core.binding(command.command_id))
    assert core.command(ACTOR.owner_key, command.command_id).status == stop
