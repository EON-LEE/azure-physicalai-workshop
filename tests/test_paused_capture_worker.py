"""CPU immutable snapshot/persistence threading checks for raw v3, not capture evidence."""

import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from test_capture_lifecycle import Backend, binding

from learning.contract import AppliedControl
from learning.paused import PausedFrameSample
from simulation.capture_worker import CaptureWorker
from tests.learning.test_paused_contract import observation


def frame(index, *, terminated=False):
    first = observation()
    offset = index * 2_000_000_000
    captured = datetime(2026, 9, 24, 0, 0, 1, tzinfo=UTC) + timedelta(seconds=index * 2)
    observed = replace(
        first,
        captured_at_utc=captured.isoformat().replace("+00:00", "Z"),
        monotonic_ns=first.monotonic_ns + offset,
        observation_started_ns=first.observation_started_ns + offset,
        joint_sample_ns=first.joint_sample_ns + offset,
        physics_step=first.physics_step + 6 * index,
        control_tick=index,
        state_revision=first.state_revision + index,
        freeze_id=f"freeze-{index}",
        simulation_time_numerator=10 + index,
        simulation_time_denominator=10,
        images={
            name: replace(
                camera,
                rendering_frame=camera.rendering_frame + index,
                physics_step=camera.physics_step + 6 * index,
                monotonic_ns=camera.monotonic_ns + offset,
                captured_at_utc=(captured - timedelta(milliseconds=100))
                .isoformat()
                .replace("+00:00", "Z"),
                simulation_time_numerator=10 + index,
                simulation_time_denominator=10,
            )
            for name, camera in first.images.items()
        },
    )
    return PausedFrameSample(
        observation=observed,
        commanded_joint_targets=observed.joint_positions,
        applied_controls=tuple(
            AppliedControl(
                observed.physics_step + tick,
                observed.monotonic_ns + tick * 100_000_000,
                observed.joint_positions,
                (0.0,) * 9,
                (1.0,) * 7 + (0.0, 0.0),
            )
            for tick in range(1, 7)
        ),
        interval_deadline_ns=observed.observation_started_ns + 5_000_000_000,
        hold_started_ns=observed.monotonic_ns,
        hold_deadline_ns=observed.monotonic_ns + 2_000_000_000,
        terminated=terminated,
    )


def test_raw_v3_frozen_mapping_and_actual_control_batch_cross_the_bounded_worker():
    context = binding()
    created = []

    def factory():
        backend = Backend(context)
        created.append(backend)
        return backend

    worker = CaptureWorker(context, factory)
    try:
        first = frame(0)
        second = frame(1, terminated=True)
        worker.append(first)
        worker.append(second)
        worker.seal()
        worker.thread.join(2)
        assert worker.snapshot().state.status == "ready"
        assert created[0].thread != threading.get_ident()
        assert created[0].frames[0].observation.sha256 == first.observation.sha256
        assert created[0].frames[1].applied_controls == second.applied_controls
        assert first.observation.images["inspection"].monotonic_ns == 1_900_000_000
    finally:
        worker.close(2)


def test_producer_mutation_after_enqueue_cannot_change_raw_v3_frame_or_hash():
    context = binding()
    release = threading.Event()
    created = []

    def factory():
        assert release.wait(5)
        backend = Backend(context)
        created.append(backend)
        return backend

    first = frame(0)
    original_hash = first.observation.sha256
    pixel_buffer = bytearray(first.observation.images["inspection"].png)
    producer_images = dict(first.observation.images)
    producer_images["inspection"] = replace(producer_images["inspection"], png=pixel_buffer)
    first = replace(first, observation=replace(first.observation, images=producer_images))
    worker = CaptureWorker(context, factory)
    try:
        worker.append(first)
        worker.append(frame(1, terminated=True))
        worker.seal()
        pixel_buffer[0] ^= 1
        producer_images.clear()
        release.set()
        worker.thread.join(2)
        assert worker.snapshot().state.status == "ready"
        actual = created[0].frames[0]
        assert actual.observation.sha256 == original_hash
        assert type(actual.observation.images["inspection"].png) is bytes
        assert set(actual.observation.images) == {"inspection", "overview"}
    finally:
        release.set()
        worker.close(2)
