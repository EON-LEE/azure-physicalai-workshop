"""CPU persistence integration: real raw-v3 writer and upload adapter, mocked Azure/GPU identity."""

import json
from dataclasses import replace

from runtime_support import ACTOR
from test_demonstration_wiring import active_capture
from test_paused_capture_worker import frame
from test_paused_dispatch import paused_core as paused_core

from learning.contract import Scope
from learning.paused.capture import PausedEpisodeBudget, validate_dataset
from simulation.capture_worker import CaptureWorker
from simulation.paused_capture import PausedDemonstration, prepare_paused_capture


def test_bound_v3_writer_stays_on_worker_and_uploads_validated_manifest_last(
    paused_core,
    monkeypatch,
    tmp_path,
):
    active_capture(monkeypatch)
    core, _, request = paused_core
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command_id)
    key = core.active_command
    budget = PausedEpisodeBudget(
        core.command_started_ns[key],
        core.monotonic_deadlines[key],
        60,
        1860,
    )
    prepared = prepare_paused_capture(core, request.command_id, budget)
    captured = core.begin_capture(request.command_id)
    uploads = []

    class Azure:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_container_client(self, name):
            assert name == "demonstrations"
            return self

        def upload_blob(self, name, stream, **kwargs):
            assert kwargs["overwrite"] is False
            uploads.append((name, stream.read()))

    monkeypatch.setattr("simulation.demonstrations.ManagedIdentityCredential", Azure)
    monkeypatch.setattr("simulation.demonstrations.BlobServiceClient", Azure)
    worker = CaptureWorker(captured, lambda: PausedDemonstration(prepared, tmp_path))
    try:
        for index in range(2):
            original = frame(index, terminated=index == 1)
            shift = budget.started_ns - 1_000_000_000
            old = original.observation
            observed = replace(
                old,
                scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
                environment_id=request.environment_id,
                revision=request.revision,
                episode_id=str(request.command_id),
                epoch=str(request.epoch),
                control_profile_sha256=core.paused_profile.sha256,
                observation_started_ns=old.observation_started_ns + shift,
                monotonic_ns=old.monotonic_ns + shift,
                joint_sample_ns=old.joint_sample_ns + shift,
                images={
                    name: replace(image, monotonic_ns=image.monotonic_ns + shift)
                    for name, image in old.images.items()
                },
            )
            worker.append(
                replace(
                    original,
                    observation=observed,
                    interval_deadline_ns=original.interval_deadline_ns + shift,
                    hold_started_ns=original.hold_started_ns + shift,
                    hold_deadline_ns=original.hold_deadline_ns + shift,
                    applied_controls=tuple(
                        replace(control, monotonic_ns=control.monotonic_ns + shift)
                        for control in original.applied_controls
                    ),
                )
            )
        worker.seal()
        worker.thread.join(5)
        assert worker.snapshot().state.status == "ready"
        assert uploads[-1][0].endswith("/manifest.json")
        manifest = json.loads(uploads[-1][1])
        assert manifest["schema"] == "physicalai.demonstrations/v3"
        assert manifest["real_time_admission"] is False
        assert manifest["purpose"] == "integration"
        assert manifest["criteria_sha256"] == prepared.authority.criteria_sha256
        validated = validate_dataset(
            tmp_path / ACTOR.owner_key / str(request.command_id),
            expected_scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
        )
        assert validated.episodes[0].metadata["frame_count"] == 2
    finally:
        worker.close(5)
