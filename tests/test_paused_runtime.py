"""CPU integration doubles. They verify sequencing, not actual pick/place quality."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest
from runtime_support import ACTOR, PNG
from test_paused_control import JOINTS, state
from test_paused_dispatch import paused_core as paused_core
from test_position_hold_control import Recorder

from learning.contract import AppliedControl, Scope
from learning.paused import FrozenCameraSample, FrozenPolicyObservation
from simulation.paused_runtime import PausedReferenceRuntime


@pytest.fixture
def running(paused_core):
    core, _, request = paused_core
    clock = [1_000_000_000]
    core.clock_ns = lambda: clock[0]
    core.tenant_id = str(ACTOR.tenant_id)
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command_id)

    class Hardware:
        def __init__(self):
            self.state = replace(state(), epoch=core.epoch, physics_step=60, world_time=1.0)
            self.steps = 0
            self.actions = []
            self.observations = []
            self.delay_camera = False
            self.goal = False

        def prepare_paused_reference(self, request, core):
            pass

        def frozen_physics_state(self, epoch):
            return replace(self.state, epoch=epoch)

        def _measured_tcp(self):
            return (0.35, 0.25, 0.38)

        def paused_reference_targets(self, point, closed):
            return JOINTS

        def paused_observation(self, request, core, episode, control_tick):
            if self.delay_camera:
                return None
            clock[0] += 10_000_000
            simulated = Fraction(self.state.physics_step, 60)
            stamp = datetime(2026, 9, 24, tzinfo=UTC) + timedelta(microseconds=clock[0] / 1000)
            stamp = stamp.isoformat().replace("+00:00", "Z")
            result = FrozenPolicyObservation(
                scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
                environment_id=request.environment_id,
                revision=request.revision,
                episode_id=str(request.command_id),
                epoch=str(core.epoch),
                captured_at_utc=stamp,
                monotonic_ns=clock[0],
                physics_step=self.state.physics_step,
                joint_positions=JOINTS,
                images={
                    name: FrozenCameraSample(
                        PNG,
                        self.state.physics_step,
                        self.state.physics_step,
                        clock[0],
                        stamp,
                        simulated.numerator,
                        simulated.denominator,
                    )
                    for name in ("inspection", "overview")
                },
                freeze_id=str(episode.freeze_id),
                state_revision=request.state_revision + control_tick,
                control_tick=control_tick,
                control_profile_sha256=core.paused_profile.sha256,
                observation_started_ns=episode.interval_started_ns,
                joint_sample_ns=clock[0],
                simulation_time_numerator=simulated.numerator,
                simulation_time_denominator=simulated.denominator,
            )
            self.observations.append(result)
            return result

        def apply_paused_tick(self, targets):
            self.actions.append(targets)
            clock[0] += 10_000_000
            self.steps += 1
            self.state = replace(
                self.state,
                physics_step=self.state.physics_step + 1,
                world_time=self.state.world_time + 1 / 60,
            )
            return AppliedControl(
                self.state.physics_step,
                clock[0],
                targets,
                (0.0,) * 9,
                (1.0,) * 7 + (0.0, 0.0),
            )

        def paused_goal_reached(self):
            return self.goal

    hardware = Hardware()
    recorder = Recorder()
    runtime = PausedReferenceRuntime(core, request, hardware, recorder)
    return runtime, core, request, hardware, recorder, clock


def test_pending_camera_leaves_physics_frozen_and_cancel_wins_before_any_action(running):
    runtime, core, request, hardware, _, clock = running
    hardware.delay_camera = True
    for _ in range(5):
        runtime.advance()
        clock[0] += 10_000_000
    assert hardware.steps == 0 and not hardware.actions
    core.cancel(ACTOR.owner_key, request.command_id)
    with pytest.raises(RuntimeError):
        runtime.advance()
    assert hardware.steps == 0


def test_actual_intervals_keep_cameras_before_six_actions_and_seal_only_complete_data(running):
    runtime, _, _, hardware, recorder, _ = running
    for _ in range(24):
        runtime.advance()
        if hardware.steps >= 12:
            break
    assert hardware.steps == 12
    assert len(hardware.observations) == 2
    assert [item.physics_step for item in hardware.observations] == [60, 66]
    assert len(hardware.actions) == 12
    runtime.stop()
    runtime.finish_recording(truncated=True)
    assert recorder.sealed
    assert len(recorder.frames) == 2
    assert recorder.frames[-1].truncated
    assert all(len(frame.applied_controls) == 6 for frame in recorder.frames)


def test_terminal_or_finalizing_episode_does_not_resume_background_physics(running):
    runtime, _, _, hardware, _, _ = running
    hardware.goal = True
    for _ in range(12):
        if runtime.advance():
            break
    assert hardware.steps == 6
    for _ in range(10):
        assert runtime.advance() is False
    assert hardware.steps == 6


def test_partial_hold_cancellation_cannot_publish_fabricated_remaining_ticks(running):
    runtime, core, request, hardware, recorder, _ = running
    for _ in range(8):
        runtime.advance()
        if hardware.steps:
            break
    assert hardware.steps == 1
    core.cancel(ACTOR.owner_key, request.command_id)
    runtime.stop()
    with pytest.raises(RuntimeError, match="partial"):
        runtime.finish_recording(truncated=True)
    assert recorder.invalid and not recorder.sealed


def test_invalid_completed_hold_cannot_be_dropped_to_publish_only_earlier_valid_frames(running):
    runtime, _, _, hardware, recorder, clock = running
    for _ in range(12):
        runtime.advance()
        if hardware.steps == 6:
            break
    for _ in range(12):
        runtime.advance()
        if hardware.steps == 11:
            break
    original = hardware.apply_paused_tick

    def over_budget(targets):
        value = original(targets)
        clock[0] += 2_001_000_000
        return value

    hardware.apply_paused_tick = over_budget
    with pytest.raises(RuntimeError):
        runtime.advance()
    assert hardware.steps == 12
    runtime.stop()
    with pytest.raises(RuntimeError, match="unrecorded|invalid"):
        runtime.finish_recording(truncated=True)
    assert recorder.invalid and not recorder.sealed


def test_main_runtime_dispatches_real_paused_reference_and_finishes_capture_off_thread(
    paused_core,
    monkeypatch,
    tmp_path,
):
    import time

    from test_demonstration_wiring import active_capture

    from simulation.run_isaac import SimulatorRuntime

    active_capture(monkeypatch)
    core, _, request = paused_core
    core.tenant_id = str(ACTOR.tenant_id)
    state_value = replace(state(), epoch=core.epoch, physics_step=60, world_time=1.0)

    class World:
        def play(self):
            raise AssertionError("Paused completion must not resume physics")

    class Hardware:
        def __init__(self):
            self.paused_driver = None
            self.recording = None
            self.world = World()
            self.target = None
            self.steps = 0
            self.state = state_value
            self.scene_epoch = core.epoch

        def prepare_paused_reference(self, command, protocol):
            self.target = command.target_station_id

        def frozen_physics_state(self, epoch):
            return replace(self.state, epoch=epoch)

        def _measured_tcp(self):
            return (0.35, 0.25, 0.38)

        def paused_reference_targets(self, point, closed):
            return self.state.joint_positions

        def paused_observation(self, command, protocol, episode, control_tick):
            fraction = Fraction(self.state.physics_step, 60)
            stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            mono = time.monotonic_ns()
            return FrozenPolicyObservation(
                scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
                environment_id=command.environment_id,
                revision=command.revision,
                episode_id=str(command.command_id),
                epoch=str(core.epoch),
                captured_at_utc=stamp,
                monotonic_ns=mono,
                physics_step=self.state.physics_step,
                joint_positions=self.state.joint_positions,
                images={
                    name: FrozenCameraSample(
                        PNG,
                        self.state.physics_step,
                        self.state.physics_step,
                        mono,
                        stamp,
                        fraction.numerator,
                        fraction.denominator,
                    )
                    for name in ("inspection", "overview")
                },
                freeze_id=str(episode.freeze_id),
                state_revision=core.state_revision,
                control_tick=control_tick,
                control_profile_sha256=core.paused_profile.sha256,
                observation_started_ns=episode.interval_started_ns,
                joint_sample_ns=mono,
                simulation_time_numerator=fraction.numerator,
                simulation_time_denominator=fraction.denominator,
            )

        def apply_paused_tick(self, targets):
            self.steps += 1
            self.state = replace(
                self.state,
                physics_step=self.state.physics_step + 1,
                world_time=self.state.world_time + 1 / 60,
            )
            return AppliedControl(
                self.state.physics_step, time.monotonic_ns(), targets, (0.0,) * 9, (0.0,) * 9
            )

        def paused_goal_reached(self):
            return self.steps >= 12

        def advance(self):
            return self.paused_driver.advance() if self.paused_driver else False

        def stop(self):
            if self.paused_driver:
                self.paused_driver.stop()

        def finish_recording(self, truncated):
            self.paused_driver.finish_recording(truncated=truncated)
            self.recording = None

        def position(self):
            return self.state.object_position

        def motion_phase(self):
            return "idle"

        def capture(self, camera):
            return None

    from simulation import run_isaac
    from simulation.paused_capture import PausedDemonstration

    monkeypatch.setattr(
        run_isaac, "PausedDemonstration", lambda capture: PausedDemonstration(capture, tmp_path)
    )
    uploaded = []

    class Blob:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_container_client(self, name):
            return self

        def upload_blob(self, name, stream, **kwargs):
            uploaded.append(name)

    monkeypatch.setattr("simulation.demonstrations.ManagedIdentityCredential", Blob)
    monkeypatch.setattr("simulation.demonstrations.BlobServiceClient", Blob)
    hardware = Hardware()
    runtime = SimulatorRuntime(core, hardware, heartbeat=tmp_path / "heartbeat")
    try:
        core.dispatch_simulation_episode(ACTOR.owner_key, request)
        runtime.tick()
        assert runtime.capture_worker.prepared.wait(2)
        for _ in range(25):
            runtime.tick()
            if core.active_command is None:
                break
        assert hardware.steps == 12
        assert core.command(ACTOR.owner_key, request.command_id).status == "succeeded"
        runtime.capture_worker.thread.join(5)
        runtime.tick()
        assert core.capture(ACTOR.owner_key, request.command_id).status == "ready"
        assert uploaded[-1].endswith("manifest.json")
        for _ in range(5):
            runtime.tick()
        assert hardware.steps == 12
    finally:
        runtime.close()
