"""CPU model/SDK doubles exercise actual runtime wiring; never GPU or learning-quality proof."""

import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from types import SimpleNamespace

import pytest
from runtime_support import ACTOR, PNG
from test_capture_lifecycle import Backend
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_teaching_runtime import teaching as teaching

from learning.contract import DemonstrationSource, Scope
from learning.paused import FrozenCameraSample, FrozenPolicyObservation
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.paused_contracts import ResolvedSimulationAuthorization
from simulation.run_isaac import SimulatorRuntime


@pytest.fixture
def learned(hardware, paused_core, monkeypatch, tmp_path):
    cell, _, _, clock = hardware
    core, _, request = paused_core
    core.clock_ns = lambda: clock[1]
    core.tenant_id = str(ACTOR.tenant_id)
    request = request.model_copy(
        update={
            "controller": "learned",
            "authorization_kind": "evaluation_grant",
            "policy_type": "smolvla",
            "model_sha256": "b" * 64,
        }
    )
    permit = ResolvedSimulationAuthorization(
        authorization_id=request.authorization_id,
        authorization_kind=request.authorization_kind,
        owner=ACTOR.owner_key,
        environment_id=request.environment_id,
        revision=request.revision,
        controller="learned",
        task=request.task,
        control_profile_sha256=core.paused_profile.sha256,
        wall_expires_at=request.wall_expires_at,
        max_episode_wall_seconds=600,
        max_simulation_steps=request.max_simulation_steps,
        purpose="evaluation",
        criteria_sha256="c" * 64,
        frozen_plan_sha256="d" * 64,
        policy_type=request.policy_type,
        model_sha256=request.model_sha256,
    )
    cell.spec, cell.scene_epoch = core.spec, core.epoch
    cell.capture = lambda name: None
    owner_thread = threading.get_ident()

    class Model:
        policy_type, execution_timing, real_time_admission = "smolvla", "paused_simulation", False
        scope = Scope(str(ACTOR.tenant_id), ACTOR.owner_key)
        model_sha256 = request.model_sha256
        profile = core.paused_profile
        task = DemonstrationSource(
            "learned", **request.task.model_dump(), source_policy_sha256=model_sha256
        )
        chunk_size, n_action_steps = 50, 1

        def __init__(self):
            self.calls = self.resets = 0
            self.entered, self.release = threading.Event(), threading.Event()
            self.release.set()
            self.offset = 0.001

        def reset(self):
            assert threading.get_ident() != owner_thread, (
                "Model reset must stay off the Isaac thread"
            )
            self.resets += 1

        def predict_chunk(self, observation, context):
            assert threading.get_ident() != owner_thread
            self.calls += 1
            self.entered.set()
            assert self.release.wait(5)
            clock[1] += 1_000_000
            joints = observation.joint_positions
            return ((joints[0] + self.offset, *joints[1:]),) * 50

    model = Model()

    class Provider:
        active = True
        port = None

        def authorize(self, owner, command, profile):
            if not self.active or owner != permit.owner or command != request:
                raise ValueError("Installed learned authority changed.")
            assert profile.sha256 == permit.control_profile_sha256
            return permit

        def create(self, owner, command, profile, *, publication_guard):
            self.authorize(owner, command, profile)
            self.port = PausedGuardedPolicyAdapter(
                model, profile=profile, clock_ns=core.clock_ns, publication_guard=publication_guard
            )
            return self.port

    provider = Provider()
    core.paused_authorizer = provider
    core.paused_policy_provider = provider

    def observation(command, protocol, episode, tick):
        assert threading.get_ident() == owner_thread
        clock[1] += 1_000_000
        observed_ns = core.clock_ns()
        state = cell.frozen_physics_state(core.epoch)
        fraction = Fraction(state.physics_step, 60)
        stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return FrozenPolicyObservation(
            scope=model.scope,
            environment_id=command.environment_id,
            revision=command.revision,
            episode_id=str(command.command_id),
            epoch=str(core.epoch),
            captured_at_utc=stamp,
            monotonic_ns=observed_ns,
            physics_step=state.physics_step,
            joint_positions=state.joint_positions,
            images={
                name: FrozenCameraSample(
                    PNG,
                    state.physics_step,
                    state.physics_step,
                    observed_ns,
                    stamp,
                    fraction.numerator,
                    fraction.denominator,
                )
                for name in ("inspection", "overview")
            },
            freeze_id=str(episode.freeze_id),
            state_revision=core.state_revision,
            control_tick=tick,
            control_profile_sha256=core.paused_profile.sha256,
            observation_started_ns=episode.interval_started_ns,
            joint_sample_ns=observed_ns,
            simulation_time_numerator=fraction.numerator,
            simulation_time_denominator=fraction.denominator,
        )

    cell.paused_observation = observation

    def forbidden(*args, **kwargs):
        raise AssertionError("Learned execution cannot construct or invoke a reference fallback")

    adapter = sys.modules["simulation.isaac_adapter"]
    monkeypatch.setattr(adapter, "RMPFlowController", forbidden)
    monkeypatch.setattr(adapter, "reference_tcp_target", forbidden)
    monkeypatch.setattr(adapter, "plan_reference_targets", forbidden)
    cell.prepare_paused_reference = forbidden
    cell.paused_reference_targets = forbidden
    monkeypatch.setattr("simulation.paused_runtime.PausedReferenceTeacher", forbidden)
    backends = []

    def capture_factory(binding):
        backend = Backend(binding)
        backends.append(backend)
        return backend

    runtime = SimulatorRuntime(
        core,
        cell,
        heartbeat=tmp_path / "heartbeat",
        clock=lambda: clock[1] / 1e9,
        capture_factory=capture_factory,
    )
    value = SimpleNamespace(
        cell=cell,
        core=core,
        request=request,
        provider=provider,
        model=model,
        runtime=runtime,
        clock=clock,
        backends=backends,
    )
    try:
        yield value
    finally:
        model.release.set()
        runtime.close()
        worker = getattr(runtime, "paused_policy_worker", None)
        if worker is not None and worker.thread is not None:
            worker.thread.join(2)
        for capture in runtime.capture_workers.values():
            capture.thread.join(2)


def start(value):
    value.core.dispatch_simulation_episode(ACTOR.owner_key, value.request)
    value.runtime.tick()
    result = value.core.command(ACTOR.owner_key, value.request.command_id)
    assert result.status == "running", result.error
    assert value.runtime.capture_worker.prepared.wait(2)
    if value.cell.paused_driver is None:
        value.runtime.tick()
    assert value.cell.paused_driver.episode.phase == "observing"


def step_to(value, steps):
    for _ in range(40):
        value.runtime.tick()
        driver = value.cell.paused_driver
        assert driver is not None
        if driver.episode.phase == "predicting":
            assert value.model.entered.wait(2)
            value.runtime.paused_policy_worker.thread.join(2)
        if value.cell.world.current_time_step_index >= steps:
            return
        result = value.core.command(ACTOR.owner_key, value.request.command_id)
        assert result.status == "running", result.error
    raise AssertionError("Actual learned ticks did not arrive within the bounded CPU test loop")


def test_paused_model_targets_reach_the_sdk_without_any_reference_controller(learned):
    start(learned)
    step_to(learned, 12)
    cell = learned.cell
    assert len(cell.robot.actions) == 12
    assert all(tuple(action.joint_velocities) == (0.0,) * 9 for action in cell.robot.actions)
    assert [action.joint_positions[0] for action in cell.robot.actions] == [0.001] * 6 + [0.002] * 6
    assert cell.controller is None and cell.route is None
    metrics = learned.core.command(ACTOR.owner_key, learned.request.command_id).simulation_runtime
    assert metrics.policy_predict_calls == learned.model.calls == 2
    assert metrics.reference_route_calls == 0
    assert metrics.applied_model_sha256 == learned.request.model_sha256
    assert metrics.applied_action_count == metrics.simulation_steps == 12
    assert learned.model.resets == 1
    learned.core.cancel(ACTOR.owner_key, learned.request.command_id)
    learned.runtime.tick()
    learned.runtime.capture_worker.thread.join(2)
    learned.runtime.tick()
    frames = learned.backends[0].frames
    assert len(frames) == 2 and frames[-1].truncated
    assert all(
        frame.policy_started_ns is not None and frame.policy_finished_ns is not None
        for frame in frames
    )
    assert all(
        control.commanded_joint_targets == frame.commanded_joint_targets
        for frame in frames
        for control in frame.applied_controls
    )


def test_pending_paused_model_keeps_the_sdk_frozen_and_the_heartbeat_advancing(learned):
    learned.model.release.clear()
    start(learned)
    for _ in range(4):
        learned.runtime.tick()
    assert learned.model.entered.wait(2)
    before = learned.cell.frozen_physics_state(learned.core.epoch)
    for _ in range(6):
        learned.clock[1] += 200_000_000
        learned.runtime.tick()
        assert learned.cell.frozen_physics_state(learned.core.epoch) == before
    assert float(learned.runtime.heartbeat.read_text()) > 2.0
    learned.core.cancel(ACTOR.owner_key, learned.request.command_id)
    learned.runtime.tick()
    learned.model.release.set()
    learned.runtime.paused_policy_worker.thread.join(2)
    learned.runtime.tick()
    assert not learned.cell.robot.actions
    assert learned.core.command(ACTOR.owner_key, learned.request.command_id).status == "cancelled"


@pytest.mark.parametrize("cause", ["expired", "model", "authority", "epoch", "invalid-target"])
def test_late_or_invalid_paused_model_never_becomes_an_actuator_command(learned, cause):
    learned.model.release.clear()
    start(learned)
    for _ in range(4):
        learned.runtime.tick()
    assert learned.model.entered.wait(2)
    if cause == "expired":
        learned.clock[1] += 2_000_000_001
    elif cause == "model":
        learned.model.model_sha256 = "f" * 64
    elif cause == "authority":
        learned.provider.active = False
    elif cause == "epoch":
        from uuid import uuid4

        learned.core.epoch = uuid4()
    else:
        learned.model.offset = 0.08
    learned.model.release.set()
    learned.runtime.paused_policy_worker.thread.join(2)
    learned.runtime.tick()
    assert not learned.cell.robot.actions
    assert learned.cell.world.current_time_step_index == 0


def test_installed_authority_problem_during_hold_stops_before_the_next_tick(learned):
    from apps.api.errors import Problem

    start(learned)
    step_to(learned, 1)

    def revoked(*args):
        raise Problem(403, "revoked", "Installed candidate authority was revoked.")

    learned.provider.authorize = revoked
    learned.runtime.tick()
    assert len(learned.cell.robot.actions) == 1
    assert learned.core.command(ACTOR.owner_key, learned.request.command_id).status == "failed"
    assert "revoked" in learned.cell.paused_driver.episode.failure


def test_model_issued_expiry_is_not_renewed_when_a_finished_prediction_is_dequeued(learned):
    start(learned)
    port = learned.provider.port
    original = port.step

    def short_command(*args):
        result = original(*args)
        return replace(result, expires_at_monotonic_ns=learned.clock[1] + 100_000_000)

    port.step = short_command
    learned.runtime.tick()
    assert learned.model.entered.wait(2)
    learned.runtime.paused_policy_worker.thread.join(2)
    learned.clock[1] += 100_000_001
    learned.runtime.tick()
    assert not learned.cell.robot.actions
    assert learned.cell.paused_driver.episode.failure is not None


def test_expired_held_command_never_finishes_its_other_five_ticks_or_pads_raw_capture(learned):
    start(learned)
    step_to(learned, 1)
    driver = learned.cell.paused_driver
    learned.clock[1] = driver.command.expires_at_monotonic_ns
    learned.runtime.tick()
    assert len(learned.cell.robot.actions) == 1
    assert learned.core.capture(ACTOR.owner_key, learned.request.command_id).status == "invalid"
    assert learned.core.command(ACTOR.owner_key, learned.request.command_id).status == "failed"


@pytest.mark.parametrize("field", ["context_sha256", "freeze_id", "model_sha256", "state_revision"])
def test_a_forged_native_command_is_independently_rejected_on_the_main_thread(learned, field):
    start(learned)
    port = learned.provider.port
    original = port.step

    def swapped(*args):
        result = original(*args)
        changed = result.state_revision + 1 if field == "state_revision" else "f" * 64
        return replace(result, **{field: changed})

    port.step = swapped
    learned.runtime.tick()
    assert learned.model.entered.wait(2)
    learned.runtime.paused_policy_worker.thread.join(2)
    learned.runtime.tick()
    assert not learned.cell.robot.actions
    result = learned.core.command(ACTOR.owner_key, learned.request.command_id)
    assert result.status == "failed" and result.simulation_runtime.applied_model_sha256 is None


def test_native_paused_socket_reply_reaches_six_sdk_actions_with_no_model_inside_isaac(
    learned,
    monkeypatch,
    tmp_path,
):
    import os
    import socket
    import time

    from learning.gr00t.ipc import receive_packet, send_packet
    from learning.paused.ipc import SocketChunkPolicy, make_response, validate_request

    learned.core.clock_ns = time.monotonic_ns
    learned.runtime.paused_policy_worker.clock_ns = time.monotonic_ns
    path = tmp_path / "native.sock"
    errors = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        os.chmod(path, 0o660)
        listener.listen(1)
        listener.settimeout(3)

        def server():
            try:
                connection, _ = listener.accept()
                with connection:
                    payload = receive_packet(
                        connection, deadline_ns=time.monotonic_ns() + 2_000_000_000
                    )
                    observation, context = validate_request(
                        payload,
                        model_sha256=learned.request.model_sha256,
                        scope=learned.model.scope,
                        profile=learned.core.paused_profile,
                        task=learned.model.task,
                    )
                    learned.model.entered.set()
                    actions = ((0.001, *observation.joint_positions[1:]),) * 50
                    send_packet(
                        connection,
                        make_response(payload, actions, inference_latency_ms=1),
                        deadline_ns=context.operation_deadline_ns,
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(exc)

        thread = threading.Thread(target=server)
        thread.start()

        def real_port(owner, request, profile, *, publication_guard):
            learned.provider.authorize(owner, request, profile)
            client = SocketChunkPolicy(
                path,
                scope=learned.model.scope,
                model_sha256=request.model_sha256,
                profile=profile,
                task=learned.model.task,
                expected_peer_uid=os.geteuid(),
            )
            learned.provider.port = PausedGuardedPolicyAdapter(
                client,
                profile=profile,
                clock_ns=time.monotonic_ns,
                publication_guard=publication_guard,
            )
            return learned.provider.port

        monkeypatch.setattr(learned.provider, "create", real_port)
        try:
            start(learned)
            step_to(learned, 6)
        finally:
            thread.join(3)
        assert not thread.is_alive() and errors == []
    assert len(learned.cell.robot.actions) == 6
    assert learned.model.calls == 0
    assert learned.provider.port.policy.predict_calls == 1
    metrics = learned.cell.paused_driver.metrics()
    assert metrics.policy_predict_calls == 1 and metrics.reference_route_calls == 0
    assert metrics.applied_model_sha256 == learned.request.model_sha256
