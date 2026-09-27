"""CPU actual actuator doubles and measured trace checks, never model/GPU quality proof."""

import importlib
from dataclasses import asdict

import pytest
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_paused_learned_runtime import learned as learned
from test_paused_learned_runtime import start, step_to
from test_teaching_runtime import teaching as teaching


def probe():
    return importlib.import_module("simulation.learned_probe")


def test_new_trace_reads_actual_sdk_state_after_learned_actions_without_reference(learned):
    trace = probe().LearnedTrace(learned.core, learned.cell, learned.request)
    start(learned)
    trace.observe()
    for step in range(1, 13):
        step_to(learned, step)
        trace.observe()
    assert len(trace.states) == 13
    assert trace.states[0].applied_action_count == trace.states[0].policy_predict_calls == 0
    last = trace.states[-1]
    assert last.applied_action_count == len(learned.cell.robot.actions) == 12
    assert last.policy_predict_calls == learned.model.calls == 2
    assert last.reference_route_calls == 0
    assert last.applied_model_sha256 == learned.request.model_sha256
    assert (
        last.joint_positions
        == learned.cell.frozen_physics_state(learned.core.epoch).joint_positions
    )
    assert last.tcp_position_m == learned.cell._measured_tcp()


def test_skipped_real_physics_cannot_be_backfilled_as_complete_task_evidence(learned):
    trace = probe().LearnedTrace(learned.core, learned.cell, learned.request)
    start(learned)
    trace.observe()
    step_to(learned, 2)
    with pytest.raises(ValueError, match="Skipped|tick"):
        trace.observe()


@pytest.mark.parametrize("change", ["epoch", "model", "reference"])
def test_trace_rejects_foreign_command_model_or_scripted_route(learned, change):
    trace = probe().LearnedTrace(learned.core, learned.cell, learned.request)
    start(learned)
    trace.observe()
    step_to(learned, 1)
    epoch = learned.core.epoch
    try:
        if change == "epoch":
            from uuid import uuid4

            learned.core.epoch = uuid4()
        elif change == "model":
            learned.cell.paused_driver.applied_model_sha256 = "f" * 64
        else:
            learned.cell.paused_driver.reference_calls = 1
        with pytest.raises(ValueError, match="model|epoch|reference|binding"):
            trace.observe()
    finally:
        learned.core.epoch = epoch
        learned.cell.paused_driver.applied_model_sha256 = learned.request.model_sha256
        learned.cell.paused_driver.reference_calls = 0


def test_final_image_codec_preserves_actual_capture_time_and_native_bytes():
    from learning.checks.fixtures import png
    from learning.paused import FrozenCameraSample

    sample = FrozenCameraSample(png(320, 320), 1, 60, 1_000_000_000, "2026-09-27T00:00:00Z", 1, 1)
    encoded = probe().encode_image(sample)
    recovered = probe().decode_image(encoded)
    assert asdict(recovered) == asdict(sample)
    assert recovered.monotonic_ns == sample.monotonic_ns
    assert recovered.png == sample.png


def test_native_trace_hook_samples_before_runtime_preview_or_terminal_io(monkeypatch):
    from types import SimpleNamespace

    from simulation.run_isaac import SimulatorRuntime

    calls = []
    monkeypatch.setattr(
        SimulatorRuntime, "_publish_policy_metrics", lambda self: calls.append("metrics")
    )
    runtime = probe().MeasuredLearnedRuntime.__new__(probe().MeasuredLearnedRuntime)
    runtime.trace = SimpleNamespace(observe=lambda: calls.append("measurement"))
    runtime._publish_policy_metrics()
    assert calls == ["metrics", "measurement"]


def test_probe_persists_original_heartbeats_and_budget_without_reconstructing_them(learned):
    trace = probe().LearnedTrace(learned.core, learned.cell, learned.request)
    start(learned)
    trace.observe()
    for step in range(1, 7):
        step_to(learned, step)
        trace.observe()
    stamps = list(trace.heartbeat_ns)
    proof = trace.recording(end_ns=stamps[-1])
    assert proof["heartbeat_ns"] == stamps
    assert proof["episode_budget"]["started_ns"] == learned.cell.paused_driver.episode.started_ns
    assert proof["episode_budget"]["wall_deadline_ns"] == (
        learned.cell.paused_driver.episode.wall_deadline_ns
    )
    assert proof["task_states"] == [asdict(state) for state in trace.states]
