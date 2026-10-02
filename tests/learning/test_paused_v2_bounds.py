from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, overwrite
from learning.common import ContractError, canonical, digest, file_digest, read_json
from learning.paused.task import TaskState, evaluate_task_states
from tests.learning.test_conversion_training import mock_converter as mock_converter
from tests.learning.test_paused_evaluation import plan, trials
from tests.learning.test_paused_profile_v2 import capture_frame, capture_writer, selected_profile
from tests.learning.test_paused_rollout import grant, runtime


def task_states(step_count):
    start = datetime(2026, 9, 25, tzinfo=UTC)
    return [
        TaskState(
            captured_at_utc=(start + timedelta(milliseconds=20 * step))
            .isoformat()
            .replace("+00:00", "Z"),
            monotonic_ns=1_000_000_000 + 20_000_000 * step,
            physics_step=step,
            object_position_m=(0.3, 0.0, 0.1),
            object_linear_velocity_m_s=(0.0, 0.0, 0.0),
            tcp_position_m=(0.4, 0.0, 0.2),
            joint_positions=JOINTS,
            command_id="source-only-reference",
            epoch="fixture-epoch",
            policy_predict_calls=0,
            applied_action_count=step,
            reference_route_calls=step,
            applied_model_sha256=None,
        )
        for step in range(step_count + 1)
    ]


def test_all_600_frames_convert_without_resampling_or_losing_original_wall_times(
    tmp_path, mock_converter, monkeypatch
):
    import numpy

    from learning.paused.dataset import convert_dataset, validate_conversion

    monkeypatch.setattr(
        numpy,
        "asarray",
        lambda value, dtype: ("fixture-image", value.size) if hasattr(value, "size") else value,
    )
    writer = capture_writer(tmp_path / "raw")
    for index in range(600):
        writer.append(capture_frame(index, terminal=index == 599))
    writer.finalize()
    output = tmp_path / "converted"
    value = convert_dataset(
        writer.root,
        output,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(writer.root / "manifest.json"),
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
        allow_test_fixture=True,
    )
    assert value["control_profile"] == asdict(selected_profile(2))
    assert value["episodes"][0]["frame_count"] == len(mock_converter[0].episodes[0]) == 600
    assert validate_conversion(output, SCOPE) == read_json(output / "conversion.json")
    import json

    timing = [
        json.loads(line) for line in (output / "source-timing.jsonl").read_text().splitlines()
    ]
    assert len(timing) == 600
    assert timing[-1]["simulation_timestamp"] == 59.9
    assert timing[-1]["applied_controls"][-1]["physics_step"] == 3600
    assert timing[-1]["monotonic_ns"] - timing[0]["monotonic_ns"] == 299_500_000_000
    value["control_profile"] = asdict(selected_profile(1))
    overwrite(output / "conversion.json", value)
    with pytest.raises(ContractError):
        validate_conversion(output, SCOPE)


def test_v2_task_trace_requires_explicit_profile_and_rejects_extra_states():
    states = task_states(3600)
    initial = states[0].object_position_m
    goal = (0.5, 0.0, 0.1)
    result = evaluate_task_states(states, initial=initial, goal=goal, profile=selected_profile(2))
    assert result["maximum_tcp_speed_m_s"] == 0
    assert result["grasp_verified"] is False and result["settled"] is False
    assert result["safety_violations"] == []
    for profile in (None, selected_profile(1)):
        with pytest.raises(ContractError, match="per-tick"):
            evaluate_task_states(states, initial=initial, goal=goal, profile=profile)
    with pytest.raises(ContractError, match="per-tick"):
        evaluate_task_states(
            task_states(3601), initial=initial, goal=goal, profile=selected_profile(2)
        )


@pytest.mark.parametrize(("version", "count"), [(1, 1800), (2, 3600)])
def test_task_trace_sidecar_reader_retains_every_state_with_profile_bound_counts(
    tmp_path, version, count
):
    from learning.paused.rollout import _read_task_states

    path = tmp_path / "task-states.jsonl"
    states = task_states(count)
    path.write_bytes(b"".join(canonical(asdict(state)) + b"\n" for state in states))
    decoded = _read_task_states(path, file_digest(path), selected_profile(version))
    assert len(decoded) == count + 1
    assert [value.physics_step for value in decoded] == list(range(count + 1))
    with path.open("ab") as stream:
        stream.write(canonical(asdict(states[-1])) + b"\n")
    with pytest.raises(ContractError, match="bounds"):
        _read_task_states(path, file_digest(path), selected_profile(version))


def test_trace_sidecar_rejects_oversized_rows_before_parsing(tmp_path):
    from learning.paused.rollout import _read_task_states

    path = tmp_path / "task-states.jsonl"
    path.write_bytes(b" " * 8193 + b"\n")
    with pytest.raises(ContractError, match="bounds"):
        _read_task_states(path, file_digest(path), selected_profile(2))


def test_v2_trace_can_use_bounded_jsonl_above_default_json_limit(tmp_path):
    from learning.paused.rollout import _read_task_states

    path = tmp_path / "task-states.jsonl"
    with path.open("wb") as stream:
        for state in task_states(3600):
            payload = canonical(asdict(state))
            assert len(payload) <= 2047
            stream.write(payload.ljust(2047, b" ") + b"\n")
    assert 4 * 1024 * 1024 < path.stat().st_size < 8 * 1024 * 1024
    with pytest.raises(ContractError, match="JSON exceeds size limit"):
        read_json(path)
    assert len(_read_task_states(path, file_digest(path), selected_profile(2))) == 3601
    with pytest.raises(ContractError, match="bounds"):
        _read_task_states(path, file_digest(path), selected_profile(1))


def test_trace_sidecar_total_size_remains_bounded_before_checksum_read(tmp_path):
    from learning.paused.rollout import MAX_TASK_STATE_BYTES, _read_task_states

    path = tmp_path / "task-states.jsonl"
    with path.open("wb") as stream:
        stream.truncate((3600 + 1) * MAX_TASK_STATE_BYTES + 1)
    with pytest.raises(ContractError, match="Oversized"):
        _read_task_states(path, "0" * 64, selected_profile(2))


@pytest.mark.parametrize(("version", "steps"), [(1, 1800), (2, 3600)])
def test_recording_grant_physics_cap_is_selected_profile_bound(version, steps):
    from learning.paused.rollout import _grant

    selected = selected_profile(version)
    native = plan()
    native["control_profile"] = asdict(selected)
    native["control_profile_sha256"] = selected.sha256
    actual_runtime = runtime()
    actual_runtime["control_profile"] = asdict(selected)
    native["runtime_sha256"] = digest(canonical(actual_runtime))
    approval = grant(native, actual_runtime)
    approval["max_episode_physics_steps"] = steps
    assert len(_grant(native, actual_runtime, approval)) == 40
    with pytest.raises(ContractError, match="physics cap"):
        _grant(native, actual_runtime, {**approval, "max_episode_physics_steps": steps + 6})
    with pytest.raises(ContractError, match="wall"):
        _grant(native, actual_runtime, {**approval, "max_episode_wall_seconds": 601})


def test_exact_ticks_define_v2_simulation_duration_without_padding_or_v1_promotion():
    from learning.paused.evaluation import compare_trials

    native = plan()
    native["control_profile"] = asdict(selected_profile(2))
    native["control_profile_sha256"] = selected_profile(2).sha256
    attempts = trials(native, before=18, after=19)
    for trial in attempts:
        trial.update(
            policy_predict_calls=600,
            applied_action_count=3600,
            latencies_wall_ms=[250.0] * 600,
            observation_wall_ms=[100.0] * 600,
            hold_wall_ms=[120.0] * 600,
            interval_wall_ms=[500.0] * 600,
            heartbeat_gap_ms=[500.0] * 600,
            wall_duration_ms=300000.0,
            simulation_duration_ms=60000.0,
        )
    report = compare_trials(native, attempts, live_gpu_verified=False)
    assert report["resource_violation_count"] == 0
    assert report["total_simulation_duration_ms"] == 40 * 60000
    assert report["latency_wall_ms"]["after"]["max"] == 250.0
    assert report["quality_gate_passed"] is False
    assert report["real_time_admission"] is False
    native["control_profile"] = asdict(selected_profile(1))
    native["control_profile_sha256"] = selected_profile(1).sha256
    legacy = compare_trials(native, attempts, live_gpu_verified=False)
    assert legacy["resource_violation_count"] == 40
    native["control_profile"] = asdict(selected_profile(2))
    native["control_profile_sha256"] = selected_profile(2).sha256
    attempts[-1]["simulation_duration_ms"] += 0.0015646
    with pytest.raises(ContractError, match="actual physics ticks"):
        compare_trials(native, attempts, live_gpu_verified=False)


def test_v2_recording_derives_full_3600_tick_trial_from_raw_not_float_world_time(tmp_path):
    from learning.paused.capture import validate_dataset
    from learning.paused.rollout import derive_trial

    writer = capture_writer(tmp_path / "raw", purpose="evaluation", seed=30001, split="test")
    for index in range(600):
        writer.append(capture_frame(index, terminal=index == 599))
    writer.finalize()
    raw = validate_dataset(writer.root, expected_scope=SCOPE)
    case = plan()["cases"][0]
    case.update(
        episode_id=writer.episode.episode_id,
        environment_id=writer.episode.environment_id,
        scene_builder_sha256=PROVENANCE.scene_builder_sha256,
    )
    states = [task_states(0)[0]]
    for frame in raw.episodes[0].frames:
        for control in frame["applied_controls"]:
            timestamp = datetime(2026, 9, 25, tzinfo=UTC) + timedelta(
                microseconds=(control["monotonic_ns"] - writer.budget.started_ns) // 1000
            )
            states.append(
                replace(
                    states[0],
                    captured_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
                    monotonic_ns=control["monotonic_ns"],
                    physics_step=control["physics_step"],
                    applied_action_count=control["physics_step"],
                    reference_route_calls=frame["frame_index"] + 1,
                )
            )
    last = states[-1]
    final_mono = last.monotonic_ns + 10_000_000
    timestamp = datetime(2026, 9, 25, tzinfo=UTC) + timedelta(
        microseconds=(final_mono - writer.budget.started_ns) // 1000
    )
    images = {
        name: replace(
            image,
            captured_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
            monotonic_ns=final_mono,
            physics_step=3600,
            rendering_frame=600,
            simulation_time_numerator=60,
            simulation_time_denominator=1,
        )
        for name, image in capture_frame(599).observation.images.items()
    }
    trial = derive_trial(
        raw,
        states,
        case=case,
        role="reference",
        model_sha256=None,
        profile=selected_profile(2),
        final_images=images,
        heartbeat_ns=tuple(range(writer.budget.started_ns, final_mono, 500_000_000)),
        destination_id="rejected",
        failure_reason="task_not_completed",
    )
    assert trial["simulation_duration_ms"] == 60000.0
    assert trial["applied_action_count"] == 3600
    assert len(trial["interval_wall_ms"]) == len(trial["hold_wall_ms"]) == 600
    assert trial["latencies_wall_ms"] == []
    assert trial["task_evidence"]["grasp_verified"] is False
    assert trial["task_evidence"]["safety_violations"] == []
    with pytest.raises(ContractError, match="profile"):
        derive_trial(
            raw,
            states,
            case=case,
            role="reference",
            model_sha256=None,
            profile=selected_profile(1),
            final_images=images,
            heartbeat_ns=(writer.budget.started_ns, final_mono),
            destination_id="rejected",
            failure_reason="task_not_completed",
        )


def test_ipc_v2_preserves_explicit_profile_v2_and_rejects_v1_server_binding():
    from learning.paused.ipc import make_request, validate_request
    from tests.learning.test_paused_contract import context, observation
    from tests.learning.test_paused_inference import Policy

    selected = selected_profile(2)
    observed = replace(observation(), control_profile_sha256=selected.sha256)
    authority = replace(
        context(),
        control_profile_sha256=selected.sha256,
        observation_sha256=observed.sha256,
        simulation_step_deadline=context().episode_initial_physics_step + 3600,
    )
    packet = make_request(
        observed,
        authority,
        model_sha256=authority.model_sha256,
        profile=selected,
        task=Policy.task,
        request_id="v2-profile-request",
        sequence=0,
        now_ns=observed.ready_ns,
    )
    decoded, current = validate_request(
        packet,
        model_sha256=authority.model_sha256,
        scope=SCOPE,
        profile=selected,
        task=Policy.task,
        now_ns=observed.ready_ns,
    )
    assert packet["schema"] == "physicalai.smolvla-request/v2"
    assert decoded.control_profile_sha256 == selected.sha256
    assert current.simulation_step_deadline == 3660
    with pytest.raises(ContractError, match="profile"):
        validate_request(
            packet,
            model_sha256=authority.model_sha256,
            scope=SCOPE,
            profile=selected_profile(1),
            task=Policy.task,
            now_ns=observed.ready_ns,
        )


def test_checkpoint_v2_loader_requires_the_selected_v2_profile_hash(tmp_path, monkeypatch):
    from learning.paused.artifacts import validate_model
    from learning.paused.model import LocalPausedSmolVLAPolicy
    from learning.smolvla import inference
    from tests.learning import test_paused_model_artifacts as model_fixture

    selected = selected_profile(2)
    monkeypatch.setattr(model_fixture, "profile", lambda: selected)
    checksum = model_fixture.fixture(tmp_path)
    metadata = validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)
    assert metadata["control_profile"] == asdict(selected)
    assert metadata["schema"] == "physicalai.smolvla-checkpoint/v2"
    calls = []

    def backbone(root, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("Stop before optional model loading in this metadata-only test")

    monkeypatch.setattr(inference, "validate_backbone", backbone)
    with pytest.raises(RuntimeError, match="before optional model loading"):
        LocalPausedSmolVLAPolicy(
            tmp_path,
            backbone_root=tmp_path / "unused-backbone",
            scope=SCOPE,
            model_sha256=checksum,
            expected_control_profile_sha256=selected.sha256,
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="e" * 64,
        )
    assert len(calls) == 1
    with pytest.raises(ContractError, match="profile"):
        LocalPausedSmolVLAPolicy(
            tmp_path,
            backbone_root=tmp_path / "unused-backbone",
            scope=SCOPE,
            model_sha256=checksum,
            expected_control_profile_sha256=selected_profile(1).sha256,
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="e" * 64,
        )
    assert len(calls) == 1
