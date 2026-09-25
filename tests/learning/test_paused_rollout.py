from dataclasses import asdict, replace

import pytest

from learning.checks.fixtures import PROVENANCE, SCOPE
from learning.common import ContractError, canonical, digest, file_digest
from learning.paused.contract import PausedControlProfile
from learning.paused.task import PREDICATE_SOURCE
from tests.learning.test_paused_evaluation import plan


def runtime():
    return {
        "schema": "physicalai.paused-evaluation-runtime/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "provenance": asdict(PROVENANCE),
        "control_profile": asdict(PausedControlProfile("f" * 64)),
        "source_files": {"simulation/control.py": PREDICATE_SOURCE["sha256"]},
        "probe_receipt_sha256": "d" * 64,
        "test_only": True,
    }


def grant(value, actual_runtime):
    return {
        "schema": "physicalai.operator-rollout-grant/v2",
        "purpose": value["comparison_kind"],
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "evaluation_run_id": "unit-evaluation",
        "scope": asdict(SCOPE),
        "operator_principal_sha256": "1" * 64,
        "plan_sha256": digest(canonical(value)),
        "runtime_sha256": digest(canonical(actual_runtime)),
        "control_profile_sha256": value["control_profile_sha256"],
        "criteria_sha256": value["criteria_sha256"],
        "frozen_plan_sha256": value["frozen_plan_sha256"],
        "issued_at_utc": "2026-09-24T00:00:00Z",
        "expires_at_utc": "2026-09-24T07:00:00Z",
        "max_episode_wall_seconds": 600,
        "max_episode_physics_steps": 1800,
        "max_total_wall_seconds": 25200,
    }


def test_recording_never_turns_an_empty_or_partial_schedule_into_results(tmp_path):
    from learning.paused.rollout import PausedRolloutRecorder

    actual_runtime = runtime()
    value = plan()
    value["runtime_sha256"] = digest(canonical(actual_runtime))
    recorder = PausedRolloutRecorder(
        tmp_path / "recording",
        plan=value,
        runtime=actual_runtime,
        grant=grant(value, actual_runtime),
        allow_test_fixture=True,
    )
    assert len(recorder.schedule) == 40
    with pytest.raises(ContractError, match="incomplete"):
        recorder.finalize()
    assert not (recorder.root / "results.json").exists()
    from learning.common import read_json

    incomplete = read_json(recorder.root / "incomplete-recording.json")
    assert incomplete["expected_trial_count"] == len(incomplete["not_recorded"]) == 40
    assert incomplete["quality_gate_passed"] is False


def test_failure_before_observation_is_persisted_without_fake_pose_or_model_execution(tmp_path):
    from learning.common import read_json
    from learning.paused.rollout import PausedRolloutRecorder

    actual_runtime = runtime()
    value = plan()
    value["runtime_sha256"] = digest(canonical(actual_runtime))
    recorder = PausedRolloutRecorder(
        tmp_path / "recorded",
        plan=value,
        runtime=actual_runtime,
        grant=grant(value, actual_runtime),
        allow_test_fixture=True,
    )
    recorder.record_failure(
        policy=recorder.schedule[0][0],
        failure_code="camera_timeout",
        phase="observation",
        observed_at_utc="2026-09-24T00:00:01Z",
        monotonic_ns=1_000_000_000,
        command_id=None,
    )
    with pytest.raises(ContractError, match="incomplete"):
        recorder.finalize()
    proof = read_json(recorder.root / "incomplete-recording.json")
    assert len(proof["incomplete_attempts"]) == 1
    assert len(proof["not_recorded"]) == 39
    assert proof["recorded_complete_trial_count"] == 0
    failure = proof["incomplete_attempts"][0]
    assert failure["actual_command_id"] is None
    assert "final_pose_m" not in failure
    assert failure["physical_outcome_verified"] is False


def test_live_recorder_refuses_fixture_runtime_before_artifact_creation(tmp_path):
    from learning.paused.rollout import PausedRolloutRecorder

    actual_runtime = runtime()
    value = plan()
    value["runtime_sha256"] = digest(canonical(actual_runtime))
    with pytest.raises(ContractError, match="live|fixture"):
        PausedRolloutRecorder(
            tmp_path / "recording",
            plan=value,
            runtime=actual_runtime,
            grant=grant(value, actual_runtime),
        )
    assert not (tmp_path / "recording").exists()


@pytest.mark.parametrize(
    "field",
    [
        "runtime_sha256",
        "control_profile_sha256",
        "criteria_sha256",
        "frozen_plan_sha256",
        "plan_sha256",
    ],
)
def test_recording_grant_binds_every_frozen_identity(tmp_path, field):
    from learning.paused.rollout import PausedRolloutRecorder

    actual_runtime = runtime()
    value = plan()
    value["runtime_sha256"] = digest(canonical(actual_runtime))
    approval = grant(value, actual_runtime)
    approval[field] = "0" * 64
    with pytest.raises(ContractError):
        PausedRolloutRecorder(
            tmp_path / "recording",
            plan=value,
            runtime=actual_runtime,
            grant=approval,
            allow_test_fixture=True,
        )


def test_raw_capture_and_per_tick_task_states_determine_trial_not_declared_success(tmp_path):
    from learning.paused.capture import validate_dataset
    from learning.paused.rollout import derive_trial
    from learning.paused.task import TaskState
    from tests.learning.test_paused_capture import shifted_sample, writer
    from tests.learning.test_paused_contract import profile

    value = writer(tmp_path / "raw", purpose="evaluation", split="test", seed=30001)
    value.append(shifted_sample(0))
    value.append(shifted_sample(1, terminal=True))
    value.finalize()
    case = plan()["cases"][0]
    case.update(
        episode_id="paused-episode",
        environment_id="paused-case",
        scene_builder_sha256=PROVENANCE.scene_builder_sha256,
    )
    raw = validate_dataset(value.root, expected_scope=SCOPE)
    states = []
    for offset in range(13):
        states.append(
            TaskState(
                captured_at_utc=f"2026-09-24T00:00:{offset:02d}Z",
                monotonic_ns=1_000_000_000 + offset * 100_000_000,
                physics_step=60 + offset,
                object_position_m=tuple(case["initial_pose_m"]),
                object_linear_velocity_m_s=(0.0, 0.0, 0.0),
                tcp_position_m=(0.3, 0.0, 0.2),
                joint_positions=shifted_sample(0).observation.joint_positions,
                command_id="actual-reference-command",
                epoch="epoch-new",
                policy_predict_calls=0,
                applied_action_count=offset,
                reference_route_calls=offset,
                applied_model_sha256=None,
            )
        )
    # Per-tick wall stamps must match the raw applied-control records, not this invented list.
    with pytest.raises(ContractError, match="tick|control"):
        derive_trial(
            raw,
            states,
            case=case,
            role="reference",
            model_sha256=None,
            profile=profile(),
            final_images={
                name: replace(image, physics_step=72, monotonic_ns=4_500_000_000)
                for name, image in shifted_sample(1).observation.images.items()
            },
            heartbeat_ns=(1_000_000_000, 2_000_000_000, 3_000_000_000, 4_500_000_000),
            destination_id="rejected",
            failure_reason=None,
        )


def test_all_attempt_recorder_roundtrip_recomputes_failures_and_rejects_forged_success(tmp_path):
    from datetime import UTC, datetime, timedelta

    from learning.common import read_json
    from learning.contract import DemonstrationSource, EpisodeSpec
    from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter
    from learning.paused.evaluation import compare_trials, expected_model
    from learning.paused.rollout import PausedRolloutRecorder, _verify, verify_recording
    from learning.paused.task import TaskState
    from tests.learning.test_paused_capture import shifted_sample
    from tests.learning.test_paused_contract import profile

    actual_runtime = runtime()
    value = plan()
    value["runtime_sha256"] = digest(canonical(actual_runtime))
    for case in value["cases"]:
        case.update(
            environment_id="paused-case", scene_builder_sha256=PROVENANCE.scene_builder_sha256
        )
    recorder = PausedRolloutRecorder(
        tmp_path / "recorded",
        plan=value,
        runtime=actual_runtime,
        grant=grant(value, actual_runtime),
        allow_test_fixture=True,
    )
    epoch = datetime(2026, 9, 24, tzinfo=UTC)

    def stamp(mono):
        return (
            (epoch + timedelta(seconds=(mono - 1_000_000_000) / 1e9))
            .isoformat()
            .replace("+00:00", "Z")
        )

    for attempt_index, (role, case) in enumerate(recorder.schedule):
        shift = attempt_index * 5_000_000_000
        model_sha = expected_model(value, role)
        raw_writer = PausedEpisodeWriter(
            tmp_path / f"raw-{attempt_index}",
            dataset_id=f"data-{attempt_index}",
            scope=SCOPE,
            episode=EpisodeSpec(
                case["episode_id"], case["environment_id"], case["revision"], case["seed"], "test"
            ),
            provenance=PROVENANCE,
            profile=profile(),
            demonstration=DemonstrationSource(
                kind="learned",
                task_id="test-task",
                instruction="Fixture only.",
                goal_id="rejected",
                source_policy_sha256=model_sha,
            ),
            budget=PausedEpisodeBudget(1_000_000_000 + shift, 601_000_000_000 + shift, 60, 1860),
            purpose="evaluation",
            criteria_sha256="d" * 64,
            frozen_plan_sha256="e" * 64,
        )
        frames = []
        for index in range(2):
            original = shifted_sample(index, terminal=index == 1)
            current = replace(
                original,
                observation=replace(
                    original.observation,
                    episode_id=case["episode_id"],
                    monotonic_ns=original.observation.monotonic_ns + shift,
                    captured_at_utc=stamp(original.observation.monotonic_ns + shift),
                    observation_started_ns=original.observation.observation_started_ns + shift,
                    joint_sample_ns=original.observation.joint_sample_ns + shift,
                    images={
                        name: replace(
                            image,
                            monotonic_ns=image.monotonic_ns + shift,
                            captured_at_utc=stamp(image.monotonic_ns + shift),
                        )
                        for name, image in original.observation.images.items()
                    },
                ),
                applied_controls=tuple(
                    replace(control, monotonic_ns=control.monotonic_ns + shift)
                    for control in original.applied_controls
                ),
                interval_deadline_ns=original.interval_deadline_ns + shift,
                hold_started_ns=original.hold_started_ns + shift,
                hold_deadline_ns=original.hold_deadline_ns + shift,
                policy_started_ns=original.observation.monotonic_ns + shift,
                policy_finished_ns=original.observation.monotonic_ns + shift + 250_000_000,
            )
            raw_writer.append(current)
            frames.append(current)
        raw_writer.finalize()
        first = frames[0].observation
        states = [
            TaskState(
                captured_at_utc=stamp(1_000_000_000 + shift),
                monotonic_ns=1_000_000_000 + shift,
                physics_step=60,
                object_position_m=tuple(case["initial_pose_m"]),
                object_linear_velocity_m_s=(0.0, 0.0, 0.0),
                tcp_position_m=(0.4, 0.0, 0.2),
                joint_positions=first.joint_positions,
                command_id=f"actual-command-{attempt_index}",
                epoch=first.epoch,
                policy_predict_calls=0,
                applied_action_count=0,
                reference_route_calls=0,
                applied_model_sha256=None,
            )
        ]
        for frame_index, frame in enumerate(frames):
            for control in frame.applied_controls:
                states.append(
                    replace(
                        states[0],
                        captured_at_utc=stamp(control.monotonic_ns),
                        monotonic_ns=control.monotonic_ns,
                        physics_step=control.physics_step,
                        applied_action_count=control.physics_step - 60,
                        policy_predict_calls=frame_index + 1,
                        applied_model_sha256=model_sha,
                    )
                )
        final_images = {
            name: replace(
                image,
                physics_step=72,
                monotonic_ns=4_500_000_000 + shift,
                captured_at_utc=stamp(4_500_000_000 + shift),
                simulation_time_numerator=12,
                simulation_time_denominator=10,
            )
            for name, image in frames[-1].observation.images.items()
        }
        recorder.record_attempt(
            policy=role,
            capture_root=raw_writer.root,
            states=states,
            final_images=final_images,
            heartbeat_ns=tuple(
                stamp + shift
                for stamp in (1_000_000_000, 2_000_000_000, 3_000_000_000, 4_500_000_000)
            ),
            destination_id="rejected",
            failure_reason="measured_task_not_completed",
        )
    path = recorder.finalize()
    result = read_json(path)
    _verify(recorder.root, result, value, SCOPE, require_live=False)
    report = compare_trials(value, result["trials"])
    assert report["counts"] == {
        "before": {"total": 20, "success": 0},
        "after": {"total": 20, "success": 0},
    }
    assert report["quality_gate_passed"] is False
    with pytest.raises(ContractError, match="live|fixture"):
        verify_recording(recorder.root, value, SCOPE, file_digest(path))
    result["trials"][0]["task_evidence"]["grasp_verified"] = True
    with pytest.raises(ContractError, match="actual source"):
        _verify(recorder.root, result, value, SCOPE, require_live=False)
