from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from learning.checks.fixtures import LIVE_PROVENANCE, SCOPE, png
from learning.checks.groot_fixtures import PROFILE, TASK
from learning.common import ContractError, canonical, digest, read_json, write_json
from learning.contract import CameraSample


def frozen(*, bootstrap=False):
    runtime = {
        "azure_resource_id": (
            "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/unit-tests"
            "/providers/Microsoft.Compute/virtualMachines/test-attestation-only"
        ),
        "run_id": "unit-test-not-gpu-proof",
        "provenance": asdict(LIVE_PROVENANCE),
    }
    plan = {
        "schema": "physicalai.smolvla-paired-plan/v1",
        "scope": asdict(SCOPE),
        "policy_before_sha256": "a" * 64,
        "policy_after_sha256": "b" * 64,
        "control_profile_sha256": PROFILE.sha256,
        "runtime_sha256": digest(canonical(runtime)),
        "minimum_pairs": 20,
        "max_latency_p95_ms": 80.0,
        "minimum_after_success_rate": 0.9,
        "cases": [
            {
                "episode_id": f"case-{i}",
                "attempt": 0,
                "environment_id": "customer-line",
                "revision": "f" * 64,
                "seed": 30001 + i,
                "scene_builder_sha256": "d" * 64,
                "initial_pose_m": [0.4 + i * 0.002, 0.0, 0.2],
                "expected_destination_id": TASK.goal_id,
                "expected_pose_m": [0.3, 0.1, 0.2],
                "tolerance_m": 0.04,
            }
            for i in range(20)
        ],
    }
    grant = {
        "schema": "physicalai.operator-rollout-grant/v1",
        "purpose": "paired_policy_eval",
        "evaluation_run_id": "eval-unit-test",
        "scope": asdict(SCOPE),
        "operator_principal_sha256": "1" * 64,
        "plan_sha256": digest(canonical(plan)),
        "runtime_sha256": plan["runtime_sha256"],
        "control_profile_sha256": PROFILE.sha256,
        "task": {"task_id": TASK.task_id, "instruction": TASK.instruction, "goal_id": TASK.goal_id},
        "issued_at_utc": "2026-09-23T10:00:00Z",
        "expires_at_utc": "2026-09-23T11:00:00Z",
        "max_episode_seconds": 30,
        "max_total_seconds": 3600,
    }
    if bootstrap:
        plan["schema"] = "physicalai.smolvla-bootstrap-plan/v1"
        plan["candidate_model_sha256"] = plan.pop("policy_after_sha256")
        plan["reference_controller_sha256"] = plan.pop("policy_before_sha256")
        plan["minimum_candidate_success_rate"] = plan.pop("minimum_after_success_rate")
        grant["purpose"] = "reference_bootstrap"
        grant["plan_sha256"] = digest(canonical(plan))
    return plan, runtime, grant


def recorder(root, *, bootstrap=False):
    from learning.smolvla.rollout import PhysicalRolloutRecorder

    plan, runtime, grant = frozen(bootstrap=bootstrap)
    return PhysicalRolloutRecorder(
        root,
        plan=plan,
        runtime=runtime,
        grant=grant,
        expected_plan_sha256=digest(canonical(plan)),
        expected_grant_sha256=digest(canonical(grant)),
    )


def begin(capture, index=0):
    from learning.smolvla.rollout import CaseBinding, MeasuredState

    planned = capture.schedule[index]
    case, role = planned["case"], planned["policy"]
    timestamp = 1_000_000_000 + index * 1_000_000_000
    utc = (
        (datetime(2026, 9, 23, 10, tzinfo=UTC) + timedelta(seconds=index))
        .isoformat()
        .replace("+00:00", "Z")
    )
    binding = CaseBinding(
        evaluation_run_id="eval-unit-test",
        purpose=capture.grant["purpose"],
        scope=SCOPE,
        policy=role,
        model_sha256=None if role == "reference" else ("a" if role == "before" else "b") * 64,
        episode_id=case["episode_id"],
        attempt=0,
        environment_id=case["environment_id"],
        revision=case["revision"],
        seed=case["seed"],
        scene_builder_sha256="d" * 64,
        goal_id=TASK.goal_id,
        command_id=f"command-{index}",
        epoch=f"epoch-{index}",
        control_profile_sha256=PROFILE.sha256,
        runtime_sha256=capture.plan["runtime_sha256"],
        deadline_at_utc=(datetime.fromisoformat(utc.replace("Z", "+00:00")) + timedelta(seconds=30))
        .isoformat()
        .replace("+00:00", "Z"),
        deadline_monotonic_ns=timestamp + 30_000_000_000,
    )
    initial = MeasuredState(utc, timestamp, 0, tuple(case["initial_pose_m"]))
    cameras = {
        camera: CameraSample(png(), 1, 0, timestamp) for camera in ("inspection", "overview")
    }
    return binding, initial, cameras


def complete(capture, index=0, *, status="succeeded", cycle_ms=80.0):
    from learning.smolvla.rollout import ControlTrace, TerminalOutcome

    binding, initial, cameras = begin(capture, index)
    capture.start_case(binding, initial, cameras)
    for step in (6, 12):
        time_ns = initial.monotonic_ns + (step // 6) * 100_000_000
        capture.append_control(
            ControlTrace(
                command_id=binding.command_id,
                epoch=binding.epoch,
                monotonic_ns=time_ns,
                physics_step=step,
                policy_predict_calls=0 if binding.policy == "reference" else step // 6,
                applied_action_count=step,
                reference_route_calls=step // 6 if binding.policy == "reference" else 0,
                applied_model_sha256=binding.model_sha256,
                latencies_ms=(20.0,),
                cycle_durations_ms=(cycle_ms,),
                last_applied_monotonic_ns=time_ns,
            )
        )
    final = replace(
        initial,
        captured_at_utc=initial.captured_at_utc.replace("Z", ".250Z"),
        monotonic_ns=initial.monotonic_ns + 250_000_000,
        physics_step=12,
        object_position_m=(0.3, 0.1, 0.2),
    )
    images = {
        camera: CameraSample(png(color=100), 3, 12, final.monotonic_ns)
        for camera in ("inspection", "overview")
    }
    raw_manifest = (
        canonical(
            {
                "schema": "physicalai.demonstrations/v2",
                "scope": asdict(SCOPE),
                "control_profile": asdict(PROFILE),
                "episodes": [
                    {
                        "episode_id": binding.command_id,
                        "environment_id": binding.environment_id,
                        "revision": binding.revision,
                        "seed": binding.seed,
                        "split": "test",
                        "frame_count": 2,
                        "provenance": asdict(LIVE_PROVENANCE),
                        "demonstration": {
                            "kind": "reference_controller"
                            if binding.policy == "reference"
                            else "learned",
                            "task_id": TASK.task_id,
                            "instruction": TASK.instruction,
                            "goal_id": TASK.goal_id,
                            "source_policy_sha256": binding.model_sha256,
                        },
                    }
                ],
            }
        )
        + b"\n"
    )
    outcome = TerminalOutcome(
        status=status,
        destination_id=TASK.goal_id,
        failure_reason=None if status == "succeeded" else "cancelled",
        capture_id=f"capture-{index}",
        capture_manifest_uri=(
            f"https://teststorage.blob.core.windows.net/demonstrations/"
            f"{SCOPE.owner_id}/{binding.command_id}/manifest.json"
        ),
        capture_manifest_sha256=digest(raw_manifest),
        capture_frame_count=2,
    )
    capture.finish_case(outcome, final, images, raw_manifest)
    return binding


def test_rollout_preserves_all_frozen_attempts_and_emits_trace_bound_v2_evidence(tmp_path):
    capture = recorder(tmp_path / "run")
    assert len(capture.schedule) == 40
    assert [entry["policy"] for entry in capture.schedule[:4]] == [
        "before",
        "after",
        "after",
        "before",
    ]
    for index in range(40):
        complete(capture, index, status="cancelled" if index == 3 else "succeeded")
    receipt = capture.finalize()
    result = read_json(receipt)
    assert result["schema"] == "physicalai.smolvla-paired-results/v2"
    assert len(result["trials"]) == 40
    assert result["trials"][3]["truncated"] is True
    assert result["recording"]["sha256"]
    assert "learning_quality_verified" not in result


def test_incomplete_or_pre_scene_failed_attempt_is_durable_but_not_fabricated(tmp_path):
    capture = recorder(tmp_path / "run")
    capture.abort_case("scene_activation", "Camera initialization failed")
    with pytest.raises(ContractError, match="Incomplete"):
        capture.finalize()
    collection = read_json(tmp_path / "run" / "collection.json")
    assert collection["complete"] is False
    assert collection["failed_attempts"] == 1
    assert not (tmp_path / "run" / "results.json").exists()


def test_bootstrap_records_a_scripted_reference_without_inventing_a_policy_sha(tmp_path):
    from learning.smolvla.evaluation import compare_bootstrap_trials

    capture = recorder(tmp_path / "bootstrap", bootstrap=True)
    for index in range(40):
        complete(capture, index)
    result = read_json(capture.finalize())
    assert result["schema"] == "physicalai.smolvla-bootstrap-results/v2"
    assert result["trials"][0]["policy"] == "reference"
    assert result["trials"][0]["model_sha256"] is None
    assert result["trials"][0]["reference_route_calls"] == 2
    metrics = compare_bootstrap_trials(capture.plan, result["trials"], live_gpu_verified=False)
    assert metrics["quality_gate_passed"] is False


def test_fast_standalone_inference_does_not_hide_a_failed_whole_control_cycle(tmp_path):
    from learning.smolvla.evaluation import compare_trials

    capture = recorder(tmp_path / "run")
    for index in range(40):
        complete(capture, index, cycle_ms=110.0 if index == 1 else 80.0)
    result = read_json(capture.finalize())
    assert result["trials"][1]["latencies_ms"] == [20.0, 20.0]
    assert "whole_control_cycle_exceeded_100ms" in result["trials"][1]["safety_violations"]
    assert (
        compare_trials(capture.plan, result["trials"], live_gpu_verified=False)["quality_gate"]
        is False
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_sha256": "c" * 64},
        {"goal_id": "other"},
        {"epoch": ""},
        {"evaluation_run_id": "other"},
        {"purpose": "production_release"},
        {"deadline_monotonic_ns": 32_000_000_001},
        {"seed": 10001},
    ],
)
def test_attempt_must_match_exact_operator_case_model_goal_and_budget(tmp_path, change):
    capture = recorder(tmp_path / "run")
    binding, initial, cameras = begin(capture)
    with pytest.raises(ContractError):
        capture.start_case(replace(binding, **change), initial, cameras)


@pytest.mark.parametrize(
    "change",
    [
        {"applied_model_sha256": "c" * 64},
        {"applied_action_count": 7},
        {"reference_route_calls": 1},
        {"last_applied_monotonic_ns": 40_000_000_000},
        {"latencies_ms": (float("nan"),)},
        {"command_id": "other"},
    ],
)
def test_trace_cannot_invent_actuation_switch_model_or_exceed_deadline(tmp_path, change):
    from learning.smolvla.rollout import ControlTrace

    capture = recorder(tmp_path / "run")
    binding, initial, cameras = begin(capture)
    capture.start_case(binding, initial, cameras)
    event = ControlTrace(
        command_id=binding.command_id,
        epoch=binding.epoch,
        monotonic_ns=1_100_000_000,
        physics_step=6,
        policy_predict_calls=1,
        applied_action_count=6,
        reference_route_calls=0,
        applied_model_sha256=binding.model_sha256,
        latencies_ms=(20.0,),
        cycle_durations_ms=(80.0,),
        last_applied_monotonic_ns=1_100_000_000,
    )
    with pytest.raises(ContractError):
        capture.append_control(replace(event, **change))


def test_result_reader_rechecks_source_artifact_hashes_not_just_summary_metrics(tmp_path):
    from learning.common import file_digest
    from learning.gr00t.evaluation import verify_evidence
    from learning.smolvla.rollout import verify_recording

    capture = recorder(tmp_path / "run")
    for index in range(40):
        complete(capture, index)
    result = read_json(capture.finalize())
    verify_recording(tmp_path / "run", result, capture.plan)
    assert (
        verify_evidence(
            capture.plan,
            tmp_path / "run",
            SCOPE,
            file_digest(tmp_path / "run" / "results.json"),
            result["schema"],
        )
        == result
    )
    trace = next((tmp_path / "run").rglob("control.jsonl"))
    trace.write_bytes(
        trace.read_bytes().replace(b'"applied_action_count":6', b'"applied_action_count":5', 1)
    )
    with pytest.raises(ContractError, match="checksum|inventory"):
        verify_recording(tmp_path / "run", result, capture.plan)


def test_operator_grant_is_checksum_pinned_and_cannot_authorize_arbitrary_goal(tmp_path):
    from learning.smolvla.rollout import PhysicalRolloutRecorder

    plan, runtime, grant = frozen()
    approved_sha = digest(canonical(grant))
    grant["task"]["goal_id"] = "other"
    with pytest.raises(ContractError):
        PhysicalRolloutRecorder(
            tmp_path / "run",
            plan=plan,
            runtime=runtime,
            grant=grant,
            expected_plan_sha256=digest(canonical(plan)),
            expected_grant_sha256=approved_sha,
        )
    assert not (tmp_path / "run").exists()


def test_frozen_attempt_order_cannot_be_changed_by_a_runtime_caller(tmp_path):
    capture = recorder(tmp_path / "run")
    exposed = capture.schedule
    exposed[0]["policy"] = "after"
    exposed[0]["case"]["seed"] = 1
    assert capture.schedule[0]["policy"] == "before"
    assert capture.schedule[0]["case"]["seed"] == 30001


def test_production_verifier_does_not_accept_legacy_ready_json_in_place_of_a_recorder(tmp_path):
    from learning.gr00t.evaluation import verify_evidence

    plan, runtime, _ = frozen()
    result = {
        "schema": "physicalai.smolvla-paired-results/v2",
        "scope": asdict(SCOPE),
        "plan_sha256": digest(canonical(plan)),
        "runtime": runtime,
        "trials": [],
    }
    write_json(tmp_path / "results.json", result)
    from learning.common import file_digest

    with pytest.raises(ContractError, match="record|missing"):
        verify_evidence(
            plan, tmp_path, SCOPE, file_digest(tmp_path / "results.json"), result["schema"]
        )
