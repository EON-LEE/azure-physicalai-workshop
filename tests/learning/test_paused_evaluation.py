from copy import deepcopy

import pytest

from learning.checks.fixtures import JOINTS, SCOPE
from learning.common import ContractError
from tests.learning.test_paused_contract import profile


def plan(*, bootstrap=False):
    from dataclasses import asdict

    from learning.paused.evaluation import QUALITY_LIMITS

    value = {
        "schema": "physicalai.smolvla-bootstrap-plan/v2"
        if bootstrap
        else "physicalai.smolvla-paired-plan/v2",
        "comparison_kind": "reference_bootstrap" if bootstrap else "paired_policy_eval",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "scope": asdict(SCOPE),
        "control_profile": asdict(profile()),
        "control_profile_sha256": profile().sha256,
        "runtime_sha256": "a" * 64,
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "quality_limits": dict(QUALITY_LIMITS),
        "cases": [
            {
                "episode_id": f"heldout-{index}",
                "seed": 30001 + index,
                "attempt": 0,
                "environment_id": f"paused-case-{index}",
                "revision": "b" * 64,
                "expected_destination_id": "rejected",
                "expected_pose_m": [0.5, 0.2, 0.1],
                "tolerance_m": 0.04,
                "initial_pose_m": [0.3 + index * 0.002, 0.0, 0.1],
                "scene_builder_sha256": "c" * 64,
            }
            for index in range(20)
        ],
    }
    if bootstrap:
        value.update(candidate_model_sha256="f" * 64, reference_controller_sha256="9" * 64)
    else:
        value.update(policy_before_sha256="f" * 64, policy_after_sha256="9" * 64)
    return value


def trials(value, *, before=17, after=18):
    roles = (
        ("reference", "candidate")
        if value["comparison_kind"] == "reference_bootstrap"
        else ("before", "after")
    )
    result = []
    for role, successes in zip(roles, (before, after), strict=True):
        for index, case in enumerate(value["cases"]):
            learned = role != "reference"
            result.append(
                {
                    "episode_id": case["episode_id"],
                    "seed": case["seed"],
                    "attempt": 0,
                    "policy": role,
                    "model_sha256": (
                        None
                        if not learned
                        else value["candidate_model_sha256"]
                        if role == "candidate"
                        else value[f"policy_{role}_sha256"]
                    ),
                    "environment_id": case["environment_id"],
                    "revision": case["revision"],
                    "observed_initial_pose_m": case["initial_pose_m"],
                    "scene_builder_sha256": case["scene_builder_sha256"],
                    "final_pose_m": case["expected_pose_m"]
                    if index < successes
                    else [0.1, 0.1, 0.1],
                    "destination_id": "rejected",
                    "terminated": True,
                    "truncated": False,
                    "failure_reason": None if index < successes else "task_not_completed",
                    "safety_violations": [],
                    "policy_predict_calls": 2 if learned else 0,
                    "reference_route_calls": 0 if learned else 2,
                    "applied_action_count": 12,
                    "final_images": {"inspection": "a" * 64, "overview": "b" * 64},
                    "latencies_wall_ms": [250.0, 260.0] if learned else [],
                    "observation_wall_ms": [100.0, 100.0],
                    "hold_wall_ms": [200.0, 200.0],
                    "interval_wall_ms": [600.0, 600.0],
                    "heartbeat_gap_ms": [20.0] * 12,
                    "wall_duration_ms": 1400.0,
                    "simulation_duration_ms": 200.0,
                    "task_evidence": {
                        "grasp_verified": index < successes,
                        "settled": index < successes,
                        "grasp_evidence_kind": (
                            "measured_lift_proximity_finger_gap_no_contact_sensor"
                        ),
                        "maximum_tcp_speed_m_s": 0.1,
                    },
                }
            )
    return result


def test_paused_metrics_count_all_twenty_pairs_and_absolute_improvement_without_rt_claim():
    from learning.paused.evaluation import compare_trials

    value = plan()
    report = compare_trials(value, trials(value), live_gpu_verified=False)
    assert report["counts"] == {
        "before": {"total": 20, "success": 17},
        "after": {"total": 20, "success": 18},
    }
    assert report["success_rates"]["after"] == 0.9
    assert report["absolute_success_rate_improvement"] == pytest.approx(0.05)
    assert report["quality_gate_passed"] is False
    assert report["conclusion"] == "inconclusive"
    assert report["real_time_admission"] is False
    assert report["latency_wall_ms"]["after"]["max"] == 260.0
    assert report["total_wall_duration_ms"] == 56000.0
    assert report["total_simulation_duration_ms"] == 8000.0


def test_matched_bootstrap_reference_is_not_a_fictional_before_model():
    from learning.paused.evaluation import compare_trials

    value = plan(bootstrap=True)
    report = compare_trials(value, trials(value, before=18, after=18), live_gpu_verified=False)
    assert set(report["counts"]) == {"reference", "candidate"}
    assert "policy_before_sha256" not in report
    assert report["comparison_kind"] == "reference_bootstrap"


@pytest.mark.parametrize(
    ("before", "after", "passed", "conclusion"),
    [
        (17, 18, True, "improved"),
        (18, 18, False, "not_improved"),
        (16, 17, False, "not_improved"),
        (19, 18, False, "not_improved"),
    ],
)
def test_pure_metric_threshold_reference_does_not_relax_ninety_percent_or_absolute_gain(
    before, after, passed, conclusion
):
    from learning.paused.evaluation import compare_trials

    value = plan()
    # This tests only the pure arithmetic branch; no fixture is accepted by the live verifier.
    report = compare_trials(
        value, trials(value, before=before, after=after), live_gpu_verified=True
    )
    assert report["quality_gate_passed"] is passed
    assert report["conclusion"] == conclusion


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed", "old-profile", "lower-threshold"]
)
def test_new_eval_plan_and_exact_attempt_count_fail_closed(change):
    from learning.paused.evaluation import compare_trials

    value = plan()
    attempts = trials(value)
    if change == "missing":
        attempts.pop()
    elif change == "duplicate":
        attempts[-1] = deepcopy(attempts[0])
    elif change == "seed":
        value["cases"][0]["seed"] = 900002
    elif change == "old-profile":
        value["control_profile"]["profile_id"] = "franka-position-hold-10hz-v1"
    else:
        value["quality_limits"]["minimum_success_rate"] = 0.8
    with pytest.raises(ContractError):
        compare_trials(value, attempts, live_gpu_verified=False)


@pytest.mark.parametrize(
    "field",
    [
        "latencies_wall_ms",
        "observation_wall_ms",
        "hold_wall_ms",
        "interval_wall_ms",
        "heartbeat_gap_ms",
    ],
)
def test_measured_deadline_violations_cannot_be_hidden_by_percentiles(field):
    from learning.paused.evaluation import compare_trials

    value = plan()
    attempts = trials(value)
    attempts[-1][field][0] = 5001.0
    attempts[-1]["wall_duration_ms"] = 20000.0
    report = compare_trials(value, attempts, live_gpu_verified=False)
    assert report["quality_gate_passed"] is False
    assert report["resource_violation_count"] > 0


def test_apparent_goal_pose_without_measured_grasp_and_settle_is_not_success():
    from learning.paused.evaluation import compare_trials

    value = plan()
    attempts = trials(value, before=20, after=20)
    for trial in attempts:
        trial["task_evidence"]["grasp_verified"] = False
        trial["task_evidence"]["settled"] = False
    report = compare_trials(value, attempts)
    assert report["success_rates"] == {"before": 0.0, "after": 0.0}
    assert report["quality_gate_passed"] is False


def test_existing_task_watchdog_semantics_are_recomputed_from_actual_tick_states():
    from learning.paused.task import TaskState, evaluate_task_states

    initial = [0.3, 0.0, 0.1]
    goal = [0.3, 0.0, 0.2]
    states = []
    # This fixture tests the physical predicate, not safety-qualified motion or real grasp sensing.
    for step in range(31):
        part = initial if step == 0 else [0.3, 0.0, 0.2]
        tcp = [0.3, 0.0, 0.2] if step < 6 else [0.3, 0.0, 0.3]
        joints = JOINTS if step < 6 else (*JOINTS[:7], 0.04, 0.04)
        states.append(
            TaskState(
                captured_at_utc=f"2026-09-24T00:00:{step:02d}Z",
                monotonic_ns=1_000_000_000 + step * 20_000_000,
                physics_step=step,
                object_position_m=tuple(part),
                object_linear_velocity_m_s=(0.0, 0.0, 0.0),
                tcp_position_m=tuple(tcp),
                joint_positions=joints,
                command_id="test-command",
                epoch="test-epoch",
                policy_predict_calls=0,
                applied_action_count=step,
                reference_route_calls=step,
                applied_model_sha256=None,
            )
        )
    evidence = evaluate_task_states(states, initial=initial, goal=goal)
    assert evidence["grasp_verified"] is True and evidence["settled"] is True
    assert evidence["maximum_tcp_speed_m_s"] > 0.2
    assert evidence["grasp_evidence_kind"].endswith("no_contact_sensor")
    assert evidence["safety_violations"]
