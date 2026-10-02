import pytest

from learning.common import ContractError


def data():
    plan = {
        "schema": "physicalai.gr00t-bootstrap-plan/v1",
        "candidate_model_sha256": "a" * 64,
        "reference_controller_sha256": "b" * 64,
        "scope": {"tenant_id": "11111111-1111-4111-8111-111111111111", "owner_id": "c" * 64},
        "control_profile_sha256": "d" * 64,
        "runtime_sha256": "e" * 64,
        "minimum_pairs": 20,
        "max_latency_p95_ms": 80.0,
        "minimum_candidate_success_rate": 0.9,
        "cases": [],
    }
    trials = []
    for seed in range(100, 120):
        case = {
            "episode_id": f"heldout-{seed}",
            "seed": seed,
            "attempt": 0,
            "environment_id": "line",
            "revision": "f" * 64,
            "initial_pose_m": [0.4 + 0.002 * (seed - 100), 0, 0.2],
            "scene_builder_sha256": "7" * 64,
            "expected_destination_id": "tray",
            "expected_pose_m": [0.3, 0.1, 0.2],
            "tolerance_m": 0.04,
        }
        plan["cases"].append(case)
        for policy in ("reference", "candidate"):
            trials.append(
                {
                    "episode_id": case["episode_id"],
                    "seed": seed,
                    "attempt": 0,
                    "policy": policy,
                    "model_sha256": None if policy == "reference" else "a" * 64,
                    "environment_id": "line",
                    "revision": "f" * 64,
                    "observed_initial_pose_m": case["initial_pose_m"],
                    "scene_builder_sha256": "7" * 64,
                    "final_pose_m": [0.3, 0.1, 0.2],
                    "destination_id": "tray",
                    "terminated": True,
                    "truncated": False,
                    "failure_reason": None,
                    "latencies_ms": [10.0, 20.0],
                    "duration_ms": 200,
                    "safety_violations": [],
                    "policy_predict_calls": 0 if policy == "reference" else 2,
                    "reference_route_calls": 1 if policy == "reference" else 0,
                    "applied_action_count": 12,
                    "final_images": {"inspection": "1" * 64, "overview": "2" * 64},
                }
            )
    return plan, trials


def test_bootstrap_reference_is_never_a_fictional_before_policy():
    from learning.gr00t.bootstrap import compare_bootstrap_trials

    plan, trials = data()
    result = compare_bootstrap_trials(plan, trials, live_gpu_verified=True)
    assert result["comparison_kind"] == "reference_bootstrap"
    assert result["quality_gate_passed"] is True
    assert result["candidate_model_sha256"] == "a" * 64
    assert result["reference_controller_sha256"] == "b" * 64
    assert "policy_before_sha256" not in result and "conclusion" not in result
    assert result["trials"][0]["model_sha256"] is None
    assert not compare_bootstrap_trials(plan, trials, live_gpu_verified=False)[
        "quality_gate_passed"
    ]


def test_candidate_cannot_borrow_scripted_routes_during_bootstrap():
    from learning.gr00t.bootstrap import compare_bootstrap_trials

    plan, trials = data()
    trials[1]["reference_route_calls"] = 1
    with pytest.raises(ContractError, match="reference route"):
        compare_bootstrap_trials(plan, trials, live_gpu_verified=True)
