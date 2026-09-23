import pytest

from learning.common import ContractError


def trials(*, after_success=True, reference_calls=0):
    return [
        {
            "episode_id": f"eval-{seed}",
            "seed": seed,
            "attempt": 0,
            "policy": policy,
            "model_sha256": ("a" if policy == "before" else "b") * 64,
            "environment_id": "line",
            "revision": "c" * 64,
            "observed_initial_pose_m": [0.4 + (seed - 100) * 0.002, 0.0, 0.2],
            "scene_builder_sha256": "7" * 64,
            "final_pose_m": [0.3, 0.1, 0.2]
            if policy == "after" and after_success
            else [0.6, 0.1, 0.2],
            "destination_id": "tray",
            "terminated": True,
            "truncated": False,
            "failure_reason": None,
            "latencies_ms": [10.0, 20.0],
            "duration_ms": 200.0,
            "safety_violations": [],
            "policy_predict_calls": 2,
            "applied_action_count": 12,
            "reference_route_calls": reference_calls,
            "final_images": {"inspection": "d" * 64, "overview": "e" * 64},
        }
        for seed in range(100, 120)
        for policy in ("before", "after")
    ]


def plan():
    return {
        "schema": "physicalai.gr00t-paired-plan/v1",
        "policy_before_sha256": "a" * 64,
        "policy_after_sha256": "b" * 64,
        "scope": {"tenant_id": "11111111-1111-4111-8111-111111111111", "owner_id": "f" * 64},
        "control_profile_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "minimum_pairs": 20,
        "max_latency_p95_ms": 80.0,
        "minimum_after_success_rate": 0.9,
        "cases": [
            {
                "episode_id": f"eval-{seed}",
                "seed": seed,
                "attempt": 0,
                "environment_id": "line",
                "revision": "c" * 64,
                "initial_pose_m": [0.4 + (seed - 100) * 0.002, 0.0, 0.2],
                "scene_builder_sha256": "7" * 64,
                "expected_destination_id": "tray",
                "expected_pose_m": [0.3, 0.1, 0.2],
                "tolerance_m": 0.04,
            }
            for seed in range(100, 120)
        ],
    }


def test_before_after_is_distinct_from_scripted_baseline_and_reports_no_improvement_honestly():
    from learning.gr00t.evaluation import compare_trials

    improved = compare_trials(plan(), trials(), live_gpu_verified=True)
    assert improved["conclusion"] == "improved"
    assert improved["counts"] == {
        "before": {"total": 20, "success": 0},
        "after": {"total": 20, "success": 20},
    }
    assert improved["quality_gate"] is True
    unchanged = compare_trials(plan(), trials(after_success=False), live_gpu_verified=True)
    assert unchanged["conclusion"] == "not_improved" and unchanged["quality_gate"] is False
    assert compare_trials(plan(), trials(), live_gpu_verified=False)["conclusion"] == "inconclusive"


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "wrong-model", "reference-route", "NaN"]
)
def test_paired_policy_comparison_never_drops_failures_or_substitutes_reference(change):
    from learning.gr00t.evaluation import compare_trials

    data = trials()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = data[0]
    elif change == "wrong-model":
        data[0]["model_sha256"] = "0" * 64
    elif change == "reference-route":
        data[0]["reference_route_calls"] = 1
    else:
        data[0]["duration_ms"] = float("nan")
    with pytest.raises(ContractError):
        compare_trials(plan(), data, live_gpu_verified=True)


def test_twenty_defect_only_seeds_are_not_heldout_placement_generalization():
    from learning.gr00t.evaluation import compare_trials

    frozen = plan()
    for case in frozen["cases"]:
        case["initial_pose_m"] = [0.4, 0.0, 0.2]
    with pytest.raises(ContractError, match="placement"):
        compare_trials(frozen, trials(), live_gpu_verified=True)


def test_before_and_after_cannot_change_the_observed_start_pose():
    from learning.gr00t.evaluation import compare_trials

    data = trials()
    data[1]["observed_initial_pose_m"] = [0.6, 0.0, 0.2]
    with pytest.raises(ContractError, match="initial pose"):
        compare_trials(plan(), data, live_gpu_verified=True)
