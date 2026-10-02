from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from learning.common import canonical, digest, finite, integer, keys, require, sha256
from learning.contract import Scope
from learning.gr00t.evaluation import TRIAL_KEYS, score_trial, validate_cases
from learning.paused.artifacts import validate_model
from learning.paused.contract import (
    BOOTSTRAP_PLAN_SCHEMA,
    BOOTSTRAP_REPORT_SCHEMA,
    BOOTSTRAP_RESULT_SCHEMA,
    EXECUTION_TIMING,
    PLAN_SCHEMA,
    REPORT_SCHEMA,
    RESULT_SCHEMA,
    PausedControlProfile,
)

QUALITY_LIMITS = {
    "heldout_case_count": 20,
    "minimum_success_rate": 0.9,
    "minimum_absolute_improvement": 0.05,
    "maximum_safety_violations": 0,
}
PAUSED_TRIAL_KEYS = (TRIAL_KEYS - {"duration_ms", "latencies_ms"}) | {
    "wall_duration_ms",
    "simulation_duration_ms",
    "latencies_wall_ms",
    "observation_wall_ms",
    "hold_wall_ms",
    "interval_wall_ms",
    "heartbeat_gap_ms",
    "task_evidence",
}


def roles(plan: dict) -> tuple[str, str]:
    return (
        ("reference", "candidate")
        if plan["comparison_kind"] == "reference_bootstrap"
        else ("before", "after")
    )


def expected_model(plan: dict, role: str) -> str | None:
    if role == "reference":
        return None
    return plan["candidate_model_sha256"] if role == "candidate" else plan[f"policy_{role}_sha256"]


def validate_plan(plan: dict) -> dict:
    bootstrap = plan.get("comparison_kind") == "reference_bootstrap"
    keys(
        plan,
        {
            "schema",
            "comparison_kind",
            "execution_timing",
            "real_time_admission",
            "scope",
            "control_profile",
            "control_profile_sha256",
            "runtime_sha256",
            "criteria_sha256",
            "frozen_plan_sha256",
            "quality_limits",
            "cases",
        }
        | (
            {"candidate_model_sha256", "reference_controller_sha256"}
            if bootstrap
            else {"policy_before_sha256", "policy_after_sha256"}
        ),
        "paused evaluation plan",
    )
    require(
        plan["schema"] == (BOOTSTRAP_PLAN_SCHEMA if bootstrap else PLAN_SCHEMA)
        and plan["comparison_kind"] in ("reference_bootstrap", "paired_policy_eval")
        and plan["execution_timing"] == EXECUTION_TIMING
        and plan["real_time_admission"] is False,
        "Unknown or real-time evaluation mode/schema",
    )
    Scope(**keys(plan["scope"], {"tenant_id", "owner_id"}, "evaluation scope")).validate()
    profile = PausedControlProfile(
        **keys(
            plan["control_profile"],
            set(PausedControlProfile.__dataclass_fields__),
            "paused profile",
        )
    )
    require(profile.sha256 == sha256(plan["control_profile_sha256"]), "Wrong evaluation profile")
    for name in ("runtime_sha256", "criteria_sha256", "frozen_plan_sha256"):
        sha256(plan[name], name)
    for role in roles(plan):
        if role != "reference":
            sha256(expected_model(plan, role), "actual evaluated model")
    if bootstrap:
        sha256(plan["reference_controller_sha256"], "reference controller source")
    else:
        require(
            plan["policy_before_sha256"] != plan["policy_after_sha256"], "No actual policy change"
        )
    require(
        canonical(plan["quality_limits"]) == canonical(QUALITY_LIMITS),
        "Frozen quality thresholds changed",
    )
    cases = validate_cases(plan["cases"])
    require(
        len(cases) == 20
        and {case["seed"] for case in cases.values()} == set(range(30001, 30021))
        and all(case["attempt"] == 0 and case["tolerance_m"] == 0.04 for case in cases.values()),
        "Exactly twenty predeclared held-out cases without retries are required",
    )
    require(
        any(
            max(case["initial_pose_m"][axis] for case in cases.values())
            - min(case["initial_pose_m"][axis] for case in cases.values())
            >= 0.02
            for axis in (0, 1)
        ),
        "Held-out initial XY positions do not span the frozen two-centimetre requirement",
    )
    return cases


def _trial(trial: dict, case: dict, *, learned: bool, profile: PausedControlProfile):
    keys(trial, PAUSED_TRIAL_KEYS, "paused physical trial")
    common = {key: trial[key] for key in TRIAL_KEYS - {"duration_ms", "latencies_ms"}}
    common.update(
        duration_ms=trial["wall_duration_ms"],
        latencies_ms=trial["latencies_wall_ms"] if learned else trial["interval_wall_ms"],
    )
    success, error, latencies = score_trial(common, case, learned=learned)
    steps = integer(trial["applied_action_count"], "actual applied physics ticks")
    simulation_ms = finite(trial["simulation_duration_ms"], "measured simulation duration")
    require(
        abs(simulation_ms - steps * 1000 / profile.physics_hz) <= 1e-6,
        "Simulation duration must derive from actual physics ticks, not divided wall latency",
    )
    resource_violations = int(
        trial["wall_duration_ms"] > profile.max_episode_wall_ms
        or steps > profile.max_simulation_steps
    )
    complete_intervals = steps // profile.hold_steps
    for field, cap in (
        ("latencies_wall_ms", profile.max_policy_wall_ms),
        ("observation_wall_ms", profile.max_observation_wall_ms),
        ("hold_wall_ms", profile.max_hold_wall_ms),
        ("interval_wall_ms", profile.max_interval_wall_ms),
        ("heartbeat_gap_ms", profile.max_heartbeat_wall_ms),
    ):
        values = trial[field]
        require(isinstance(values, list), "Measured wall samples must be explicit lists")
        measured = [finite(item, field) for item in values]
        require(all(item >= 0 for item in measured), "Negative measured wall duration")
        resource_violations += sum(item > cap for item in measured)
        if field in ("observation_wall_ms", "hold_wall_ms", "interval_wall_ms"):
            require(
                len(measured) == complete_intervals, "Missing complete-interval timing evidence"
            )
        elif field == "heartbeat_gap_ms":
            require(
                bool(measured) if steps else not measured,
                "Missing actual main-thread heartbeat samples",
            )
    task = trial["task_evidence"]
    require(
        isinstance(task, dict)
        and type(task.get("grasp_verified")) is bool
        and type(task.get("settled")) is bool
        and task.get("grasp_evidence_kind")
        == "measured_lift_proximity_finger_gap_no_contact_sensor",
        "Missing recomputed physical grasp/release/settling evidence",
    )
    speed = finite(task.get("maximum_tcp_speed_m_s"), "actual TCP speed")
    require(speed >= 0, "Negative measured speed")
    success = (
        success
        and task["grasp_verified"]
        and task["settled"]
        and speed <= 0.2 + 1e-9
        and resource_violations == 0
    )
    if success and learned:
        require(
            steps == trial["policy_predict_calls"] * profile.hold_steps,
            "Successful learned trial contains controls without a fresh bound model prediction",
        )
    return success, error, latencies if learned else [], resource_violations


def _latency(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50": ordered[math.ceil(len(ordered) * 0.5) - 1] if ordered else None,
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
        "max": max(ordered) if ordered else None,
    }


def compare_trials(plan: dict, trials: list[dict], *, live_gpu_verified: bool = False) -> dict:
    """Pure metric projection; use evaluate_pair/evaluate_bootstrap to verify physical artifacts."""
    cases = validate_plan(plan)
    require(type(live_gpu_verified) is bool, "Explicit evidence-verification flag required")
    require(
        isinstance(trials, list) and len(trials) == 40, "Missing/extra frozen physical attempts"
    )
    profile = PausedControlProfile(**plan["control_profile"])
    counts = {role: {"total": 0, "success": 0} for role in roles(plan)}
    latency = {role: [] for role in counts}
    seen, details, safety, resources = set(), [], 0, 0
    for trial in trials:
        keys(trial, PAUSED_TRIAL_KEYS, "paused trial")
        role, key = trial["policy"], (trial["episode_id"], trial["attempt"])
        require(
            role in counts and key in cases and (role, *key) not in seen,
            "Unknown/duplicate controller/case attempt",
        )
        require(
            trial["model_sha256"] == expected_model(plan, role), "Wrong actual controller/model"
        )
        success, error, values, violations = _trial(
            trial, cases[key], learned=role != "reference", profile=profile
        )
        seen.add((role, *key))
        counts[role]["total"] += 1
        counts[role]["success"] += int(success)
        latency[role].extend(values)
        safety += len(trial["safety_violations"])
        resources += violations
        details.append({**trial, "physical_success": success, "position_error_m": error})
    require(
        seen == {(role, *key) for role in counts for key in cases}, "Unpaired held-out attempts"
    )
    rates = {role: value["success"] / 20 for role, value in counts.items()}
    first, second = roles(plan)
    improvement = (counts[second]["success"] - counts[first]["success"]) / 20
    bootstrap = plan["comparison_kind"] == "reference_bootstrap"
    passed = (
        live_gpu_verified
        and safety == resources == 0
        and rates[second] >= QUALITY_LIMITS["minimum_success_rate"]
        and (
            rates[first] >= QUALITY_LIMITS["minimum_success_rate"]
            if bootstrap
            else improvement >= QUALITY_LIMITS["minimum_absolute_improvement"]
        )
    )
    conclusion = (
        "inconclusive"
        if not live_gpu_verified
        else ("improved" if not bootstrap and passed else "not_improved")
    )
    report = {
        "schema": BOOTSTRAP_REPORT_SCHEMA if bootstrap else REPORT_SCHEMA,
        "policy_type": "smolvla",
        "comparison_kind": plan["comparison_kind"],
        "execution_timing": EXECUTION_TIMING,
        "real_time_admission": False,
        "scope": plan["scope"],
        "control_profile_sha256": plan["control_profile_sha256"],
        "runtime_sha256": plan["runtime_sha256"],
        "criteria_sha256": plan["criteria_sha256"],
        "frozen_plan_sha256": plan["frozen_plan_sha256"],
        "evaluation_plan_sha256": digest(canonical(plan)),
        "results_sha256": None,
        "quality_gate_passed": passed,
        "conclusion": conclusion,
        "counts": counts,
        "success_rates": rates,
        "absolute_success_rate_improvement": improvement,
        "latency_wall_ms": {role: _latency(values) for role, values in latency.items()},
        "total_wall_duration_ms": sum(trial["wall_duration_ms"] for trial in trials),
        "total_simulation_duration_ms": sum(trial["simulation_duration_ms"] for trial in trials),
        "safety_violation_count": safety,
        "resource_violation_count": resources,
        "total_trial_count": len(trials),
        "trials": details,
        "live_gpu_verified": live_gpu_verified,
    }
    if bootstrap:
        report.update(
            candidate_model_sha256=plan["candidate_model_sha256"],
            reference_controller_sha256=plan["reference_controller_sha256"],
        )
    else:
        report.update(
            policy_before_sha256=plan["policy_before_sha256"],
            policy_after_sha256=plan["policy_after_sha256"],
        )
    return report


def validate_models(plan: dict, models: dict[str, Path], *, scope: Scope) -> dict:
    cases = validate_plan(plan)
    loaded = {}
    for role, path in models.items():
        model = validate_model(
            path, expected_scope=scope, expected_model_sha256=expected_model(plan, role)
        )
        require(
            model["control_profile"] == plan["control_profile"]
            and model["criteria_sha256"] == plan["criteria_sha256"]
            and model["frozen_plan_sha256"] == plan["frozen_plan_sha256"],
            "Evaluated model differs from new profile or frozen training/evaluation conditions",
        )
        for episode in model["training"]["episodes"]:
            require(
                all(
                    episode["episode_id"] != case["episode_id"] and episode["seed"] != case["seed"]
                    for case in cases.values()
                ),
                "Training/final-held-out leakage",
            )
        loaded[role] = model
    if "before" in loaded:
        require(
            loaded["before"]["task"] == loaded["after"]["task"]
            and (
                loaded["after"]["training"]["parent_model_sha256"] == plan["policy_before_sha256"]
                or plan["policy_before_sha256"]
                in loaded["after"]["training"].get("ancestor_model_sha256s", [])
            ),
            "Paired task or actual incremental model lineage differs",
        )
    return loaded


def _evaluate(
    plan: dict,
    evidence_root: Path,
    models: dict[str, Path],
    *,
    scope: Scope,
    expected_plan_sha256: str,
    expected_results_sha256: str,
) -> dict:
    validate_plan(plan)
    require(
        digest(canonical(plan)) == sha256(expected_plan_sha256) and plan["scope"] == asdict(scope),
        "Frozen evaluation plan/scope changed",
    )
    validate_models(plan, models, scope=scope)
    from learning.paused.rollout import verify_recording

    result = verify_recording(evidence_root, plan, scope, expected_results_sha256)
    require(
        result["schema"] == (BOOTSTRAP_RESULT_SCHEMA if "candidate" in models else RESULT_SCHEMA),
        "Wrong paused result kind",
    )
    report = compare_trials(plan, result["trials"], live_gpu_verified=True)
    report["results_sha256"] = expected_results_sha256
    return report


def evaluate_pair(plan, evidence_root, before_root, after_root, **kwargs):
    require(plan.get("comparison_kind") == "paired_policy_eval", "Not a paired policy plan")
    return _evaluate(plan, evidence_root, {"before": before_root, "after": after_root}, **kwargs)


def evaluate_bootstrap(plan, evidence_root, candidate_root, **kwargs):
    require(plan.get("comparison_kind") == "reference_bootstrap", "Not a reference bootstrap plan")
    return _evaluate(plan, evidence_root, {"candidate": candidate_root}, **kwargs)
