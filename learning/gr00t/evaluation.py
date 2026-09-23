from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    finite,
    integer,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
    token,
    vector,
)
from learning.contract import ControlProfile, Provenance, Scope, png_dimensions
from learning.gr00t.artifacts import validate_model

PLAN_SCHEMA = "physicalai.gr00t-paired-plan/v1"
RESULT_SCHEMA = "physicalai.gr00t-paired-results/v1"
REPORT_SCHEMA = "physicalai.gr00t-paired-report/v1"
CASE_KEYS = {
    "episode_id",
    "seed",
    "attempt",
    "environment_id",
    "revision",
    "expected_destination_id",
    "expected_pose_m",
    "tolerance_m",
    "initial_pose_m",
    "scene_builder_sha256",
}
TRIAL_KEYS = {
    "episode_id",
    "seed",
    "attempt",
    "policy",
    "model_sha256",
    "environment_id",
    "revision",
    "final_pose_m",
    "destination_id",
    "terminated",
    "truncated",
    "failure_reason",
    "latencies_ms",
    "duration_ms",
    "safety_violations",
    "policy_predict_calls",
    "applied_action_count",
    "reference_route_calls",
    "final_images",
    "observed_initial_pose_m",
    "scene_builder_sha256",
}


def validate_plan(plan: dict, *, schema: str = PLAN_SCHEMA) -> dict:
    keys(
        plan,
        {
            "schema",
            "policy_before_sha256",
            "policy_after_sha256",
            "scope",
            "control_profile_sha256",
            "runtime_sha256",
            "minimum_pairs",
            "max_latency_p95_ms",
            "minimum_after_success_rate",
            "cases",
        },
        "paired P0/P1 plan",
    )
    require(plan["schema"] == schema, "Unknown paired evaluation schema")
    Scope(**keys(plan["scope"], {"tenant_id", "owner_id"}, "scope")).validate()
    for name in (
        "policy_before_sha256",
        "policy_after_sha256",
        "control_profile_sha256",
        "runtime_sha256",
    ):
        sha256(plan[name], name)
    require(
        plan["policy_before_sha256"] != plan["policy_after_sha256"],
        "P0/P1 cannot be the same artifact",
    )
    integer(plan["minimum_pairs"], "minimum complete pairs", 20, 10000)
    require(
        0 < finite(plan["max_latency_p95_ms"], "latency threshold") <= 80, "Invalid latency gate"
    )
    require(
        0.9 <= finite(plan["minimum_after_success_rate"], "success threshold") <= 1,
        "Invalid success gate",
    )
    return validate_cases(plan["cases"])


def validate_cases(values: list[dict]) -> dict:
    require(isinstance(values, list) and bool(values), "No frozen held-out cases")
    cases, seen_seeds = {}, set()
    initial_poses = []
    for case in values:
        keys(case, CASE_KEYS, "paired case")
        for name in ("episode_id", "environment_id", "expected_destination_id"):
            token(case[name], name)
        sha256(case["revision"])
        integer(case["seed"], "held-out seed", 0, 2**32 - 1)
        integer(case["attempt"], "paired attempt", 0, 100)
        vector(case["expected_pose_m"], 3, "approved goal pose")
        initial_poses.append(vector(case["initial_pose_m"], 3, "approved initial pose"))
        sha256(case["scene_builder_sha256"], "reviewed placement builder")
        require(
            0 < finite(case["tolerance_m"], "pose tolerance") <= 0.04, "Unapproved pose tolerance"
        )
        key = (case["episode_id"], case["attempt"])
        seed_key = (case["seed"], case["attempt"])
        require(key not in cases and seed_key not in seen_seeds, "Duplicate frozen case")
        cases[key] = case
        seen_seeds.add(seed_key)
    require(
        any(
            max(pose[axis] for pose in initial_poses) - min(pose[axis] for pose in initial_poses)
            >= 0.02
            for axis in range(3)
        ),
        "Held-out placement must span at least 2cm; defect-only seeds are not generalization",
    )
    return cases


def score_trial(trial: dict, case: dict, *, learned: bool) -> tuple[bool, list[float], list[float]]:
    keys(trial, TRIAL_KEYS, "physical trial")
    integer(trial["attempt"], "actual attempt", 0, 100)
    integer(trial["seed"], "actual seed", 0, 2**32 - 1)
    require(
        all(trial[name] == case[name] for name in ("seed", "environment_id", "revision")),
        "Physical trial environment/seed mismatch",
    )
    observed_start = vector(trial["observed_initial_pose_m"], 3, "observed initial pose")
    require(
        trial["scene_builder_sha256"] == case["scene_builder_sha256"]
        and all(
            abs(actual - expected) <= 0.001
            for actual, expected in zip(observed_start, case["initial_pose_m"], strict=True)
        ),
        "Observed initial pose/builder differs from the frozen paired case",
    )
    reference_calls = integer(trial["reference_route_calls"], "reference route calls")
    predictions = integer(trial["policy_predict_calls"], "prediction calls")
    applied = integer(trial["applied_action_count"], "actual applied physics actions")
    require(
        reference_calls == 0 if learned else predictions == 0,
        "Scripted reference route substituted for a learned policy",
    )
    duration = finite(trial["duration_ms"], "episode duration")
    require(duration > 0, "Missing measured episode duration")
    require(
        isinstance(trial["latencies_ms"], list)
        and len(trial["latencies_ms"]) <= (predictions if learned else applied),
        "Latency sample count exceeds actual controller calls",
    )
    values = [finite(value, "measured latency") for value in trial["latencies_ms"]]
    require(
        all(value >= 0 for value in values) and sum(values) <= duration, "Invalid measured latency"
    )
    require(
        type(trial["terminated"]) is bool
        and type(trial["truncated"]) is bool
        and trial["terminated"] != trial["truncated"],
        "Trial must terminate or truncate exactly once",
    )
    require(
        trial["failure_reason"] is None
        or (isinstance(trial["failure_reason"], str) and 0 < len(trial["failure_reason"]) <= 512),
        "Invalid explicit failure reason",
    )
    require(
        isinstance(trial["safety_violations"], list)
        and all(isinstance(item, str) and bool(item) for item in trial["safety_violations"]),
        "Invalid safety trace",
    )
    for value in keys(
        trial["final_images"], {"inspection", "overview"}, "final image evidence"
    ).values():
        sha256(value, "final image checksum")
    position = vector(trial["final_pose_m"], 3, "measured final object pose")
    error = [abs(a - b) for a, b in zip(position, case["expected_pose_m"], strict=True)]
    success = (
        trial["terminated"]
        and not trial["truncated"]
        and trial["failure_reason"] is None
        and not trial["safety_violations"]
        and trial["destination_id"] == case["expected_destination_id"]
        and max(error) <= case["tolerance_m"]
    )
    if success and learned:
        require(
            predictions > 0 and applied >= predictions * 6 and len(values) == predictions,
            "Successful trial lacks actual ten-Hz model/control evidence",
        )
    elif success:
        require(
            reference_calls > 0 and applied > 0 and bool(values),
            "Missing actual reference evidence",
        )
    return success, error, values


def compare_trials(
    plan: dict,
    trials: list[dict],
    *,
    live_gpu_verified: bool,
    plan_schema: str = PLAN_SCHEMA,
    report_schema: str = REPORT_SCHEMA,
    policy_type: str = "gr00t_n1_5",
) -> dict:
    """Pure metrics; production must use evaluate_pair to verify artifacts."""
    cases = validate_plan(plan, schema=plan_schema)
    require(
        isinstance(trials, list) and len(trials) == 2 * len(cases), "Missing/extra paired trials"
    )
    seen, details, results = set(), [], {}
    counts = {policy: {"total": 0, "success": 0} for policy in ("before", "after")}
    latency = {"before": [], "after": []}
    safety_count = 0
    for trial in trials:
        keys(trial, TRIAL_KEYS, "policy trial")
        policy = trial["policy"]
        require(policy in ("before", "after"), "A scripted reference controller is not P0/P1")
        key = (trial["episode_id"], trial["attempt"])
        binding = (policy, *key)
        require(key in cases and binding not in seen, "Unknown/duplicate paired trial")
        seen.add(binding)
        case = cases[key]
        require(
            all(trial[name] == case[name] for name in ("seed", "environment_id", "revision"))
            and trial["model_sha256"] == plan[f"policy_{policy}_sha256"],
            "Policy/checkpoint/environment/seed mismatch",
        )
        success, error, values = score_trial(trial, case, learned=True)
        counts[policy]["total"] += 1
        counts[policy]["success"] += int(success)
        safety_count += len(trial["safety_violations"])
        latency[policy].extend(values)
        results[binding] = success
        details.append({**trial, "physical_success": success, "position_error_m": error})
    expected = {(role, *key) for role in counts for key in cases}
    require(seen == expected, "Unpaired held-out results")
    wins = sum(not results[("before", *key)] and results[("after", *key)] for key in cases)
    losses = sum(results[("before", *key)] and not results[("after", *key)] for key in cases)
    rates = {role: value["success"] / value["total"] for role, value in counts.items()}
    p95 = {
        role: sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None
        for role, values in latency.items()
    }
    enough = len(cases) >= plan["minimum_pairs"]
    conclusion = (
        "inconclusive"
        if not live_gpu_verified or not enough
        else ("improved" if wins > losses else "not_improved")
    )
    quality = (
        live_gpu_verified
        and enough
        and safety_count == 0
        and rates["after"] >= plan["minimum_after_success_rate"]
        and rates["after"] >= rates["before"]
        and p95["after"] is not None
        and p95["after"] <= plan["max_latency_p95_ms"]
    )
    return {
        "schema": report_schema,
        "policy_type": policy_type,
        "plan_sha256": digest(canonical(plan)),
        "scope": plan["scope"],
        "control_profile_sha256": plan["control_profile_sha256"],
        "runtime_sha256": plan["runtime_sha256"],
        "policy_before_sha256": plan["policy_before_sha256"],
        "policy_after_sha256": plan["policy_after_sha256"],
        "conclusion": conclusion,
        "quality_gate": quality,
        "counts": counts,
        "paired_wins": wins,
        "paired_losses": losses,
        "success_rates": rates,
        "latency_p95_ms": p95,
        "safety_violation_count": safety_count,
        "total_trial_count": len(trials),
        "trials": details,
        "live_gpu_verified": live_gpu_verified,
    }


def evaluate_pair(
    plan: dict,
    evidence_root: Path,
    before_root: Path,
    after_root: Path,
    *,
    scope: Scope,
    expected_plan_sha256: str,
    expected_results_sha256: str,
    model_validator=validate_model,
    plan_schema: str = PLAN_SCHEMA,
    result_schema: str = RESULT_SCHEMA,
    report_schema: str = REPORT_SCHEMA,
    policy_type: str = "gr00t_n1_5",
) -> dict:
    require(digest(canonical(plan)) == sha256(expected_plan_sha256), "Paired plan changed")
    cases = validate_plan(plan, schema=plan_schema)
    require(plan["scope"] == asdict(scope), "Paired plan owner mismatch")
    models = {
        role: model_validator(
            root, expected_scope=scope, expected_model_sha256=plan[f"policy_{role}_sha256"]
        )
        for role, root in (("before", before_root), ("after", after_root))
    }
    require(
        models["after"]["training"]["parent_model_sha256"] == plan["policy_before_sha256"]
        or plan["policy_before_sha256"]
        in models["after"]["training"].get("ancestor_model_sha256s", []),
        "P1 is not a descendant of the approved actual P0",
    )
    require(models["before"]["task"] == models["after"]["task"], "P0/P1 task changed")
    heldout_ids = {case["episode_id"] for case in cases.values()}
    heldout_seeds = {case["seed"] for case in cases.values()}
    for model in models.values():
        require(
            ControlProfile(**model["control_profile"]).sha256 == plan["control_profile_sha256"],
            "P0/P1 servo profile differs",
        )
        for episode in model["training"]["episodes"]:
            require(
                episode["episode_id"] not in heldout_ids and episode["seed"] not in heldout_seeds,
                "Training/held-out episode or seed leakage in P0/P1 comparison",
            )
    result = verify_evidence(plan, evidence_root, scope, expected_results_sha256, result_schema)
    return compare_trials(
        plan,
        result["trials"],
        live_gpu_verified=True,
        plan_schema=plan_schema,
        report_schema=report_schema,
        policy_type=policy_type,
    )


def verify_evidence(plan: dict, root: Path, scope: Scope, checksum: str, schema: str) -> dict:
    path = safe_path(root, "results.json")
    require(file_digest(path) == sha256(checksum), "Physical result checksum mismatch")
    result = keys(
        read_json(path, max_bytes=64 * 1024 * 1024),
        {
            "schema",
            "scope",
            "plan_sha256",
            "runtime",
            "trials",
        },
        "paired evidence",
    )
    require(
        result["schema"] == schema
        and result["scope"] == asdict(scope)
        and result["plan_sha256"] == digest(canonical(plan))
        and digest(canonical(result["runtime"])) == plan["runtime_sha256"],
        "Unbound live paired runtime evidence",
    )
    from learning.evaluation import validate_runtime

    validate_runtime(result["runtime"])
    Provenance(**result["runtime"]["provenance"]).validate(require_live=True)
    require(isinstance(result["trials"], list), "Missing physical trial evidence")
    for trial in result["trials"]:
        keys(trial, TRIAL_KEYS, "physical trial")
        token(trial["policy"], "evidence controller")
        token(trial["episode_id"], "evidence episode")
        integer(trial["attempt"], "evidence attempt", 0, 100)
        for camera, checksum in trial["final_images"].items():
            relative = f"{trial['policy']}/{trial['episode_id']}/{trial['attempt']}/{camera}.png"
            path = safe_path(root, relative)
            require(path.stat().st_size <= 32 * 1024 * 1024, "Oversized final camera evidence")
            payload = path.read_bytes()
            require(digest(payload) == checksum, "Paired final-image checksum mismatch")
            png_dimensions(payload)
    return result
