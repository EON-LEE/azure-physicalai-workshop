from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from learning.common import canonical, digest, finite, integer, keys, require, sha256
from learning.contract import ControlProfile, Scope
from learning.gr00t.artifacts import validate_model
from learning.gr00t.evaluation import score_trial, validate_cases, verify_evidence

PLAN_SCHEMA = "physicalai.gr00t-bootstrap-plan/v1"
RESULT_SCHEMA = "physicalai.gr00t-bootstrap-results/v1"


def validate_plan(plan: dict, *, schema: str = PLAN_SCHEMA) -> dict:
    keys(
        plan,
        {
            "schema",
            "candidate_model_sha256",
            "reference_controller_sha256",
            "scope",
            "control_profile_sha256",
            "runtime_sha256",
            "minimum_pairs",
            "max_latency_p95_ms",
            "minimum_candidate_success_rate",
            "cases",
        },
        "privileged bootstrap evaluation plan",
    )
    require(plan["schema"] == schema, "Wrong bootstrap plan kind")
    Scope(**plan["scope"]).validate()
    for name in (
        "candidate_model_sha256",
        "reference_controller_sha256",
        "control_profile_sha256",
        "runtime_sha256",
    ):
        sha256(plan[name], name)
    integer(plan["minimum_pairs"], "bootstrap minimum pairs", 20, 10000)
    require(0 < finite(plan["max_latency_p95_ms"], "latency limit") <= 80, "Invalid latency limit")
    require(
        0.9 <= finite(plan["minimum_candidate_success_rate"], "success limit") <= 1,
        "Invalid bootstrap success threshold",
    )
    return validate_cases(plan["cases"])


def compare_bootstrap_trials(
    plan: dict,
    trials: list[dict],
    *,
    live_gpu_verified: bool,
    plan_schema: str = PLAN_SCHEMA,
    report_schema: str = "physicalai.gr00t-bootstrap-report/v1",
    policy_type: str = "gr00t_n1_5",
) -> dict:
    cases = validate_plan(plan, schema=plan_schema)
    require(len(trials) == 2 * len(cases), "Incomplete reference/candidate bootstrap trials")
    seen, evidence, latencies = set(), [], []
    counts = {name: {"total": 0, "success": 0} for name in ("reference", "candidate")}
    safety = 0
    for trial in trials:
        role = trial["policy"]
        require(role in counts, "Bootstrap has no fictional before/after model")
        key = (trial["episode_id"], trial["attempt"])
        pair_key = (role, *key)
        require(key in cases and pair_key not in seen, "Unknown/duplicate bootstrap case")
        expected_model = plan["candidate_model_sha256"] if role == "candidate" else None
        require(trial["model_sha256"] == expected_model, "Reference is not a model checkpoint")
        success, error, values = score_trial(trial, cases[key], learned=role == "candidate")
        seen.add(pair_key)
        counts[role]["total"] += 1
        counts[role]["success"] += int(success)
        safety += len(trial["safety_violations"])
        if role == "candidate":
            latencies.extend(values)
        evidence.append({**trial, "physical_success": success, "position_error_m": error})
    require(
        seen == {(role, *key) for role in counts for key in cases}, "Unpaired bootstrap results"
    )
    rates = {role: value["success"] / value["total"] for role, value in counts.items()}
    p95 = sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1] if latencies else None
    passed = (
        live_gpu_verified
        and len(cases) >= plan["minimum_pairs"]
        and safety == 0
        and rates["candidate"] >= plan["minimum_candidate_success_rate"]
        and rates["reference"] >= 0.9
        and p95 is not None
        and p95 <= plan["max_latency_p95_ms"]
    )
    return {
        "schema": report_schema,
        "policy_type": policy_type,
        "comparison_kind": "reference_bootstrap",
        "scope": plan["scope"],
        "control_profile_sha256": plan["control_profile_sha256"],
        "runtime_sha256": plan["runtime_sha256"],
        "evaluation_plan_sha256": digest(canonical(plan)),
        "candidate_model_sha256": plan["candidate_model_sha256"],
        "reference_controller_sha256": plan["reference_controller_sha256"],
        "quality_gate_passed": passed,
        "counts": counts,
        "success_rates": rates,
        "candidate_latency_p95_ms": p95,
        "safety_violation_count": safety,
        "trials": evidence,
        "live_gpu_verified": live_gpu_verified,
    }


def evaluate_bootstrap(
    plan: dict,
    evidence_root: Path,
    candidate_root: Path,
    *,
    scope: Scope,
    expected_plan_sha256: str,
    expected_results_sha256: str,
    model_validator=validate_model,
    plan_schema: str = PLAN_SCHEMA,
    result_schema: str = RESULT_SCHEMA,
    report_schema: str = "physicalai.gr00t-bootstrap-report/v1",
    policy_type: str = "gr00t_n1_5",
) -> dict:
    require(digest(canonical(plan)) == sha256(expected_plan_sha256), "Bootstrap plan changed")
    cases = validate_plan(plan, schema=plan_schema)
    require(plan["scope"] == asdict(scope), "Bootstrap owner mismatch")
    model = model_validator(
        candidate_root,
        expected_scope=scope,
        expected_model_sha256=plan["candidate_model_sha256"],
    )
    require(
        ControlProfile(**model["control_profile"]).sha256 == plan["control_profile_sha256"],
        "Bootstrap candidate servo mismatch",
    )
    for trained in model["training"]["episodes"]:
        require(
            all(
                trained["episode_id"] != case["episode_id"] and trained["seed"] != case["seed"]
                for case in cases.values()
            ),
            "Bootstrap held-out training leakage",
        )
    result = verify_evidence(plan, evidence_root, scope, expected_results_sha256, result_schema)
    return compare_bootstrap_trials(
        plan,
        result["trials"],
        live_gpu_verified=True,
        plan_schema=plan_schema,
        report_schema=report_schema,
        policy_type=policy_type,
    )
