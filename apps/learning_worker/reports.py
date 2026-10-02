import math
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import BootstrapReport, BootstrapTrial, PairedReport, TrialOutcome
from apps.api.learning_service import validate_bootstrap_report, validate_paired_report


def project_report(actor, specification, native, *, report_sha256, expected_plan_sha256):
    """Project exact native results; retain every attempt, including failed physical trials."""
    kind = specification.run.comparison_kind
    schema = (
        "physicalai.smolvla-bootstrap-report/v1"
        if kind == "reference_bootstrap"
        else "physicalai.smolvla-paired-report/v1"
    )
    plan_key = "evaluation_plan_sha256" if kind == "reference_bootstrap" else "plan_sha256"
    if (
        not isinstance(native, dict)
        or native.get("schema") != schema
        or native.get("policy_type") != specification.project.policy_type
        or native.get("scope") != {"tenant_id": str(actor.tenant_id), "owner_id": actor.owner_key}
        or native.get("azure_job_id") != specification.run.azure_job_id
        or native.get("specification_sha256") != specification.run.specification_sha256
        or native.get(plan_key) != expected_plan_sha256
        or native.get("live_gpu_verified") is not True
    ):
        raise Problem(
            503, "native_report_scope", "The native report is not the exact verified physical job."
        )
    rows = native.get("trials")
    if not isinstance(rows, list) or not 40 <= len(rows) <= 1000:
        raise Problem(
            503, "native_report_incomplete", "All bounded physical trials must be retained."
        )
    trials = []
    try:
        for row in rows:
            latencies = row["latencies_ms"]
            if not isinstance(latencies, list) or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in latencies
            ):
                raise ValueError("Invalid measured latency")
            p95 = sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1] if latencies else None
            succeeded = row["physical_success"] is True
            if succeeded and (row["terminated"] is not True or row["truncated"] is not False):
                raise ValueError("Nonterminal physical success")
            reason = row["failure_reason"]
            status = (
                "succeeded"
                if succeeded
                else (reason if reason in ("cancelled", "timed_out") else "failed")
            )
            violations = row["safety_violations"]
            if not isinstance(violations, list):
                raise ValueError("Missing safety evidence")
            trial = {
                "seed": row["seed"],
                "attempt": row["attempt"],
                "policy": row["policy"],
                "status": status,
                "model_sha256": row["model_sha256"],
                "environment_id": row["environment_id"],
                "revision": row["revision"],
                "physical_success": succeeded,
                "axis_error_m": row["position_error_m"],
                "duration_seconds": row["duration_ms"] / 1000,
                "safety_violations": len(violations),
                "inference_p95_ms": p95,
                "applied_action_count": row["applied_action_count"],
                "policy_predict_calls": row["policy_predict_calls"],
                "reference_route_calls": row["reference_route_calls"],
                "observed_initial_pose_m": row["observed_initial_pose_m"],
                "scene_builder_sha256": row["scene_builder_sha256"],
                "episode_id": row["episode_id"],
                "final_inspection_sha256": row["final_images"]["inspection"],
                "final_overview_sha256": row["final_images"]["overview"],
                "recording_id": None,
                "message": str(reason) if reason else "Physical terminal result recorded.",
            }
            model = BootstrapTrial if kind == "reference_bootstrap" else TrialOutcome
            trials.append(model.model_validate(trial))
        common = {
            "evaluation_plan_sha256": specification.project.evaluation_plan.sha256,
            "native_plan_sha256": expected_plan_sha256,
            "runtime_sha256": native["runtime_sha256"],
            "control_profile_sha256": native["control_profile_sha256"],
            "trials": tuple(trials),
            "report_sha256": report_sha256,
            "artifact_id": uuid5(NAMESPACE_URL, f"{actor.owner_key}:evaluation:{report_sha256}"),
        }
        if kind == "reference_bootstrap":
            report = BootstrapReport(
                **common,
                candidate_model_sha256=native["candidate_model_sha256"],
                reference_controller_sha256=native["reference_controller_sha256"],
                quality_gate_passed=native["quality_gate_passed"],
            )
            validate_bootstrap_report(specification.project, specification.candidate, report)
        else:
            if native.get("total_trial_count") != len(trials):
                raise ValueError("Native trial count differs")
            report = PairedReport(
                **common,
                before_model_sha256=native["policy_before_sha256"],
                after_model_sha256=native["policy_after_sha256"],
                conclusion=native["conclusion"],
                quality_gate_passed=native["quality_gate"],
            )
            validate_paired_report(
                specification.project, specification.baseline, specification.candidate, report
            )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise Problem(
            503, "native_report_invalid", "Native measured results failed strict API projection."
        ) from exc
    return report
