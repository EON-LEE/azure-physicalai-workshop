"""Source-backed paused report projection; a native output flag is not verification."""

from __future__ import annotations

import math
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from azure.core.exceptions import AzureError, ResourceNotFoundError
from pydantic import ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import PausedEvaluationPlan, fingerprint
from apps.api.simulation_reports import SimulationReport, validate_report_binding

JOB_FIELDS = frozenset({"azure_job_id", "azure_component_job_id", "specification_sha256"})
PHASE_FIELDS = {
    "policy": "latencies_wall_ms",
    "observation": "observation_wall_ms",
    "hold": "hold_wall_ms",
    "interval": "interval_wall_ms",
    "heartbeat": "heartbeat_gap_ms",
}


def summarize_wall(values):
    if not isinstance(values, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("Wall-clock samples must be explicit finite measured numbers.")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50": ordered[math.ceil(len(values) * 0.5) - 1] if ordered else None,
        "p95": ordered[math.ceil(len(values) * 0.95) - 1] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def compact_trial(row):
    value = {key: item for key, item in row.items() if key not in PHASE_FIELDS.values()}
    value["phase_wall_ms"] = {
        phase: summarize_wall(row[field]) for phase, field in PHASE_FIELDS.items()
    }
    value["safety_violation_count"] = len(value.pop("safety_violations"))
    task = dict(value["task_evidence"])
    task["safety_violation_count"] = len(task.pop("safety_violations"))
    value["task_evidence"] = task
    return value


def verifier_version(config):
    from learning.common import file_digest
    from learning.smolvla.azure import code_files

    root = Path(__file__).resolve().parents[2]
    files = (
        *code_files(config),
        "apps/learning_worker/paused_reports.py",
        "apps/learning_worker/artifacts.py",
        "apps/learning_worker/registry.py",
        "apps/api/simulation_reports.py",
        "apps/api/models.py",
        "apps/api/learning_models.py",
    )
    return fingerprint({name: file_digest(root / name) for name in files})


def _read_metadata(verifier, config, specification, output):
    from learning.azure import registered_input_matches
    from learning.gr00t.azure import workspace_id
    from learning.smolvla.azure import validate_config

    try:
        validate_config(config)
        client = verifier._metadata_client(config)
        datastore = client.datastores.get(config["datastore"])
        credentials = getattr(getattr(datastore, "credentials", None), "type", None)
        if (
            str(getattr(datastore.type, "value", datastore.type)) != "AzureBlob"
            or datastore.account_name != config["storage_account_name"]
            or datastore.container_name != config["blob_container"]
            or getattr(credentials, "value", credentials) != "None"
        ):
            raise Problem(503, "paused_datastore_changed", "Reviewed datastore mapping changed.")
        for value in config["inputs"].values():
            asset = client.data.get(value["name"], version=value["version"])
            if not registered_input_matches(asset.type, asset.path, value):
                raise Problem(503, "paused_asset_changed", "Registered input version/path changed.")
        parent = client.jobs.get(specification.run.backend_job_name)
        child_id = output.get("azure_component_job_id")
        if (
            not isinstance(child_id, str)
            or child_id.rsplit("/jobs/", 1)[0].lower() != workspace_id(config).lower()
        ):
            raise Problem(503, "paused_job_mismatch", "Report component is outside the workspace.")
        child = client.jobs.get(child_id.rsplit("/", 1)[-1])
        tags = {
            "scope_owner": specification.owner_key,
            "scope_tenant": str(specification.project.tenant_id),
            "specification_sha256": specification.run.specification_sha256,
            "config_schema": config["schema"],
            "job_deadline_utc": config["job_deadline_utc"],
            "execution_timing": "paused_simulation",
            "real_time_admission": "false",
            "criteria_sha256": config["criteria_sha256"],
            "frozen_plan_sha256": config["frozen_plan_sha256"],
        }
        if (
            parent.id != specification.run.azure_job_id
            or output.get("azure_job_id") != parent.id
            or child.id != child_id
            or child.parent_job_name != specification.run.backend_job_name
            or parent.status not in ("Completed", "Failed", "Canceled")
            or child.status not in ("Completed", "Failed", "Canceled")
            or any(
                (job.tags or {}).get(key) != value
                for job in (parent, child)
                for key, value in tags.items()
            )
        ):
            raise Problem(503, "paused_job_mismatch", "Actual parent/child job binding changed.")
    except (AzureError, ValueError, KeyError, AttributeError) as exc:
        raise unavailable("Read-only evaluation job/datastore metadata") from exc


def _locations(verifier, actor, config, specification):
    from learning.azure import datastore_prefix
    from learning.common import relative_path

    account = f"https://{config['storage_account_name']}.blob.core.windows.net"
    container = config["blob_container"]
    prefix = datastore_prefix(config)
    owner_prefix = f"tenants/{actor.tenant_id}/owners/{actor.owner_key}/"
    result = {
        "report": (
            account,
            container,
            f"{config['output_prefix'].rstrip('/')}/{specification.run.backend_job_name}/report/report.json",
            True,
        ),
    }
    for name in config["inputs"]:
        uri = config["inputs"][name]["uri"]
        if not uri.startswith(prefix):
            raise Problem(
                503, "paused_asset_scope", "Only the exact approved datastore is allowed."
            )
        key = uri[len(prefix) :]
        relative_path(key)
        if not key.startswith(owner_prefix):
            raise Problem(503, "paused_asset_scope", "Input does not belong to the original owner.")
        result[name] = (
            account,
            container,
            key if name == "plan" else key.rstrip("/") + "/",
            name == "plan",
        )
    for role, record in (
        ("candidate", specification.candidate),
        ("baseline", specification.baseline),
    ):
        if record is not None:
            result[f"registered_{role}"] = (
                verifier.registry.client.url.rstrip("/"),
                verifier.registry.container.container_name,
                verifier.registry.key(actor, f"artifacts/{record.artifact_id}/"),
                False,
            )
    return result


def _inventory(verifier, locations):
    return {
        name: verifier._blob_inventory(account, container, key, exact=exact)
        for name, (account, container, key, exact) in locations.items()
    }


def _source_rescore(verifier, actor, specification, config, locations, directory):
    from learning.common import canonical, digest, read_json
    from learning.paused.evaluation import evaluate_bootstrap, evaluate_pair, validate_plan

    plan_path = directory / "native-plan.json"
    account, container, key, _ = locations["plan"]
    verifier._download_file(account, container, key, plan_path)
    plan = read_json(plan_path)
    validate_plan(plan)
    if (
        digest(canonical(plan)) != config["inputs"]["plan"]["sha256"]
        or plan["control_profile"]["profile_id"] != specification.project.control_profile_id
        or any(
            plan[name] != getattr(specification.project, name)
            for name in ("control_profile_sha256", "criteria_sha256", "frozen_plan_sha256")
        )
    ):
        raise Problem(
            503, "paused_plan_mismatch", "Canonical native plan or frozen provenance differs."
        )
    evidence_root = directory / "evidence"
    account, container, key, _ = locations["evidence"]
    verifier._download_prefix(account, container, key, evidence_root)
    bootstrap = specification.run.comparison_kind == "reference_bootstrap"
    roots = {"candidate": directory / "candidate"}
    records = {"candidate": specification.candidate}
    if not bootstrap:
        roots["baseline"] = directory / "baseline"
        records["baseline"] = specification.baseline
    task = {
        "task_id": specification.project.task_id,
        "instruction": specification.project.instruction,
        "goal_id": specification.project.goal_station_id,
    }
    for role, root in roots.items():
        record = records[role]
        input_name = (
            "candidate" if bootstrap else "policy_after" if role == "candidate" else "policy_before"
        )
        if record is None or config["inputs"][input_name]["sha256"] != record.model_sha256:
            raise Problem(
                503, "paused_model_mismatch", "Original registered model differs from input."
            )
        index = verifier.registry.artifact_index(actor, record.artifact_id)
        if index.get("manifest_sha256") != record.model_sha256:
            raise Problem(503, "paused_model_mismatch", "Registered model manifest changed.")
        account, container, key, _ = locations[input_name]
        verifier._download_prefix(account, container, key, root)
        model = verifier._model(
            actor, root, record.model_sha256, "smolvla", execution_timing="paused_simulation"
        )
        if (
            model["task"] != task
            or digest(canonical(task)) != config["task_sha256"]
            or verifier._model_timing(model) != specification.project.timing_fields()
        ):
            raise Problem(
                503,
                "paused_model_task",
                "Evaluated model task or timing provenance differs from the project.",
            )
    kwargs = {
        "scope": verifier._scope(actor),
        "expected_plan_sha256": config["inputs"]["plan"]["sha256"],
        "expected_results_sha256": config["inputs"]["evidence"]["sha256"],
    }
    if bootstrap:
        return evaluate_bootstrap(plan, evidence_root, roots["candidate"], **kwargs)
    if "baseline" not in roots:
        raise Problem(503, "paused_model_missing", "Paired evaluation requires the original P0.")
    return evaluate_pair(plan, evidence_root, roots["baseline"], roots["candidate"], **kwargs)


def complete_verified_report(verifier, actor, specification, azure_job_id, *, required=True):
    from learning.common import file_digest

    config = verifier.registry.job_configuration(actor, specification)
    if not isinstance(config, dict) or config.get("execution_timing") != "paused_simulation":
        raise unavailable("Original paused evaluation configuration")
    from learning.smolvla.azure import validate_config

    try:
        validate_config(config)
    except (ValueError, KeyError) as exc:
        raise unavailable("Original closed native evaluation config") from exc
    if (
        config["tenant_id"] != str(actor.tenant_id)
        or config["owner_id"] != actor.owner_key
        or specification.owner_key != actor.owner_key
    ):
        raise Problem(403, "paused_report_owner", "Evaluation belongs to a different owner.")
    run = specification.run.model_copy(update={"azure_job_id": azure_job_id})
    specification = specification.model_copy(update={"run": run})
    locations = _locations(verifier, actor, config, specification)
    with TemporaryDirectory(prefix="physicalai-paused-report-") as temporary:
        directory = Path(temporary)
        before = _inventory(verifier, locations)
        output_path = directory / "report.json"
        account, container, key, _ = locations["report"]
        try:
            verifier._download_file(account, container, key, output_path)
        except Problem as exc:
            if not required and isinstance(exc.__cause__, ResourceNotFoundError):
                return None
            raise
        output = verifier._read_json(output_path)
        _read_metadata(verifier, config, specification, output)
        report_sha = file_digest(output_path)
        binding = {
            "owner": actor.owner_key,
            "job": azure_job_id,
            "specification_sha256": run.specification_sha256,
            "configuration_sha256": fingerprint(config),
            "report_sha256": report_sha,
            "verifier_sha256": verifier_version(config),
            "inventory_sha256": fingerprint(before),
        }
        cache = f"jobs/{run.backend_job_name}/report-verification/{fingerprint(binding)}"
        certificate = {"binding": binding, "verified": True}
        cached = verifier.registry.get(actor, f"{cache}/verified.json")
        if cached is not None:
            if cached != certificate or before != _inventory(verifier, locations):
                raise Problem(
                    503, "paused_report_cache_changed", "Verified content binding changed."
                )
            verified = {key: value for key, value in output.items() if key not in JOB_FIELDS}
        else:
            if not verifier.registry.put(actor, f"{cache}/claim.json", {"binding": binding}):
                failure = verifier.registry.get(actor, f"{cache}/failure.json")
                code = (
                    "paused_report_unverified"
                    if failure
                    else "paused_report_verification_unconfirmed"
                )
                raise Problem(
                    503, code, "Reconcile the original verification; no repeated rescore."
                )
            try:
                verified = _source_rescore(
                    verifier, actor, specification, config, locations, directory
                )
                if before != _inventory(verifier, locations):
                    raise Problem(
                        503,
                        "paused_report_source_changed",
                        "Artifacts changed during verification.",
                    )
                project_verified_report(
                    actor,
                    specification,
                    output,
                    verified_native=verified,
                    report_sha256=report_sha,
                    expected_plan_sha256=config["inputs"]["plan"]["sha256"],
                    expected_results_sha256=config["inputs"]["evidence"]["sha256"],
                )
                verifier.registry.put(actor, f"{cache}/verified.json", certificate)
            except (Problem, ValueError, OSError) as exc:
                code = exc.code if isinstance(exc, Problem) else "paused_report_source_invalid"
                verifier.registry.put(
                    actor,
                    f"{cache}/failure.json",
                    {
                        "binding": binding,
                        "verified": False,
                        "error_code": code,
                    },
                )
                raise Problem(
                    503, code, "Independent physical report verification failed."
                ) from exc
        report = project_verified_report(
            actor,
            specification,
            output,
            verified_native=verified,
            report_sha256=report_sha,
            expected_plan_sha256=config["inputs"]["plan"]["sha256"],
            expected_results_sha256=config["inputs"]["evidence"]["sha256"],
        )
        existing = verifier.registry.get(actor, f"artifacts/{report.artifact_id}/index.json")
        if existing is None:
            publication = directory / "publication"
            publication.mkdir()
            copyfile(output_path, publication / "report.json")
            verifier.registry.upload(
                actor,
                report.artifact_id,
                publication,
                {"manifest_sha256": report.report_sha256, "role": "paused_evaluation_report"},
            )
        elif (
            existing.get("owner_key") != actor.owner_key
            or existing.get("manifest_sha256") != report.report_sha256
            or existing.get("files") != {"report.json": report.report_sha256}
        ):
            raise Problem(503, "paused_report_cache_changed", "Published report bytes differ.")
        verifier.registry.put(
            actor,
            f"jobs/{run.backend_job_name}/reports/{report_sha}.json",
            {
                "specification_sha256": run.specification_sha256,
                "report": report.model_dump(mode="json"),
            },
        )
        return report


def project_verified_report(
    actor,
    specification,
    output,
    *,
    verified_native,
    report_sha256,
    expected_plan_sha256,
    expected_results_sha256,
):
    if not isinstance(verified_native, dict) or not isinstance(output, dict):
        raise Problem(
            503, "paused_report_unverified", "An independent physical rescore is required."
        )
    project, run = specification.project, specification.run
    bootstrap = run.comparison_kind == "reference_bootstrap"
    if (
        {key: value for key, value in output.items() if key not in JOB_FIELDS} != verified_native
        or not JOB_FIELDS.issubset(output)
        or output["azure_job_id"] != run.azure_job_id
        or output["specification_sha256"] != run.specification_sha256
        or not isinstance(output["azure_component_job_id"], str)
        or output.get("scope") != {"tenant_id": str(actor.tenant_id), "owner_id": actor.owner_key}
        or output.get("live_gpu_verified") is not True
        or output.get("real_time_admission") is not False
        or output.get("execution_timing") != "paused_simulation"
        or output.get("results_sha256") != expected_results_sha256
        or output.get("evaluation_plan_sha256") != expected_plan_sha256
        or output.get("criteria_sha256") != project.criteria_sha256
        or output.get("frozen_plan_sha256") != project.frozen_plan_sha256
        or output.get("control_profile_sha256") != project.control_profile_sha256
        or output.get("policy_type") != project.policy_type
        or output.get("comparison_kind")
        != ("reference_bootstrap" if bootstrap else "paired_policy_eval")
        or not isinstance(project.evaluation_plan, PausedEvaluationPlan)
    ):
        raise Problem(503, "paused_report_scope", "Output differs from its verified source or job.")
    try:
        values = {
            key: output[key]
            for key in (
                "execution_timing",
                "real_time_admission",
                "control_profile_sha256",
                "criteria_sha256",
                "frozen_plan_sha256",
                "results_sha256",
                "runtime_sha256",
                "counts",
                "success_rates",
                "absolute_success_rate_improvement",
                "latency_wall_ms",
                "total_wall_duration_ms",
                "total_simulation_duration_ms",
                "safety_violation_count",
                "resource_violation_count",
                "total_trial_count",
                "quality_gate_passed",
                "conclusion",
                "live_gpu_verified",
            )
        }
        values.update(
            native_schema=output["schema"],
            comparison_kind=run.comparison_kind,
            control_profile_id=project.control_profile_id,
            evaluation_plan_sha256=project.evaluation_plan.sha256,
            native_plan_sha256=expected_plan_sha256,
            report_sha256=report_sha256,
            artifact_id=uuid5(
                NAMESPACE_URL, f"{actor.owner_key}:paused-evaluation:{report_sha256}"
            ),
            trials=[compact_trial(row) for row in output["trials"]],
        )
        if bootstrap:
            values.update(
                candidate_model_sha256=output["candidate_model_sha256"],
                reference_controller_sha256=output["reference_controller_sha256"],
            )
        else:
            values.update(
                before_model_sha256=output["policy_before_sha256"],
                after_model_sha256=output["policy_after_sha256"],
            )
        report = SimulationReport.model_validate(values)
        validate_report_binding(specification, report)
        if len(report.model_dump_json().encode()) > 512 * 1024:
            raise ValueError("Compact report exceeds its bounded Cosmos projection budget.")
        return report
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise Problem(
            503, "paused_report_invalid", "Verified report projection is incomplete."
        ) from exc
