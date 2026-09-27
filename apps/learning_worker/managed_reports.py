"""Operator-bound imports of complete managed evaluations; never an Azure job submitter."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import AwareDatetime, Field, model_validator

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    EvaluationRun,
    Frozen,
    PausedEvaluationPlan,
    PolicyCandidate,
    fingerprint,
)
from apps.api.learning_ports import JobSpecification
from apps.api.models import Revision, utcnow
from apps.api.simulation_reports import (
    MANAGED_REPORT_SCHEMA,
    ManagedEvaluationReceipt,
    ManagedImportReference,
    SimulationReport,
    validate_report_binding,
)
from apps.learning_worker.paused_reports import compact_trial
from learning.common import canonical, digest, file_digest, parse_json, read_json, safe_path, utc


class ManagedEvaluationBinding(Frozen):
    schema_version: Literal["physicalai.managed-evaluation-binding/v1"] = Field(alias="schema")
    provider: Literal["managed_batch"]
    created_at: AwareDatetime
    study_max_wall_seconds: int = Field(strict=True, ge=1, le=28800)
    study_approved_cost_usd: Decimal = Field(gt=0, le=20, decimal_places=2)
    mapping_sha256: Revision
    specification: JobSpecification

    @model_validator(mode="after")
    def predeclared_evaluation(self):
        spec = self.specification
        run = spec.run
        before = spec.baseline_candidate or spec.baseline
        if (
            not isinstance(run, EvaluationRun)
            or run.provider != "managed_batch"
            or run.status != "awaiting_import"
            or run.import_operation_id is not None
            or spec.candidate is None
            or before is None
            or (spec.baseline is not None and spec.baseline_candidate is not None)
            or spec.training_parent is not None
            or run.candidate_id != spec.candidate.id
            or run.baseline_release_id != (spec.baseline.id if spec.baseline else None)
            or run.before_candidate_id
            != (spec.baseline_candidate.id if spec.baseline_candidate else None)
            or run.project_id != spec.project.id
            or spec.candidate.project_id != spec.project.id
            or spec.project.policy_type != "smolvla"
            or run.policy_type != "smolvla"
            or spec.candidate.policy_type != "smolvla"
            or before.policy_type != "smolvla"
            or spec.project.execution_timing != "paused_simulation"
            or run.evaluation_plan_sha256 != spec.project.evaluation_plan.sha256
            or spec.candidate.parent_model_sha256 != before.model_sha256
            or not run.created_at <= self.created_at < run.deadline
            or spec.project.created_at > self.created_at
            or not 0
            < (run.deadline - run.created_at).total_seconds()
            <= self.study_max_wall_seconds
            or run.approved_cost_usd != self.study_approved_cost_usd
            or self.study_approved_cost_usd > spec.project.budget.maximum_cost_usd
            or any(
                record.owner_key != spec.owner_key
                or record.tenant_id != spec.project.tenant_id
                or not record.matches_timing(spec.project)
                for record in (run, spec.candidate, before)
            )
        ):
            raise ValueError("Binding requires the original scoped evaluation and actual models.")
        return self


class ManagedImportCompletion(Frozen):
    schema_version: Literal["physicalai.managed-evaluation-import/v1"] = Field(alias="schema")
    created_at: AwareDatetime
    operation_id: UUID
    binding_sha256: Revision
    evidence_sha256: Revision
    report_sha256: Revision


@dataclass(frozen=True)
class ImportContext:
    prefix: str
    binding: ManagedEvaluationBinding
    completion: ManagedImportCompletion
    reference: ManagedImportReference
    registered_at: datetime


def prefix(project_id: UUID, evaluation_id: UUID) -> str:
    return f"projects/{UUID(str(project_id))}/managed-evaluations/{UUID(str(evaluation_id))}"


def _record(model, content):
    try:
        return model.model_validate(parse_json(content))
    except ValueError as exc:
        raise Problem(
            503, "managed_import_invalid", "Original import metadata is invalid."
        ) from exc


def context(registry, actor, project_id, evaluation_id, *, expected=None) -> ImportContext:
    base = prefix(project_id, evaluation_id)
    raw, registered = registry.import_document(actor, f"{base}/binding.json")
    binding = _record(ManagedEvaluationBinding, raw)
    spec = binding.specification
    if (
        spec.project.id != project_id
        or spec.run.id != evaluation_id
        or spec.owner_key != actor.owner_key
        or spec.project.owner_key != actor.owner_key
        or spec.project.tenant_id != actor.tenant_id
        or spec.project.actor_id != actor.object_id
        or spec.run.actor_id != actor.object_id
        or registry.job(actor, spec.run.backend_job_name) != spec
    ):
        raise Problem(403, "managed_import_scope", "Original evaluation registration differs.")
    payload, completed = registry.import_document(actor, f"{base}/completion.json")
    completion = _record(ManagedImportCompletion, payload)
    reference = ManagedImportReference(
        operation_id=completion.operation_id,
        owner_key=actor.owner_key,
        project_id=project_id,
        evaluation_run_id=evaluation_id,
        specification_sha256=spec.run.specification_sha256,
        binding_sha256=digest(raw),
        completion_sha256=digest(payload),
    )
    if (
        completion.binding_sha256 != reference.binding_sha256
        or binding.created_at >= registered + timedelta(seconds=1)
        or not registered + timedelta(seconds=1) <= completion.created_at
        or completion.created_at >= completed + timedelta(seconds=1)
        or completed > utcnow()
        or (expected is not None and expected != reference)
    ):
        raise Problem(
            409,
            "managed_import_binding",
            "Original binding/completion bytes or chronology changed.",
        )
    # HTTP Last-Modified has one-second precision; same-second claims are not proven later.
    return ImportContext(base, binding, completion, reference, registered + timedelta(seconds=1))


def input_locations(registry, actor, value: ImportContext):
    spec = value.binding.specification
    before = spec.baseline_candidate or spec.baseline
    return (
        registry.key(actor, f"{value.prefix}/files/"),
        registry.key(actor, f"artifacts/{before.artifact_id}/files/"),
        registry.key(actor, f"artifacts/{spec.candidate.artifact_id}/files/"),
    )


def _version():
    root = Path(__file__).resolve().parents[2]
    paths = [
        path
        for folder in ("apps/api", "apps/learning_worker", "learning", "simulation", "contracts")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.suffix in (".py", ".json")
    ]
    return fingerprint(
        {path.relative_to(root).as_posix(): file_digest(path) for path in sorted(paths)}
    )


def _inventory(verifier, actor, value):
    registry = verifier.registry
    inventory = {
        location: verifier._blob_inventory(
            registry.client.url.rstrip("/"),
            registry.container.container_name,
            location,
        )
        for location in input_locations(registry, actor, value)
    }
    spec = value.binding.specification
    for role, record in (
        ("before", spec.baseline_candidate or spec.baseline),
        ("after", spec.candidate),
    ):
        index = registry.artifact_index(actor, record.artifact_id)
        if (
            index.get("manifest_sha256") != record.model_sha256
            or index.get("files", {}).get("model.json") != record.model_sha256
        ):
            raise Problem(409, "managed_import_model", "Original registered model index differs.")
        inventory[f"model-index:{role}"] = fingerprint(index)
    return fingerprint(inventory)


def _mapping_scope(value, mapping):
    spec, native = value.binding.specification, mapping.evaluation_plan
    project, run = spec.project, spec.run
    before = spec.baseline_candidate or spec.baseline
    plan = project.evaluation_plan
    if (
        not isinstance(plan, PausedEvaluationPlan)
        or native["scope"] != {"tenant_id": str(project.tenant_id), "owner_id": spec.owner_key}
        or native["control_profile"]["profile_id"] != project.control_profile_id
        or any(
            native[key] != getattr(project, key)
            for key in ("control_profile_sha256", "criteria_sha256", "frozen_plan_sha256")
        )
        or native["policy_before_sha256"] != before.model_sha256
        or native["policy_after_sha256"] != spec.candidate.model_sha256
        or {(case["seed"], case["environment_id"], case["revision"]) for case in native["cases"]}
        != {(case.seed, case.environment_id, case.revision) for case in plan.cases}
        or any(
            case["expected_destination_id"] != project.goal_station_id
            or case["tolerance_m"] > plan.maximum_axis_error_m
            for case in native["cases"]
        )
        or native["quality_limits"]["minimum_success_rate"] < plan.minimum_success_rate
        or native["quality_limits"]["minimum_absolute_improvement"]
        < plan.minimum_absolute_improvement
        or native["control_profile"]["max_cartesian_speed_m_s"] > plan.max_cartesian_speed_m_s
        or not utc(mapping.issued_at_utc) <= value.registered_at < run.deadline
    ):
        raise Problem(409, "managed_import_plan", "Mapping differs from the original project/plan.")


def _models(verifier, actor, value, folder):
    spec = value.binding.specification
    roots = {}
    for role, record in (
        ("before", spec.baseline_candidate or spec.baseline),
        ("after", spec.candidate),
    ):
        roots[role] = folder / role
        index = verifier.registry.download(actor, record.artifact_id, roots[role])
        if index.get("manifest_sha256") != record.model_sha256:
            raise Problem(409, "managed_import_model", "Registered model manifest changed.")
        model = verifier._model(
            actor, roots[role], record.model_sha256, "smolvla", execution_timing="paused_simulation"
        )
        if (
            verifier._model_timing(model) != spec.project.timing_fields()
            or record.manifest_sha256 != record.model_sha256
            or record.processor_sha256 != model["processor_sha256"]
            or model["task"]
            != {
                "task_id": spec.project.task_id,
                "instruction": spec.project.instruction,
                "goal_id": spec.project.goal_station_id,
            }
        ):
            raise Problem(409, "managed_import_model", "Actual model task or provenance differs.")
        if isinstance(record, PolicyCandidate) and (
            record.azure_job_id != model["training"]["azure_pipeline_job_id"]
            or record.optimizer_steps != model["training"]["optimizer_steps"]
            or record.parent_model_sha256 != model["training"]["parent_model_sha256"]
            or record.source_commit != model["upstream"]["source_commit"]
            or record.model_revision != model["upstream"]["model_revision"]
        ):
            raise Problem(409, "managed_import_model", "Actual native training metadata differs.")
    return roots


def _chronology(value, verified, root):
    spec = value.binding.specification
    plan = spec.project.evaluation_plan
    for attempt in verified["attempts"]:
        claimed = utc(attempt["claimed_at_utc"].replace("+00:00", "Z"))
        if (
            claimed < value.registered_at
            or claimed >= spec.run.deadline
            or utc(attempt["ended_at_utc"]) > spec.run.deadline
        ):
            raise Problem(
                409, "managed_import_late_binding", "Physical claims exceed original registration."
            )
        grant = read_json(
            root / "attempts" / attempt["physical_attempt_id"] / "inputs" / "grant.json",
            max_bytes=1024**2,
        )["authorization"]
        if (
            grant["max_simulation_steps"] > plan.max_simulation_seconds * 60
            or grant["max_episode_wall_seconds"] > plan.max_wall_seconds
        ):
            raise Problem(
                409, "managed_import_budget", "Original physical grant exceeds the project budget."
            )


def _project(value, mapping, verified):
    from learning.paused.evaluation import compare_trials

    spec = value.binding.specification
    attempts = verified["attempts"]
    scored = compare_trials(
        mapping.evaluation_plan,
        [{**item["physical_trial"], "episode_id": item["logical_case_id"]} for item in attempts],
        live_gpu_verified=True,
    )
    if canonical(
        {key: item for key, item in scored.items() if key not in ("schema", "trials")}
    ) != canonical(verified["comparison"]):
        raise Problem(503, "managed_import_rescore", "Verified numeric projection differs.")
    retained = (
        "execution_timing",
        "real_time_admission",
        "control_profile_sha256",
        "criteria_sha256",
        "frozen_plan_sha256",
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
    report = SimulationReport(
        **{key: scored[key] for key in retained},
        native_schema=MANAGED_REPORT_SCHEMA,
        comparison_kind="paired_policy",
        control_profile_id=spec.project.control_profile_id,
        evaluation_plan_sha256=spec.project.evaluation_plan.sha256,
        native_plan_sha256=verified["native_plan_sha256"],
        mapping_sha256=value.binding.mapping_sha256,
        evidence_sha256=value.completion.evidence_sha256,
        report_sha256=value.completion.report_sha256,
        artifact_id=uuid5(
            NAMESPACE_URL,
            f"{spec.owner_key}:managed-evaluation:{value.completion.report_sha256}",
        ),
        before_model_sha256=scored["policy_before_sha256"],
        after_model_sha256=scored["policy_after_sha256"],
        trials=[
            {
                **compact_trial(row),
                "episode_id": item["physical_attempt_id"],
                "physical_attempt_id": item["physical_attempt_id"],
                "logical_case_id": item["logical_case_id"],
            }
            for row, item in zip(scored["trials"], attempts, strict=True)
        ],
    )
    validate_report_binding(spec, report)
    if len(report.model_dump_json().encode()) > 512 * 1024:
        raise Problem(503, "managed_import_size", "Compact report exceeds its existing bound.")
    return ManagedEvaluationReceipt(**value.reference.model_dump(), report=report)


def verify_local(verifier, actor, value, root, model_roots):
    from simulation.paired_evaluation import PairingEvidence, PairingPlan, aggregate

    if value.reference.owner_key != actor.owner_key:
        raise Problem(403, "managed_import_scope", "Import belongs to another owner.")
    if {path.name for path in root.iterdir()} != {
        "mapping.json",
        "evidence.json",
        "report.json",
        "attempts",
    }:
        raise Problem(
            409, "managed_import_inventory", "Unexpected files outside the original snapshot."
        )
    documents = []
    for name, ceiling in (
        ("mapping.json", 4 * 1024**2),
        ("evidence.json", 32 * 1024**2),
        ("report.json", 32 * 1024**2),
    ):
        path = safe_path(root, name)
        if path.stat().st_size > ceiling:
            raise Problem(409, "managed_import_size", "Import document exceeds its bound.")
        documents.append(path.read_bytes())
    mapping_bytes, evidence_bytes, report_bytes = documents
    if (
        len(mapping_bytes) > 4 * 1024**2
        or len(evidence_bytes) > 32 * 1024**2
        or len(report_bytes) > 32 * 1024**2
        or digest(mapping_bytes) != value.binding.mapping_sha256
        or digest(evidence_bytes) != value.completion.evidence_sha256
        or digest(report_bytes) != value.completion.report_sha256
    ):
        raise Problem(
            409, "managed_import_digest", "Original mapping/evidence/report bytes differ."
        )
    mapping = PairingPlan.model_validate(parse_json(mapping_bytes))
    evidence = PairingEvidence.model_validate(parse_json(evidence_bytes))
    _mapping_scope(value, mapping)
    if utc(evidence.snapshot_at_utc) > value.completion.created_at:
        raise Problem(409, "managed_import_snapshot", "Completion predates the evidence snapshot.")
    verified = aggregate(
        root,
        mapping,
        evidence,
        mapping_sha256=value.binding.mapping_sha256,
        evidence_sha256=value.completion.evidence_sha256,
        before_root=model_roots["before"],
        after_root=model_roots["after"],
    )
    if canonical(parse_json(report_bytes)) != canonical(verified):
        raise Problem(409, "managed_import_rescore", "Declared report differs from actual rescore.")
    if not verified["complete"] or verified["verified_trial_count"] != 40 or verified["missing"]:
        raise Problem(
            409, "managed_import_incomplete", "All forty original physical trials must be verified."
        )
    _chronology(value, verified, root)
    return _project(value, mapping, verified)


def verified_receipt(verifier, actor, reference):
    value = context(
        verifier.registry,
        actor,
        reference.project_id,
        reference.evaluation_run_id,
        expected=reference,
    )
    saved = verifier.registry.get(actor, f"{value.prefix}/verified.json")
    if not isinstance(saved, dict) or set(saved) != {
        "reference",
        "verifier",
        "inventory",
        "receipt",
    }:
        raise unavailable("Complete source-verified managed evaluation certificate")
    receipt = ManagedEvaluationReceipt.model_validate(saved["receipt"])
    if (
        saved["reference"] != reference.model_dump(mode="json")
        or receipt.model_dump(exclude={"report"}) != reference.model_dump()
        or receipt.report.report_sha256 != value.completion.report_sha256
        or saved["verifier"] != _version()
        or saved["inventory"] != _inventory(verifier, actor, value)
    ):
        raise Problem(409, "managed_import_certificate", "Verified import source binding changed.")
    validate_report_binding(value.binding.specification, receipt.report)
    index = verifier.registry.artifact_index(actor, receipt.report.artifact_id)
    if index.get("manifest_sha256") != receipt.report.report_sha256 or index.get("files") != {
        "report.json": receipt.report.report_sha256
    }:
        raise Problem(
            409, "managed_import_certificate", "Private original report artifact changed."
        )
    return receipt


def complete(verifier, actor, work):
    reference = work.managed_import
    value = context(verifier.registry, actor, work.project.id, work.target_id, expected=reference)
    if value.binding.specification.project != work.project:
        raise Problem(409, "managed_import_project", "Queued project differs from its binding.")
    if verifier.registry.get(actor, f"{value.prefix}/verified.json") is not None:
        return verified_receipt(verifier, actor, reference)
    if verifier.budget is None:
        raise unavailable("Bounded resident managed artifact processor")
    from simulation.paired_evaluation import PairingPlan

    mapping_bytes, _ = verifier.registry.import_document(
        actor, f"{value.prefix}/files/mapping.json", max_bytes=4 * 1024**2
    )
    if digest(mapping_bytes) != value.binding.mapping_sha256:
        raise Problem(409, "managed_import_digest", "The predeclared mapping bytes changed.")
    _mapping_scope(value, _record(PairingPlan, mapping_bytes))
    before = _inventory(verifier, actor, value)
    with TemporaryDirectory(prefix="physicalai-managed-report-") as directory:
        folder = Path(directory)
        root = folder / "evidence"
        verifier._download_prefix(
            verifier.registry.client.url.rstrip("/"),
            verifier.registry.container.container_name,
            input_locations(verifier.registry, actor, value)[0],
            root,
            max_bytes=work.max_bytes,
        )
        roots = _models(verifier, actor, value, folder)
        try:
            receipt = verify_local(verifier, actor, value, root, roots)
        except ValueError as exc:
            raise Problem(
                503, "managed_import_source_invalid", "Original native artifact rescore failed."
            ) from exc
        verifier.budget.check()
        if _inventory(verifier, actor, value) != before:
            raise Problem(409, "managed_import_source_changed", "Inputs changed during rescore.")
        publication = folder / "publication"
        publication.mkdir()
        copyfile(root / "report.json", publication / "report.json")
        verifier.registry.upload(
            actor,
            receipt.report.artifact_id,
            publication,
            {"manifest_sha256": receipt.report.report_sha256, "role": "managed_evaluation_report"},
        )
        verifier.registry.put(
            actor,
            f"{value.prefix}/verified.json",
            {
                "reference": reference.model_dump(mode="json"),
                "verifier": _version(),
                "inventory": before,
                "receipt": receipt.model_dump(mode="json"),
            },
        )
        verifier.registry.put(
            actor,
            f"jobs/{value.binding.specification.run.backend_job_name}/reports/"
            f"{receipt.report.report_sha256}.json",
            {
                "specification_sha256": reference.specification_sha256,
                "report": receipt.report.model_dump(mode="json"),
            },
        )
        return receipt
