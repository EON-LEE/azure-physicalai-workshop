from __future__ import annotations

import copy
import importlib
import logging
import re
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from azure.core.exceptions import AzureError, ResourceNotFoundError
from pydantic import AwareDatetime, TypeAdapter, ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    JOB_TERMINAL,
    BootstrapReport,
    PairedReport,
    TrainingMetrics,
    fingerprint,
)
from apps.api.learning_ports import BackendJob, JobSpecification
from apps.api.models import utcnow
from apps.api.simulation_reports import SimulationReport
from apps.learning_worker.policies import implementation
from apps.learning_worker.registry import ReconciliationTarget

log = logging.getLogger(__name__)
NATIVE_DEADLINE_SCHEMA = "physicalai.smolvla-azure/v2"


class PolicyLearningWorker:
    def __init__(
        self,
        registry,
        artifact_verifier,
        caller_client_id,
        *,
        allowed_policy_types=(),
        sdk_factory=None,
        reconciliation_enabled=False,
        reconciliation_actor_ids=frozenset(),
        reconciliation_targets=(),
        paused_training_enabled=False,
        paused_evaluation_enabled=False,
    ):
        self.registry = registry
        self.artifacts = artifact_verifier
        self.caller_client_id = caller_client_id
        self.allowed_policy_types = tuple(allowed_policy_types)
        self.sdk_factory = sdk_factory
        self.reconciliation_enabled = reconciliation_enabled
        self.reconciliation_actor_ids = frozenset(reconciliation_actor_ids)
        self.reconciliation_targets = tuple(reconciliation_targets)
        self.paused_training_enabled = paused_training_enabled
        self.paused_evaluation_enabled = paused_evaluation_enabled

    def _authorize_specification(self, actor, specification: JobSpecification):
        if (
            specification.owner_key != actor.owner_key
            or specification.project.owner_key != actor.owner_key
            or specification.run.owner_key != actor.owner_key
            or specification.project.actor_id != actor.object_id
            or specification.run.project_id != specification.project.id
            or specification.project.tenant_id != actor.tenant_id
            or specification.run.tenant_id != actor.tenant_id
            or specification.run.actor_id != actor.object_id
        ):
            raise Problem(403, "worker_scope_mismatch", "Worker request scope is inconsistent.")
        if specification.run.policy_type != specification.project.policy_type:
            raise Problem(409, "worker_policy_mismatch", "The job's exact model family changed.")
        if not specification.run.matches_timing(specification.project):
            raise Problem(
                409, "worker_timing_mismatch", "Job mode and frozen project timing differ."
            )
        for record in (
            specification.dataset,
            specification.candidate,
            specification.baseline,
            specification.training_parent,
        ):
            if record is not None and (
                record.owner_key != actor.owner_key or record.tenant_id != actor.tenant_id
            ):
                raise Problem(
                    403, "worker_scope_mismatch", "An input artifact belongs to another owner."
                )

    def _configuration(self, actor, specification):
        self._authorize_specification(actor, specification)
        approval = self.registry.approved_plan(actor, specification)
        required = {"expires_at", "maximum_cost_usd", "gpu_hourly_usd", "config"}
        if (
            not isinstance(approval, dict)
            or set(approval) != required
            or not isinstance(approval["config"], dict)
        ):
            raise unavailable("Operator-reviewed job cost and configuration")
        try:
            expiry = TypeAdapter(AwareDatetime).validate_python(approval["expires_at"])
            ceiling, hourly = (
                Decimal(str(approval["maximum_cost_usd"])),
                Decimal(str(approval["gpu_hourly_usd"])),
            )
        except (ValidationError, ValueError, ArithmeticError) as exc:
            raise unavailable("Reviewed job cost admission") from exc
        now = utcnow()
        deadline = min(specification.run.deadline, expiry)
        remaining = (deadline - now).total_seconds()
        if (
            expiry <= now
            or remaining <= 0
            or not hourly.is_finite()
            or hourly <= 0
            or not ceiling.is_finite()
        ):
            raise Problem(
                503, "job_approval_expired", "The original job/time/cost approval is not current."
            )
        if (
            specification.run.approved_cost_usd > ceiling
            or Decimal(str(remaining)) * hourly / 3600 > specification.run.approved_cost_usd
        ):
            raise Problem(
                422,
                "job_cost_not_approved",
                "Approved compute time exceeds the explicit USD ceiling.",
            )
        config = copy.deepcopy(approval["config"])
        if config.get("execution_timing") != specification.project.execution_timing or (
            specification.project.execution_timing is not None
            and (
                config.get("real_time_admission") is not False
                or config.get("criteria_sha256") != specification.project.criteria_sha256
                or config.get("frozen_plan_sha256") != specification.project.frozen_plan_sha256
                or config.get("control_profile_sha256")
                != specification.project.control_profile_sha256
            )
        ):
            raise Problem(
                409,
                "worker_timing_mismatch",
                "Approved native mode/profile/criteria differ from project.",
            )
        timeout = config.get("parameters", {}).get("timeout_seconds")
        maximum_duration = (
            specification.project.budget.training_seconds
            if specification.run.kind == "training"
            else specification.project.budget.evaluation_seconds
        )
        if type(timeout) is not int or not 0 < timeout <= min(maximum_duration, remaining):
            raise Problem(
                422,
                "worker_time_budget",
                "The complete Azure job timeout must fit the original remaining authorization.",
            )
        if config.get("schema") != NATIVE_DEADLINE_SCHEMA or config.get(
            "job_deadline_utc"
        ) != deadline.astimezone(UTC).isoformat().replace("+00:00", "Z"):
            raise Problem(
                409,
                "worker_deadline_mismatch",
                "Register the exact original run/operator not-after in the reviewed v2 config.",
            )
        if (
            config.get("tenant_id") != str(actor.tenant_id)
            or config.get("owner_id") != actor.owner_key
        ):
            raise Problem(
                403, "worker_scope_mismatch", "Registered Azure plan belongs to another owner."
            )
        if config.get("specification_sha256") != specification.run.specification_sha256:
            raise Problem(
                409,
                "worker_plan_mismatch",
                "Register the exact approved immutable job specification.",
            )
        if (
            config.get("policy_type", specification.project.policy_type)
            != specification.project.policy_type
        ):
            raise Problem(409, "worker_policy_mismatch", "Policy families cannot be relabeled.")
        inputs = config.get("inputs", {})
        if specification.run.kind == "training":
            if not specification.dataset:
                raise Problem(422, "dataset_missing", "Training requires the sealed dataset.")
            parent = specification.training_parent or specification.baseline
            if (
                parent is None
                or inputs.get("demonstrations", {}).get("sha256")
                != specification.dataset.manifest_sha256
                or inputs.get("parent_model", {}).get("sha256") != parent.model_sha256
            ):
                raise Problem(
                    409,
                    "worker_input_mismatch",
                    "Dataset and parent model must match approved content digests.",
                )
            if config.get("parameters", {}).get("max_steps") != specification.run.optimizer_steps:
                raise Problem(
                    409, "worker_plan_mismatch", "Optimizer count differs from explicit approval."
                )
        elif (
            specification.run.evaluation_plan_sha256 != specification.project.evaluation_plan.sha256
            or not re.fullmatch(r"[a-f0-9]{64}", str(inputs.get("plan", {}).get("sha256", "")))
            or inputs.get("plan", {}).get("type") != "uri_file"
        ):
            raise Problem(
                409, "worker_plan_mismatch", "Evaluation must use the frozen held-out plan."
            )
        return config

    def _target(self, actor, specification, configuration):
        return ReconciliationTarget(
            actor_id=actor.object_id,
            job_name=specification.run.backend_job_name,
            specification_sha256=specification.run.specification_sha256,
            configuration_sha256=fingerprint(configuration),
            job_deadline_utc=self._job_deadline(specification, configuration),
        )

    def _authorized_target(self, actor, target):
        if (
            not self.reconciliation_enabled
            or actor.object_id not in self.reconciliation_actor_ids
            or target.actor_id != actor.object_id
            or target not in self.reconciliation_targets
            or not 0 < len(self.reconciliation_targets) <= 20
        ):
            raise Problem(
                503,
                "job_reconciliation_unenrolled",
                "An authorized exact job deadline monitor is required before submission.",
            )

    def _enrollment(self, actor, specification, configuration):
        target = self._target(actor, specification, configuration)
        self._authorized_target(actor, target)
        record = self.registry.heartbeat(actor, target)
        now = utcnow()
        if (
            record is None
            or record.value.target != target
            or record.value.worker_client_id != self.caller_client_id
            or not -5 <= (now - record.value.observed_at).total_seconds() <= 90
            or now >= target.job_deadline_utc
        ):
            raise Problem(
                503,
                "job_reconciliation_unenrolled",
                "The exact deadline monitor has no fresh verified enrollment heartbeat.",
            )

    @staticmethod
    def _job_deadline(specification, configuration):
        value = configuration.get("job_deadline_utc")
        if value is None and str(configuration.get("schema", "")).endswith("/v1"):
            return specification.run.deadline
        try:
            parsed = TypeAdapter(AwareDatetime).validate_python(value)
        except ValidationError as exc:
            raise unavailable("Frozen job deadline") from exc
        if parsed > specification.run.deadline:
            raise Problem(409, "worker_deadline_mismatch", "Frozen job deadline exceeds approval.")
        return parsed

    def _sdk(self, config, policy_type, *, model_use=False):
        if self.sdk_factory is not None:
            return self.sdk_factory(config)
        module_name, jobs_name = implementation(policy_type, model_use=model_use)
        try:
            sdk = importlib.import_module(f"{module_name}.azure")
        except ModuleNotFoundError as exc:
            raise unavailable("Pinned policy Azure worker dependencies") from exc
        clients = sdk.clients_for_managed_identity(
            config, caller_client_id=str(self.caller_client_id)
        )
        jobs = getattr(sdk, jobs_name)(clients[0], config, storage_client=clients[1])
        return jobs, sdk.create_plan

    def preflight(self, actor, specification):
        if specification.project.execution_timing == "paused_simulation" and not (
            self.paused_training_enabled
            if specification.run.kind == "training"
            else self.paused_evaluation_enabled
        ):
            raise Problem(
                503,
                "paused_learning_unavailable",
                "Paused runtime/data/model/report adapters are not admitted; no legacy fallback.",
            )
        if specification.project.policy_type not in self.allowed_policy_types:
            raise Problem(
                503,
                "learning_policy_unapproved",
                "Model license and hardware admission are not verified.",
            )
        config = self._configuration(actor, specification)
        self._enrollment(actor, specification, config)
        jobs, _ = self._sdk(config, specification.project.policy_type, model_use=True)
        jobs.preflight()

    def submit(self, actor, specification):
        self.preflight(actor, specification)
        config = self._configuration(actor, specification)
        self._enrollment(actor, specification, config)
        if not self.registry.claim_job(actor, specification, config):
            existing = self.status(actor, specification.run)
            if existing is None:
                raise Problem(
                    503,
                    "worker_submission_unconfirmed",
                    "An existing claim cannot be submitted again.",
                )
            return existing
        jobs, create_plan = self._sdk(config, specification.project.policy_type, model_use=True)
        with TemporaryDirectory(prefix="physicalai-job-plan-") as folder:
            plan = Path(folder) / "plan"
            digest = create_plan(
                config, plan, deterministic_job_name=specification.run.backend_job_name
            )
            self._enrollment(actor, specification, config)
            receipt = jobs.submit(
                plan,
                approved_plan_sha256=digest,
                deterministic_job_name=specification.run.backend_job_name,
            )
        return self._receipt(actor, specification, receipt, configuration=config)

    def _validated_receipt(self, actor, specification, receipt, configuration=None):
        try:
            status = receipt["status"]
            if receipt.get("azure_status") in ("NotResponding", "Paused", "Unknown"):
                status = "running"
            deadline = (
                self._job_deadline(specification, configuration)
                if configuration is not None
                else None
            )
            result = BackendJob.model_validate(
                {
                    "job_name": receipt["job_name"],
                    "azure_job_id": receipt["azure_job_id"],
                    "owner_key": receipt["owner_key"],
                    "specification_sha256": receipt["specification_sha256"],
                    "status": status,
                    "azure_status": receipt.get("azure_status"),
                    "job_deadline_utc": deadline,
                    "metrics": {"optimizer_steps": None, "loss": None, "measured_at": None},
                }
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise unavailable("Actual Azure ML job receipt") from exc
        if (
            result.owner_key != actor.owner_key
            or result.job_name != specification.run.backend_job_name
            or result.specification_sha256 != specification.run.specification_sha256
        ):
            raise Problem(
                503, "worker_receipt_mismatch", "Azure job tags differ from the durable claim."
            )
        if (
            configuration is not None
            and receipt.get("job_deadline_utc") is not None
            and (receipt["job_deadline_utc"] != configuration.get("job_deadline_utc"))
        ):
            raise Problem(503, "worker_receipt_mismatch", "Azure job deadline tag differs.")
        return result

    def _receipt(self, actor, specification, receipt, *, configuration=None):
        result = self._validated_receipt(actor, specification, receipt, configuration)
        if result.status == "succeeded" and specification.run.kind == "training":
            candidate = self.artifacts.completed_candidate(
                actor, specification, result.azure_job_id
            )
            result = result.model_copy(
                update={
                    "candidate": candidate,
                    "metrics": TrainingMetrics(
                        optimizer_steps=candidate.optimizer_steps,
                        measured_at=candidate.updated_at,
                    ),
                }
            )
        if specification.run.kind == "evaluation" and result.status in (
            "succeeded",
            "failed",
            "cancelled",
            "timed_out",
        ):
            try:
                report = self.artifacts.completed_report(
                    actor, specification, result.azure_job_id, required=result.status == "succeeded"
                )
                if report is not None and not isinstance(
                    report, (PairedReport, BootstrapReport, SimulationReport)
                ):
                    raise unavailable("Verified complete physical evaluation report")
                if result.status == "succeeded" and report is None:
                    raise unavailable("Final physical evaluation report")
                result = result.model_copy(update={"report": report})
            except Problem as exc:
                if result.status == "succeeded":
                    raise
                result = result.model_copy(
                    update={
                        "error_code": "evaluation_report_unverified",
                        "message": (
                            f"Azure job is {result.status}; its report is unverified ({exc.code})."
                        ),
                    }
                )
        return result

    def _job(self, actor, run):
        specification = self.registry.job(actor, run.backend_job_name)
        if specification is None:
            return None
        self._authorize_specification(actor, specification)
        if any(
            getattr(specification.run, field) != getattr(run, field)
            for field in (
                "id",
                "kind",
                "project_id",
                "policy_type",
                "deadline",
                "backend_job_name",
                "specification_sha256",
            )
        ):
            raise Problem(
                409, "worker_job_mismatch", "Requested job differs from its original claim."
            )
        return specification

    def _job_configuration(self, actor, specification):
        config = self.registry.job_configuration(actor, specification)
        if config is None:
            config = self.registry.approved_plan(actor, specification)["config"]
            if not str(config.get("schema", "")).endswith("/v1"):
                raise unavailable("Original frozen job configuration")
        self._job_deadline(specification, config)
        return config

    @staticmethod
    def _azure_error(error):
        forbidden = getattr(error, "status_code", None) == 403
        log.error("Owned Azure learning operation failed: %s", type(error).__name__)
        return Problem(
            403 if forbidden else 503,
            "job_access_forbidden" if forbidden else "job_operation_unconfirmed",
            "Azure denied this operation."
            if forbidden
            else "The Azure operation is unconfirmed; no mutation was retried.",
        )

    def _read_status(self, jobs, name):
        try:
            return jobs.status(name)
        except ResourceNotFoundError:
            return None
        except AzureError as exc:
            raise self._azure_error(exc) from exc
        except ValueError as exc:
            log.error("Owned job read failed immutable scope validation")
            raise Problem(
                409, "worker_job_mismatch", "Azure job tags differ from the owned frozen scope."
            ) from exc

    def _with_cancellation(self, actor, specification, config, result):
        marker = self.registry.cancellation(actor, specification, config)
        if marker is None:
            return result
        return result.model_copy(
            update={
                "cancellation_state": marker.value.state,
                "error_code": marker.value.error_code or result.error_code,
                "message": result.message
                if result.status in JOB_TERMINAL
                else (
                    f"Cancellation {marker.value.state}; "
                    f"Azure reports {result.azure_status or result.status}."
                ),
            }
        )

    def status(self, actor, run):
        specification = self._job(actor, run)
        if specification is None:
            return None
        config = self._job_configuration(actor, specification)
        jobs, _ = self._sdk(config, specification.project.policy_type)
        receipt = self._read_status(jobs, run.backend_job_name)
        if receipt is None:
            return None
        return self._with_cancellation(
            actor,
            specification,
            config,
            self._receipt(actor, specification, receipt, configuration=config),
        )

    def _cancel_receipt(self, actor, specification, config, jobs):
        name = specification.run.backend_job_name
        raw = self._read_status(jobs, name)
        if raw is None:
            raise Problem(404, "worker_job_missing", "The owned Azure job is not yet confirmed.")
        observed = self._validated_receipt(actor, specification, raw, config)
        if observed.status in JOB_TERMINAL or observed.status == "cancelling":
            return raw
        claim, first = self.registry.claim_cancellation(actor, specification, config)
        if not first:
            return raw
        try:
            result = jobs.cancel(name)
            self._validated_receipt(actor, specification, result, config)
        except AzureError as exc:
            failure = self._azure_error(exc)
            self.registry.record_cancellation(
                actor, claim, "forbidden" if failure.status == 403 else "uncertain", failure.code
            )
            raise failure from exc
        except (Problem, ValueError) as exc:
            self.registry.record_cancellation(actor, claim, "uncertain", "job_cancel_unconfirmed")
            if isinstance(exc, Problem):
                raise
            raise unavailable("Verified owned cancellation") from exc
        self.registry.record_cancellation(actor, claim, "acknowledged")
        return result

    def cancel(self, actor, run):
        specification = self._job(actor, run)
        if specification is None:
            raise Problem(404, "worker_job_missing", "No owned durable job claim exists.")
        config = self._job_configuration(actor, specification)
        jobs, _ = self._sdk(config, specification.project.policy_type)
        raw = self._cancel_receipt(actor, specification, config, jobs)
        return self._with_cancellation(
            actor,
            specification,
            config,
            self._receipt(actor, specification, raw, configuration=config),
        )

    def reconcile_deadline(self, actor, target):
        self._authorized_target(actor, target)
        specification = self.registry.job(actor, target.job_name)
        if specification is None:
            return {
                "job_name": target.job_name,
                "status": "awaiting_submission",
                "deadline_expired": utcnow() >= target.job_deadline_utc,
            }
        self._authorize_specification(actor, specification)
        config = self._job_configuration(actor, specification)
        if self._target(actor, specification, config) != target:
            raise Problem(409, "worker_job_mismatch", "Monitor target differs from the frozen job.")
        jobs, _ = self._sdk(config, specification.project.policy_type)
        raw = self._read_status(jobs, target.job_name)
        if raw is None:
            return {
                "job_name": target.job_name,
                "status": "unconfirmed",
                "deadline_expired": utcnow() >= target.job_deadline_utc,
            }
        result = self._validated_receipt(actor, specification, raw, config)
        expired = utcnow() >= target.job_deadline_utc
        if expired and result.status not in JOB_TERMINAL and result.status != "cancelling":
            raw = self._cancel_receipt(actor, specification, config, jobs)
            result = self._validated_receipt(actor, specification, raw, config)
        result = self._with_cancellation(actor, specification, config, result)
        return {
            "job_name": result.job_name,
            "status": result.status,
            "azure_status": result.azure_status,
            "deadline_expired": expired,
            "job_deadline_utc": target.job_deadline_utc.isoformat(),
            "cancellation_state": result.cancellation_state,
            "error_code": result.error_code,
        }
