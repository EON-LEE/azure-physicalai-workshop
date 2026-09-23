from __future__ import annotations

import copy
import importlib
import re
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import BootstrapReport, PairedReport, TrainingMetrics
from apps.api.learning_ports import BackendJob, JobSpecification
from apps.api.models import utcnow
from apps.learning_worker.policies import implementation


class PolicyLearningWorker:
    def __init__(
        self,
        registry,
        artifact_verifier,
        caller_client_id,
        *,
        allowed_policy_types=(),
        sdk_factory=None,
    ):
        self.registry = registry
        self.artifacts = artifact_verifier
        self.caller_client_id = caller_client_id
        self.allowed_policy_types = tuple(allowed_policy_types)
        self.sdk_factory = sdk_factory

    def _authorize_specification(self, actor, specification: JobSpecification):
        if (
            specification.owner_key != actor.owner_key
            or specification.project.owner_key != actor.owner_key
            or specification.run.owner_key != actor.owner_key
            or specification.project.actor_id != actor.object_id
            or specification.run.project_id != specification.project.id
            or specification.project.tenant_id != actor.tenant_id
        ):
            raise Problem(403, "worker_scope_mismatch", "Worker request scope is inconsistent.")
        if specification.run.policy_type != specification.project.policy_type:
            raise Problem(409, "worker_policy_mismatch", "The job's exact model family changed.")
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
        if not isinstance(approval, dict) or set(approval) != required:
            raise unavailable("Operator-reviewed job cost and configuration")
        from pydantic import AwareDatetime, TypeAdapter

        try:
            expiry = TypeAdapter(AwareDatetime).validate_python(approval["expires_at"])
            ceiling, hourly = (
                Decimal(str(approval["maximum_cost_usd"])),
                Decimal(str(approval["gpu_hourly_usd"])),
            )
        except (ValidationError, ValueError, ArithmeticError) as exc:
            raise unavailable("Reviewed job cost admission") from exc
        remaining = (specification.run.deadline - utcnow()).total_seconds()
        if (
            expiry <= utcnow()
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

    def _sdk(self, config, policy_type, *, model_use=False):
        if self.sdk_factory is not None:
            return self.sdk_factory(config)
        module_name, jobs_name = implementation(policy_type, model_use=model_use)
        try:
            sdk = importlib.import_module(f"{module_name}.azure")
        except ModuleNotFoundError as exc:
            raise unavailable("Pinned policy Azure worker dependencies") from exc
        clients = sdk.clients_for_managed_identity(config, str(self.caller_client_id))
        jobs = getattr(sdk, jobs_name)(clients[0], config, storage_client=clients[1])
        return jobs, sdk.create_plan

    def preflight(self, actor, specification):
        if specification.project.policy_type not in self.allowed_policy_types:
            raise Problem(
                503,
                "learning_policy_unapproved",
                "Model license and hardware admission are not verified.",
            )
        config = self._configuration(actor, specification)
        jobs, _ = self._sdk(config, specification.project.policy_type, model_use=True)
        jobs.preflight()

    def submit(self, actor, specification):
        self.preflight(actor, specification)
        config = self._configuration(actor, specification)
        if not self.registry.claim_job(actor, specification):
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
            digest = create_plan(config, plan, specification.run.backend_job_name)
            receipt = jobs.submit(
                plan,
                approved_plan_sha256=digest,
                deterministic_job_name=specification.run.backend_job_name,
            )
        return self._receipt(actor, specification, receipt)

    def _receipt(self, actor, specification, receipt):
        try:
            result = BackendJob.model_validate(
                {
                    "job_name": receipt["job_name"],
                    "azure_job_id": receipt["azure_job_id"],
                    "owner_key": receipt["owner_key"],
                    "specification_sha256": receipt["specification_sha256"],
                    "status": receipt["status"],
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
        if result.status == "succeeded":
            # Completed Azure state alone never manufactures a checkpoint or a quality result.
            if specification.run.kind == "training":
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
            else:
                report = self.artifacts.completed_report(actor, specification, result.azure_job_id)
                if not isinstance(report, (PairedReport, BootstrapReport)):
                    raise unavailable("Verified complete physical evaluation report")
                result = result.model_copy(update={"report": report})
        return result

    def status(self, actor, run):
        specification = self.registry.job(actor, run.backend_job_name)
        if specification is None:
            return None
        self._authorize_specification(actor, specification)
        approval = self.registry.approved_plan(actor, specification)
        jobs, _ = self._sdk(approval["config"], specification.project.policy_type)
        try:
            receipt = jobs.status(run.backend_job_name)
        except Problem:
            raise
        return self._receipt(actor, specification, receipt)

    def cancel(self, actor, run):
        specification = self.registry.job(actor, run.backend_job_name)
        if specification is None:
            raise Problem(404, "worker_job_missing", "No owned durable job claim exists.")
        self._authorize_specification(actor, specification)
        approval = self.registry.approved_plan(actor, specification)
        jobs, _ = self._sdk(approval["config"], specification.project.policy_type)
        return self._receipt(actor, specification, jobs.cancel(run.backend_job_name))
