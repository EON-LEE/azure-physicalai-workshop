from __future__ import annotations

import logging
from datetime import timedelta
from typing import TypeVar
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from apps.api.artifact_models import ArtifactOperationRecord, ArtifactStatus, ArtifactWork
from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    JOB_TERMINAL,
    PAUSED_PROFILE_ID,
    PROFILE_ID,
    TEACHING_TERMINAL,
    ArmTeaching,
    BootstrapReport,
    CoachRecord,
    ControlGrant,
    CreateDataset,
    CreateProject,
    DatasetVersion,
    EvaluationRun,
    JobCancellation,
    JogIntent,
    JogTeaching,
    LearningMutation,
    LearningProject,
    LearningRecord,
    PairedReport,
    PolicyRelease,
    PolicyScene,
    ReleasePolicy,
    StartEvaluation,
    StartTeaching,
    StartTraining,
    TeachingControl,
    TeachingSession,
    TrainingParent,
    TrainingRun,
    fingerprint,
    transition,
)
from apps.api.learning_ports import (
    BackendJob,
    JobSpecification,
    LearningArtifacts,
    LearningJobs,
    LearningStore,
    PolicyCatalog,
    RuntimeCapture,
    TeachingRuntime,
    TeachingRuntimeState,
    TeachingStartSpec,
    TeachingTask,
)
from apps.api.models import (
    TERMINAL,
    LearnedPolicyType,
    Principal,
    ReleasedPolicyBinding,
    Stored,
    utcnow,
)
from apps.api.reference_models import (
    REFERENCE_TERMINAL,
    ReferenceCollection,
    StartReferenceCollection,
)
from apps.api.service import FactoryService, check_fresh, content_hash
from apps.api.simulation_models import SimulationEpisodeCommand, SimulationEpisodeExecution
from apps.api.simulation_reports import SimulationReport, validate_report_binding
from contracts.validate_environment import (
    validate_environment,
    validate_paused_learning_environment,
)

log = logging.getLogger(__name__)
R = TypeVar("R", bound=LearningRecord | ArtifactOperationRecord | ReferenceCollection)


def require_etag(stored: Stored, expected: str | None) -> None:
    if expected is None:
        raise Problem(428, "learning_precondition_required", "Supply the loaded If-Match ETag.")
    if expected != stored.etag:
        raise Problem(409, "revision_conflict", "Learning state changed; reload before acting.")


def metadata(actor: Principal, request_id: UUID, digest: str) -> dict:
    now = utcnow()
    return {
        "id": request_id,
        "owner_key": actor.owner_key,
        "actor_id": actor.object_id,
        "tenant_id": actor.tenant_id,
        "created_at": now,
        "updated_at": now,
        "fingerprint": digest,
    }


def operation_hash(project_id: UUID, operation: str, request) -> str:
    return fingerprint(
        {
            "context": str(project_id),
            "operation": operation,
            "request": request.model_dump(mode="json"),
        }
    )


class LearningService:
    def __init__(
        self,
        factory: FactoryService,
        store: LearningStore | None = None,
        jobs: LearningJobs | None = None,
        artifacts: LearningArtifacts | None = None,
        catalog: PolicyCatalog | None = None,
        *,
        enabled: bool = False,
        runtime: TeachingRuntime | None = None,
        coach=None,
        bootstrap_principal_ids: frozenset[UUID] = frozenset(),
        allowed_policy_types: tuple[LearnedPolicyType, ...] = (),
        reference_collections_enabled: bool = False,
        paused_training_enabled: bool = False,
        paused_evaluation_enabled: bool = False,
        paused_release_enabled: bool = False,
    ) -> None:
        self.factory = factory
        self.store = store
        self.jobs = jobs
        self.artifacts = artifacts
        self.catalog = catalog
        self.enabled = enabled
        self.runtime = runtime
        self.coach = coach
        self.bootstrap_principal_ids = bootstrap_principal_ids
        self.allowed_policy_types = allowed_policy_types
        self.reference_collections_enabled = reference_collections_enabled
        self.paused_training_enabled = paused_training_enabled
        self.paused_evaluation_enabled = paused_evaluation_enabled
        self.paused_release_enabled = paused_release_enabled

    def capabilities(self, actor: Principal | None = None) -> dict:
        integrated = all(
            (
                self.store,
                self.jobs,
                self.artifacts,
                self.catalog,
                self.runtime,
                self.allowed_policy_types,
            )
        )
        paused_configured = self.enabled and integrated
        return {
            "enabled": self.enabled and integrated,
            "status": "configured"
            if self.enabled and integrated
            else ("blocked" if self.enabled else "disabled"),
            "message": (
                "Real training and learned execution require separately verified Azure integration."
                if integrated
                else "Learning dependencies are not integrated; no fallback exists."
            ),
            "policy_types": list(self.allowed_policy_types),
            "control_profiles": [PROFILE_ID],
            "training_verified": False,
            "coach_configured": self.coach is not None,
            "bootstrap_allowed": bool(actor and actor.object_id in self.bootstrap_principal_ids),
            "model_admission": "configured_not_verified"
            if self.allowed_policy_types
            else "license_or_hardware_unapproved",
            "simulation_learning": {
                "execution_timing": "paused_simulation",
                "real_time_admission": False,
                "supported": True,
                "enabled": False,
                "reference_generation_enabled": paused_configured
                and self.reference_collections_enabled,
                "training_enabled": paused_configured and self.paused_training_enabled,
                "evaluation_enabled": paused_configured and self.paused_evaluation_enabled,
                "release_enabled": paused_configured and self.paused_release_enabled,
                "status": "staged_configuration_not_verification"
                if paused_configured
                and any(
                    (
                        self.reference_collections_enabled,
                        self.paused_training_enabled,
                        self.paused_evaluation_enabled,
                        self.paused_release_enabled,
                    )
                )
                else "producer_verifier_unavailable",
                "message": (
                    "Each simulation-only stage requires separate operator admission and exact "
                    "runtime/artifact authority. Configuration is not learned-policy proof."
                ),
            },
        }

    def _enabled(self) -> None:
        if not self.enabled:
            raise Problem(
                503, "learning_disabled", "Learning is disabled until verified integration."
            )
        if self.store is None:
            raise unavailable("Durable learning store")

    def coach_proposal(self, actor, project_id, body):
        from agents.learning_coach import CoachContext

        digest = operation_hash(project_id, "coach", body)
        existing = self._existing(actor, "coach", body.request_id, digest)
        if existing:
            return self._coach_response(actor, existing)
        project = self.get(actor, "project", project_id).value
        dataset = self.get(actor, "dataset", body.dataset_id).value if body.dataset_id else None
        evaluation = (
            self.get(actor, "evaluation", body.evaluation_run_id).value
            if body.evaluation_run_id
            else None
        )
        if (dataset and dataset.project_id != project_id) or (
            evaluation and evaluation.project_id != project_id
        ):
            raise Problem(404, "learning_resource_missing", "Coach context is not in this project.")
        releases = self.list(actor, "release", project_id)
        context = CoachContext(
            project_id=project_id,
            task_id=project.task_id,
            instruction=project.instruction,
            human_teleop_count=dataset.human_teleop_count if dataset else 0,
            reference_controller_count=dataset.reference_controller_count if dataset else 0,
            learned_policy_count=dataset.learned_policy_count if dataset else 0,
            optimizer_step_limit=project.budget.optimizer_steps,
            dataset_id=dataset.id if dataset else None,
            evaluation_conclusion=evaluation.report.conclusion
            if evaluation and isinstance(evaluation.report, PairedReport)
            else None,
            approved_release_ids=tuple(item.value.id for item in releases),
        )
        coach = self._dependency(self.coach, "Separate Foundry learning coach")
        claim = CoachRecord(
            **metadata(actor, body.request_id, digest),
            project_id=project_id,
            status="planning",
        )
        stored, first = self._claim(actor, claim)
        if not first:
            return self._coach_response(actor, stored)
        try:
            proposal, response_id = coach.propose(body.instruction, context)
        except Problem as exc:
            self._save(actor, stored, status="failed", error_code=exc.code, message=exc.message)
            raise
        recorded = self._save(
            actor,
            stored,
            status="recorded",
            proposal=proposal,
            model_response_id=response_id,
        )
        return self._coach_response(actor, recorded)

    def _coach_response(self, actor, stored):
        record = stored.value
        if record.status == "planning" and (utcnow() - record.created_at).total_seconds() > 150:
            record = self._save(
                actor,
                stored,
                status="failed",
                error_code="coach_unconfirmed",
                message="The original model response was not confirmed; it was not resubmitted.",
            ).value
        if record.status != "recorded" or record.proposal is None or not record.model_response_id:
            raise Problem(
                409 if record.status == "planning" else 503,
                record.error_code or "coach_in_progress",
                record.message or "The original coach request is still awaiting its response.",
            )
        return {
            "request_id": str(record.id),
            "model_response_id": record.model_response_id,
            "authority": "proposal_only",
            "proposal": record.proposal.model_dump(mode="json"),
        }

    def _dependency(self, component, name: str):
        if component is None:
            raise unavailable(name)
        return component

    def _policy(self, policy_type):
        if policy_type not in self.allowed_policy_types:
            raise Problem(
                503,
                "learning_policy_unapproved",
                "No verified license and hardware admission exists for this pinned policy type.",
            )

    def _timing_admission(self, project: LearningProject, stage="reference") -> None:
        admitted = {
            "project": self.reference_collections_enabled or self.paused_training_enabled,
            "reference": self.reference_collections_enabled,
            "training": self.paused_training_enabled,
            "evaluation": self.paused_evaluation_enabled,
            "release": self.paused_release_enabled,
        }[stage]
        if project.execution_timing == "paused_simulation" and not admitted:
            raise Problem(
                503,
                "paused_learning_unavailable",
                "Paused runtime/model/report adapters are not admitted; no fallback was used.",
            )

    def get(self, actor: Principal, kind: str, resource_id: UUID) -> Stored:
        self._enabled()
        stored = self.store.get_learning(actor.owner_key, kind, resource_id)
        if stored is None or stored.value.owner_key != actor.owner_key:
            raise Problem(404, "learning_resource_missing", "Learning resource not found.")
        return stored

    def list(self, actor: Principal, kind: str, project_id: UUID | None = None) -> list[Stored]:
        self._enabled()
        if project_id:
            self.get(actor, "project", project_id)
        return self.store.list_learning(actor.owner_key, kind, project_id)

    def _existing(self, actor: Principal, kind: str, resource_id: UUID, digest: str):
        self._enabled()
        stored = self.store.get_learning(actor.owner_key, kind, resource_id)
        if stored and stored.value.fingerprint != digest:
            raise Problem(409, "request_id_reused", "Use a new request ID for changed input.")
        return stored

    def _claim(self, actor: Principal, record: R) -> tuple[Stored[R], bool]:
        try:
            return self.store.put_learning(actor.owner_key, record, None), True
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            winner = self._existing(actor, record.kind, record.id, record.fingerprint)
            if winner is None:
                raise
            return winner, False

    def _save(self, actor: Principal, stored: Stored[R], **changes) -> Stored[R]:
        updated = type(stored.value).model_validate(
            {
                **stored.value.model_dump(),
                "updated_at": utcnow(),
                **changes,
            }
        )
        return self.store.put_learning(actor.owner_key, updated, stored.etag)

    def _begin_artifact(
        self,
        actor,
        project,
        operation_id,
        target_id,
        digest,
        *,
        session=None,
        receipt=None,
        captures=(),
        managed_import=None,
        external_import=None,
    ):
        existing = self._existing(actor, "artifact_operation", operation_id, digest)
        if existing:
            return self.get_artifact_operation(actor, operation_id)
        client = self._dependency(self.artifacts, "Resident artifact processor")
        policy = client.artifact_policy(actor)
        now = utcnow()
        operation = (
            "external_training"
            if external_import is not None
            else "managed_evaluation"
            if managed_import is not None
            else "capture"
            if session is not None
            else "dataset"
        )
        work = ArtifactWork(
            id=operation_id,
            actor=actor,
            operation=operation,
            project=project,
            target_id=target_id,
            created_at=now,
            deadline=now + timedelta(seconds=policy.maximum_seconds),
            max_bytes=policy.capture_bytes if operation == "capture" else policy.dataset_bytes,
            max_files=policy.maximum_files,
            session=session,
            receipt=receipt,
            captures=captures,
            managed_import=managed_import,
            external_import=external_import,
        )
        state = ArtifactStatus(
            id=work.id,
            owner_key=actor.owner_key,
            project_id=project.id,
            target_id=target_id,
            operation=operation,
            work_sha256=work.sha256,
            created_at=now,
            updated_at=now,
            deadline=work.deadline,
            max_bytes=work.max_bytes,
            max_files=work.max_files,
            status="queued",
        )
        record = ArtifactOperationRecord.model_validate(
            {
                **metadata(actor, operation_id, digest),
                **state.model_dump(),
                "work_document": work.model_dump(mode="json"),
            }
        )
        stored, first = self._claim(actor, record)
        if not first:
            return self.get_artifact_operation(actor, operation_id)
        try:
            response = client.begin_artifact(actor, work)
        except Problem as exc:
            return self._save(
                actor,
                stored,
                status="uncertain",
                error_code=exc.code,
                message="Artifact admission is unconfirmed; reconcile only the original operation.",
            )
        return self._apply_artifact(actor, stored, response)

    def _apply_artifact(self, actor, stored, response: ArtifactStatus):
        original = stored.value
        if any(
            getattr(response, field) != getattr(original, field)
            for field in (
                "id",
                "owner_key",
                "project_id",
                "target_id",
                "operation",
                "work_sha256",
                "created_at",
                "deadline",
                "max_bytes",
                "max_files",
            )
        ):
            raise Problem(
                503, "artifact_receipt_mismatch", "Artifact operation scope or budget changed."
            )
        if original.status in ("ready", "failed", "timed_out"):
            return stored
        if original.operation == "managed_evaluation":
            self._apply_managed_import(actor, original, response)
        if original.operation == "external_training" and response.status == "ready":
            self._apply_external_import(actor, original, response)
        if response.status == "ready" and original.operation == "dataset":
            work = ArtifactWork.model_validate(original.work_document)
            self._claim(
                actor,
                self._dataset_record(
                    actor,
                    work.project,
                    work.target_id,
                    original.fingerprint,
                    work.captures,
                    response.result.artifact_id,
                    response.result.manifest_sha256,
                ),
            )
        updates = {
            field: getattr(response, field)
            for field in (
                "status",
                "phase",
                "result",
                "claim_id",
                "heartbeat_at",
                "error_code",
                "message",
            )
        }
        if all(getattr(original, field) == value for field, value in updates.items()):
            return stored
        try:
            return self._save(actor, stored, **updates)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self.get(actor, "artifact_operation", original.id)

    def get_artifact_operation(self, actor, operation_id):
        stored = self.get(actor, "artifact_operation", operation_id)
        if stored.value.status in ("ready", "failed", "timed_out"):
            return stored
        state = self._dependency(self.artifacts, "Resident artifact processor").artifact_status(
            actor, operation_id
        )
        if state is None:
            if (
                stored.value.status == "queued"
                and (utcnow() - stored.value.created_at).total_seconds() <= 20
            ):
                return stored
            if stored.value.status == "uncertain":
                return stored
            return self._save(
                actor,
                stored,
                status="uncertain",
                error_code="artifact_worker_receipt_missing",
                message="No owned worker receipt exists; heavy work was not replayed.",
            )
        return self._apply_artifact(actor, stored, state)

    def import_managed_evaluation(self, actor, evaluation_id, body, etag):
        digest = operation_hash(evaluation_id, "managed_evaluation", body)
        if self._existing(actor, "artifact_operation", body.request_id, digest):
            return self.get_artifact_operation(actor, body.request_id)
        stored = self.get(actor, "evaluation", evaluation_id)
        require_etag(stored, etag)
        run = stored.value
        if run.provider != "managed_batch" or run.status != "awaiting_import":
            raise Problem(
                409, "managed_import_required", "An original managed evaluation is required."
            )
        project = self.get(actor, "project", run.project_id).value
        self._timing_admission(project, "evaluation")
        self._policy(project.policy_type)
        client = self._dependency(self.artifacts, "Resident artifact processor")
        reference = client.managed_import_reference(actor, project.id, run.id)
        if (
            reference.operation_id != body.request_id
            or reference.owner_key != actor.owner_key
            or reference.project_id != project.id
            or reference.evaluation_run_id != run.id
            or reference.specification_sha256 != run.specification_sha256
            or run.import_operation_id not in (None, body.request_id)
        ):
            raise Problem(409, "managed_import_binding", "The original import request differs.")
        return self._begin_artifact(
            actor, project, body.request_id, run.id, digest, managed_import=reference
        )

    def import_external_training(self, actor, project_id, import_id, body, etag):
        if body.request_id != import_id:
            raise Problem(
                409, "external_import_identity", "Import URL and request identity differ."
            )
        digest = operation_hash(project_id, "external_training", body)
        if self._existing(actor, "artifact_operation", import_id, digest):
            return self.get_artifact_operation(actor, import_id)
        stored = self.get(actor, "project", project_id)
        require_etag(stored, etag)
        self._bootstrap_actor(actor)
        self._timing_admission(stored.value, "training")
        self._policy(stored.value.policy_type)
        client = self._dependency(self.artifacts, "Bounded external training verifier")
        reference = client.external_import_reference(actor, stored.value, import_id)
        if (
            reference.import_id != import_id
            or reference.project_id != project_id
            or reference.owner_key != actor.owner_key
        ):
            raise Problem(
                409, "external_import_identity", "Original post-hoc import reference differs."
            )
        return self._begin_artifact(
            actor, stored.value, import_id, import_id, digest, external_import=reference
        )

    def _apply_external_import(self, actor, operation, response):
        work = ArtifactWork.model_validate(operation.work_document)
        result = response.result.external_training
        if (
            result is None
            or result.record.id != work.target_id
            or result.record.project_id != work.project.id
            or result.record.owner_key != actor.owner_key
            or result.record.completion_sha256 != work.external_import.completion_sha256
        ):
            raise Problem(
                503, "external_import_receipt", "Original verified external receipt differs."
            )
        self._dependency(self.artifacts, "External training verifier").verify_external_import(
            actor, work.project, result
        )
        existing = self.store.get_learning(actor.owner_key, "dataset", result.dataset.id)
        if existing is not None:
            excluded = {"created_at", "updated_at", "fingerprint"}
            if existing.value.model_dump(exclude=excluded) != result.dataset.model_dump(
                exclude=excluded
            ):
                raise Problem(
                    409, "external_import_dataset", "An existing dataset registration differs."
                )
        else:
            self._claim(actor, result.dataset)
        self._claim(actor, result.record)
        self._claim(actor, result.candidate)

    def _apply_managed_import(self, actor, operation, response):
        stored = self.get(actor, "evaluation", operation.target_id)
        run = stored.value
        work = ArtifactWork.model_validate(operation.work_document)
        reference = work.managed_import
        if (
            run.provider != "managed_batch"
            or run.project_id != work.project.id
            or run.specification_sha256 != reference.specification_sha256
            or run.import_operation_id not in (None, operation.id)
        ):
            raise Problem(409, "managed_import_binding", "Evaluation/import identity changed.")
        if run.status == "succeeded":
            if response.result is None or run.import_receipt != response.result.managed_evaluation:
                raise Problem(409, "managed_import_binding", "A completed import cannot change.")
            return
        if response.status == "ready":
            receipt = response.result.managed_evaluation
            if receipt is None or receipt.model_dump(exclude={"report"}) != reference.model_dump():
                raise Problem(503, "managed_import_receipt", "Complete managed receipt is missing.")
            project = self.get(actor, "project", run.project_id).value
            candidate = self.get(actor, "candidate", run.candidate_id).value
            before = (
                self.get(actor, "candidate", run.before_candidate_id).value
                if run.before_candidate_id is not None
                else None
            )
            baseline = (
                None
                if before is not None
                else self._baseline(
                    actor,
                    run.baseline_release_id,
                    execution_timing="paused_simulation",
                    control_profile_id=project.control_profile_id,
                )
            )
            validate_report_binding(
                JobSpecification(
                    owner_key=actor.owner_key,
                    project=project,
                    run=run,
                    candidate=candidate,
                    baseline=baseline,
                    baseline_candidate=before,
                ),
                receipt.report,
            )
            self._dependency(self.artifacts, "Verified learning artifacts").verify_report(
                actor, project, run, receipt.report
            )
            self._save(
                actor,
                stored,
                status="succeeded",
                import_operation_id=operation.id,
                import_receipt=receipt,
                report=receipt.report,
                error_code=None,
                message="Complete managed Batch artifacts verified; physical quality is separate.",
            )
        else:
            changes = {
                "import_operation_id": operation.id,
                "error_code": response.error_code,
                "message": response.message
                or "Awaiting a complete verified managed artifact import.",
            }
            if any(getattr(run, key) != item for key, item in changes.items()):
                self._save(actor, stored, **changes)

    def _baseline(
        self, actor: Principal, release_id: UUID, *, execution_timing=None, control_profile_id=None
    ) -> PolicyRelease:
        local = self.store.get_learning(actor.owner_key, "release", release_id)
        if local:
            release = local.value
        else:
            catalog = self._dependency(self.catalog, "Reviewed policy catalog")
            release = catalog.resolve(actor, release_id)
        if not isinstance(release, PolicyRelease):
            raise Problem(
                409,
                "training_parent_not_executable",
                "Train-only weights are not a reviewed policy.",
            )
        if release.owner_key != actor.owner_key or release.id != release_id:
            raise Problem(404, "policy_release_missing", "Reviewed policy not found.")
        self._policy(release.policy_type)
        profile = control_profile_id or (
            PAUSED_PROFILE_ID if execution_timing == "paused_simulation" else PROFILE_ID
        )
        if release.control_profile_id != profile or release.execution_timing != execution_timing:
            raise Problem(
                409, "policy_type_mismatch", "A reviewed compatible learned policy is required."
            )
        return release

    def _bootstrap_actor(self, actor: Principal):
        if actor.object_id not in self.bootstrap_principal_ids:
            raise Problem(
                403,
                "bootstrap_forbidden",
                "Only an explicitly configured bootstrap operator may act.",
            )

    def _training_parent(self, actor, artifact_id):
        catalog = self._dependency(self.catalog, "Verified training-parent catalog")
        method = getattr(catalog, "training_parent", None)
        if not callable(method):
            raise unavailable("Verified training-parent catalog")
        parent = method(actor, artifact_id)
        if (
            not isinstance(parent, TrainingParent)
            or parent.owner_key != actor.owner_key
            or parent.id != artifact_id
        ):
            raise Problem(404, "training_parent_missing", "Registered train-only parent not found.")
        self._policy(parent.policy_type)
        return parent

    def _saved_scene(self, actor, environment_id, revision):
        environment = self.factory.environment(actor, environment_id).value
        if environment.revision != revision:
            raise Problem(
                409, "revision_conflict", "The case's pinned saved revision is no longer current."
            )
        if (
            environment.environment_id != environment_id
            or environment.document.get("environment_id") != environment_id
            or validate_environment(environment.document)
            or content_hash(environment.document) != revision
        ):
            raise Problem(
                409, "case_environment_corrupted", "Saved case content does not match its revision."
            )
        if environment.document["execution"]["mode"] != "live":
            raise Problem(409, "replay_not_live", "Only a saved LIVE case can authorize teaching.")
        return environment

    def _case_environment(self, actor, case, *, goal_station_id, robot_profile, split):
        environment = self._saved_scene(actor, case.environment_id, case.revision)
        execution = environment.document["execution"]
        if execution.get("demonstration_split") != split or (
            split != "test" and execution.get("record_demonstration") is not True
        ):
            raise Problem(
                422,
                "case_split_mismatch",
                "The saved scene must explicitly authorize this capture split.",
            )
        scene = environment.document["scene"]
        if (
            scene["template_id"] != "inspection-cell-learning-v1"
            or scene["seed"] != case.seed
            or scene["robot_profile"] != robot_profile
            or goal_station_id
            not in {station["id"] for station in environment.document["stations"]}
        ):
            raise Problem(
                422,
                "learning_case_not_bound",
                "Cases require the saved reviewed pose builder, exact seed, robot and task goal.",
            )
        return environment

    @staticmethod
    def _check_timing_scene(project, environment):
        if project.execution_timing == "paused_simulation" and (
            validate_paused_learning_environment(environment.document)
            or environment.document["learning_execution"]["profile_id"]
            != project.control_profile_id
        ):
            raise Problem(
                409,
                "paused_execution_required",
                "Every frozen case needs explicit paused authority.",
            )

    def _verify_cases(self, actor, plan, *, goal_station_id, robot_profile, project=None):
        for case in plan.cases:
            environment = self._case_environment(
                actor,
                case,
                goal_station_id=goal_station_id,
                robot_profile=robot_profile,
                split="test",
            )
            if project is not None:
                self._check_timing_scene(project, environment)

    def create_project(self, actor: Principal, body: CreateProject) -> Stored[LearningProject]:
        self._enabled()
        record = LearningProject.create(actor, body)
        existing = self._existing(actor, "project", record.id, record.fingerprint)
        if existing:
            return existing
        self._timing_admission(record, "project")
        environment = self._saved_scene(actor, body.environment_id, body.revision)
        self._check_timing_scene(record, environment)
        if body.goal_station_id not in {item["id"] for item in environment.document["stations"]}:
            raise Problem(422, "unknown_task_goal", "Select a goal in the pinned environment.")
        if body.project_kind == "bootstrap":
            self._bootstrap_actor(actor)
            parent = self._training_parent(actor, body.pretrained_artifact_id)
            if parent.policy_type != body.policy_type:
                raise Problem(
                    409,
                    "policy_type_mismatch",
                    "The exact training-parent model family must match.",
                )
        else:
            baseline = self._baseline(
                actor,
                body.baseline_release_id,
                execution_timing=body.execution_timing,
                control_profile_id=body.control_profile_id,
            )
            if baseline.task_id != body.task_id or baseline.policy_type != body.policy_type:
                raise Problem(
                    409, "baseline_task_mismatch", "The baseline belongs to a different task."
                )
        self._policy(body.policy_type)
        parent = parent if body.project_kind == "bootstrap" else baseline
        if not parent.matches_timing(record):
            raise Problem(
                409, "parent_timing_mismatch", "The immutable parent timing/provenance differs."
            )
        robot_profile = environment.document["scene"]["robot_profile"]
        self._verify_cases(
            actor,
            body.evaluation_plan,
            goal_station_id=body.goal_station_id,
            robot_profile=robot_profile,
            project=record,
        )
        for case in body.teaching_cases:
            case_environment = self._case_environment(
                actor,
                case,
                goal_station_id=body.goal_station_id,
                robot_profile=robot_profile,
                split=case.split,
            )
            self._check_timing_scene(record, case_environment)
        return self._claim(actor, record)[0]

    def train(
        self,
        actor: Principal,
        project_id: UUID,
        body: StartTraining,
        etag: str | None,
    ) -> Stored[TrainingRun]:
        digest = operation_hash(project_id, "train", body)
        existing = self._existing(actor, "training", body.request_id, digest)
        if existing:
            return existing
        context = self.get(actor, "project", project_id)
        require_etag(context, etag)
        project = context.value
        self._timing_admission(project, "training")
        self._policy(project.policy_type)
        if body.policy_type != project.policy_type:
            raise Problem(
                409, "policy_type_mismatch", "Model versions cannot be relabeled or substituted."
            )
        dataset = self.get(actor, "dataset", body.dataset_id).value
        if (
            dataset.project_id != project_id
            or dataset.evaluation_plan_sha256 != project.evaluation_plan.sha256
        ):
            raise Problem(
                409, "dataset_project_mismatch", "The dataset is not pinned to this project."
            )
        self._check_split(project, dataset)
        if (
            body.parent_release_id != project.baseline_release_id
            or body.pretrained_artifact_id != project.pretrained_artifact_id
        ):
            raise Problem(409, "baseline_mismatch", "Training must use the frozen parent policy.")
        baseline, parent = None, None
        if project.project_kind == "bootstrap":
            self._bootstrap_actor(actor)
            parent = self._training_parent(actor, project.pretrained_artifact_id)
        else:
            baseline = self._baseline(
                actor,
                body.parent_release_id,
                execution_timing=project.execution_timing,
                control_profile_id=project.control_profile_id,
            )
        if (parent or baseline).policy_type != project.policy_type:
            raise Problem(
                409,
                "policy_type_mismatch",
                "No cross-family fallback or artifact relabeling is allowed.",
            )
        if not (parent or baseline).matches_timing(project) or not dataset.matches_timing(project):
            raise Problem(
                409,
                "training_timing_mismatch",
                "Dataset and parent must retain the frozen timing pins.",
            )
        if body.optimizer_steps > project.budget.optimizer_steps:
            raise Problem(
                422, "learning_budget_exceeded", "Optimizer steps exceed project approval."
            )
        self._cost(project, body.maximum_cost_usd)
        record = TrainingRun(
            **metadata(actor, body.request_id, digest),
            **project.timing_fields(),
            project_id=project_id,
            policy_type=project.policy_type,
            status="submitting",
            backend_job_name=self._job_name(actor, body.request_id),
            deadline=utcnow() + timedelta(seconds=project.budget.training_seconds),
            approved_cost_usd=body.maximum_cost_usd,
            specification_sha256=digest,
            dataset_id=dataset.id,
            parent_release_id=baseline.id if baseline else None,
            pretrained_artifact_id=parent.id if parent else None,
            optimizer_steps=body.optimizer_steps,
        )
        specification = JobSpecification(
            owner_key=actor.owner_key,
            project=project,
            run=record,
            baseline=baseline,
            training_parent=parent,
            dataset=dataset,
        )
        return self._submit(actor, specification)

    @staticmethod
    def _job_name(actor: Principal, request_id: UUID) -> str:
        return f"learning-{actor.owner_key[:12]}-{request_id.hex}"

    @staticmethod
    def _cost(project: LearningProject, approved) -> None:
        if approved > project.budget.maximum_cost_usd:
            raise Problem(
                422, "learning_budget_exceeded", "Cost approval exceeds the project ceiling."
            )

    @staticmethod
    def _check_split(project: LearningProject, dataset: DatasetVersion) -> None:
        if set(dataset.seeds) & set(project.evaluation_plan.seeds) or set(
            dataset.episode_ids
        ) & set(project.evaluation_plan.held_out_episode_ids):
            raise Problem(422, "held_out_overlap", "Held-out conditions must not enter training.")
        if not dataset.captures:
            raise Problem(
                409, "dataset_case_unverified", "Reseal verified case/split-bound captures."
            )
        for capture in dataset.captures:
            capture.authorized_case(project)
        if not any(capture.split == "train" for capture in dataset.captures):
            raise Problem(
                422, "training_split_empty", "Validation-only data cannot become optimizer input."
            )

    def _submit(self, actor: Principal, specification: JobSpecification) -> Stored:
        jobs = self._dependency(self.jobs, "Azure ML learning backend")
        jobs.preflight(actor, specification)
        stored, claimed = self._claim(actor, specification.run)
        if not claimed:
            return stored
        try:
            receipt = jobs.submit(actor, specification)
            self._check_job_receipt(actor, stored.value, receipt)
        except Problem as exc:
            self._save(
                actor,
                stored,
                status="submission_unknown",
                error_code=exc.code,
                message=(
                    "Submission may have reached Azure. Reconcile the named job; do not resubmit."
                ),
            )
            raise Problem(
                503,
                "learning_submission_unconfirmed",
                "Azure job submission is unconfirmed.",
                [{"job_id": str(stored.value.id)}],
            ) from exc
        submitted = self._save(actor, stored, status="submitted", azure_job_id=receipt.azure_job_id)
        return self._apply_job(actor, submitted, receipt)

    @staticmethod
    def _check_job_receipt(actor: Principal, run, receipt: BackendJob) -> None:
        if (
            receipt.owner_key != actor.owner_key
            or receipt.job_name != run.backend_job_name
            or receipt.specification_sha256 != run.specification_sha256
            or not receipt.azure_job_id.startswith("/subscriptions/")
            or "/providers/Microsoft.MachineLearningServices/workspaces/"
            not in receipt.azure_job_id
            or not receipt.azure_job_id.endswith(f"/jobs/{run.backend_job_name}")
            or (run.azure_job_id is not None and receipt.azure_job_id != run.azure_job_id)
            or (receipt.job_deadline_utc is not None and receipt.job_deadline_utc > run.deadline)
            or (
                run.job_deadline_utc is not None
                and receipt.job_deadline_utc != run.job_deadline_utc
            )
        ):
            raise Problem(
                503, "learning_receipt_mismatch", "Job receipt has different immutable scope."
            )

    def get_job(self, actor: Principal, job_id: UUID) -> Stored:
        self._enabled()
        stored = self.store.get_learning(actor.owner_key, "training", job_id)
        if stored is None:
            stored = self.get(actor, "evaluation", job_id)
        if stored.value.status in JOB_TERMINAL:
            return stored
        if isinstance(stored.value, EvaluationRun) and stored.value.provider == "managed_batch":
            if stored.value.import_operation_id is not None:
                self.get_artifact_operation(actor, stored.value.import_operation_id)
                return self.get(actor, "evaluation", job_id)
            return stored
        jobs = self._dependency(self.jobs, "Azure ML learning backend")
        receipt = jobs.status(actor, stored.value)
        if receipt is None:
            if utcnow() >= (stored.value.job_deadline_utc or stored.value.deadline):
                if stored.value.error_code == "job_receipt_missing":
                    return stored
                return self._save(
                    actor,
                    stored,
                    error_code="job_receipt_missing",
                    message=(
                        "Deadline elapsed without a verified Azure receipt. The named job remains "
                        "unconfirmed; reconciliation must continue. No job was resubmitted."
                    ),
                )
            return stored
        run = stored.value
        self._check_job_receipt(actor, run, receipt)
        if (
            receipt.status not in JOB_TERMINAL
            and run.status != "cancelling"
            and run.cancellation is None
            and utcnow() >= (receipt.job_deadline_utc or run.job_deadline_utc or run.deadline)
        ):
            request_id = uuid5(
                NAMESPACE_URL,
                f"learning-cancel:{actor.owner_key}:{run.id}:{run.specification_sha256}:deadline",
            )
            try:
                return self._request_job_cancellation(
                    actor, stored, request_id, "deadline", observed=receipt
                )
            except Problem as exc:
                if exc.code != "revision_conflict":
                    raise
                return self.get(actor, run.kind, run.id)
        return self._apply_job(actor, stored, receipt)

    def _apply_job(self, actor: Principal, stored: Stored, receipt: BackendJob) -> Stored:
        run = stored.value
        self._check_job_receipt(actor, run, receipt)
        if run.status in JOB_TERMINAL:
            return stored
        if run.status in ("submitting", "submission_unknown"):
            stored = self._save(
                actor, stored, status="submitted", azure_job_id=receipt.azure_job_id
            )
            run = stored.value
        target = transition(
            "job",
            run.status,
            "cancelling"
            if run.status == "cancelling" and receipt.status not in JOB_TERMINAL
            else receipt.status,
        )
        changes = {
            "status": target,
            "azure_job_id": receipt.azure_job_id,
            "job_deadline_utc": receipt.job_deadline_utc,
            "backend_status": receipt.status,
            "azure_status": receipt.azure_status,
            "metrics": receipt.metrics,
            "error_code": receipt.error_code,
            "message": receipt.message,
        }
        if run.cancellation is not None and target not in JOB_TERMINAL:
            changes["error_code"] = run.cancellation.error_code or receipt.error_code
            changes["message"] = receipt.message or (
                f"Cancellation {run.cancellation.state}; Azure currently reports "
                f"{receipt.status}. Termination is not confirmed."
            )
        if receipt.metrics.optimizer_steps is not None and run.metrics.optimizer_steps is not None:
            if receipt.metrics.optimizer_steps < run.metrics.optimizer_steps:
                raise Problem(503, "regressing_job_metrics", "Worker step count regressed.")
        # A transient worker-side progress read failure reports no history this poll; that must
        # never erase or be treated as regressing previously verified checkpoint samples.
        if receipt.metrics.history:
            if len(receipt.metrics.history) < len(run.metrics.history) or receipt.metrics.history[
                : len(run.metrics.history)
            ] != run.metrics.history:
                raise Problem(
                    503,
                    "regressing_job_metrics",
                    "A verified training progress sample cannot be dropped or relabeled.",
                )
        elif run.metrics.history:
            changes["metrics"] = receipt.metrics.model_copy(
                update={"history": run.metrics.history}
            )
        if target == "succeeded" and isinstance(run, TrainingRun):
            project = self.get(actor, "project", run.project_id).value
            artifacts = self._dependency(self.artifacts, "Verified learning artifacts")
            candidate = receipt.candidate
            self._check_candidate(actor, project, run, candidate)
            artifacts.verify_candidate(actor, project, run, candidate)
            self._claim(actor, candidate)
            changes["candidate_id"] = candidate.id
        if isinstance(run, EvaluationRun) and target in JOB_TERMINAL:
            if target == "succeeded" and receipt.report is None:
                raise Problem(503, "missing_evaluation_evidence", "A paired report is required.")
            if receipt.report is not None:
                project = self.get(actor, "project", run.project_id).value
                artifacts = self._dependency(self.artifacts, "Verified learning artifacts")
                candidate = self.get(actor, "candidate", run.candidate_id).value
                if isinstance(receipt.report, SimulationReport):
                    baseline = (
                        None
                        if run.comparison_kind == "reference_bootstrap"
                        else self._baseline(
                            actor,
                            run.baseline_release_id,
                            execution_timing="paused_simulation",
                            control_profile_id=project.control_profile_id,
                        )
                    )
                    validate_report_binding(
                        JobSpecification(
                            owner_key=actor.owner_key,
                            project=project,
                            run=run,
                            candidate=candidate,
                            baseline=baseline,
                        ),
                        receipt.report,
                    )
                elif run.comparison_kind == "reference_bootstrap":
                    validate_bootstrap_report(project, candidate, receipt.report)
                else:
                    baseline = self._baseline(actor, run.baseline_release_id)
                    validate_paired_report(project, baseline, candidate, receipt.report)
                artifacts.verify_report(actor, project, run, receipt.report)
                changes["report"] = receipt.report
        if all(getattr(run, key) == value for key, value in changes.items()):
            return stored
        try:
            return self._save(actor, stored, **changes)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self.get(actor, run.kind, run.id)

    def report_document(self, actor: Principal, job_id: UUID):
        stored = self.get(actor, "evaluation", job_id)
        report = stored.value.report
        if not isinstance(report, SimulationReport):
            raise Problem(409, "report_not_verified", "No verified simulation report is recorded.")
        method = getattr(self.jobs, "report_document", None)
        if not callable(method):
            raise unavailable("Owner-scoped verified report download")
        return report, method(actor, stored.value, report.report_sha256)

    def start_reference_collection(self, actor, project_id, body: StartReferenceCollection, etag):
        digest = operation_hash(project_id, "reference_collection", body)
        existing = self._existing(actor, "reference_collection", body.request_id, digest)
        if existing:
            return self.get_reference_collection(actor, body.request_id)
        if not self.reference_collections_enabled:
            raise Problem(503, "reference_phase_disabled", "Reference generation is not admitted.")
        context = self.get(actor, "project", project_id)
        require_etag(context, etag)
        project = context.value
        if project.execution_timing != "paused_simulation":
            raise Problem(
                409,
                "reference_mode_mismatch",
                "This route is only for explicit paused reference data.",
            )
        case = project.selected_case(body.case_id)
        environment = self._saved_scene(actor, case.environment_id, case.revision)
        self._case_environment(
            actor,
            case,
            goal_station_id=project.goal_station_id,
            robot_profile=environment.document["scene"]["robot_profile"],
            split=case.split,
        )
        if validate_paused_learning_environment(environment.document):
            raise Problem(
                409,
                "paused_execution_required",
                "The saved case does not authorize paused execution.",
            )
        catalog = self._dependency(self.catalog, "Operator-installed reference catalog")
        authorization = catalog.reference_authorization(actor, project.id, case.case_id)
        permit = authorization.authorize(actor, project, case)
        observation = self.factory.bridge.observe(
            actor.owner_key, case.environment_id, case.revision
        )
        self.factory._check_scene(observation, case.environment_id, case.revision)
        check_fresh(
            observation, min(2000, environment.document["execution"]["max_observation_age_ms"])
        )
        limits = environment.document["learning_execution"]
        now = utcnow()
        authorization.authorize(actor, project, case)
        command = SimulationEpisodeCommand(
            schema="physicalai.simulation-episode-command/v1",
            execution_timing="paused_simulation",
            real_time_admission=False,
            profile_id=project.control_profile_id,
            command_id=body.request_id,
            environment_id=case.environment_id,
            revision=case.revision,
            epoch=observation.epoch,
            state_revision=observation.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=project.goal_station_id,
            task=permit.task,
            wall_expires_at=min(
                permit.wall_expires_at,
                authorization.operator_grant.expires_at,
                now
                + timedelta(
                    seconds=min(
                        permit.max_episode_wall_seconds,
                        limits["max_wall_seconds"],
                        project.evaluation_plan.max_wall_seconds,
                    )
                ),
            ),
            max_simulation_steps=min(
                permit.max_simulation_steps,
                limits["max_simulation_seconds"] * 60,
                project.evaluation_plan.max_simulation_seconds * 60,
            ),
            controller="reference_controller",
            authorization_kind="reference_collection",
            authorization_id=permit.authorization_id,
        )
        record = ReferenceCollection(
            **metadata(actor, body.request_id, digest),
            **project.timing_fields(),
            project_id=project.id,
            teaching_case=case,
            command_id=command.command_id,
            epoch=command.epoch,
            command=command,
            status="starting",
            target_position_m=next(
                item["position_m"]
                for item in environment.document["stations"]
                if item["id"] == project.goal_station_id
            ),
            goal_tolerance_m=project.evaluation_plan.maximum_axis_error_m,
            runtime_catalog_record_sha256=authorization.runtime_catalog_record_sha256,
            source_revision=authorization.operator_grant.source_revision,
            simulator_image_digest=authorization.operator_grant.simulator_image_digest,
        )
        stored, first = self._claim(actor, record)
        if not first:
            return stored
        try:
            result = self.factory.bridge.dispatch_simulation_episode(actor.owner_key, command)
        except Problem as exc:
            return self._save(
                actor,
                stored,
                status="unconfirmed",
                error_code=exc.code,
                message="Reference dispatch is unconfirmed; no command is resubmitted.",
            )
        return self._apply_reference(actor, stored, result)

    def _apply_reference(self, actor, stored, result: SimulationEpisodeExecution):
        record, metrics = stored.value, result.simulation_runtime
        if record.status in REFERENCE_TERMINAL:
            return stored
        if (
            result.command_id != record.command_id
            or metrics.controller != "reference_controller"
            or metrics.profile_id != record.control_profile_id
            or metrics.control_profile_sha256 != record.control_profile_sha256
            or metrics.applied_model_sha256 is not None
            or metrics.policy_predict_calls
            or metrics.simulation_steps > record.command.max_simulation_steps
        ):
            raise Problem(
                503, "reference_receipt_mismatch", "Runtime reference provenance differs."
            )
        if result.status == "succeeded" and (
            result.error is not None
            or metrics.simulation_steps % 6 != 0
            or metrics.simulation_steps != metrics.applied_action_count
            or abs(metrics.simulation_elapsed_seconds - metrics.simulation_steps / 60) > 1e-6
            or metrics.wall_elapsed_ms
            > (record.command.wall_expires_at - record.created_at).total_seconds() * 1000
            or result.final_position is None
            or result.completed_at is None
            or result.completed_at < record.created_at
            or result.completed_at > utcnow()
            or result.completed_at > record.command.wall_expires_at
            or metrics.reference_route_calls <= 0
            or metrics.applied_action_count <= 0
            or any(
                abs(actual - target) > record.goal_tolerance_m
                for actual, target in zip(
                    result.final_position, record.target_position_m, strict=True
                )
            )
        ):
            raise Problem(
                503,
                "reference_result_unverified",
                "Actual completed reference evidence is missing.",
            )
        status = {"queued": "running", "running": "running"}.get(result.status, result.status)
        if record.status == "cancelling" and result.status in ("queued", "running", "cancelling"):
            status = "cancelling"
        changes = {
            "status": status,
            "execution": result,
            "error_code": result.error.code if result.error else None,
            "message": result.error.message
            if result.error
            else "Reference controller result received; not human teaching.",
        }
        if all(getattr(record, key) == value for key, value in changes.items()):
            return stored
        return self._save_reference(actor, stored, **changes)

    def _save_reference(self, actor, stored, **changes):
        try:
            return self._save(actor, stored, **changes)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self.get(actor, "reference_collection", stored.value.id)

    def get_reference_collection(self, actor, collection_id):
        stored = self.get(actor, "reference_collection", collection_id)
        record = stored.value
        if record.status not in ("succeeded", "failed", "cancelled", "timed_out"):
            try:
                result = self.factory.bridge.simulation_episode(actor.owner_key, record.command_id)
                stored = self._apply_reference(actor, stored, result)
            except Problem as exc:
                if exc.status != 404:
                    raise
                if utcnow() >= record.command.wall_expires_at:
                    return self._save(
                        actor,
                        stored,
                        status="timed_out",
                        error_code="reference_command_unconfirmed",
                        message="No result was confirmed before the original grant expired.",
                    )
                return stored
        record = stored.value
        if record.status == "succeeded" and record.capture_status not in ("ready", "invalid"):
            capture = self.factory.bridge.capture(actor.owner_key, record.command_id)
            if capture.command_id != record.command_id or capture.epoch != record.epoch:
                raise Problem(
                    503, "capture_scope_mismatch", "Reference capture belongs to another command."
                )
            if capture.status == "invalid":
                return self._save_reference(
                    actor,
                    stored,
                    capture_status="invalid",
                    error_code="reference_capture_invalid",
                    message=capture.message or "The reference capture failed validation.",
                )
            if capture.status == "ready":
                if capture.receipt is None or capture.receipt.status != "uploaded":
                    raise Problem(
                        503, "capture_not_verified", "A complete uploaded capture is required."
                    )
                if capture.receipt.episode_id != record.command_id:
                    raise Problem(
                        503, "capture_scope_mismatch", "Capture is not the original command."
                    )
                project = self.get(actor, "project", record.project_id).value
                operation_id = uuid5(
                    NAMESPACE_URL,
                    f"reference-artifact:{actor.owner_key}:{record.id}:{capture.receipt.manifest_sha256}",
                )
                if (
                    record.artifact_operation_id is not None
                    and record.artifact_operation_id != operation_id
                ):
                    raise Problem(
                        409, "reference_capture_changed", "The original capture manifest changed."
                    )
                operation = self._begin_artifact(
                    actor,
                    project,
                    operation_id,
                    record.id,
                    fingerprint(
                        {"reference": str(record.id), "manifest": capture.receipt.manifest_sha256}
                    ),
                    session=record,
                    receipt=capture.receipt,
                ).value
                if operation.status == "ready" and operation.result and operation.result.capture:
                    receipt = operation.result.capture
                    if (
                        receipt.authorized_case(project) != record.teaching_case
                        or receipt.source != "reference_controller"
                        or receipt.episode_id != record.command_id
                        or receipt.manifest_sha256 != capture.receipt.manifest_sha256
                        or receipt.frame_count != capture.receipt.frame_count
                    ):
                        raise Problem(
                            503, "capture_scope_mismatch", "Verified reference capture differs."
                        )
                    return self._save_reference(
                        actor,
                        stored,
                        capture_status="ready",
                        capture=operation.result.capture,
                        artifact_operation_id=operation.id,
                    )
                return self._save_reference(
                    actor,
                    stored,
                    capture_status="invalid"
                    if operation.status in ("failed", "timed_out")
                    else "verifying",
                    artifact_operation_id=operation.id,
                    error_code=operation.error_code,
                    message=operation.message,
                )
        return stored

    def cancel_reference_collection(self, actor, collection_id):
        stored = self.get(actor, "reference_collection", collection_id)
        if stored.value.status in ("succeeded", "failed", "cancelled", "timed_out", "cancelling"):
            return stored
        reserved = self._save(actor, stored, status="cancelling")
        try:
            result = self.factory.bridge.cancel_simulation_episode(
                actor.owner_key, stored.value.command_id
            )
            return self._apply_reference(actor, reserved, result)
        except Problem as exc:
            return self._save_reference(
                actor,
                reserved,
                error_code=exc.code,
                message="Reference cancellation is unconfirmed; no cancel POST is replayed.",
            )

    def _check_candidate(self, actor, project, run, candidate) -> None:
        if candidate is not None and candidate.training_origin != "api_training_run":
            raise Problem(
                503, "candidate_evidence_mismatch", "External imports are not API training results."
            )
        parent = (
            self._training_parent(actor, run.pretrained_artifact_id)
            if project.project_kind == "bootstrap"
            else self._baseline(
                actor,
                run.parent_release_id,
                execution_timing=project.execution_timing,
                control_profile_id=project.control_profile_id,
            )
        )
        if candidate is None or (
            candidate.owner_key != actor.owner_key
            or candidate.project_id != project.id
            or candidate.dataset_id != run.dataset_id
            or candidate.training_run_id != run.id
            or candidate.parent_release_id != run.parent_release_id
            or candidate.pretrained_artifact_id != run.pretrained_artifact_id
            or candidate.parent_model_sha256 != parent.model_sha256
            or candidate.policy_type != project.policy_type
            or candidate.azure_job_id != run.azure_job_id
            or candidate.optimizer_steps > run.optimizer_steps
            or candidate.control_profile_id != project.control_profile_id
            or not candidate.matches_timing(project)
        ):
            raise Problem(
                503, "candidate_evidence_mismatch", "Training output provenance is incomplete."
            )

    def _mutation(self, actor, stored, request_id, operation, digest, etag, expires=None):
        existing = self._existing(actor, "mutation", request_id, digest)
        if existing:
            if existing.value.status != "recorded":
                raise Problem(
                    409, "learning_operation_unconfirmed", "Reconcile the original operation."
                )
            return existing, False
        require_etag(stored, etag)
        claim = LearningMutation(
            **metadata(actor, request_id, digest),
            resource_kind=stored.value.kind,
            resource_id=stored.value.id,
            operation=operation,
            server_expires_at=expires,
        )
        result, first = self._claim(actor, claim)
        if not first and result.value.status != "recorded":
            raise Problem(
                409, "learning_operation_unconfirmed", "The operation is already reserved."
            )
        return result, first

    def cancel_job(self, actor: Principal, job_id: UUID, request_id: UUID, etag: str | None):
        stored = self.get_job(actor, job_id)
        if isinstance(stored.value, EvaluationRun) and stored.value.provider == "managed_batch":
            raise Problem(
                409,
                "managed_operator_required",
                "This is not an Azure ML job; managed physical cancellation is operator-owned.",
            )
        if stored.value.status in JOB_TERMINAL or stored.value.cancellation is not None:
            return stored
        digest = fingerprint(
            {"job_id": str(job_id), "request_id": str(request_id), "action": "cancel"}
        )
        mutation, first = self._mutation(actor, stored, request_id, "cancel", digest, etag)
        if not first:
            return stored
        try:
            result = self._request_job_cancellation(actor, stored, request_id, "user")
        except Problem as exc:
            self._save(actor, mutation, status="uncertain", error_code=exc.code)
            raise
        self._save(actor, mutation, status="recorded")
        return result

    def _request_job_cancellation(
        self, actor, stored, request_id, reason, *, observed: BackendJob | None = None
    ):
        if stored.value.status in JOB_TERMINAL or stored.value.cancellation is not None:
            return stored
        reservation = JobCancellation(request_id=request_id, reason=reason, requested_at=utcnow())
        changes = {}
        if observed is not None:
            self._check_job_receipt(actor, stored.value, observed)
            changes = {
                "azure_job_id": observed.azure_job_id,
                "job_deadline_utc": observed.job_deadline_utc,
                "backend_status": observed.status,
                "azure_status": observed.azure_status,
            }
        reserved = self._save(
            actor,
            stored,
            status=transition("job", stored.value.status, "cancelling"),
            cancellation=reservation,
            **changes,
        )
        try:
            receipt = self._dependency(self.jobs, "Azure ML learning backend").cancel(
                actor, reserved.value
            )
            result = self._apply_job(actor, reserved, receipt)
        except Problem as exc:
            self._record_job_cancellation(
                actor, reserved, "forbidden" if exc.status == 403 else "uncertain", exc.code
            )
            raise
        return self._record_job_cancellation(
            actor, result, receipt.cancellation_state or "acknowledged", receipt.error_code
        )

    def _record_job_cancellation(self, actor, stored, state, error_code=None):
        request_id = stored.value.cancellation.request_id
        for attempt in range(2):
            run = stored.value
            if run.cancellation is None or run.cancellation.request_id != request_id:
                raise Problem(409, "cancellation_scope_changed", "Cancellation binding changed.")
            outcome = run.cancellation.model_copy(update={"state": state, "error_code": error_code})
            changes = {"cancellation": outcome}
            if run.status not in JOB_TERMINAL:
                changes.update(
                    error_code=error_code,
                    message=(
                        f"Cancellation {state}; actual Azure termination is not confirmed. "
                        "The cancellation POST will not be replayed."
                    ),
                )
            try:
                return self._save(actor, stored, **changes)
            except Problem as exc:
                if exc.code != "revision_conflict" or attempt:
                    raise
                stored = self.get(actor, run.kind, run.id)

    def evaluate(self, actor, project_id, body: StartEvaluation, etag):
        digest = operation_hash(project_id, "evaluate", body)
        existing = self._existing(actor, "evaluation", body.request_id, digest)
        if existing:
            return existing
        context = self.get(actor, "project", project_id)
        require_etag(context, etag)
        project = context.value
        self._timing_admission(project, "evaluation")
        candidate = self.get(actor, "candidate", body.candidate_id).value
        if (
            candidate.project_id != project_id
            or body.baseline_release_id != project.baseline_release_id
            or not candidate.matches_timing(project)
        ):
            raise Problem(
                409, "evaluation_scope_mismatch", "Select this project's candidate and P0."
            )
        if body.evaluation_plan_sha256 != project.evaluation_plan.sha256:
            raise Problem(409, "evaluation_plan_mismatch", "Use the original held-out plan.")
        dataset = self.get(actor, "dataset", candidate.dataset_id).value
        self._check_split(project, dataset)
        baseline = None
        expected_kind = (
            "reference_bootstrap" if project.project_kind == "bootstrap" else "paired_policy"
        )
        if body.comparison_kind != expected_kind:
            raise Problem(
                409,
                "comparison_kind_mismatch",
                "Bootstrap reference and paired policies are different evaluations.",
            )
        if project.project_kind == "bootstrap":
            self._bootstrap_actor(actor)
        else:
            baseline = self._baseline(
                actor,
                body.baseline_release_id,
                execution_timing=project.execution_timing,
                control_profile_id=project.control_profile_id,
            )
        self._cost(project, body.maximum_cost_usd)
        run = EvaluationRun(
            **metadata(actor, body.request_id, digest),
            **project.timing_fields(),
            project_id=project_id,
            policy_type=project.policy_type,
            status="submitting",
            backend_job_name=self._job_name(actor, body.request_id),
            deadline=utcnow() + timedelta(seconds=project.budget.evaluation_seconds),
            approved_cost_usd=body.maximum_cost_usd,
            specification_sha256=digest,
            candidate_id=candidate.id,
            baseline_release_id=baseline.id if baseline else None,
            comparison_kind=expected_kind,
            evaluation_plan_sha256=project.evaluation_plan.sha256,
        )
        return self._submit(
            actor,
            JobSpecification(
                owner_key=actor.owner_key,
                project=project,
                run=run,
                baseline=baseline,
                dataset=dataset,
                candidate=candidate,
            ),
        )

    def release(self, actor: Principal, body: ReleasePolicy, etag: str | None):
        digest = fingerprint(body.model_dump(mode="json"))
        existing = self._existing(actor, "release", body.request_id, digest)
        if existing:
            return existing
        evaluation = self.get(actor, "evaluation", body.evaluation_run_id)
        require_etag(evaluation, etag)
        run = evaluation.value
        if run.status != "succeeded" or run.report is None or run.candidate_id != body.candidate_id:
            raise Problem(409, "release_gate_failed", "A completed paired evaluation is required.")
        candidate = self.get(actor, "candidate", body.candidate_id).value
        project = self.get(actor, "project", candidate.project_id).value
        self._timing_admission(project, "release")
        if isinstance(run.report, SimulationReport):
            before = (
                self.get(actor, "candidate", run.before_candidate_id).value
                if run.before_candidate_id is not None
                else None
            )
            baseline = (
                None
                if project.project_kind == "bootstrap" or before is not None
                else self._baseline(
                    actor,
                    run.baseline_release_id,
                    execution_timing="paused_simulation",
                    control_profile_id=project.control_profile_id,
                )
            )
            if project.project_kind == "bootstrap":
                self._bootstrap_actor(actor)
            validate_report_binding(
                JobSpecification(
                    owner_key=actor.owner_key,
                    project=project,
                    run=run,
                    candidate=candidate,
                    baseline=baseline,
                    baseline_candidate=before,
                ),
                run.report,
            )
            eligible = run.report.quality_gate_passed and (
                project.project_kind == "bootstrap" or run.report.conclusion == "improved"
            )
        elif project.project_kind == "bootstrap":
            self._bootstrap_actor(actor)
            validate_bootstrap_report(project, candidate, run.report)
            eligible = run.report.quality_gate_passed
        else:
            baseline = self._baseline(actor, run.baseline_release_id)
            validate_paired_report(project, baseline, candidate, run.report)
            eligible = run.report.quality_gate_passed and run.report.conclusion == "improved"
        if not eligible:
            raise Problem(
                409, "release_gate_failed", "Safety, physical quality and improvement are required."
            )
        self._dependency(self.artifacts, "Verified learning artifacts").verify_report(
            actor, project, run, run.report
        )
        record = PolicyRelease(
            **metadata(actor, body.request_id, digest),
            project_id=project.id,
            candidate_id=candidate.id,
            evaluation_run_id=run.id,
            policy_type=candidate.policy_type,
            model_sha256=candidate.model_sha256,
            processor_sha256=candidate.processor_sha256,
            manifest_sha256=candidate.manifest_sha256,
            artifact_id=candidate.artifact_id,
            environment_id=project.environment_id,
            revision=project.revision,
            task_id=project.task_id,
            goal_station_id=project.goal_station_id,
            instruction=project.instruction,
            **({"control_profile_id": project.control_profile_id} | project.timing_fields()),
            evaluation_plan_sha256=project.evaluation_plan.sha256,
            reviewed_by=actor.object_id,
            comparison_kind=run.comparison_kind,
            environment_cases=tuple(
                PolicyScene(environment_id=case.environment_id, revision=case.revision)
                for case in project.evaluation_plan.cases
            ),
        )
        return self._claim(actor, record)[0]

    def resolve_for_run(self, actor, release_id, environment_id, revision):
        self._enabled()
        release = self._baseline(actor, release_id)
        if (release.environment_id, release.revision) != (environment_id, revision) and not any(
            (case.environment_id, case.revision) == (environment_id, revision)
            for case in release.environment_cases
        ):
            raise Problem(
                409, "policy_scene_mismatch", "The reviewed release pins a different scene."
            )
        return ReleasedPolicyBinding(
            policy_release_id=release.id,
            policy_type=release.policy_type,
            model_sha256=release.model_sha256,
            processor_sha256=release.processor_sha256,
            manifest_sha256=release.manifest_sha256,
            control_profile_id=release.control_profile_id,
            task_id=release.task_id,
            goal_station_id=release.goal_station_id,
            instruction=release.instruction,
        )

    def start_teaching(self, actor, project_id, body: StartTeaching, etag):
        digest = operation_hash(project_id, "teach", body)
        existing = self._existing(actor, "teaching", body.request_id, digest)
        if existing:
            return existing
        context = self.get(actor, "project", project_id)
        require_etag(context, etag)
        project = context.value
        if project.execution_timing == "paused_simulation":
            raise Problem(
                503,
                "paused_learning_unavailable",
                "Manual paused teaching is unavailable; use reviewed reference collection.",
            )
        self._timing_admission(project)
        case = project.selected_case(body.case_id)
        runtime = self._dependency(self.runtime, "Verified teaching runtime")
        anchor = self._saved_scene(actor, project.environment_id, project.revision)
        environment = self._case_environment(
            actor,
            case,
            goal_station_id=project.goal_station_id,
            robot_profile=anchor.document["scene"]["robot_profile"],
            split=case.split,
        )
        observation = self.factory.bridge.observe(
            actor.owner_key, case.environment_id, case.revision
        )
        self.factory._check_scene(observation, case.environment_id, case.revision)
        check_fresh(observation, environment.document["execution"]["max_observation_age_ms"])
        duration = project.budget.teaching_seconds
        expires = utcnow() + timedelta(seconds=duration)
        session = TeachingSession(
            **metadata(actor, body.request_id, digest),
            **project.timing_fields(),
            project_id=project_id,
            teaching_case=case,
            source=body.source,
            status="starting",
            lease_id=uuid4(),
            epoch=observation.epoch,
            command_id=body.request_id,
            expires_at=expires,
        )
        command = TeachingStartSpec(
            session_id=session.id,
            lease_id=session.lease_id,
            command_id=session.command_id,
            environment_id=case.environment_id,
            revision=case.revision,
            epoch=observation.epoch,
            state_revision=observation.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=project.goal_station_id,
            session_expires_at=expires,
            control_profile_id=project.control_profile_id,
            task=TeachingTask(
                task_id=project.task_id,
                instruction=project.instruction,
                goal_id=project.goal_station_id,
            ),
            demonstrator_kind=body.source,
            split=case.split,
        )
        stored, first = self._claim(actor, session)
        if not first:
            return stored
        try:
            state = runtime.start_teaching(actor.owner_key, command)
            return self._apply_teaching(actor, stored, state)
        except Problem as exc:
            self._save(
                actor,
                stored,
                error_code=exc.code,
                message="Teaching start unconfirmed; no POST retry.",
            )
            raise

    def get_teaching(self, actor, session_id):
        stored = self.get(actor, "teaching", session_id)
        if stored.value.status in TEACHING_TERMINAL:
            return stored
        runtime = self._dependency(self.runtime, "Teaching runtime")
        if stored.value.physical_status in TERMINAL:
            capture = runtime.capture(actor.owner_key, stored.value.command_id)
            status, receipt, operation = self._capture_status(
                actor, stored.value, capture, stored.value.status, stored.value.physical_status
            )
            changes = {
                "status": status,
                "capture": receipt,
                "message": operation.message if operation else capture.message,
                "artifact_operation_id": operation.id
                if operation
                else stored.value.artifact_operation_id,
                "verification_status": operation.status
                if operation
                else stored.value.verification_status,
                "error_code": operation.error_code if operation else stored.value.error_code,
            }
            if all(getattr(stored.value, key) == value for key, value in changes.items()):
                return stored
            return self._save(actor, stored, **changes)
        state = runtime.teaching(actor.owner_key, session_id)
        return self._apply_teaching(actor, stored, state)

    def _capture_status(
        self,
        actor,
        session,
        state: RuntimeCapture,
        status,
        physical_status,
    ):
        if state.command_id != session.command_id or state.epoch != session.epoch:
            raise Problem(503, "capture_scope_mismatch", "Capture belongs to a different command.")
        receipt = session.capture
        operation = None
        if state.status == "ready":
            if physical_status != "succeeded":
                return "invalid", receipt, operation
            if not state.receipt or state.receipt.status != "uploaded":
                raise Problem(503, "capture_not_verified", "Capture is not an uploaded manifest.")
            project = self.get(actor, "project", session.project_id).value
            artifacts = self._dependency(self.artifacts, "Capture verifier")
            if callable(getattr(artifacts, "begin_artifact", None)):
                operation_id = uuid5(
                    NAMESPACE_URL,
                    f"artifact-capture:{actor.owner_key}:{session.id}:{state.receipt.manifest_sha256}",
                )
                operation = self._begin_artifact(
                    actor,
                    project,
                    operation_id,
                    session.id,
                    fingerprint(
                        {"session": str(session.id), "manifest": state.receipt.manifest_sha256}
                    ),
                    session=session,
                    receipt=state.receipt,
                ).value
                if operation.status != "ready":
                    return (
                        "invalid" if operation.status in ("failed", "timed_out") else "uploading",
                        receipt,
                        operation,
                    )
                if operation.result is None or operation.result.capture is None:
                    raise Problem(
                        503, "artifact_receipt_mismatch", "Verified capture result is missing."
                    )
                receipt = operation.result.capture
            else:
                receipt = artifacts.verify_capture(
                    actor, project, session, state.receipt.model_dump(mode="json")
                )
            actual_case = receipt.authorized_case(project)
            if (
                receipt.source != session.source
                or session.teaching_case is None
                or actual_case != session.teaching_case
                or receipt.episode_id != state.receipt.episode_id
                or receipt.manifest_sha256 != state.receipt.manifest_sha256
                or receipt.frame_count != state.receipt.frame_count
            ):
                raise Problem(
                    503, "capture_scope_mismatch", "Capture provenance differs from teaching."
                )
            status = "ready"
        elif state.status in ("finalizing", "uploading", "invalid"):
            status = state.status
        return status, receipt, operation

    def _apply_teaching(self, actor, stored, state: TeachingRuntimeState):
        session = stored.value
        if (
            state.session_id,
            state.lease_id,
            state.epoch,
            state.command_id,
            state.execution.command_id,
        ) != (
            session.id,
            session.lease_id,
            session.epoch,
            session.command_id,
            session.command_id,
        ) or state.control_profile_id != PROFILE_ID:
            raise Problem(
                503, "teaching_receipt_mismatch", "Teaching receipt belongs to another lease."
            )
        if session.status in TEACHING_TERMINAL:
            return stored
        status = {
            "queued": "starting",
            "running": "recording",
            "finishing": "finishing",
            "succeeded": "finalizing",
            "failed": "invalid",
            "cancelled": "cancelled",
            "timed_out": "invalid",
            "cancelling": "cancelling",
        }[state.status]
        capture = session.capture
        operation = None
        if state.capture:
            status, capture, operation = self._capture_status(
                actor, session, state.capture, status, state.execution.status
            )
        if session.status == "cancelling":
            status = "cancelled" if state.status in ("cancelled", "timed_out") else "cancelling"
        changes = {
            "status": status,
            "physical_status": state.execution.status,
            "capture": capture,
            "message": operation.message
            if operation
            else state.capture.message
            if state.capture
            else None,
            "artifact_operation_id": operation.id if operation else session.artifact_operation_id,
            "verification_status": operation.status if operation else session.verification_status,
            "error_code": operation.error_code if operation else session.error_code,
        }
        if all(getattr(session, key) == value for key, value in changes.items()):
            return stored
        return self._save(actor, stored, **changes)

    def arm(self, actor, session_id, body: ArmTeaching, etag):
        digest = operation_hash(session_id, "arm", body)
        grant_id = uuid5(
            NAMESPACE_URL, f"{actor.owner_key}:teaching:{session_id}:{body.request_id}"
        )
        existing = self._existing(actor, "control_grant", grant_id, digest)
        if existing:
            return existing
        stored = self.get(actor, "teaching", session_id)
        require_etag(stored, etag)
        session = stored.value
        self._teaching_authority(session, body.lease_id, body.epoch)
        if session.status != "recording" or body.sequence != session.last_sequence + 1:
            raise Problem(409, "teaching_sequence", "Arm only the next active-session input.")
        grant = ControlGrant(
            **metadata(actor, grant_id, digest),
            session_id=session_id,
            lease_id=session.lease_id,
            epoch=session.epoch,
            sequence=body.sequence,
            delta_xyz_m=body.delta_xyz_m,
            gripper=body.gripper,
            expires_at=min(session.expires_at, utcnow() + timedelta(seconds=1)),
        )
        return self._claim(actor, grant)[0]

    def jog(self, actor, session_id, body: JogIntent, etag):
        stored = self.get(actor, "teaching", session_id)
        digest = operation_hash(session_id, "jog", body)
        existing = self._existing(actor, "mutation", body.request_id, digest)
        if existing:
            if existing.value.status != "recorded":
                raise Problem(
                    409, "learning_operation_unconfirmed", "Do not renew or resend this jog."
                )
            return stored
        session = stored.value
        self._teaching_authority(session, body.lease_id, body.epoch, require_unexpired=body.deadman)
        if not body.deadman and session.status in TEACHING_TERMINAL:
            return stored
        valid_sequence = (
            body.sequence == session.last_sequence + 1
            if body.deadman
            else body.sequence > session.last_sequence
        )
        if (body.deadman and session.status != "recording") or not valid_sequence:
            raise Problem(
                409, "teaching_sequence", "The next sequence of an active session is required."
            )
        expires = utcnow() + timedelta(milliseconds=250)
        grant = None
        if body.deadman:
            grant = self.get(actor, "control_grant", body.grant_id)
            value = grant.value
            if value.expires_at <= utcnow():
                raise Problem(409, "teaching_grant_expired", "The original input grant expired.")
            if value.consumed_by is not None:
                raise Problem(409, "teaching_grant_consumed", "This input grant was consumed.")
            if (
                value.session_id != session_id
                or value.lease_id != body.lease_id
                or value.epoch != body.epoch
                or value.sequence != body.sequence
                or value.delta_xyz_m != body.delta_xyz_m
                or value.gripper != body.gripper
            ):
                raise Problem(409, "teaching_grant_mismatch", "The grant binds a different input.")
            expires = min(expires, value.expires_at)
        mutation, first = self._mutation(
            actor,
            stored,
            body.request_id,
            "jog",
            digest,
            etag if body.deadman else stored.etag,
            expires,
        )
        if not first:
            return stored
        reserved = self._save(
            actor,
            stored,
            last_sequence=body.sequence,
            last_input_fingerprint=digest,
            input_expires_at=expires,
        )
        if grant:
            self._save(actor, grant, consumed_by=body.request_id)
        wire = JogTeaching(
            **body.model_dump(),
            expires_at=expires,
            grant_expires_at=grant.value.expires_at if grant else None,
        )
        try:
            wire.check_time(utcnow())
            state = self._dependency(self.runtime, "Teaching runtime").teaching_input(
                actor.owner_key, session_id, wire.model_dump(mode="json", exclude={"request_id"})
            )
            if state.last_sequence != body.sequence:
                raise Problem(
                    503, "teaching_sequence", "The runtime did not acknowledge this input."
                )
            result = self._apply_teaching(actor, reserved, state)
        except Problem as exc:
            self._save(actor, mutation, status="uncertain", error_code=exc.code)
            raise
        self._save(actor, mutation, status="recorded")
        return result

    @staticmethod
    def _teaching_authority(session, lease_id, epoch, *, require_unexpired=True):
        if session.lease_id != lease_id or session.epoch != epoch:
            raise Problem(409, "teaching_lease_mismatch", "The teaching lease or epoch changed.")
        if require_unexpired and utcnow() >= session.expires_at:
            raise Problem(409, "teaching_expired", "The original teaching authorization expired.")

    def control_teaching(self, actor, session_id, body: TeachingControl, etag, action):
        stored = self.get(actor, "teaching", session_id)
        session = stored.value
        self._teaching_authority(
            session, body.lease_id, body.epoch, require_unexpired=action != "cancel"
        )
        digest = operation_hash(session_id, action, body)
        mutation, first = self._mutation(actor, stored, body.request_id, action, digest, etag)
        if not first or session.status in TEACHING_TERMINAL:
            return stored
        target = "cancelling" if action == "cancel" else "finishing"
        reserved = self._save(actor, stored, status=transition("teaching", session.status, target))
        runtime = self._dependency(self.runtime, "Teaching runtime")
        method = runtime.cancel_teaching if action == "cancel" else runtime.finish_teaching
        try:
            state = method(
                actor.owner_key,
                session_id,
                {
                    "lease_id": str(body.lease_id),
                    "epoch": str(body.epoch),
                },
            )
            result = self._apply_teaching(actor, reserved, state)
        except Problem as exc:
            self._save(actor, mutation, status="uncertain", error_code=exc.code)
            raise
        self._save(actor, mutation, status="recorded")
        return result

    def dataset(self, actor, project_id, body: CreateDataset, etag):
        digest = operation_hash(project_id, "dataset", body)
        existing = self._existing(actor, "dataset", body.request_id, digest)
        if existing:
            return existing
        pending = self._existing(actor, "artifact_operation", body.request_id, digest)
        if pending:
            operation = self.get_artifact_operation(actor, body.request_id)
            return (
                self.get(actor, "dataset", body.request_id)
                if operation.value.status == "ready"
                else operation
            )
        context = self.get(actor, "project", project_id)
        require_etag(context, etag)
        captures = []
        project = context.value
        self._timing_admission(project)
        for session_id in body.teaching_session_ids:
            session = self.get(actor, "teaching", session_id).value
            if (
                session.project_id != project_id
                or session.status != "ready"
                or session.capture is None
            ):
                raise Problem(
                    409,
                    "capture_not_ready",
                    "Only this project's verified uploaded captures can be sealed.",
                )
            case = session.capture.authorized_case(project)
            if (
                session.teaching_case is None
                or case != session.teaching_case
                or session.capture.source != session.source
            ):
                raise Problem(
                    409, "capture_case_mismatch", "Ready capture differs from its approved session."
                )
            captures.append(session.capture)
        for collection_id in body.reference_collection_ids:
            collection = self.get(actor, "reference_collection", collection_id).value
            if (
                collection.project_id != project_id
                or collection.status != "succeeded"
                or collection.capture_status != "ready"
                or collection.capture is None
                or collection.source != "reference_controller"
            ):
                raise Problem(
                    409,
                    "capture_not_ready",
                    "Reference data must be physically complete and verified.",
                )
            collection.capture.authorized_case(project)
            captures.append(collection.capture)
        episodes = tuple(item.episode_id for item in captures)
        seeds = tuple(item.seed for item in captures)
        if (
            len(set(episodes)) != len(episodes)
            or set(episodes) & set(project.evaluation_plan.held_out_episode_ids)
            or set(seeds) & set(project.evaluation_plan.seeds)
        ):
            raise Problem(
                422, "held_out_overlap", "Duplicate or held-out evidence cannot enter training."
            )
        artifacts = self._dependency(self.artifacts, "Dataset verifier")
        if callable(getattr(artifacts, "begin_artifact", None)):
            return self._begin_artifact(
                actor, project, body.request_id, body.request_id, digest, captures=tuple(captures)
            )
        artifact_id, digest_manifest = artifacts.seal_dataset(
            actor, project, body.request_id, tuple(captures)
        )
        return self._claim(
            actor,
            self._dataset_record(
                actor,
                project,
                body.request_id,
                digest,
                tuple(captures),
                artifact_id,
                digest_manifest,
            ),
        )[0]

    @staticmethod
    def _dataset_record(actor, project, dataset_id, digest, captures, artifact_id, digest_manifest):
        return DatasetVersion(
            **metadata(actor, dataset_id, digest),
            **project.timing_fields(),
            project_id=project.id,
            artifact_id=artifact_id,
            manifest_sha256=digest_manifest,
            episode_ids=tuple(item.episode_id for item in captures),
            seeds=tuple(item.seed for item in captures),
            human_teleop_count=sum(item.source == "human_teleop" for item in captures),
            reference_controller_count=sum(
                item.source == "reference_controller" for item in captures
            ),
            learned_policy_count=sum(item.source == "learned" for item in captures),
            evaluation_plan_sha256=project.evaluation_plan.sha256,
            captures=tuple(captures),
        )


def validate_paired_report(project, baseline, candidate, report: PairedReport) -> None:
    plan = project.evaluation_plan
    if not isinstance(report, PairedReport):
        raise Problem(503, "evaluation_kind_mismatch", "Reference bootstrap is not P0/P1 evidence.")
    if (
        report.evaluation_plan_sha256 != plan.sha256
        or report.before_model_sha256 != baseline.model_sha256
        or report.after_model_sha256 != candidate.model_sha256
    ):
        raise Problem(
            503, "evaluation_evidence_mismatch", "Paired policies or held-out plan changed."
        )
    expected = {(seed, policy) for seed in plan.seeds for policy in ("before", "after")}
    actual = {(trial.seed, trial.policy) for trial in report.trials}
    keys = {(trial.seed, trial.policy, trial.attempt) for trial in report.trials}
    if actual != expected or len(keys) != len(report.trials):
        raise Problem(
            503,
            "evaluation_incomplete",
            "Every held-out condition and all attempts must be retained.",
        )
    for seed, policy in expected:
        attempts = sorted(
            trial.attempt
            for trial in report.trials
            if trial.seed == seed and trial.policy == policy
        )
        if attempts != list(range(1, len(attempts) + 1)):
            raise Problem(
                503, "evaluation_incomplete", "Intermediate failed retries cannot be omitted."
            )
    first = {(trial.seed, trial.policy): trial for trial in report.trials if trial.attempt == 1}
    if set(first) != expected:
        raise Problem(
            503, "evaluation_incomplete", "First attempts cannot be replaced by selected retries."
        )
    successes = {"before": 0, "after": 0}
    candidate_safe = True
    for trial in report.trials:
        case = next(item for item in plan.cases if item.seed == trial.seed)
        if (trial.environment_id, trial.revision) != (case.environment_id, case.revision):
            raise Problem(
                503, "evaluation_pose_mismatch", "Trial did not use the frozen scene revision."
            )
        expected_sha = baseline.model_sha256 if trial.policy == "before" else candidate.model_sha256
        if trial.model_sha256 != expected_sha:
            raise Problem(503, "evaluation_evidence_mismatch", "Trial used a different policy.")
        measured = (
            trial.status == "succeeded"
            and trial.physical_success
            and trial.axis_error_m is not None
            and all(abs(error) <= plan.maximum_axis_error_m for error in trial.axis_error_m)
            and trial.duration_seconds <= plan.max_step_seconds
            and trial.safety_violations == 0
            and trial.inference_p95_ms is not None
            and trial.inference_p95_ms <= plan.maximum_inference_p95_ms
            and trial.applied_action_count > 0
            and trial.policy_predict_calls > 0
            and trial.reference_route_calls == 0
        )
        if trial.attempt == 1 and measured:
            successes[trial.policy] += 1
        if trial.policy == "after" and trial.safety_violations:
            candidate_safe = False
    quality = candidate_safe and successes["after"] / len(plan.seeds) >= plan.minimum_success_rate
    conclusion = "improved" if successes["after"] > successes["before"] else "not_improved"
    if report.quality_gate_passed != quality or (
        report.conclusion != "inconclusive" and report.conclusion != conclusion
    ):
        raise Problem(
            503,
            "evaluation_evidence_mismatch",
            "Reported quality/improvement disagrees with complete trial evidence.",
        )


def validate_bootstrap_report(project, candidate, report) -> None:
    if not isinstance(report, BootstrapReport) or (
        report.evaluation_plan_sha256 != project.evaluation_plan.sha256
        or report.candidate_model_sha256 != candidate.model_sha256
    ):
        raise Problem(
            503, "bootstrap_evidence_mismatch", "Complete reference-bootstrap evidence is required."
        )
    plan = project.evaluation_plan
    expected = {(case.seed, policy) for case in plan.cases for policy in ("reference", "candidate")}
    keys = {(trial.seed, trial.policy, trial.attempt) for trial in report.trials}
    if (
        len(keys) != len(report.trials)
        or {(trial.seed, trial.policy) for trial in report.trials} != expected
    ):
        raise Problem(
            503,
            "evaluation_incomplete",
            "Bootstrap must retain every reference and candidate trial.",
        )
    for seed, policy in expected:
        attempts = sorted(t.attempt for t in report.trials if (t.seed, t.policy) == (seed, policy))
        if attempts != list(range(1, len(attempts) + 1)):
            raise Problem(503, "evaluation_incomplete", "Bootstrap attempts cannot be omitted.")
    successes = 0
    safe = True
    for trial in report.trials:
        case = next(item for item in plan.cases if item.seed == trial.seed)
        if (trial.environment_id, trial.revision) != (case.environment_id, case.revision):
            raise Problem(
                503, "evaluation_pose_mismatch", "Bootstrap scene pose is not the frozen case."
            )
        if trial.policy == "reference":
            if trial.status == "succeeded" and trial.reference_route_calls < 1:
                raise Problem(
                    503, "bootstrap_evidence_mismatch", "Reference motion evidence is missing."
                )
            continue
        if trial.model_sha256 != candidate.model_sha256:
            raise Problem(
                503, "bootstrap_evidence_mismatch", "Candidate differs from actual training."
            )
        safe = safe and trial.safety_violations == 0
        if trial.attempt == 1 and (
            trial.status == "succeeded"
            and trial.physical_success
            and trial.axis_error_m is not None
            and all(abs(error) <= plan.maximum_axis_error_m for error in trial.axis_error_m)
            and trial.duration_seconds <= plan.max_step_seconds
            and trial.inference_p95_ms is not None
            and trial.inference_p95_ms <= plan.maximum_inference_p95_ms
            and trial.applied_action_count > 0
            and trial.policy_predict_calls > 0
            and trial.reference_route_calls == 0
            and not trial.safety_violations
        ):
            successes += 1
    if report.quality_gate_passed != (
        safe and successes / len(plan.cases) >= plan.minimum_success_rate
    ):
        raise Problem(
            503,
            "bootstrap_evidence_mismatch",
            "Bootstrap quality claim contradicts physical evidence.",
        )
