from datetime import timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from apps.api.learning_models import (
    CaptureReceipt,
    Frozen,
    LearningProject,
    OwnedRecord,
    TeachingSession,
    fingerprint,
)
from apps.api.models import DemonstrationResult, Principal, Revision
from apps.api.reference_models import ReferenceCollection
from apps.api.simulation_reports import ManagedEvaluationReceipt, ManagedImportReference


class ArtifactPolicy(Frozen):
    maximum_seconds: int = Field(strict=True, ge=1, le=1800)
    capture_bytes: int = Field(strict=True, ge=1, le=4 * 1024**3)
    dataset_bytes: int = Field(strict=True, ge=1, le=20 * 1024**3)
    maximum_files: int = Field(strict=True, ge=1, le=100000)


class ArtifactWork(Frozen):
    id: UUID
    actor: Principal
    operation: Literal["capture", "dataset", "managed_evaluation"]
    project: LearningProject
    target_id: UUID
    created_at: AwareDatetime
    deadline: AwareDatetime
    max_bytes: int = Field(strict=True, ge=1, le=20 * 1024**3)
    max_files: int = Field(strict=True, ge=1, le=100000)
    session: TeachingSession | ReferenceCollection | None = None
    receipt: DemonstrationResult | None = None
    captures: tuple[CaptureReceipt, ...] = Field(default=(), max_length=1000)
    managed_import: ManagedImportReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def exact_inputs(self):
        if (
            self.project.owner_key != self.actor.owner_key
            or self.project.actor_id != self.actor.object_id
            or self.project.tenant_id != self.actor.tenant_id
            or not timedelta() < self.deadline - self.created_at <= timedelta(seconds=1800)
        ):
            raise ValueError("Artifact operation owner and original wall authority must match.")
        if self.operation == "managed_evaluation":
            if (
                self.managed_import is None
                or self.project.execution_timing != "paused_simulation"
                or self.project.policy_type != "smolvla"
                or self.managed_import.operation_id != self.id
                or self.managed_import.owner_key != self.actor.owner_key
                or self.managed_import.project_id != self.project.id
                or self.managed_import.evaluation_run_id != self.target_id
                or self.session is not None
                or self.receipt is not None
                or self.captures
            ):
                raise ValueError("Managed verification requires its original scoped import pins.")
        elif self.managed_import is not None:
            raise ValueError("Capture/dataset work cannot acquire managed evaluation authority.")
        elif self.operation == "capture":
            if (
                self.session is None
                or self.receipt is None
                or self.captures
                or self.session.id != self.target_id
                or self.session.project_id != self.project.id
                or self.session.owner_key != self.actor.owner_key
                or self.max_bytes > 4 * 1024**3
            ):
                raise ValueError("Capture validation requires its original owned session/receipt.")
        elif self.session is not None or self.receipt is not None or not self.captures:
            raise ValueError("Dataset sealing requires the immutable verified capture tuple.")
        for capture in self.captures:
            capture.authorized_case(self.project)
        return self

    @property
    def sha256(self):
        return fingerprint(self.model_dump(mode="json"))


class ArtifactResult(Frozen):
    artifact_id: UUID
    manifest_sha256: Revision
    capture: CaptureReceipt | None = None
    managed_evaluation: ManagedEvaluationReceipt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def matching_capture(self):
        if self.managed_evaluation is not None and (
            self.capture is not None
            or self.managed_evaluation.report.artifact_id != self.artifact_id
            or self.managed_evaluation.report.report_sha256 != self.manifest_sha256
        ):
            raise ValueError("Managed receipt and original report artifact must match.")
        if self.capture is not None and (
            self.capture.artifact_id != self.artifact_id
            or self.capture.manifest_sha256 != self.manifest_sha256
        ):
            raise ValueError("Capture and committed artifact identity must match.")
        return self


class ArtifactStatus(Frozen):
    kind: Literal["artifact_operation"] = "artifact_operation"
    id: UUID
    owner_key: Revision
    project_id: UUID
    target_id: UUID
    operation: Literal["capture", "dataset", "managed_evaluation"]
    work_sha256: Revision
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deadline: AwareDatetime
    max_bytes: int
    max_files: int
    status: Literal["queued", "running", "ready", "failed", "timed_out", "uncertain"]
    phase: Literal["queued", "processing", "manifest_committed", "report_committed", "stopped"] = (
        "queued"
    )
    claim_id: UUID | None = None
    heartbeat_at: AwareDatetime | None = None
    result: ArtifactResult | None = None
    error_code: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def ready_requires_result(self):
        if (self.status == "ready") != (self.result is not None):
            raise ValueError("Only a committed verified manifest can be ready.")
        if self.result is not None and (
            (self.operation == "managed_evaluation") != (self.result.managed_evaluation is not None)
        ):
            raise ValueError("Artifact result must match its declared operation.")
        return self

    def public(self):
        return self.model_dump(mode="json", exclude={"owner_key", "claim_id", "heartbeat_at"})


class ArtifactOperationRecord(OwnedRecord, ArtifactStatus):
    work_document: dict[str, JsonValue]

    def public(self):
        return self.model_dump(
            mode="json",
            exclude={
                "owner_key",
                "tenant_id",
                "fingerprint",
                "claim_id",
                "heartbeat_at",
                "work_document",
            },
        )
