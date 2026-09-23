from __future__ import annotations

from typing import Literal, Protocol
from uuid import UUID

from pydantic import Field

from apps.api.learning_models import (
    CaptureReceipt,
    DatasetVersion,
    EvaluationRun,
    Frozen,
    LearningProject,
    LearningRecord,
    PairedReport,
    PolicyCandidate,
    PolicyRelease,
    TeachingSession,
    TrainingMetrics,
    TrainingRun,
)
from apps.api.models import Principal, Revision, Stored


class LearningStore(Protocol):
    def get_learning(self, owner: str, kind: str, resource_id: UUID) -> Stored | None: ...
    def put_learning(self, owner: str, record: LearningRecord, etag: str | None) -> Stored: ...
    def list_learning(
        self, owner: str, kind: str, project_id: UUID | None = None
    ) -> list[Stored]: ...


class JobSpecification(Frozen):
    owner_key: Revision
    project: LearningProject
    run: TrainingRun | EvaluationRun
    baseline: PolicyRelease
    dataset: DatasetVersion | None = None
    candidate: PolicyCandidate | None = None


class BackendJob(Frozen):
    job_name: str
    azure_job_id: str = Field(min_length=1, max_length=2048)
    owner_key: Revision
    specification_sha256: Revision
    status: Literal[
        "submitted", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out"
    ]
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    candidate: PolicyCandidate | None = None
    report: PairedReport | None = None
    error_code: str | None = None
    message: str | None = None


class LearningJobs(Protocol):
    def preflight(self, actor: Principal, specification: JobSpecification) -> None: ...
    def submit(self, actor: Principal, specification: JobSpecification) -> BackendJob: ...
    def status(self, actor: Principal, run: TrainingRun | EvaluationRun) -> BackendJob | None: ...
    def cancel(self, actor: Principal, run: TrainingRun | EvaluationRun) -> BackendJob: ...


class LearningArtifacts(Protocol):
    def verify_capture(
        self, actor: Principal, project: LearningProject, session: TeachingSession, receipt: dict
    ) -> CaptureReceipt: ...
    def seal_dataset(
        self,
        actor: Principal,
        project: LearningProject,
        dataset_id: UUID,
        captures: tuple[CaptureReceipt, ...],
    ) -> tuple[UUID, str]: ...
    def verify_candidate(
        self,
        actor: Principal,
        project: LearningProject,
        run: TrainingRun,
        candidate: PolicyCandidate,
    ) -> None: ...
    def verify_report(
        self,
        actor: Principal,
        project: LearningProject,
        run: EvaluationRun,
        report: PairedReport,
    ) -> None: ...


class PolicyCatalog(Protocol):
    def resolve(self, actor: Principal, release_id: UUID) -> PolicyRelease: ...
