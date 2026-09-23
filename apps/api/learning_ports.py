from __future__ import annotations

from typing import Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field

from apps.api.learning_models import (
    BootstrapReport,
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
    TrainingParent,
    TrainingRun,
)
from apps.api.models import DemonstrationResult, Execution, Identifier, Principal, Revision, Stored


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
    baseline: PolicyRelease | None = None
    training_parent: TrainingParent | None = None
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
    report: PairedReport | BootstrapReport | None = None
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
    def training_parent(self, actor: Principal, artifact_id: UUID) -> TrainingParent: ...


class RuntimeCapture(Frozen):
    capture_id: UUID
    command_id: UUID
    epoch: UUID
    status: Literal["recording", "finalizing", "uploading", "ready", "invalid"]
    receipt: DemonstrationResult | None = None
    message: str | None = None


class TeachingRuntimeState(Frozen):
    session_id: UUID
    lease_id: UUID
    epoch: UUID
    command_id: UUID
    control_profile_id: str
    status: Literal[
        "queued",
        "running",
        "finishing",
        "cancelling",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    ]
    last_sequence: int = Field(ge=0)
    input_expires_at: AwareDatetime | None = None
    execution: Execution
    capture: RuntimeCapture | None = None


class TeachingRuntime(Protocol):
    def start_teaching(self, owner: str, body: TeachingStartSpec) -> TeachingRuntimeState: ...
    def teaching(self, owner: str, session_id: UUID) -> TeachingRuntimeState: ...
    def teaching_input(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState: ...
    def finish_teaching(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState: ...
    def cancel_teaching(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState: ...
    def capture(self, owner: str, command_id: UUID) -> RuntimeCapture: ...


class TeachingTask(Frozen):
    task_id: Identifier
    instruction: str = Field(min_length=1, max_length=512, pattern=r"^[^\r\n]*\S[^\r\n]*$")
    goal_id: Identifier


class TeachingStartSpec(Frozen):
    session_id: UUID
    lease_id: UUID
    command_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(ge=0)
    observation_id: UUID
    object_id: str
    target_station_id: Identifier
    session_expires_at: AwareDatetime
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    task: TeachingTask
    demonstrator_kind: Literal["human_teleop", "reference_controller"]
