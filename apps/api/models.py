from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated, Generic, Literal, TypeVar
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*(?![\s\S])", max_length=64)]
Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
LearnedPolicyType = Literal["gr00t_n1_5", "gr00t_n1_7", "smolvla"]
Position = tuple[float, float, float]
RunStatus = Literal[
    "planning",
    "awaiting_approval",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
]
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out"})


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, regex_engine="python-re")


class Principal(Model):
    tenant_id: UUID
    object_id: UUID

    @property
    def owner_key(self) -> str:
        return hashlib.sha256(f"{self.tenant_id}:{self.object_id}".encode()).hexdigest()


class SaveEnvironment(Model):
    document_json: str = Field(min_length=2, max_length=131072)
    expected_revision: Revision | None = None


class ActivateEnvironment(Model):
    revision: Revision


class EnvironmentRecord(Model):
    environment_id: Identifier
    display_name: str
    revision: Revision
    document: dict[str, JsonValue]
    created_at: AwareDatetime
    updated_at: AwareDatetime


class StartRun(Model):
    request_id: UUID
    environment_id: Identifier
    revision: Revision
    instruction: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    policy_release_id: UUID | None = None

    def fingerprint_document(self) -> dict:
        value = self.model_dump(mode="json")
        if self.policy_release_id is None:
            value.pop("policy_release_id")
        return value


class ApproveRun(Model):
    plan_response_id: str = Field(min_length=1, max_length=256)


class Event(Model):
    kind: str
    message: str
    at: AwareDatetime = Field(default_factory=utcnow)


class RunError(Model):
    code: str
    message: str
    retryable: bool = False


class Observation(Model):
    observation_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(ge=0)
    captured_at: AwareDatetime
    physics_steps: int = Field(ge=0)
    object_id: str = Field(min_length=1, max_length=64)
    object_position: Position
    camera: Literal["overview", "inspection"]
    image_base64: str = Field(min_length=16, max_length=7000000)


class Evidence(Model):
    observation_id: UUID
    epoch: UUID
    state_revision: int
    captured_at: AwareDatetime
    object_id: str
    object_position: Position
    blob_name: str
    sha256: str


class Decision(Model):
    classification: Literal["accepted", "rejected"]
    object_id: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class Plan(Decision):
    target_station_id: Identifier
    observation_id: UUID
    epoch: UUID
    state_revision: int
    model_response_id: str
    expires_at: AwareDatetime


class DemonstrationResult(Model):
    status: Literal["uploaded", "failed"]
    manifest_uri: str | None = None
    manifest_sha256: Revision | None = None
    episode_id: UUID | None = None
    frame_count: int | None = Field(default=None, ge=2)
    message: str | None = None

    @model_validator(mode="after")
    def valid_receipt(self):
        if self.status == "uploaded":
            parsed = urlsplit(self.manifest_uri or "")
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or not parsed.hostname.endswith(".blob.core.windows.net")
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or self.manifest_sha256 is None
                or self.episode_id is None
                or self.frame_count is None
            ):
                raise ValueError("An uploaded demonstration requires complete Azure evidence.")
        elif not self.message:
            raise ValueError("A failed demonstration requires an explicit reason.")
        return self


class PolicyRuntime(Model):
    execution_mode: Literal["learned"] = "learned"
    policy_type: LearnedPolicyType
    policy_release_id: UUID
    applied_model_sha: Revision | None = None
    control_profile_id: Identifier | None = None
    policy_predict_calls: int = Field(strict=True, ge=0)
    applied_action_count: int = Field(strict=True, ge=0)
    reference_route_calls: Literal[0]

    @model_validator(mode="after")
    def application_is_measured(self):
        if self.applied_action_count > 0 and (
            self.applied_model_sha is None or self.policy_predict_calls == 0
        ):
            raise ValueError("Applied actions require the actual model and prediction evidence.")
        return self


class ReleasedPolicyBinding(Model):
    model_config = ConfigDict(frozen=True)

    policy_release_id: UUID
    policy_type: LearnedPolicyType
    model_sha256: Revision
    processor_sha256: Revision
    manifest_sha256: Revision
    control_profile_id: Identifier
    task_id: Identifier
    goal_station_id: Identifier
    instruction: str = Field(min_length=1, max_length=512, pattern=r"^[^\r\n]*\S[^\r\n]*$")


class Execution(Model):
    command_id: UUID
    status: Literal[
        "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out"
    ]
    final_position: Position | None = None
    completed_at: AwareDatetime | None = None
    error: RunError | None = None
    demonstration: DemonstrationResult | None = None
    policy_runtime: PolicyRuntime | None = None


class RunRecord(Model):
    id: UUID
    environment_id: Identifier
    revision: Revision
    instruction: str
    status: RunStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    request_fingerprint: str
    environment_document: dict[str, JsonValue]
    command_deadline: AwareDatetime | None = None
    plan: Plan | None = None
    evidence: Evidence | None = None
    execution: Execution | None = None
    error: RunError | None = None
    events: list[Event] = Field(default_factory=list)
    policy: ReleasedPolicyBinding | None = None

    def public(self) -> dict:
        return self.model_dump(
            mode="json",
            exclude={"request_fingerprint", "evidence", "environment_document", "command_deadline"},
        )


class Activation(Model):
    activation_id: UUID
    environment_id: Identifier
    revision: Revision
    status: Literal["loading"] = "loading"


MotionPhase = Literal[
    "idle",
    "approaching",
    "grasping",
    "lifting",
    "inspection_station",
    "transporting",
    "releasing",
    "returning",
    "complete",
    "stopped",
]


class MotionTelemetry(Model):
    command_id: UUID | None
    phase: MotionPhase
    object_position: Position
    target_station_id: Identifier | None


class SimulationStatus(Model):
    backend: Literal["isaac_sim"] = "isaac_sim"
    status: Literal["ready", "loading", "unavailable", "occupied"]
    environment_id: Identifier | None = None
    revision: Revision | None = None
    epoch: UUID | None = None
    physics_steps: int | None = None
    message: str | None = None
    motion: MotionTelemetry | None = None


PresentationStatus = Literal[
    "preparing", "inspecting", "awaiting_motion", "moving", "completed", "stopped", "failed"
]


class PresentationResult(Model):
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    physical_success: bool
    inspection_correct: bool | None
    final_position_m: Position | None
    completed_at: AwareDatetime
    message: Literal[
        "Inspection and physical sorting completed.",
        "Inspection disagreed with the reference evaluation; motion was not authorized.",
        "Physical completion was not verified.",
        "The authorized presentation time expired.",
        "The reference cycle was cancelled.",
    ]

    @model_validator(mode="after")
    def evidence_consistent(self):
        if self.physical_success and self.final_position_m is None:
            raise ValueError("Physical success requires a final position.")
        if (self.status == "succeeded") != (
            self.physical_success and self.inspection_correct is True
        ):
            raise ValueError("Success requires correct inspection and physical completion.")
        return self


class PresentationOutcome(Model):
    cycle: int = Field(ge=1, le=1000)
    run_id: UUID
    result: PresentationResult


class PresentationRecord(Model):
    """Dedicated publication state, never an index of an operator's private history."""

    id: Identifier
    owner_key: Revision
    normal_environment_id: Identifier
    normal_revision: Revision
    defect_environment_id: Identifier
    defect_revision: Revision
    runner_id: UUID
    started_at: AwareDatetime
    expires_at: AwareDatetime
    updated_at: AwareDatetime
    total_cycles: int = Field(ge=1, le=1000)
    cycle: int = Field(ge=1, le=1000)
    status: PresentationStatus
    scene_epoch: UUID | None = None
    run_id: UUID | None = None
    outcomes: list[PresentationOutcome] = Field(default_factory=list, max_length=1000)
    stop_reason: (
        Literal[
            "time_limit",
            "repeated_failures",
            "dependency_unavailable",
            "scene_changed",
            "runner_interrupted",
            "cancellation_unconfirmed",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def bounded_history(self):
        if not self.started_at <= self.updated_at or not (
            30 <= (self.expires_at - self.started_at).total_seconds() <= 21600
        ):
            raise ValueError("Invalid presentation authorization timestamps.")
        if self.cycle > self.total_cycles:
            raise ValueError("Current cycle exceeds authorization.")
        if [outcome.cycle for outcome in self.outcomes] != list(
            range(1, len(self.outcomes) + 1)
        ) or len(self.outcomes) > self.cycle:
            raise ValueError("Presentation history must be ordered and contiguous.")
        return self


class PresentationDecision(Model):
    classification: Literal["accepted", "rejected"]
    summary: str
    target_station_id: Identifier
    observation_id: UUID
    captured_at: AwareDatetime
    image_url: Literal["/api/demo/evidence"] = "/api/demo/evidence"


class PresentationMotion(Model):
    status: Literal[
        "queued", "running", "succeeded", "failed", "cancelled", "timed_out", "cancelling"
    ]
    phase: MotionPhase | None
    part_position_m: Position | None
    target_position_m: Position


class PresentationCounts(Model):
    attempted: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    inspected_correctly: int = Field(ge=0)
    physically_completed: int = Field(ge=0)


class PublicPresentation(Model):
    id: str
    status: PresentationStatus
    cycle: int
    total_cycles: int
    scenario: Literal["normal", "surface_defect"]
    instruction: str
    updated_at: AwareDatetime
    expires_at: AwareDatetime
    scene_epoch: UUID | None
    run_id: UUID | None
    decision: PresentationDecision | None
    motion: PresentationMotion | None
    result: PresentationResult | None
    counts: PresentationCounts


class PublicDemoCase(Model):
    kind: Literal["normal_route", "defect_route", "withheld"]
    cycle: int = Field(ge=1, le=1000)
    scenario: Literal["normal", "surface_defect"]
    classification: Literal["accepted", "rejected"]
    summary: str
    target_station_id: Identifier
    target_position_m: Position
    observation_id: UUID
    captured_at: AwareDatetime
    image_url: Literal["/api/demo/cases/evidence"] = "/api/demo/cases/evidence"
    motion_authorized: bool
    physical_duration_seconds: float | None = Field(default=None, ge=0, le=30)
    result: PresentationResult


class PublicDemoCases(Model):
    api_version: Literal["public-demo-cases-v1"] = "public-demo-cases-v1"
    source: Literal["recorded_reference_runs"] = "recorded_reference_runs"
    presentation_id: Identifier | None
    cases: list[PublicDemoCase] = Field(default_factory=list, max_length=3)


class MotionCommand(Model):
    command_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(ge=0)
    observation_id: UUID
    object_id: str
    target_station_id: Identifier
    deadline: AwareDatetime


T = TypeVar("T")


class Stored(BaseModel, Generic[T]):
    value: T
    etag: str
