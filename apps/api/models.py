from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*(?![\s\S])", max_length=64)]
Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
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


class Execution(Model):
    command_id: UUID
    status: Literal[
        "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out"
    ]
    final_position: Position | None = None
    completed_at: AwareDatetime | None = None
    error: RunError | None = None


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


class SimulationStatus(Model):
    backend: Literal["isaac_sim"] = "isaac_sim"
    status: Literal["ready", "loading", "unavailable", "occupied"]
    environment_id: Identifier | None = None
    revision: Revision | None = None
    epoch: UUID | None = None
    physics_steps: int | None = None
    message: str | None = None


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
