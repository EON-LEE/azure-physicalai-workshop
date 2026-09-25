import hashlib
from datetime import timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from apps.api.errors import Problem
from apps.api.learning_models import (
    Approval,
    CaptureReceipt,
    Frozen,
    OwnedRecord,
    TeachingCase,
    TimingMetadata,
)
from apps.api.models import Identifier, Position, Revision, utcnow
from apps.api.simulation_models import (
    ResolvedSimulationAuthorization,
    SimulationEpisodeCommand,
    SimulationEpisodeExecution,
)
from contracts.validate_environment import parse_document

REFERENCE_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out"})


class PausedOperatorGrant(Frozen):
    model_config = ConfigDict(serialize_by_alias=True)
    schema_version: Literal["physicalai.paused-operator-grant/v1"] = Field(alias="schema")
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    tenant_id: UUID
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    simulator_image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    authorization: ResolvedSimulationAuthorization

    @field_validator("real_time_admission", mode="before")
    @classmethod
    def never_realtime(cls, value):
        if value is not False:
            raise ValueError("Reference collection is explicitly non-real-time.")
        return value


class ReferenceAuthorization(Frozen):
    model_config = ConfigDict(serialize_by_alias=True)
    schema_version: Literal["physicalai.reference-authorization/v1"] = Field(alias="schema")
    project_id: UUID
    case_id: Identifier
    operator_grant: PausedOperatorGrant
    grant_document_json: str = Field(max_length=1024 * 1024)
    runtime_catalog_record_sha256: Revision
    control_profile_sha256: Revision
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    reference_g0_proof_artifact_id: UUID | None = None
    reference_g0_proof_sha256: Revision | None = None

    @model_validator(mode="after")
    def original_installed_bytes(self):
        content = self.grant_document_json.encode("utf-8")
        if (
            len(content) > 1024 * 1024
            or hashlib.sha256(content).hexdigest() != self.runtime_catalog_record_sha256
            or PausedOperatorGrant.model_validate(parse_document(self.grant_document_json))
            != self.operator_grant
        ):
            raise ValueError(
                "Operator catalog must retain the exact runtime-installed grant bytes."
            )
        if (self.reference_g0_proof_artifact_id is None) != (
            self.reference_g0_proof_sha256 is None
        ):
            raise ValueError("G0 proof artifact identity and checksum must be supplied together.")
        return self

    def authorize(self, actor, project, case):
        grant, permit, now = self.operator_grant, self.operator_grant.authorization, utcnow()
        if (
            not grant.issued_at <= now < grant.expires_at
            or not timedelta() < grant.expires_at - grant.issued_at <= timedelta(seconds=600)
            or not now < permit.wall_expires_at <= grant.expires_at
        ):
            raise Problem(409, "reference_grant_expired", "The original reference grant expired.")
        if (
            grant.tenant_id != actor.tenant_id
            or permit.owner != actor.owner_key
            or self.project_id != project.id
            or self.case_id != case.case_id
            or project.execution_timing != "paused_simulation"
            or permit.controller != "reference_controller"
            or permit.authorization_kind != "reference_collection"
            or permit.purpose != "demonstration"
            or permit.policy_type is not None
            or permit.model_sha256 is not None
            or permit.environment_id != case.environment_id
            or permit.revision != case.revision
            or permit.task.task_id != project.task_id
            or permit.task.instruction != project.instruction
            or permit.task.goal_id != project.goal_station_id
            or any(
                getattr(self, field) != getattr(project, field)
                or getattr(permit, field) != getattr(project, field)
                for field in ("control_profile_sha256", "criteria_sha256", "frozen_plan_sha256")
            )
        ):
            raise Problem(
                409,
                "reference_grant_mismatch",
                "Operator reference scope differs from the frozen project/case.",
            )
        return permit


class StartReferenceCollection(Approval):
    case_id: Identifier
    motion_approved: Literal[True]


class ReferenceCollection(OwnedRecord, TimingMetadata):
    kind: Literal["reference_collection"] = "reference_collection"
    project_id: UUID
    teaching_case: TeachingCase
    source: Literal["reference_controller"] = "reference_controller"
    command_id: UUID
    epoch: UUID
    command: SimulationEpisodeCommand
    target_position_m: Position
    goal_tolerance_m: float = Field(gt=0, le=0.04)
    runtime_catalog_record_sha256: Revision
    source_revision: str
    simulator_image_digest: str
    status: Literal[
        "starting",
        "running",
        "cancelling",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
        "unconfirmed",
    ]
    execution: SimulationEpisodeExecution | None = None
    capture_status: Literal["pending", "verifying", "ready", "invalid"] = "pending"
    capture: CaptureReceipt | None = None
    artifact_operation_id: UUID | None = None
    error_code: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def reference_only(self):
        if (
            self.execution_timing != "paused_simulation"
            or self.command.controller != "reference_controller"
        ):
            raise ValueError("Reference collection cannot impersonate human or learned execution.")
        if self.capture_status == "ready" and self.capture is None:
            raise ValueError("A ready reference capture requires verified artifact evidence.")
        if (
            self.command.command_id != self.command_id
            or self.command.epoch != self.epoch
            or self.command.environment_id != self.teaching_case.environment_id
            or self.command.revision != self.teaching_case.revision
            or self.command.authorization_kind != "reference_collection"
        ):
            raise ValueError("The original reference command and approved case must match.")
        return self
