"""CPU-safe bridge contracts; importing these never loads an actuator or a model."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StrictBool, field_validator, model_validator

from apps.api.models import (
    DemonstrationResult,
    Execution,
    Identifier,
    Model,
    MotionCommand,
    Revision,
)

RuntimeMode = Literal["reference", "human_teaching", "learned"]
PolicyType = Literal["smolvla", "gr00t_n1_5", "gr00t_n1_7"]
CapturePhase = Literal["recording", "finalizing", "uploading", "ready", "invalid"]


@dataclass(frozen=True)
class CommandBinding:
    owner: str
    environment_id: str
    revision: str
    epoch: UUID
    command_id: UUID


@dataclass(frozen=True)
class CaptureBinding(CommandBinding):
    capture_id: UUID


@dataclass(frozen=True)
class CaptureReceipt:
    manifest_uri: str
    manifest_sha256: str
    episode_id: str
    frame_count: int


class CaptureStatus(Model):
    capture_id: UUID
    command_id: UUID
    epoch: UUID
    status: CapturePhase
    receipt: DemonstrationResult | None = None
    message: str | None = None


@dataclass(frozen=True)
class CaptureUpdate:
    binding: CaptureBinding
    state: CaptureStatus


CONTROL_PROFILE_ID = "franka-position-hold-10hz-v1"


class TaskDefinition(Model):
    task_id: Identifier
    instruction: str = Field(min_length=1, max_length=512)
    goal_id: Identifier

    @field_validator("instruction")
    @classmethod
    def bounded_instruction(cls, value: str) -> str:
        if not value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("Task instruction must be a bounded, nonempty single line.")
        return value


class TeachingStart(Model):
    session_id: UUID
    lease_id: UUID
    command_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(ge=0, strict=True)
    observation_id: UUID
    object_id: str = Field(min_length=1, max_length=64)
    target_station_id: Identifier
    session_expires_at: AwareDatetime
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    task: TaskDefinition
    demonstrator_kind: Literal["human_teleop", "reference_controller"]

    @model_validator(mode="after")
    def matching_goal(self):
        if self.task.goal_id != self.target_station_id:
            raise ValueError("The approved task goal must match the commanded station.")
        return self


class TeachingLease(Model):
    lease_id: UUID
    epoch: UUID


class TeachingInput(TeachingLease):
    sequence: int = Field(ge=1, le=10000, strict=True)
    expires_at: AwareDatetime
    deadman: StrictBool
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["hold", "open", "close"]
    grant_id: UUID | None = None
    grant_expires_at: AwareDatetime | None = None

    @field_validator("delta_xyz_m", mode="before")
    @classmethod
    def numeric_displacement(cls, value):
        if not isinstance(value, (tuple, list)) or any(type(v) not in (int, float) for v in value):
            raise ValueError("Cartesian displacement must contain only finite numbers.")
        return value

    @model_validator(mode="after")
    def bounded_motion(self):
        if dist(self.delta_xyz_m, (0.0, 0.0, 0.0)) > 0.01:
            raise ValueError("Cartesian jog exceeds the one-centimetre displacement limit.")
        if not self.deadman and (any(self.delta_xyz_m) or self.gripper != "hold"):
            raise ValueError("Releasing the deadman cannot request motion or gripper actuation.")
        return self


class TeachingState(Model):
    session_id: UUID
    lease_id: UUID
    epoch: UUID
    command_id: UUID
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    demonstrator_kind: Literal["human_teleop", "reference_controller"]
    session_expires_at: AwareDatetime
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
    input_expires_at: AwareDatetime | None
    execution: Execution
    capture: CaptureStatus | None


@dataclass(frozen=True)
class TeachingIntent:
    sequence: int
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["hold", "open", "close"]
    expires_at_monotonic_ns: int


class PolicyCommand(Model):
    command: MotionCommand
    policy_type: PolicyType
    policy_release_id: UUID
    model_sha256: Revision
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    task: TaskDefinition

    @model_validator(mode="after")
    def matching_goal(self):
        if self.task.goal_id != self.command.target_station_id:
            raise ValueError("The released policy task must match the commanded station.")
        return self


class PolicyRuntime(Model):
    execution_mode: Literal["learned"] = "learned"
    policy_type: PolicyType
    policy_release_id: UUID
    applied_model_sha: Revision | None = None
    control_profile_id: Literal["franka-position-hold-10hz-v1"] = CONTROL_PROFILE_ID
    policy_predict_calls: int = Field(default=0, ge=0)
    applied_action_count: int = Field(default=0, ge=0)
    reference_route_calls: Literal[0] = 0

    @model_validator(mode="after")
    def actual_application_evidence(self):
        if self.applied_action_count and (
            self.applied_model_sha is None or self.policy_predict_calls == 0
        ):
            raise ValueError("An applied policy action needs actual model/prediction evidence.")
        if not self.applied_action_count and self.applied_model_sha is not None:
            raise ValueError("A model cannot be reported as applied before actuator submission.")
        return self


class PolicyExecution(Execution):
    policy_runtime: PolicyRuntime
