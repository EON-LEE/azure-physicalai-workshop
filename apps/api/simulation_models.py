"""Strict API-side paused-episode wire types, without importing simulator or model code."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from apps.api.models import (
    PAUSED_PROFILE_STEPS,
    PAUSED_PROFILE_V1,
    Execution,
    Identifier,
    LearnedPolicyType,
    Model,
    PausedProfileId,
    Revision,
)


class EpisodeTask(Model):
    task_id: Identifier
    instruction: str = Field(min_length=1, max_length=512)
    goal_id: Identifier

    @field_validator("instruction")
    @classmethod
    def single_line_instruction(cls, value):
        if not value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("The approved task instruction must be a nonempty single line.")
        return value


class SimulationEpisodeCommand(Model):
    model_config = ConfigDict(serialize_by_alias=True)

    schema_version: Literal["physicalai.simulation-episode-command/v1"] = Field(alias="schema")
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    profile_id: PausedProfileId
    command_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(strict=True, ge=0)
    observation_id: UUID
    object_id: str = Field(min_length=1, max_length=64)
    target_station_id: Identifier
    task: EpisodeTask
    wall_expires_at: AwareDatetime
    max_simulation_steps: int = Field(strict=True, ge=6, le=3600, multiple_of=6)
    controller: Literal["reference_controller", "learned"]
    authorization_kind: Literal["reference_collection", "policy_release", "evaluation_grant"]
    authorization_id: UUID
    policy_type: LearnedPolicyType | None = None
    model_sha256: Revision | None = None

    @field_validator("real_time_admission", mode="before")
    @classmethod
    def no_realtime_claim(cls, value):
        if value is not False:
            raise ValueError("Paused simulation requires explicit real_time_admission=false.")
        return value

    @model_validator(mode="after")
    def exact_authorized_controller(self):
        if self.max_simulation_steps > PAUSED_PROFILE_STEPS[self.profile_id]:
            raise ValueError("Command exceeds its explicitly declared paused profile.")
        if self.task.goal_id != self.target_station_id:
            raise ValueError("Task and commanded goal must match.")
        if self.controller == "reference_controller":
            if (
                self.policy_type is not None
                or self.model_sha256 is not None
                or self.authorization_kind != "reference_collection"
            ):
                raise ValueError("Reference collection requires its own non-model authorization.")
        elif (
            self.policy_type is None
            or self.model_sha256 is None
            or self.authorization_kind == "reference_collection"
        ):
            raise ValueError("Learned episodes require the reviewed model and release/grant.")
        return self


class PausedRuntimeMetrics(Model):
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    display_label: Literal["NON_REALTIME_SIMULATION"]
    profile_id: PausedProfileId
    control_profile_sha256: Revision
    controller: Literal["reference_controller", "learned"]
    phase: Literal["queued", "observing", "predicting", "applying", "idle", "stopped"]
    wall_elapsed_ms: float = Field(ge=0)
    simulation_steps: int = Field(strict=True, ge=0, le=3600)
    simulation_elapsed_seconds: float = Field(ge=0, le=60)
    policy_predict_calls: int = Field(strict=True, ge=0)
    applied_action_count: int = Field(strict=True, ge=0)
    applied_model_sha256: Revision | None
    reference_route_calls: int = Field(strict=True, ge=0)

    @field_validator("real_time_admission", mode="before")
    @classmethod
    def no_realtime_claim(cls, value):
        if value is not False:
            raise ValueError("Simulation metrics must explicitly deny real-time admission.")
        return value

    @model_validator(mode="after")
    def actual_application_only(self):
        if (
            self.simulation_steps > PAUSED_PROFILE_STEPS[self.profile_id]
            or self.simulation_elapsed_seconds != self.simulation_steps / 60
        ):
            raise ValueError(
                "Simulation time must derive from actual ticks within the declared profile."
            )
        if self.controller == "learned":
            if self.reference_route_calls or (
                self.applied_action_count
                and (self.applied_model_sha256 is None or not self.policy_predict_calls)
            ):
                raise ValueError(
                    "Learned actions require actual model evidence, no reference route."
                )
        elif self.applied_model_sha256 is not None or self.policy_predict_calls:
            raise ValueError("Reference episodes do not report model predictions.")
        if not self.applied_action_count and self.applied_model_sha256 is not None:
            raise ValueError("No applied-model claim is valid before actuator submission.")
        return self


class SimulationEpisodeExecution(Execution):
    simulation_runtime: PausedRuntimeMetrics

    @model_validator(mode="after")
    def separate_timing_evidence(self):
        if self.policy_runtime is not None:
            raise ValueError("Paused simulation cannot reuse legacy real-time policy evidence.")
        return self


class ResolvedSimulationAuthorization(Model):
    model_config = ConfigDict(frozen=True)

    authorization_id: UUID
    profile_id: PausedProfileId = Field(
        default=PAUSED_PROFILE_V1, exclude_if=lambda value: value == PAUSED_PROFILE_V1
    )
    authorization_kind: Literal["reference_collection", "policy_release", "evaluation_grant"]
    owner: Revision
    environment_id: Identifier
    revision: Revision
    controller: Literal["reference_controller", "learned"]
    task: EpisodeTask
    control_profile_sha256: Revision
    wall_expires_at: AwareDatetime
    max_episode_wall_seconds: int = Field(strict=True, ge=1, le=600)
    max_simulation_steps: int = Field(strict=True, ge=6, le=3600, multiple_of=6)
    purpose: Literal["integration", "demonstration", "evaluation"]
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    policy_type: LearnedPolicyType | None = None
    model_sha256: Revision | None = None

    @model_validator(mode="after")
    def profile_bounds(self):
        if self.max_simulation_steps > PAUSED_PROFILE_STEPS[self.profile_id]:
            raise ValueError("Resolved authority cannot exceed its declared profile.")
        return self
