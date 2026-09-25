"""Closed non-real-time bridge contracts, separate from legacy wall-time MotionCommand."""

from __future__ import annotations

from typing import Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from apps.api.models import Execution, Identifier, Model, Revision
from learning.paused import PausedControlProfile
from simulation.runtime_contracts import PolicyType, TaskDefinition


class SimulationEpisodeCommand(Model):
    schema_version: Literal["physicalai.simulation-episode-command/v1"] = Field(alias="schema")
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    profile_id: Literal["franka-position-hold-10hz-paused-v1"]
    command_id: UUID
    environment_id: Identifier
    revision: Revision
    epoch: UUID
    state_revision: int = Field(ge=0, strict=True)
    observation_id: UUID
    object_id: str = Field(min_length=1, max_length=64)
    target_station_id: Identifier
    task: TaskDefinition
    wall_expires_at: AwareDatetime
    max_simulation_steps: int = Field(ge=6, le=1800, multiple_of=6, strict=True)
    controller: Literal["reference_controller", "learned"]
    authorization_kind: Literal["reference_collection", "policy_release", "evaluation_grant"]
    authorization_id: UUID
    policy_type: PolicyType | None = None
    model_sha256: Revision | None = None

    @field_validator("real_time_admission", mode="before")
    @classmethod
    def explicitly_non_realtime(cls, value):
        if value is not False:
            raise ValueError("Paused simulation requires explicit real_time_admission=false.")
        return value

    @model_validator(mode="after")
    def authorized_controller(self):
        if self.task.goal_id != self.target_station_id:
            raise ValueError("The approved task goal must match the commanded station.")
        if self.controller == "reference_controller":
            if self.policy_type is not None or self.model_sha256 is not None:
                raise ValueError("A scripted reference cannot claim a learned checkpoint.")
            if self.authorization_kind != "reference_collection":
                raise ValueError("Scripted collection requires its own reference authorization.")
        elif (
            self.policy_type is None
            or self.model_sha256 is None
            or self.authorization_kind == "reference_collection"
        ):
            raise ValueError(
                "Learned execution requires an actual authorized model and release/grant."
            )
        return self


class PausedRuntimeMetrics(Model):
    execution_timing: Literal["paused_simulation"] = "paused_simulation"
    real_time_admission: Literal[False] = False
    display_label: Literal["NON_REALTIME_SIMULATION"] = "NON_REALTIME_SIMULATION"
    profile_id: Literal["franka-position-hold-10hz-paused-v1"] = (
        "franka-position-hold-10hz-paused-v1"
    )
    control_profile_sha256: Revision
    controller: Literal["reference_controller", "learned"]
    phase: Literal["queued", "observing", "predicting", "applying", "idle", "stopped"] = "queued"
    wall_elapsed_ms: float = Field(default=0, ge=0)
    simulation_steps: int = Field(default=0, ge=0, le=1800)
    simulation_elapsed_seconds: float = Field(default=0, ge=0, le=30.000001)
    policy_predict_calls: int = Field(default=0, ge=0)
    applied_action_count: int = Field(default=0, ge=0)
    applied_model_sha256: Revision | None = None
    reference_route_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def genuine_model_evidence(self):
        if self.controller == "learned":
            if self.reference_route_calls != 0:
                raise ValueError("A learned paused episode cannot run the reference route.")
            if self.applied_action_count and (
                self.applied_model_sha256 is None or not self.policy_predict_calls
            ):
                raise ValueError(
                    "Applied learned actions require real model and prediction evidence."
                )
        elif self.applied_model_sha256 is not None or self.policy_predict_calls:
            raise ValueError("Reference collection cannot claim model predictions.")
        if not self.applied_action_count and self.applied_model_sha256 is not None:
            raise ValueError(
                "A checkpoint cannot be reported as applied before actuator submission."
            )
        return self


class SimulationEpisodeExecution(Execution):
    simulation_runtime: PausedRuntimeMetrics


class ResolvedSimulationAuthorization(Model):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, regex_engine="python-re"
    )
    authorization_id: UUID
    authorization_kind: Literal["reference_collection", "policy_release", "evaluation_grant"]
    owner: Revision
    environment_id: Identifier
    revision: Revision
    controller: Literal["reference_controller", "learned"]
    task: TaskDefinition
    control_profile_sha256: Revision
    wall_expires_at: AwareDatetime
    max_episode_wall_seconds: int = Field(ge=1, le=600, strict=True)
    max_simulation_steps: int = Field(ge=6, le=1800, multiple_of=6, strict=True)
    purpose: Literal["integration", "demonstration", "evaluation"]
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    policy_type: PolicyType | None = None
    model_sha256: Revision | None = None


class PausedEpisodeAuthorizer(Protocol):
    def authorize(
        self, owner: str, request: SimulationEpisodeCommand, profile: PausedControlProfile
    ) -> ResolvedSimulationAuthorization: ...
