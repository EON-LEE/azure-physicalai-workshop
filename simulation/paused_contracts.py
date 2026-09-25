"""Closed non-real-time bridge contracts, separate from legacy wall-time MotionCommand."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol
from uuid import UUID

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from apps.api.models import Execution, Identifier, Model, Revision
from learning.paused import (
    CONTROL_PROFILE_ID,
    FrozenPolicyObservation,
    InitialFrozenPublication,
    PausedControlContext,
    PausedControlProfile,
)
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.paused_profiles import PausedProfileId, profile_step_limit
from simulation.runtime_contracts import PolicyType, TaskDefinition


class SimulationEpisodeCommand(Model):
    schema_version: Literal["physicalai.simulation-episode-command/v1"] = Field(alias="schema")
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    profile_id: PausedProfileId
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
    max_simulation_steps: int = Field(ge=6, le=3600, multiple_of=6, strict=True)
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
        if self.max_simulation_steps > profile_step_limit(self.profile_id):
            raise ValueError("The command exceeds its explicitly selected paused profile budget.")
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
    profile_id: PausedProfileId = CONTROL_PROFILE_ID
    control_profile_sha256: Revision
    controller: Literal["reference_controller", "learned"]
    phase: Literal["queued", "observing", "predicting", "applying", "idle", "stopped"] = "queued"
    wall_elapsed_ms: float = Field(default=0, ge=0)
    simulation_steps: int = Field(default=0, ge=0, le=3600)
    simulation_elapsed_seconds: float = Field(default=0, ge=0, le=60.000001)
    policy_predict_calls: int = Field(default=0, ge=0)
    applied_action_count: int = Field(default=0, ge=0)
    applied_model_sha256: Revision | None = None
    reference_route_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def genuine_model_evidence(self):
        maximum = profile_step_limit(self.profile_id)
        if self.simulation_steps > maximum or self.simulation_elapsed_seconds > maximum / 60 + 1e-6:
            raise ValueError("Metrics exceed their explicitly selected paused profile budget.")
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
    profile_id: PausedProfileId = CONTROL_PROFILE_ID
    wall_expires_at: AwareDatetime
    max_episode_wall_seconds: int = Field(ge=1, le=600, strict=True)
    max_simulation_steps: int = Field(ge=6, le=3600, multiple_of=6, strict=True)
    purpose: Literal["integration", "demonstration", "evaluation"]
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    policy_type: PolicyType | None = None
    model_sha256: Revision | None = None

    @model_validator(mode="after")
    def versioned_budget(self):
        if self.max_simulation_steps > profile_step_limit(self.profile_id):
            raise ValueError("Resolved authority exceeds its explicit paused profile budget.")
        return self

    @model_serializer(mode="wrap")
    def preserve_v1_wire(self, handler):
        value = handler(self)
        if self.profile_id == CONTROL_PROFILE_ID:
            value.pop("profile_id", None)
        return value


class PausedEpisodeAuthorizer(Protocol):
    def authorize(
        self, owner: str, request: SimulationEpisodeCommand, profile: PausedControlProfile
    ) -> ResolvedSimulationAuthorization: ...


class PausedPolicyProvider(PausedEpisodeAuthorizer, Protocol):
    def create(
        self,
        owner: str,
        request: SimulationEpisodeCommand,
        profile: PausedControlProfile,
        *,
        publication_guard: Callable[
            [InitialFrozenPublication, FrozenPolicyObservation, PausedControlContext], bool
        ],
    ) -> PausedGuardedPolicyAdapter: ...
