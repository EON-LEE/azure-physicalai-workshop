"""Verified non-real-time report DTOs; never aliases for real-time policy qualification."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, StrictBool, field_validator, model_validator

from apps.api.errors import Problem
from apps.api.models import Identifier, Model, Position, Revision

Role = Literal["before", "after", "reference", "candidate"]
Duration = Annotated[float, Field(ge=0)]


class FrozenReport(Model):
    model_config = ConfigDict(frozen=True)


class WallLatency(FrozenReport):
    samples: int = Field(strict=True, ge=0)
    p50: Duration | None
    p95: Duration | None
    max: Duration | None

    @model_validator(mode="after")
    def sample_consistency(self):
        if self.samples == 0:
            if any(value is not None for value in (self.p50, self.p95, self.max)):
                raise ValueError("An empty phase has no invented percentile.")
        elif (
            self.p50 is None
            or self.p95 is None
            or self.max is None
            or not self.p50 <= self.p95 <= self.max
        ):
            raise ValueError("Measured percentile order is invalid.")
        return self


class PredicateSource(FrozenReport):
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    path: str = Field(min_length=1, max_length=256)
    git_blob: str = Field(pattern=r"^[a-f0-9]{40}$")
    sha256: Revision
    symbol: Literal["TaskWatchdog"]


class MeasuredTaskEvidence(FrozenReport):
    predicate_version: Literal["physicalai.measured-grasp-transport/v1"]
    predicate_source: PredicateSource
    grasp_evidence_kind: Literal["measured_lift_proximity_finger_gap_no_contact_sensor"]
    grasp_verified: StrictBool
    settled: StrictBool
    settled_simulation_seconds: Duration
    final_goal_error_m: Duration
    maximum_tcp_speed_m_s: Duration
    safety_violation_count: int = Field(strict=True, ge=0)


class SimulationTrial(FrozenReport):
    episode_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(strict=True, ge=0)
    attempt: Literal[0]
    policy: Role
    model_sha256: Revision | None
    environment_id: Identifier
    revision: Revision
    observed_initial_pose_m: Position
    scene_builder_sha256: Revision
    final_pose_m: Position
    destination_id: Identifier
    terminated: StrictBool
    truncated: StrictBool
    failure_reason: str | None = Field(max_length=2048)
    safety_violation_count: int = Field(strict=True, ge=0)
    policy_predict_calls: int = Field(strict=True, ge=0)
    applied_action_count: int = Field(strict=True, ge=0, le=1800)
    reference_route_calls: int = Field(strict=True, ge=0)
    final_images: dict[Literal["inspection", "overview"], Revision]
    phase_wall_ms: dict[
        Literal["policy", "observation", "hold", "interval", "heartbeat"], WallLatency
    ]
    wall_duration_ms: Duration
    simulation_duration_ms: Duration
    task_evidence: MeasuredTaskEvidence
    physical_success: StrictBool
    position_error_m: tuple[Duration, Duration, Duration]

    @model_validator(mode="after")
    def controller_and_clock_evidence(self):
        if set(self.final_images) != {"inspection", "overview"}:
            raise ValueError("Both final camera hashes are required.")
        if self.policy == "reference":
            if self.model_sha256 is not None or self.policy_predict_calls:
                raise ValueError("Reference motion is not model inference.")
        elif self.model_sha256 is None or self.reference_route_calls:
            raise ValueError("Learned trials cannot substitute reference actions.")
        if abs(self.simulation_duration_ms - self.applied_action_count * 1000 / 60) > 1e-6:
            raise ValueError("Simulation duration must match actual physics ticks.")
        if set(self.phase_wall_ms) != {"policy", "observation", "hold", "interval", "heartbeat"}:
            raise ValueError("Every wall-time phase summary must be retained.")
        for name in ("observation", "hold", "interval"):
            if self.phase_wall_ms[name].samples != self.applied_action_count // 6:
                raise ValueError("Phase counts must match recorded complete control intervals.")
        return self


class TrialCount(FrozenReport):
    total: Literal[20]
    success: int = Field(strict=True, ge=0, le=20)


class SimulationReport(FrozenReport):
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    native_schema: Literal[
        "physicalai.smolvla-paired-report/v2", "physicalai.smolvla-bootstrap-report/v2"
    ]
    comparison_kind: Literal["paired_policy", "reference_bootstrap"]
    control_profile_id: Literal["franka-position-hold-10hz-paused-v1"]
    control_profile_sha256: Revision
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    evaluation_plan_sha256: Revision
    native_plan_sha256: Revision
    results_sha256: Revision
    runtime_sha256: Revision
    report_sha256: Revision
    artifact_id: UUID
    before_model_sha256: Revision | None = None
    after_model_sha256: Revision | None = None
    candidate_model_sha256: Revision | None = None
    reference_controller_sha256: Revision | None = None
    trials: tuple[SimulationTrial, ...] = Field(min_length=40, max_length=40)
    counts: dict[Role, TrialCount]
    success_rates: dict[Role, Annotated[float, Field(ge=0, le=1)]]
    absolute_success_rate_improvement: float = Field(ge=-1, le=1)
    latency_wall_ms: dict[Role, WallLatency]
    total_wall_duration_ms: Duration
    total_simulation_duration_ms: Duration
    safety_violation_count: int = Field(strict=True, ge=0)
    resource_violation_count: int = Field(strict=True, ge=0)
    total_trial_count: Literal[40]
    quality_gate_passed: StrictBool
    conclusion: Literal["improved", "not_improved", "inconclusive"]
    live_gpu_verified: Literal[True]

    @field_validator("real_time_admission", "live_gpu_verified", mode="before")
    @classmethod
    def explicit_evidence_flags(cls, value, info):
        expected = info.field_name == "live_gpu_verified"
        if value is not expected:
            raise ValueError(
                "Physical verification and false real-time admission must be explicit."
            )
        return value

    @model_validator(mode="after")
    def complete_role_pair(self):
        bootstrap = self.comparison_kind == "reference_bootstrap"
        roles = {"reference", "candidate"} if bootstrap else {"before", "after"}
        if (
            any(
                set(values) != roles
                for values in (
                    self.counts,
                    self.success_rates,
                    self.latency_wall_ms,
                )
            )
            or {trial.policy for trial in self.trials} != roles
        ):
            raise ValueError("All frozen controller roles must be retained.")
        if bootstrap:
            valid = (
                self.native_schema == "physicalai.smolvla-bootstrap-report/v2"
                and self.candidate_model_sha256 is not None
                and self.reference_controller_sha256 is not None
                and self.before_model_sha256 is None
                and self.after_model_sha256 is None
            )
        else:
            valid = (
                self.native_schema == "physicalai.smolvla-paired-report/v2"
                and self.before_model_sha256 is not None
                and self.after_model_sha256 is not None
                and self.candidate_model_sha256 is None
                and self.reference_controller_sha256 is None
            )
        if not valid:
            raise ValueError("Bootstrap reference and paired model evidence cannot be mixed.")
        for role in roles:
            actual = [trial for trial in self.trials if trial.policy == role]
            successes = sum(trial.physical_success for trial in actual)
            if (
                len(actual) != 20
                or self.counts[role].success != successes
                or abs(self.success_rates[role] - successes / 20) > 1e-9
                or self.latency_wall_ms[role].samples
                != sum(trial.phase_wall_ms["policy"].samples for trial in actual)
            ):
                raise ValueError("Counts and sample totals must preserve every physical trial.")
        if (
            abs(self.total_wall_duration_ms - sum(trial.wall_duration_ms for trial in self.trials))
            > 1e-6
            or abs(
                self.total_simulation_duration_ms
                - sum(trial.simulation_duration_ms for trial in self.trials)
            )
            > 1e-6
            or self.safety_violation_count
            != sum(trial.safety_violation_count for trial in self.trials)
        ):
            raise ValueError("Report totals must match the complete physical trial summaries.")
        return self


def validate_report_binding(specification, report: SimulationReport) -> None:
    project = specification.project
    plan = project.evaluation_plan
    roles = (
        ("reference", "candidate")
        if report.comparison_kind == "reference_bootstrap"
        else ("before", "after")
    )
    expected = {
        (case.seed, role, case.environment_id, case.revision)
        for case in plan.cases
        for role in roles
    }
    actual = {(row.seed, row.policy, row.environment_id, row.revision) for row in report.trials}
    if len(actual) != 40 or actual != expected or report.evaluation_plan_sha256 != plan.sha256:
        raise Problem(503, "paused_report_cases", "The report does not retain every approved case.")
    for row in report.trials:
        model = (
            None
            if row.policy == "reference"
            else (
                specification.baseline.model_sha256
                if row.policy == "before"
                else specification.candidate.model_sha256
            )
        )
        if row.model_sha256 != model or row.destination_id != project.goal_station_id:
            raise Problem(
                503, "paused_report_models", "Trial model/task differs from reviewed inputs."
            )
    if (
        project.execution_timing != "paused_simulation"
        or report.control_profile_sha256 != project.control_profile_sha256
        or report.criteria_sha256 != project.criteria_sha256
        or report.frozen_plan_sha256 != project.frozen_plan_sha256
    ):
        raise Problem(503, "paused_report_scope", "The immutable report provenance changed.")
