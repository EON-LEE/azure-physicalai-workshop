from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from apps.api.errors import Problem
from apps.api.models import Identifier, LearnedPolicyType, Model, Principal, Revision, utcnow

PolicyType = Literal["gr00t_n1_5", "gr00t_n1_7", "smolvla", "act_auxiliary"]
SourceKind = Literal["human_teleop", "reference_controller", "learned"]
JobStatus = Literal[
    "submitting",
    "submission_unknown",
    "submitted",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
    "blocked",
]
TeachingStatus = Literal[
    "starting",
    "recording",
    "finishing",
    "finalizing",
    "uploading",
    "ready",
    "cancelling",
    "cancelled",
    "invalid",
    "blocked",
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
JOB_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "blocked"})
TEACHING_TERMINAL = frozenset({"ready", "cancelled", "invalid", "blocked"})
PROFILE_ID = "franka-position-hold-10hz-v1"
INTEGRATION_ONLY_SEEDS = frozenset({900002})


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


class Frozen(Model):
    model_config = ConfigDict(frozen=True)


class Budget(Frozen):
    teaching_seconds: int = Field(strict=True, ge=5, le=300)
    training_seconds: int = Field(strict=True, ge=1, le=86400)
    evaluation_seconds: int = Field(strict=True, ge=1, le=21600)
    optimizer_steps: int = Field(strict=True, ge=1, le=100000)
    maximum_cost_usd: Decimal = Field(gt=0, le=10000, max_digits=9, decimal_places=2)


class PolicyScene(Frozen):
    environment_id: Identifier
    revision: Revision


class EvaluationCase(PolicyScene):
    seed: NonnegativeInt


class TeachingCase(EvaluationCase):
    case_id: Identifier
    split: Literal["train", "validation"]


class EvaluationPlan(Frozen):
    id: UUID
    seeds: tuple[Annotated[int, Field(strict=True, ge=0, le=2147483647)], ...] = Field(
        min_length=20, max_length=100
    )
    held_out_episode_ids: tuple[UUID, ...] = Field(max_length=10000)
    cases: tuple[EvaluationCase, ...] = Field(min_length=20, max_length=100)
    minimum_success_rate: float = Field(ge=0.9, le=1)
    maximum_axis_error_m: float = Field(gt=0, le=0.04)
    maximum_inference_p95_ms: float = Field(gt=0, le=80)
    max_step_seconds: int = Field(strict=True, ge=1, le=30)
    max_cartesian_speed_m_s: float = Field(gt=0, le=0.2)

    @model_validator(mode="after")
    def unique_conditions(self):
        if set(self.seeds) & INTEGRATION_ONLY_SEEDS:
            raise ValueError("Integration-only probes cannot become held-out evaluation cases.")
        if len(set(self.seeds)) != len(self.seeds) or len(set(self.held_out_episode_ids)) != len(
            self.held_out_episode_ids
        ):
            raise ValueError("Held-out conditions must be unique and frozen before training.")
        if tuple(item.seed for item in self.cases) != self.seeds or len(
            {(item.environment_id, item.revision) for item in self.cases}
        ) != len(self.cases):
            raise ValueError("Each held-out seed must bind one distinct immutable scene revision.")
        return self

    @property
    def sha256(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def validate_teaching_partition(cases: tuple[TeachingCase, ...], plan: EvaluationPlan) -> None:
    if not cases:
        return
    if (
        len({item.case_id for item in cases}) != len(cases)
        or len({item.seed for item in cases}) != len(cases)
        or len({(item.environment_id, item.revision) for item in cases}) != len(cases)
    ):
        raise ValueError(
            "Teaching case IDs, scene revisions and seeds must be unique across splits."
        )
    if {item.seed for item in cases} & (set(plan.seeds) | INTEGRATION_ONLY_SEEDS) or {
        (item.environment_id, item.revision) for item in cases
    } & {(item.environment_id, item.revision) for item in plan.cases}:
        raise ValueError(
            "Train/validation cases cannot include held-out or integration-only cases."
        )


class CreateProject(Frozen):
    request_id: UUID
    display_name: str = Field(min_length=1, max_length=120, pattern=r"\S")
    task_id: Identifier
    policy_type: LearnedPolicyType
    instruction: str = Field(min_length=1, max_length=512, pattern=r"^[^\r\n]*\S[^\r\n]*$")
    goal_station_id: Identifier
    environment_id: Identifier
    revision: Revision
    project_kind: Literal["adaptation", "bootstrap"] = "adaptation"
    baseline_release_id: UUID | None
    pretrained_artifact_id: UUID | None = None
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    evaluation_plan: EvaluationPlan
    teaching_cases: tuple[TeachingCase, ...] = Field(min_length=1, max_length=1000)
    budget: Budget

    @model_validator(mode="after")
    def real_training_parent(self):
        validate_teaching_partition(self.teaching_cases, self.evaluation_plan)
        if self.project_kind == "bootstrap":
            if self.baseline_release_id is not None or self.pretrained_artifact_id is None:
                raise ValueError("Bootstrap uses a registered train-only artifact, not a fake P0.")
        elif self.baseline_release_id is None or self.pretrained_artifact_id is not None:
            raise ValueError("Normal adaptation requires an actual reviewed P0 release.")
        return self


class OwnedRecord(Frozen):
    id: UUID
    owner_key: Revision
    actor_id: UUID
    tenant_id: UUID
    created_at: AwareDatetime
    updated_at: AwareDatetime
    fingerprint: Revision

    def public(self) -> dict:
        return self.model_dump(mode="json", exclude={"owner_key", "fingerprint", "tenant_id"})


class LearningProject(OwnedRecord):
    kind: Literal["project"] = "project"
    display_name: str
    task_id: Identifier
    policy_type: LearnedPolicyType
    instruction: str
    goal_station_id: Identifier
    environment_id: Identifier
    revision: Revision
    project_kind: Literal["adaptation", "bootstrap"] = "adaptation"
    baseline_release_id: UUID | None
    pretrained_artifact_id: UUID | None = None
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    evaluation_plan: EvaluationPlan
    teaching_cases: tuple[TeachingCase, ...] = Field(default=(), max_length=1000)
    budget: Budget

    @model_validator(mode="after")
    def frozen_case_partition(self):
        validate_teaching_partition(self.teaching_cases, self.evaluation_plan)
        return self

    def selected_case(self, case_id: str | None) -> TeachingCase:
        if case_id is not None:
            matches = [item for item in self.teaching_cases if item.case_id == case_id]
        else:
            matches = [
                item
                for item in self.teaching_cases
                if (item.environment_id, item.revision) == (self.environment_id, self.revision)
            ]
        if len(matches) != 1:
            raise Problem(
                409,
                "teaching_case_unapproved",
                "Select an immutable approved train/validation case; the anchor is not authority.",
            )
        return matches[0]

    def public(self) -> dict:
        return {**super().public(), "evaluation_plan_sha256": self.evaluation_plan.sha256}

    @classmethod
    def create(cls, actor: Principal, request: CreateProject) -> LearningProject:
        now = utcnow()
        return cls(
            id=request.request_id,
            owner_key=actor.owner_key,
            actor_id=actor.object_id,
            tenant_id=actor.tenant_id,
            created_at=now,
            updated_at=now,
            fingerprint=fingerprint(request.model_dump(mode="json")),
            **request.model_dump(exclude={"request_id"}),
        )


class Approval(Frozen):
    request_id: UUID

    @field_validator("*", mode="before")
    @classmethod
    def explicit_approval(cls, value, info):
        if info.field_name.endswith("_approved") and value is not True:
            raise ValueError("An explicit human approval is required.")
        return value


class StartTeaching(Approval):
    source: Literal["human_teleop", "reference_controller"]
    motion_approved: Literal[True]
    case_id: Identifier | None = None


class JogTeaching(Approval):
    lease_id: UUID
    epoch: UUID
    sequence: PositiveInt
    expires_at: AwareDatetime
    deadman: bool = Field(strict=True)
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["open", "close", "hold"]
    grant_id: UUID | None = None
    grant_expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def bounded_jog(self):
        if math.sqrt(sum(value * value for value in self.delta_xyz_m)) > 0.01:
            raise ValueError("A jog must not exceed 1 cm in total Cartesian displacement.")
        if not self.deadman and (any(self.delta_xyz_m) or self.gripper != "hold"):
            raise ValueError("Released deadman permits zero-motion hold only.")
        return self

    def check_time(self, now: datetime) -> None:
        seconds = (self.expires_at - now).total_seconds()
        if not 0 < seconds <= 0.25:
            raise Problem(409, "expired_input", "Teaching input must expire within 250 ms.")
        if self.deadman and (
            self.grant_id is None
            or self.grant_expires_at is None
            or not 0 < (self.grant_expires_at - now).total_seconds() <= 1
            or self.expires_at > self.grant_expires_at
        ):
            raise Problem(
                409, "teaching_grant_expired", "A fresh server-issued input grant is required."
            )


class ArmTeaching(Approval):
    lease_id: UUID
    epoch: UUID
    sequence: PositiveInt
    deadman: Literal[True]
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["open", "close", "hold"]

    @field_validator("deadman", mode="before")
    @classmethod
    def real_deadman(cls, value):
        if value is not True:
            raise ValueError("Explicit deadman authority is required.")
        return value

    @model_validator(mode="after")
    def bounded_arm(self):
        if math.sqrt(sum(x * x for x in self.delta_xyz_m)) > 0.01:
            raise ValueError("Only a bounded 1 cm input can be armed.")
        return self


class JogIntent(Approval):
    lease_id: UUID
    epoch: UUID
    sequence: PositiveInt
    deadman: bool = Field(strict=True)
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["open", "close", "hold"]
    grant_id: UUID | None = None

    @model_validator(mode="after")
    def bounded_intent(self):
        if math.sqrt(sum(x * x for x in self.delta_xyz_m)) > 0.01:
            raise ValueError("At most 1 cm movement is allowed.")
        if self.deadman and self.grant_id is None:
            raise ValueError("Motion requires a server-issued control grant.")
        if not self.deadman and (any(self.delta_xyz_m) or self.gripper != "hold" or self.grant_id):
            raise ValueError("Released deadman permits zero-motion hold with no grant.")
        return self


class ControlGrant(OwnedRecord):
    kind: Literal["control_grant"] = "control_grant"
    session_id: UUID
    lease_id: UUID
    epoch: UUID
    sequence: PositiveInt
    delta_xyz_m: tuple[float, float, float]
    gripper: Literal["open", "close", "hold"]
    expires_at: AwareDatetime
    consumed_by: UUID | None = None


class TeachingControl(Approval):
    lease_id: UUID
    epoch: UUID


class CaptureReceipt(Frozen):
    episode_id: UUID
    manifest_sha256: Revision
    artifact_id: UUID
    frame_count: int = Field(strict=True, ge=2)
    source: SourceKind
    seed: int = Field(strict=True, ge=0)
    task_id: Identifier
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    source_model_sha256: Revision | None = None
    case_id: Identifier | None = None
    environment_id: Identifier | None = None
    revision: Revision | None = None
    split: Literal["train", "validation"] | None = None

    def authorized_case(self, project: LearningProject) -> TeachingCase:
        if self.case_id is None:
            raise Problem(
                409, "capture_case_unverified", "Capture lacks an approved case and split."
            )
        case = project.selected_case(self.case_id)
        if (
            (self.environment_id, self.revision, self.seed, self.split)
            != (case.environment_id, case.revision, case.seed, case.split)
            or self.task_id != project.task_id
            or self.control_profile_id != project.control_profile_id
        ):
            raise Problem(
                409, "capture_case_mismatch", "Capture scene, split, task or profile differs."
            )
        return case

    @model_validator(mode="after")
    def source_is_evidenced(self):
        if (self.source == "learned") != (self.source_model_sha256 is not None):
            raise ValueError("Generated demonstrations must identify their actual source model.")
        return self


class TeachingSession(OwnedRecord):
    kind: Literal["teaching"] = "teaching"
    project_id: UUID
    teaching_case: TeachingCase | None = None
    source: Literal["human_teleop", "reference_controller"]
    status: TeachingStatus
    lease_id: UUID
    epoch: UUID
    command_id: UUID
    expires_at: AwareDatetime
    last_sequence: NonnegativeInt = 0
    last_input_fingerprint: Revision | None = None
    input_expires_at: AwareDatetime | None = None
    capture: CaptureReceipt | None = None
    error_code: str | None = None
    message: str | None = None
    physical_status: (
        Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out"]
        | None
    ) = None

    @model_validator(mode="after")
    def ready_is_uploaded(self):
        if self.status == "ready" and self.capture is None:
            raise ValueError("A ready teaching session requires verified uploaded capture.")
        return self


class CreateDataset(Approval):
    teaching_session_ids: tuple[UUID, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_sessions(self):
        if len(set(self.teaching_session_ids)) != len(self.teaching_session_ids):
            raise ValueError("A teaching session may occur only once in a dataset.")
        return self


class DatasetVersion(OwnedRecord):
    kind: Literal["dataset"] = "dataset"
    project_id: UUID
    status: Literal["ready"] = "ready"
    artifact_id: UUID
    manifest_sha256: Revision
    episode_ids: tuple[UUID, ...] = Field(min_length=1, max_length=1000)
    seeds: tuple[NonnegativeInt, ...] = Field(min_length=1, max_length=1000)
    human_teleop_count: NonnegativeInt
    reference_controller_count: NonnegativeInt
    learned_policy_count: NonnegativeInt
    evaluation_plan_sha256: Revision
    captures: tuple[CaptureReceipt, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def counts_match_manifest(self):
        count = len(self.episode_ids)
        if (
            count != len(set(self.episode_ids))
            or count != len(self.seeds)
            or count
            != (
                self.human_teleop_count
                + self.reference_controller_count
                + self.learned_policy_count
            )
        ):
            raise ValueError("Dataset counts must preserve all actual source provenance.")
        if self.captures and (
            tuple(item.episode_id for item in self.captures) != self.episode_ids
            or tuple(item.seed for item in self.captures) != self.seeds
            or sum(item.source == "human_teleop" for item in self.captures)
            != self.human_teleop_count
            or sum(item.source == "reference_controller" for item in self.captures)
            != self.reference_controller_count
            or sum(item.source == "learned" for item in self.captures) != self.learned_policy_count
        ):
            raise ValueError(
                "Dataset episode order, source and split provenance must match receipts."
            )
        return self


class StartTraining(Approval):
    dataset_id: UUID
    parent_release_id: UUID | None
    pretrained_artifact_id: UUID | None = None
    policy_type: LearnedPolicyType
    optimizer_steps: int = Field(strict=True, ge=1, le=100000)
    paid_approved: Literal[True]
    maximum_cost_usd: Decimal = Field(gt=0, le=10000, max_digits=9, decimal_places=2)


class StartEvaluation(Approval):
    candidate_id: UUID
    baseline_release_id: UUID | None
    comparison_kind: Literal["paired_policy", "reference_bootstrap"] = "paired_policy"
    evaluation_plan_sha256: Revision
    motion_approved: Literal[True]
    paid_approved: Literal[True]
    maximum_cost_usd: Decimal = Field(gt=0, le=10000, max_digits=9, decimal_places=2)


class TrainingMetrics(Frozen):
    optimizer_steps: NonnegativeInt | None = None
    loss: float | None = Field(default=None, ge=0)
    measured_at: AwareDatetime | None = None


class PolicyCandidate(OwnedRecord):
    kind: Literal["candidate"] = "candidate"
    project_id: UUID
    dataset_id: UUID
    training_run_id: UUID
    parent_release_id: UUID | None
    pretrained_artifact_id: UUID | None = None
    policy_type: PolicyType
    model_sha256: Revision
    parent_model_sha256: Revision
    processor_sha256: Revision
    manifest_sha256: Revision
    artifact_id: UUID
    optimizer_steps: PositiveInt
    azure_job_id: str = Field(min_length=1, max_length=2048)
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    control_profile_id: Literal["franka-position-hold-10hz-v1"]

    @model_validator(mode="after")
    def actual_new_weights(self):
        if self.model_sha256 == self.parent_model_sha256:
            raise ValueError("Training requires changed weights, not a relabeled checkpoint.")
        return self


class TrialOutcome(Frozen):
    seed: NonnegativeInt
    environment_id: Identifier
    revision: Revision
    attempt: PositiveInt
    policy: Literal["before", "after"]
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    model_sha256: Revision
    physical_success: bool
    axis_error_m: tuple[float, float, float] | None
    duration_seconds: float = Field(ge=0)
    safety_violations: NonnegativeInt
    inference_p95_ms: float | None = Field(default=None, ge=0)
    applied_action_count: NonnegativeInt
    policy_predict_calls: NonnegativeInt
    reference_route_calls: Literal[0]
    observed_initial_pose_m: tuple[float, float, float] | None = None
    scene_builder_sha256: Revision | None = None
    episode_id: str | None = Field(default=None, min_length=1, max_length=128)
    final_inspection_sha256: Revision | None = None
    final_overview_sha256: Revision | None = None
    recording_id: UUID | None = None
    message: str


class PairedReport(Frozen):
    comparison_kind: Literal["paired_policy"] = "paired_policy"
    evaluation_plan_sha256: Revision
    before_model_sha256: Revision
    after_model_sha256: Revision
    trials: tuple[TrialOutcome, ...] = Field(min_length=40, max_length=1000)
    conclusion: Literal["improved", "not_improved", "inconclusive"]
    quality_gate_passed: bool
    report_sha256: Revision
    artifact_id: UUID
    native_plan_sha256: Revision | None = None
    runtime_sha256: Revision | None = None
    control_profile_sha256: Revision | None = None


class BootstrapTrial(TrialOutcome):
    policy: Literal["reference", "candidate"]
    model_sha256: Revision | None
    reference_route_calls: NonnegativeInt

    @model_validator(mode="after")
    def reference_is_not_a_model(self):
        if self.policy == "reference":
            if (
                self.model_sha256 is not None
                or self.policy_predict_calls
                or self.applied_action_count
            ):
                raise ValueError("Scripted reference is not a learned P0 or model prediction.")
        elif self.model_sha256 is None or self.reference_route_calls:
            raise ValueError("A candidate uses actual model actions, never reference fallback.")
        return self


class BootstrapReport(Frozen):
    comparison_kind: Literal["reference_bootstrap"] = "reference_bootstrap"
    evaluation_plan_sha256: Revision
    candidate_model_sha256: Revision
    reference_controller_sha256: Revision
    trials: tuple[BootstrapTrial, ...] = Field(min_length=40, max_length=1000)
    quality_gate_passed: bool
    report_sha256: Revision
    artifact_id: UUID
    native_plan_sha256: Revision | None = None
    runtime_sha256: Revision | None = None
    control_profile_sha256: Revision | None = None


class TrainingParent(OwnedRecord):
    kind: Literal["training_parent"] = "training_parent"
    role: Literal["pretrained_train_only"] = "pretrained_train_only"
    policy_type: LearnedPolicyType
    artifact_id: UUID
    model_sha256: Revision
    processor_sha256: Revision
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    registered_by: UUID


class LearningJob(OwnedRecord):
    kind: Literal["training", "evaluation"]
    project_id: UUID
    policy_type: LearnedPolicyType = "gr00t_n1_5"
    status: JobStatus
    backend_job_name: str = Field(pattern=r"^learning-[a-f0-9-]+$", max_length=100)
    azure_job_id: str | None = Field(default=None, min_length=1, max_length=2048)
    deadline: AwareDatetime
    approved_cost_usd: Decimal
    specification_sha256: Revision
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    error_code: str | None = None
    message: str | None = None


class LearningMutation(OwnedRecord):
    kind: Literal["mutation"] = "mutation"
    resource_kind: Literal["teaching", "training", "evaluation"]
    resource_id: UUID
    operation: Literal["jog", "finish", "cancel"]
    status: Literal["claimed", "recorded", "uncertain", "rejected"] = "claimed"
    server_expires_at: AwareDatetime | None = None
    error_code: str | None = None


class TrainingRun(LearningJob):
    kind: Literal["training"] = "training"
    dataset_id: UUID
    parent_release_id: UUID | None
    pretrained_artifact_id: UUID | None = None
    optimizer_steps: PositiveInt
    candidate_id: UUID | None = None

    @model_validator(mode="after")
    def succeeded_requires_artifact(self):
        if self.status == "succeeded" and (not self.azure_job_id or not self.candidate_id):
            raise ValueError("A completed training run must reference a verified new candidate.")
        return self


class EvaluationRun(LearningJob):
    kind: Literal["evaluation"] = "evaluation"
    candidate_id: UUID
    baseline_release_id: UUID | None
    comparison_kind: Literal["paired_policy", "reference_bootstrap"] = "paired_policy"
    evaluation_plan_sha256: Revision
    report: PairedReport | BootstrapReport | None = None

    @model_validator(mode="after")
    def succeeded_requires_report(self):
        if self.status == "succeeded" and (not self.azure_job_id or not self.report):
            raise ValueError("A completed evaluation requires the entire paired trial report.")
        return self


class ReleasePolicy(Approval):
    candidate_id: UUID
    evaluation_run_id: UUID
    release_approved: Literal[True]


class CoachRequest(Approval):
    instruction: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    dataset_id: UUID | None = None
    evaluation_run_id: UUID | None = None


class CoachProposal(Frozen):
    project_id: UUID
    action: Literal[
        "define_task",
        "review_demonstrations",
        "propose_training",
        "explain_evaluation",
        "select_release",
    ]
    summary: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    dataset_id: UUID | None
    optimizer_steps: int | None = Field(ge=1, le=100000)
    selected_release_id: UUID | None


class CoachRecord(OwnedRecord):
    kind: Literal["coach"] = "coach"
    project_id: UUID
    status: Literal["planning", "recorded", "failed"]
    model_response_id: str | None = None
    proposal: CoachProposal | None = None
    error_code: str | None = None
    message: str | None = None


class PolicyRelease(OwnedRecord):
    kind: Literal["release"] = "release"
    project_id: UUID
    candidate_id: UUID
    evaluation_run_id: UUID
    policy_type: PolicyType
    model_sha256: Revision
    processor_sha256: Revision
    manifest_sha256: Revision
    artifact_id: UUID
    environment_id: Identifier
    revision: Revision
    task_id: Identifier
    goal_station_id: Identifier
    instruction: str = Field(min_length=1, max_length=512, pattern=r"^[^\r\n]*\S[^\r\n]*$")
    control_profile_id: Literal["franka-position-hold-10hz-v1"]
    evaluation_plan_sha256: Revision
    reviewed_by: UUID
    comparison_kind: Literal["paired_policy", "reference_bootstrap"] = "paired_policy"
    environment_cases: tuple[PolicyScene, ...] = ()


LearningRecord = (
    LearningProject
    | TeachingSession
    | DatasetVersion
    | TrainingRun
    | EvaluationRun
    | PolicyCandidate
    | PolicyRelease
    | LearningMutation
    | ControlGrant
    | TrainingParent
    | CoachRecord
)

_TRANSITIONS = {
    "job": {
        "submitting": {"submitted", "submission_unknown", "blocked", "failed", "cancelling"},
        "submission_unknown": {"submitted", "running", "cancelling", "blocked"},
        "submitted": {"running", "cancelling", "succeeded", "failed", "cancelled", "timed_out"},
        "running": {"cancelling", "succeeded", "failed", "cancelled", "timed_out"},
        "cancelling": {"cancelled", "failed", "timed_out", "succeeded"},
    },
    "teaching": {
        "starting": {"recording", "blocked", "invalid", "cancelling"},
        "recording": {"finishing", "finalizing", "cancelling", "invalid"},
        "finishing": {"finalizing", "cancelling", "invalid"},
        "finalizing": {"uploading", "invalid", "cancelling"},
        "uploading": {"ready", "invalid", "cancelling"},
        "cancelling": {"cancelled", "invalid"},
    },
}


def transition(kind: Literal["job", "teaching"], current: str, target: str) -> str:
    if current == target or target in _TRANSITIONS[kind].get(current, set()):
        return target
    raise Problem(409, "invalid_learning_transition", f"Cannot transition {current} to {target}.")


def replace_record(record: LearningRecord, **changes) -> LearningRecord:
    return type(record).model_validate({**record.model_dump(), **changes})
