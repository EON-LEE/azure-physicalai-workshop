from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import (
    Budget,
    CreateProject,
    DatasetVersion,
    EvaluationPlan,
    JogTeaching,
    LearningProject,
    TrainingMetrics,
    transition,
)
from apps.api.models import utcnow
from tests.runtime_support import ACTOR


def plan():
    return EvaluationPlan(
        id=uuid4(),
        seeds=tuple(range(200, 220)),
        held_out_episode_ids=(),
        cases=tuple(
            {
                "seed": seed,
                "environment_id": f"held-out-{seed}",
                "revision": f"{seed:064x}",
            }
            for seed in range(200, 220)
        ),
        minimum_success_rate=0.9,
        maximum_axis_error_m=0.04,
        maximum_inference_p95_ms=80,
        max_step_seconds=30,
        max_cartesian_speed_m_s=0.2,
    )


def budget():
    return Budget(
        teaching_seconds=120,
        training_seconds=3600,
        evaluation_seconds=1800,
        optimizer_steps=100,
        maximum_cost_usd="10.00",
    )


def project_request():
    return CreateProject(
        request_id=uuid4(),
        display_name="제조 부품 키팅 학습",
        task_id="part-kitting-v1",
        policy_type="gr00t_n1_5",
        instruction="부품을 승인된 트레이에 놓습니다.",
        goal_station_id="accepted",
        environment_id="reference-cell",
        revision="a" * 64,
        baseline_release_id=uuid4(),
        control_profile_id="franka-position-hold-10hz-v1",
        evaluation_plan=plan(),
        budget=budget(),
    )


def test_project_request_cannot_supply_owner_urls_python_or_raw_joint_authority():
    request = project_request().model_dump(mode="json")
    for key, value in (
        ("owner_key", "b" * 64),
        ("tenant_id", str(uuid4())),
        ("model_uri", "https://example.test/model"),
        ("dataset_uri", "https://example.test/data"),
        ("python", "print('not allowed')"),
        ("joint_positions", [0] * 9),
    ):
        with pytest.raises(ValidationError):
            CreateProject.model_validate({**request, key: value})


def test_project_and_all_pins_are_immutable_and_owner_bound():
    request = project_request()
    record = LearningProject.create(ACTOR, request)
    assert record.owner_key == ACTOR.owner_key
    assert record.actor_id == ACTOR.object_id
    assert record.id == request.request_id
    assert record.evaluation_plan.sha256 == request.evaluation_plan.sha256
    assert len(record.fingerprint) == 64
    with pytest.raises(ValidationError):
        record.instruction = "silently replace the task"
    with pytest.raises(ValidationError):
        record.budget.optimizer_steps = 1000
    assert isinstance(record.evaluation_plan.seeds, tuple)


@pytest.mark.parametrize("count", [0, 1, 19, 101])
def test_evaluation_plan_cannot_reduce_the_paired_held_out_gate(count):
    fields = plan().model_dump()
    fields["seeds"] = tuple(range(count))
    with pytest.raises(ValidationError):
        EvaluationPlan.model_validate(fields)


def test_evaluation_plan_rejects_duplicate_conditions_and_relaxed_safety_bounds():
    fields = plan().model_dump()
    for changes in (
        {"seeds": tuple([200] * 20)},
        {"maximum_axis_error_m": 0.041},
        {"max_step_seconds": 31},
        {"max_cartesian_speed_m_s": 0.21},
        {"minimum_success_rate": 0.89},
        {"maximum_inference_p95_ms": 81},
    ):
        with pytest.raises(ValidationError):
            EvaluationPlan.model_validate({**fields, **changes})


def test_dataset_requires_a_content_digest_and_split_source_provenance():
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate(
            {
                "id": str(uuid4()),
                "owner_key": ACTOR.owner_key,
                "actor_id": str(ACTOR.object_id),
                "project_id": str(uuid4()),
                "created_at": utcnow(),
                "updated_at": utcnow(),
                "fingerprint": "a" * 64,
                "status": "ready",
                "episode_ids": [],
                "manifest_sha256": None,
                "human_teleop_count": 1,
                "reference_controller_count": 0,
                "learned_policy_count": 0,
            }
        )


def test_jog_requires_deadman_short_expiry_and_bounded_cartesian_commands():
    now = utcnow()
    body = {
        "request_id": uuid4(),
        "lease_id": uuid4(),
        "epoch": uuid4(),
        "sequence": 1,
        "expires_at": now + timedelta(milliseconds=200),
        "deadman": True,
        "grant_id": uuid4(),
        "grant_expires_at": now + timedelta(seconds=1),
        "delta_xyz_m": (0.005, 0.0, 0.0),
        "gripper": "hold",
    }
    request = JogTeaching.model_validate(body)
    request.check_time(now)
    for invalid in (
        {"deadman": False},
        {"delta_xyz_m": (0.011, 0.0, 0.0)},
        {"delta_xyz_m": (0.008, 0.008, 0.0)},
        {"sequence": 0},
        {"joints": [0] * 9},
        {"delta_xyz_m": (float("nan"), 0, 0)},
    ):
        with pytest.raises(ValidationError):
            JogTeaching.model_validate({**body, **invalid})
    for expires in (now, now + timedelta(milliseconds=251)):
        with pytest.raises(Problem) as failure:
            JogTeaching.model_validate({**body, "expires_at": expires}).check_time(now)
        assert failure.value.status == 409


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("submitting", "submission_unknown"),
        ("submission_unknown", "submitted"),
        ("submitted", "running"),
        ("running", "cancelling"),
        ("cancelling", "cancelled"),
        ("running", "succeeded"),
        ("running", "failed"),
        ("running", "timed_out"),
    ],
)
def test_job_transitions_preserve_truthful_submission_and_cancellation(current, target):
    assert transition("job", current, target) == target


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("succeeded", "running"),
        ("cancelled", "succeeded"),
        ("failed", "submitted"),
        ("timed_out", "running"),
        ("submission_unknown", "submitting"),
        ("submitting", "succeeded"),
    ],
)
def test_job_transitions_never_revive_terminal_runs_or_retry_uncertain_paid_submission(
    current, target
):
    with pytest.raises(Problem) as failure:
        transition("job", current, target)
    assert failure.value.status == 409


def test_capture_state_is_not_physical_or_training_success():
    assert transition("teaching", "recording", "finalizing") == "finalizing"
    assert transition("teaching", "finalizing", "uploading") == "uploading"
    assert transition("teaching", "uploading", "ready") == "ready"
    for current, target in (("recording", "ready"), ("cancelled", "ready"), ("invalid", "ready")):
        with pytest.raises(Problem):
            transition("teaching", current, target)


def test_training_metrics_never_invent_progress_or_accept_nonfinite_loss():
    assert TrainingMetrics().optimizer_steps is None
    assert TrainingMetrics().loss is None
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValidationError):
            TrainingMetrics(optimizer_steps=1, loss=value)
