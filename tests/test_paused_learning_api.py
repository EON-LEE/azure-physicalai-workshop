from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_api import api as api
from test_api import headers

from apps.api.learning_models import (
    CreateProject,
    LearningProject,
    StartEvaluation,
    StartTeaching,
    fingerprint,
)
from tests.runtime_support import ACTOR
from tests.test_learning_lifecycle_contracts import project_request
from tests.test_learning_service import setup

TIMING_FIELDS = {
    "execution_timing",
    "real_time_admission",
    "control_profile_sha256",
    "criteria_sha256",
    "frozen_plan_sha256",
}


def paused_project_payload():
    value = project_request().model_dump(mode="json")
    value.update(
        policy_type="smolvla",
        control_profile_id="franka-position-hold-10hz-paused-v1",
        execution_timing="paused_simulation",
        real_time_admission=False,
        control_profile_sha256="b" * 64,
        criteria_sha256="c" * 64,
        frozen_plan_sha256="d" * 64,
    )
    plan = value["evaluation_plan"]
    plan.pop("maximum_inference_p95_ms")
    plan.pop("max_step_seconds")
    plan.update(
        execution_timing="paused_simulation",
        real_time_admission=False,
        max_simulation_seconds=30,
        max_wall_seconds=600,
        max_observation_wall_ms=2000,
        max_policy_wall_ms=2000,
        max_hold_wall_ms=2000,
        max_interval_wall_ms=5000,
        max_heartbeat_wall_ms=2000,
        minimum_absolute_improvement=0.05,
    )
    value["budget"]["evaluation_seconds"] = 7200
    return value


def test_new_paused_project_retains_distinct_wall_simulation_and_frozen_conditions():
    payload = paused_project_payload()
    request = CreateProject.model_validate(payload)
    project = LearningProject.create(ACTOR, request)
    assert project.execution_timing == "paused_simulation"
    assert project.real_time_admission is False
    assert project.control_profile_sha256 == payload["control_profile_sha256"]
    assert project.criteria_sha256 == payload["criteria_sha256"]
    assert project.frozen_plan_sha256 == payload["frozen_plan_sha256"]
    assert project.budget.evaluation_seconds == 7200
    plan = project.evaluation_plan.model_dump(mode="json")
    assert plan["max_simulation_seconds"] == 30 and plan["max_wall_seconds"] == 600
    assert "maximum_inference_p95_ms" not in plan and "max_step_seconds" not in plan
    assert project.evaluation_plan.sha256 != project.frozen_plan_sha256
    assert project.public()["real_time_admission"] is False


@pytest.mark.parametrize("field", sorted(TIMING_FIELDS))
def test_paused_project_cannot_default_any_authority_or_provenance_pin(field):
    value = paused_project_payload()
    value.pop(field)
    with pytest.raises(ValidationError):
        CreateProject.model_validate(value)


@pytest.mark.parametrize(
    "change",
    [
        {"real_time_admission": True},
        {"real_time_admission": 0},
        {"execution_timing": "realtime"},
        {"timing_mode": "paused_simulation"},
        {"control_profile_id": "franka-position-hold-10hz-v1"},
        {"policy_type": "gr00t_n1_5"},
    ],
)
def test_new_mode_cannot_relabel_old_policy_profile_or_claim_realtime(change):
    with pytest.raises(ValidationError):
        CreateProject.model_validate(paused_project_payload() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"max_wall_seconds": 601},
        {"max_simulation_seconds": 31},
        {"max_observation_wall_ms": 2001},
        {"max_policy_wall_ms": 2001},
        {"max_interval_wall_ms": 5001},
        {"max_heartbeat_wall_ms": 2001},
        {"maximum_axis_error_m": 0.041},
        {"minimum_success_rate": 0.89},
        {"minimum_absolute_improvement": 0.04},
        {"maximum_inference_p95_ms": 80},
        {"real_time_admission": True},
    ],
)
def test_paused_plan_cannot_relax_frozen_caps_or_mix_rt_gate_fields(change):
    value = paused_project_payload()
    value["evaluation_plan"].update(change)
    with pytest.raises(ValidationError):
        CreateProject.model_validate(value)


def test_legacy_project_dump_and_fingerprint_have_no_new_mode_defaults():
    request = project_request()
    original = request.model_dump(mode="json")
    assert not TIMING_FIELDS & original.keys()
    project = LearningProject.create(ACTOR, request)
    assert project.fingerprint == fingerprint(original)
    assert not TIMING_FIELDS & project.model_dump(mode="json").keys()
    assert project.evaluation_plan.maximum_inference_p95_ms == 80
    assert project.budget.evaluation_seconds == 1800
    value = paused_project_payload()
    value["evaluation_plan"] = original["evaluation_plan"]
    with pytest.raises(ValidationError):
        CreateProject.model_validate(value)


def test_total_evaluation_wall_budget_does_not_change_the_per_episode_caps():
    value = paused_project_payload()
    value["budget"]["evaluation_seconds"] = 21600
    request = CreateProject.model_validate(value)
    assert request.evaluation_plan.max_wall_seconds == 600
    value["budget"]["evaluation_seconds"] = 21601
    with pytest.raises(ValidationError):
        CreateProject.model_validate(value)


def test_paused_mutations_remain_blocked_without_integrated_producers_and_verifiers(api):
    client, _, token = api
    service, store, jobs, _ = setup()
    client.app.state.learning = service
    value = paused_project_payload()
    response = client.post("/api/learning/projects", json=value, headers=headers(token))
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "paused_learning_unavailable"
    assert jobs.submissions == []
    assert store.get_learning(ACTOR.owner_key, "project", value["request_id"]) is None


@pytest.mark.parametrize(
    "reference,training,admitted",
    [(False, True, True), (True, False, True), (True, True, True), (False, False, False)],
)
def test_project_metadata_admission_is_separate_from_motion_permissions(
    monkeypatch, reference, training, admitted
):
    from apps.api.errors import Problem

    service, store, jobs, _ = setup()
    service.reference_collections_enabled = reference
    service.paused_training_enabled = training
    body = CreateProject.model_validate(paused_project_payload())
    project = LearningProject.create(ACTOR, body)

    class SceneValidationReached(Exception):
        pass

    def validate_scene(*args, **kwargs):
        raise SceneValidationReached

    monkeypatch.setattr(service, "_saved_scene", validate_scene)
    if admitted:
        with pytest.raises(SceneValidationReached):
            service.create_project(ACTOR, body)
    else:
        with pytest.raises(Problem) as failure:
            service.create_project(ACTOR, body)
        assert failure.value.code == "paused_learning_unavailable"
    if not reference:
        with pytest.raises(Problem) as failure:
            service._timing_admission(project, "reference")
        assert failure.value.code == "paused_learning_unavailable"
    if not training:
        with pytest.raises(Problem) as failure:
            service._timing_admission(project, "training")
        assert failure.value.code == "paused_learning_unavailable"
    assert jobs.submissions == []
    assert store.get_learning(ACTOR.owner_key, "project", project.id) is None


def test_capability_does_not_treat_native_type_support_as_model_or_runtime_admission():
    service, _, _, _ = setup()
    capability = service.capabilities(ACTOR)
    paused = capability["simulation_learning"]
    assert paused["execution_timing"] == "paused_simulation"
    assert paused["real_time_admission"] is False
    assert paused["enabled"] is False and paused["status"] == "producer_verifier_unavailable"
    assert "franka-position-hold-10hz-paused-v1" not in capability["control_profiles"]


def test_injected_paused_project_cannot_reach_existing_rt_motion_or_model_paths():
    from apps.api.errors import Problem

    service, store, jobs, _ = setup()
    project = LearningProject.create(ACTOR, CreateProject.model_validate(paused_project_payload()))
    saved = store.put_learning(ACTOR.owner_key, project, None)
    with pytest.raises(Problem) as teaching:
        service.start_teaching(
            ACTOR,
            project.id,
            StartTeaching(request_id=uuid4(), source="human_teleop", motion_approved=True),
            saved.etag,
        )
    assert teaching.value.code == "paused_learning_unavailable"
    with pytest.raises(Problem) as evaluation:
        service.evaluate(
            ACTOR,
            project.id,
            StartEvaluation(
                request_id=uuid4(),
                candidate_id=uuid4(),
                baseline_release_id=project.baseline_release_id,
                evaluation_plan_sha256=project.evaluation_plan.sha256,
                motion_approved=True,
                paid_approved=True,
                maximum_cost_usd="10",
            ),
            saved.etag,
        )
    assert evaluation.value.code == "paused_learning_unavailable"
    assert jobs.submissions == []


def test_internal_worker_cannot_admit_paused_project_through_legacy_model_allowlist():
    from apps.api.errors import Problem
    from apps.learning_worker.backend import PolicyLearningWorker
    from tests.test_learning_worker import specification

    original, _ = specification()
    project = LearningProject.create(ACTOR, CreateProject.model_validate(paused_project_payload()))
    spec = original.model_copy(
        update={
            "project": project,
            "run": original.run.model_copy(
                update={"project_id": project.id, "policy_type": "smolvla"}
            ),
        }
    )

    class NoAccess:
        def __getattr__(self, name):
            pytest.fail("Unadmitted paused mode must not reach model/registry/Azure clients.")

    worker = PolicyLearningWorker(
        NoAccess(), NoAccess(), uuid4(), allowed_policy_types=("smolvla",)
    )
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, spec)
    assert failure.value.code == "paused_learning_unavailable"
