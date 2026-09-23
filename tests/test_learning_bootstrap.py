import json
from uuid import uuid4

import pytest
from test_api import api as api
from test_api import headers

from apps.api.learning_models import (
    BootstrapReport,
    BootstrapTrial,
    CaptureReceipt,
    CreateProject,
    DatasetVersion,
    PolicyCandidate,
    ReleasePolicy,
    StartEvaluation,
    StartTraining,
    TrainingParent,
    fingerprint,
)
from apps.api.learning_service import LearningService
from apps.api.models import SaveEnvironment, utcnow
from tests.learning_api_support import learning_setup
from tests.runtime_support import ACTOR


def bootstrap_body(request):
    return {
        **request.model_dump(mode="json"),
        "project_kind": "bootstrap",
        "baseline_release_id": None,
        "pretrained_artifact_id": str(uuid4()),
    }


def test_empty_store_bootstrap_is_not_available_to_an_ordinary_operator(api):
    client, _, token = api
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    client.app.state.learning = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, allowed_policy_types=("gr00t_n1_5",)
    )
    response = client.post(
        "/api/learning/projects", json=bootstrap_body(request), headers=headers(token)
    )
    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "bootstrap_forbidden"
    assert store.list_learning(ACTOR.owner_key, "project") == []
    assert jobs.submissions == []


def test_authorized_bootstrap_still_requires_a_verified_registered_train_only_artifact(api):
    client, _, token = api
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    backend = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
        bootstrap_principal_ids=frozenset({ACTOR.object_id}),
        allowed_policy_types=("gr00t_n1_5",),
    )
    client.app.state.learning = backend
    response = client.post(
        "/api/learning/projects", json=bootstrap_body(request), headers=headers(token)
    )
    assert response.status_code in (404, 503), response.text
    assert store.list_learning(ACTOR.owner_key, "release") == []
    assert jobs.submissions == []


def test_normal_customer_project_never_relabels_train_only_vendor_weights_as_a_policy(api):
    client, _, token = api
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    client.app.state.learning = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, allowed_policy_types=("gr00t_n1_5",)
    )
    body = bootstrap_body(request)
    body["project_kind"] = "adaptation"
    response = client.post("/api/learning/projects", json=body, headers=headers(token))
    assert response.status_code == 422
    assert jobs.submissions == []


def test_empty_release_catalog_can_train_evaluate_then_explicitly_initialize_real_p0():
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    now = utcnow()
    parent = TrainingParent(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="a" * 64,
        artifact_id=uuid4(),
        model_sha256="b" * 64,
        processor_sha256="c" * 64,
        policy_type="gr00t_n1_5",
        source_commit="d" * 40,
        model_revision="e" * 40,
        registered_by=ACTOR.object_id,
    )
    catalog.training_parent = lambda actor, artifact_id: parent
    body = CreateProject.model_validate(
        {
            **request.model_dump(),
            "project_kind": "bootstrap",
            "baseline_release_id": None,
            "pretrained_artifact_id": parent.id,
        }
    )
    base = factory.environment(ACTOR, request.environment_id).value
    evaluated_cases = []
    for case in body.evaluation_plan.cases:
        document = {
            **base.document,
            "environment_id": case.environment_id,
            "scene": {
                **base.document["scene"],
                "template_id": "inspection-cell-learning-v1",
                "seed": case.seed,
            },
            "execution": {**base.document["execution"], "demonstration_split": "test"},
        }
        saved = factory.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(document)))
        evaluated_cases.append(case.model_copy(update={"revision": saved.revision}))
    body = body.model_copy(
        update={
            "evaluation_plan": body.evaluation_plan.model_copy(
                update={"cases": tuple(evaluated_cases)}
            ),
        }
    )
    service = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
        bootstrap_principal_ids=frozenset({ACTOR.object_id}),
        allowed_policy_types=("gr00t_n1_5",),
    )
    project = service.create_project(ACTOR, body)
    assert project.value.baseline_release_id is None
    assert store.list_learning(ACTOR.owner_key, "release") == []
    case = project.value.teaching_cases[0]
    capture = CaptureReceipt(
        episode_id=uuid4(),
        artifact_id=uuid4(),
        manifest_sha256="2" * 64,
        frame_count=20,
        source="reference_controller",
        seed=case.seed,
        task_id=project.value.task_id,
        control_profile_id=project.value.control_profile_id,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        split=case.split,
    )
    dataset = DatasetVersion(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="a" * 64,
        project_id=project.value.id,
        artifact_id=uuid4(),
        manifest_sha256="2" * 64,
        episode_ids=(capture.episode_id,),
        seeds=(42,),
        captures=(capture,),
        human_teleop_count=0,
        reference_controller_count=1,
        learned_policy_count=0,
        evaluation_plan_sha256=project.value.evaluation_plan.sha256,
    )
    store.put_learning(ACTOR.owner_key, dataset, None)
    training = service.train(
        ACTOR,
        project.value.id,
        StartTraining(
            request_id=uuid4(),
            dataset_id=dataset.id,
            parent_release_id=None,
            pretrained_artifact_id=parent.id,
            optimizer_steps=100,
            paid_approved=True,
            policy_type="gr00t_n1_5",
            maximum_cost_usd="10.00",
        ),
        project.etag,
    )
    assert jobs.submissions[0].baseline is None
    assert jobs.submissions[0].training_parent.role == "pretrained_train_only"
    candidate = PolicyCandidate(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="3" * 64,
        project_id=project.value.id,
        dataset_id=dataset.id,
        training_run_id=training.value.id,
        parent_release_id=None,
        pretrained_artifact_id=parent.id,
        policy_type="gr00t_n1_5",
        model_sha256="4" * 64,
        parent_model_sha256=parent.model_sha256,
        processor_sha256="5" * 64,
        manifest_sha256="4" * 64,
        artifact_id=uuid4(),
        optimizer_steps=100,
        azure_job_id=training.value.azure_job_id,
        source_commit=parent.source_commit,
        model_revision=parent.model_revision,
        control_profile_id=project.value.control_profile_id,
    )
    receipt = jobs.receipts[training.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(
        update={"status": "succeeded", "candidate": candidate}
    )
    service.get_job(ACTOR, training.value.id)
    assert store.list_learning(ACTOR.owner_key, "release") == []
    evaluated = service.evaluate(
        ACTOR,
        project.value.id,
        StartEvaluation(
            request_id=uuid4(),
            candidate_id=candidate.id,
            baseline_release_id=None,
            comparison_kind="reference_bootstrap",
            evaluation_plan_sha256=project.value.evaluation_plan.sha256,
            paid_approved=True,
            motion_approved=True,
            maximum_cost_usd="10.00",
        ),
        project.etag,
    )
    release = ReleasePolicy(
        request_id=uuid4(),
        candidate_id=candidate.id,
        evaluation_run_id=evaluated.value.id,
        release_approved=True,
    )
    with pytest.raises(Exception, match="completed paired evaluation"):
        service.release(ACTOR, release, evaluated.etag)
    trials = tuple(
        BootstrapTrial(
            seed=case.seed,
            environment_id=case.environment_id,
            revision=case.revision,
            attempt=1,
            policy=policy,
            status="succeeded",
            model_sha256=candidate.model_sha256 if policy == "candidate" else None,
            physical_success=True,
            axis_error_m=(0.01, 0, 0),
            duration_seconds=20,
            safety_violations=0,
            inference_p95_ms=70 if policy == "candidate" else None,
            applied_action_count=20 if policy == "candidate" else 0,
            policy_predict_calls=20 if policy == "candidate" else 0,
            reference_route_calls=0 if policy == "candidate" else 1,
            message="Test-only physical trial",
        )
        for case in project.value.evaluation_plan.cases
        for policy in ("reference", "candidate")
    )
    report = BootstrapReport(
        evaluation_plan_sha256=project.value.evaluation_plan.sha256,
        candidate_model_sha256=candidate.model_sha256,
        reference_controller_sha256="6" * 64,
        trials=trials,
        quality_gate_passed=True,
        report_sha256=fingerprint([x.model_dump(mode="json") for x in trials]),
        artifact_id=uuid4(),
    )
    receipt = jobs.receipts[evaluated.value.backend_job_name]
    jobs.receipts[receipt.job_name] = receipt.model_copy(
        update={"status": "succeeded", "report": report}
    )
    result = service.get_job(ACTOR, evaluated.value.id)
    assert "conclusion" not in result.value.report.model_dump()
    first_p0 = service.release(ACTOR, release, result.etag)
    assert first_p0.value.comparison_kind == "reference_bootstrap"
    assert first_p0.value.model_sha256 == candidate.model_sha256 != parent.model_sha256
    assert len(first_p0.value.environment_cases) == 20
    assert len(store.list_learning(ACTOR.owner_key, "release")) == 1
