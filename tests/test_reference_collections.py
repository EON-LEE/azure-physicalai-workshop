import hashlib
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_api import api as api

from apps.api.errors import Problem
from apps.api.learning_models import CreateProject, LearningProject
from apps.api.models import utcnow
from tests.runtime_support import ACTOR
from tests.test_paused_learning_api import paused_project_payload


def envelope():
    project = LearningProject.create(ACTOR, CreateProject.model_validate(paused_project_payload()))
    case = project.teaching_cases[0]
    now = utcnow()
    grant = {
        "schema": "physicalai.paused-operator-grant/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "tenant_id": str(ACTOR.tenant_id),
        "source_revision": "a" * 40,
        "simulator_image_digest": "sha256:" + "b" * 64,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=600)).isoformat(),
        "authorization": {
            "authorization_id": str(uuid4()),
            "authorization_kind": "reference_collection",
            "owner": ACTOR.owner_key,
            "environment_id": case.environment_id,
            "revision": case.revision,
            "controller": "reference_controller",
            "task": {
                "task_id": project.task_id,
                "instruction": project.instruction,
                "goal_id": project.goal_station_id,
            },
            "control_profile_sha256": project.control_profile_sha256,
            "wall_expires_at": (now + timedelta(seconds=600)).isoformat(),
            "max_episode_wall_seconds": 600,
            "max_simulation_steps": 1800,
            "purpose": "demonstration",
            "criteria_sha256": project.criteria_sha256,
            "frozen_plan_sha256": project.frozen_plan_sha256,
            "policy_type": None,
            "model_sha256": None,
        },
    }
    raw = json.dumps(grant, indent=2) + "\n"
    return (
        project,
        case,
        {
            "schema": "physicalai.reference-authorization/v1",
            "project_id": str(project.id),
            "case_id": case.case_id,
            "operator_grant": grant,
            "grant_document_json": raw,
            "runtime_catalog_record_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "control_profile_sha256": project.control_profile_sha256,
            "criteria_sha256": project.criteria_sha256,
            "frozen_plan_sha256": project.frozen_plan_sha256,
            "reference_g0_proof_artifact_id": None,
            "reference_g0_proof_sha256": None,
        },
    )


def catalog_model():
    from apps.api.reference_models import ReferenceAuthorization

    return ReferenceAuthorization


def test_catalog_retains_exact_runtime_file_bytes_and_roundtrips_real_grant_class():
    from simulation.paused_configuration import PausedOperatorGrant

    project, case, value = envelope()
    parsed = catalog_model().model_validate(value)
    parsed.authorize(ACTOR, project, case)
    native = PausedOperatorGrant.model_validate(
        parsed.operator_grant.model_dump(mode="json", by_alias=True)
    )
    assert native.model_dump(mode="json", by_alias=True) == parsed.operator_grant.model_dump(
        mode="json", by_alias=True
    )
    assert (
        parsed.runtime_catalog_record_sha256
        == hashlib.sha256(value["grant_document_json"].encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_catalog_record_sha256", "f" * 64),
        ("control_profile_sha256", "f" * 64),
        ("criteria_sha256", "f" * 64),
        ("frozen_plan_sha256", "f" * 64),
    ],
)
def test_catalog_cannot_replace_installed_bytes_profile_or_frozen_conditions(field, value):
    project, case, source = envelope()
    source[field] = value
    with pytest.raises((Problem, ValidationError)):
        catalog_model().model_validate(source).authorize(ACTOR, project, case)


def test_original_raw_json_is_not_parsed_and_reserialized_to_hide_duplicates():
    _, _, source = envelope()
    raw = source["grant_document_json"].replace(
        '"execution_timing": "paused_simulation",',
        '"execution_timing": "legacy", "execution_timing": "paused_simulation",',
    )
    source["grant_document_json"] = raw
    source["runtime_catalog_record_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(ValidationError):
        catalog_model().model_validate(source)


def test_expired_or_integration_authority_never_authorizes_a_customer_demonstration():
    for change in ("expired", "integration"):
        project, case, value = envelope()
        grant = value["operator_grant"]
        if change == "expired":
            grant["expires_at"] = (utcnow() - timedelta(seconds=1)).isoformat()
        else:
            grant["authorization"]["purpose"] = "integration"
        value["grant_document_json"] = json.dumps(grant)
        value["runtime_catalog_record_sha256"] = hashlib.sha256(
            value["grant_document_json"].encode()
        ).hexdigest()
        with pytest.raises((Problem, ValidationError)):
            catalog_model().model_validate(value).authorize(ACTOR, project, case)


def reference_service():
    from types import SimpleNamespace

    from apps.api.models import SaveEnvironment
    from apps.api.reference_models import StartReferenceCollection
    from apps.api.simulation_models import SimulationEpisodeExecution
    from simulation.paused_contracts import PausedRuntimeMetrics
    from tests.test_learning_service import setup
    from tests.test_paused_environment import paused_document

    service, store, _, _ = setup()
    project, _, value = envelope()
    document = paused_document()
    document["scene"]["seed"] = project.teaching_cases[0].seed
    document["execution"].update(record_demonstration=True, demonstration_split="train")
    saved = service.factory.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(document),
            expected_revision=service.factory.environment(
                ACTOR, document["environment_id"]
            ).value.revision,
        ),
    )
    project = project.model_copy(
        update={
            "revision": saved.revision,
            "teaching_cases": tuple(
                case.model_copy(update={"revision": saved.revision})
                for case in project.teaching_cases
            ),
        }
    )
    service.factory.activate(ACTOR, saved.environment_id, saved.revision)
    value["operator_grant"]["authorization"]["revision"] = saved.revision
    value["grant_document_json"] = json.dumps(value["operator_grant"], indent=2) + "\n"
    value["runtime_catalog_record_sha256"] = hashlib.sha256(
        value["grant_document_json"].encode()
    ).hexdigest()
    authorization = catalog_model().model_validate(value)
    service.catalog = SimpleNamespace(reference_authorization=lambda *_: authorization)
    service.reference_collections_enabled = True
    stored = store.put_learning(ACTOR.owner_key, project, None)
    calls = []

    def dispatch(owner, command):
        calls.append(command)
        return SimulationEpisodeExecution.model_validate(
            {
                "command_id": str(command.command_id),
                "status": "queued",
                "simulation_runtime": PausedRuntimeMetrics(
                    control_profile_sha256=project.control_profile_sha256,
                    controller="reference_controller",
                ).model_dump(mode="json"),
            }
        )

    service.factory.bridge.dispatch_simulation_episode = dispatch
    service.factory.bridge.simulation_episode = lambda owner, command_id: dispatch(owner, calls[0])
    body = StartReferenceCollection(
        request_id=uuid4(), case_id=project.teaching_cases[0].case_id, motion_approved=True
    )
    return service, stored, body, calls, authorization


def test_reference_request_never_uses_human_or_legacy_motion_and_is_idempotent():
    service, project, body, calls, authority = reference_service()
    created = service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    assert created.value.source == "reference_controller"
    assert created.value.execution_timing == "paused_simulation"
    assert created.value.status == "running" and created.value.capture is None
    assert len(calls) == 1
    command = calls[0]
    assert command.controller == "reference_controller"
    assert command.authorization_id == authority.operator_grant.authorization.authorization_id
    assert command.wall_expires_at <= authority.operator_grant.expires_at
    assert command.max_simulation_steps <= 1800
    service.factory.bridge.simulation_episode = lambda *_: created.value.execution
    repeated = service.start_reference_collection(ACTOR, project.value.id, body, "stale-retry")
    assert repeated.value.id == body.request_id
    assert len(calls) == 1 and service.factory.bridge.dispatches == 0


def test_reference_stage_does_not_admit_training_or_manual_paused_controls():
    from apps.api.learning_models import StartTeaching, StartTraining

    service, project, _, calls, _ = reference_service()
    with pytest.raises(Problem) as manual:
        service.start_teaching(
            ACTOR,
            project.value.id,
            StartTeaching(
                request_id=uuid4(),
                source="human_teleop",
                motion_approved=True,
            ),
            project.etag,
        )
    assert manual.value.code == "paused_learning_unavailable"
    with pytest.raises(Problem) as training:
        service.train(
            ACTOR,
            project.value.id,
            StartTraining(
                request_id=uuid4(),
                dataset_id=uuid4(),
                parent_release_id=project.value.baseline_release_id,
                policy_type="smolvla",
                optimizer_steps=1,
                paid_approved=True,
                maximum_cost_usd="1",
            ),
            project.etag,
        )
    assert training.value.code == "paused_learning_unavailable"
    assert calls == []


def test_all_paused_stage_settings_default_off():
    from apps.learning_worker.main import WorkerSettings
    from tests.runtime_support import AUDIENCE, TENANT, settings

    api_settings = settings()
    for field in (
        "learning_reference_collections_enabled",
        "learning_paused_training_enabled",
        "learning_paused_evaluation_enabled",
        "learning_paused_release_enabled",
    ):
        assert getattr(api_settings, field) is False
    worker = WorkerSettings(
        tenant_id=TENANT,
        audience=AUDIENCE,
        managed_identity_client_id=uuid4(),
        registry_account_url="https://testregistry.blob.core.windows.net",
        registry_container="artifacts",
        capture_account_url="https://testcapture.blob.core.windows.net",
    )
    assert not worker.reference_collections_enabled
    assert not worker.paused_training_enabled
    assert not worker.paused_evaluation_enabled


def test_private_catalog_rejects_arbitrary_paths_before_any_registry_read():
    from apps.learning_worker.registry import BlobRegistry

    registry = BlobRegistry.__new__(BlobRegistry)
    registry.get = lambda *_: pytest.fail("Invalid case must not read a registry path.")
    for case_id in ("../another-owner", "case/elsewhere", "case\n", "case%2fother"):
        with pytest.raises(Problem) as failure:
            registry.reference_authorization(ACTOR, uuid4(), case_id)
        assert failure.value.status == 422


def test_reference_default_disabled_never_reads_catalog_or_moves():
    service, project, body, calls, _ = reference_service()
    service.reference_collections_enabled = False
    with pytest.raises(Problem) as failure:
        service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    assert failure.value.code == "reference_phase_disabled"
    assert calls == []


def test_runtime_independently_rechecks_the_catalog_grant_at_dispatch():
    from types import SimpleNamespace

    from simulation.paused_configuration import OperatorPausedAuthority, PausedOperatorGrant
    from simulation.paused_contracts import SimulationEpisodeCommand

    service, project, body, calls, authority = reference_service()
    service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    native = OperatorPausedAuthority(
        PausedOperatorGrant.model_validate(
            authority.operator_grant.model_dump(mode="json", by_alias=True)
        )
    )
    command = SimulationEpisodeCommand.model_validate(calls[0].model_dump(mode="json"))
    profile = SimpleNamespace(
        sha256=project.value.control_profile_sha256, profile_id=project.value.control_profile_id
    )
    assert (
        native.authorize(ACTOR.owner_key, command, profile).authorization_id
        == command.authorization_id
    )
    for changed in (
        {"authorization_id": uuid4()},
        {"revision": "f" * 64},
        {"profile_id": "franka-position-hold-10hz-paused-v2"},
        {"wall_expires_at": authority.operator_grant.expires_at + timedelta(seconds=1)},
    ):
        with pytest.raises(Problem):
            native.authorize(ACTOR.owner_key, command.model_copy(update=changed), profile)
    with pytest.raises(Problem):
        native.authorize(
            ACTOR.owner_key,
            command,
            SimpleNamespace(
                sha256=profile.sha256, profile_id="franka-position-hold-10hz-paused-v2"
            ),
        )


def test_lower_saved_scene_caps_limit_the_new_command_without_changing_legacy_wall_cap():
    from apps.api.models import SaveEnvironment

    service, project, body, calls, authority = reference_service()
    record = service.factory.environment(ACTOR, project.value.environment_id).value
    document = json.loads(json.dumps(record.document))
    document["learning_execution"].update(max_simulation_seconds=10, max_wall_seconds=20)
    saved = service.factory.save_environment(
        ACTOR,
        SaveEnvironment(document_json=json.dumps(document), expected_revision=record.revision),
    )
    service.factory.activate(ACTOR, saved.environment_id, saved.revision)
    changed = project.value.model_copy(
        update={
            "revision": saved.revision,
            "teaching_cases": tuple(
                case.model_copy(update={"revision": saved.revision})
                for case in project.value.teaching_cases
            ),
        }
    )
    current = service.store.put_learning(ACTOR.owner_key, changed, project.etag)
    raw = authority.model_dump(mode="json", by_alias=True)
    raw["operator_grant"]["authorization"]["revision"] = saved.revision
    raw["grant_document_json"] = json.dumps(raw["operator_grant"])
    raw["runtime_catalog_record_sha256"] = hashlib.sha256(
        raw["grant_document_json"].encode()
    ).hexdigest()
    from types import SimpleNamespace

    service.catalog = SimpleNamespace(
        reference_authorization=lambda *_: catalog_model().model_validate(raw)
    )
    created = service.start_reference_collection(ACTOR, changed.id, body, current.etag)
    assert calls[0].max_simulation_steps == 600
    assert 0 < (calls[0].wall_expires_at - created.value.created_at).total_seconds() <= 20
    assert saved.document["execution"]["max_step_seconds"] == 30


@pytest.mark.parametrize("changed", ["missing", "stale", "wrong-owner"])
def test_no_motion_when_catalog_or_original_observation_cannot_be_verified(changed):
    from tests.runtime_support import OTHER

    service, project, body, calls, _ = reference_service()
    if changed == "missing":

        def missing(*_):
            raise Problem(503, "reference_authority_missing", "No installed record")

        service.catalog.reference_authorization = missing
    elif changed == "stale":
        observe = service.factory.bridge.observe
        service.factory.bridge.observe = lambda *args: observe(*args).model_copy(
            update={
                "captured_at": utcnow() - timedelta(seconds=5),
            }
        )
    with pytest.raises(Problem):
        service.start_reference_collection(
            OTHER if changed == "wrong-owner" else ACTOR, project.value.id, body, project.etag
        )
    assert calls == []


def test_concurrent_start_and_uncertain_post_never_dispatch_twice():
    from concurrent.futures import ThreadPoolExecutor

    service, project, body, calls, _ = reference_service()

    def uncertain(owner, command):
        calls.append(command)
        raise Problem(503, "bridge_unconfirmed", "Test-only response lost")

    service.factory.bridge.dispatch_simulation_episode = uncertain

    def not_found(*_):
        raise Problem(404, "command_missing", "Not observed yet")

    service.factory.bridge.simulation_episode = not_found
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: service.start_reference_collection(
                    ACTOR, project.value.id, body, project.etag
                ),
                range(4),
            )
        )
    assert {item.value.id for item in results} == {body.request_id}
    assert len(calls) == 1
    service.start_reference_collection(ACTOR, project.value.id, body, "stale-retry")
    assert len(calls) == 1


@pytest.mark.parametrize("changed", ["duration", "ticks", "error", "clock", "goal"])
def test_success_requires_matching_clock_ticks_and_error_free_physical_evidence(changed):
    from apps.api.simulation_models import SimulationEpisodeExecution
    from simulation.paused_contracts import PausedRuntimeMetrics

    service, project, body, _, _ = reference_service()
    stored = service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    metrics = PausedRuntimeMetrics(
        control_profile_sha256=project.value.control_profile_sha256,
        controller="reference_controller",
        phase="stopped",
        wall_elapsed_ms=100,
        simulation_steps=6,
        simulation_elapsed_seconds=0.1,
        applied_action_count=6,
        reference_route_calls=1,
    ).model_dump(mode="json")
    payload = {
        "command_id": stored.value.command_id,
        "status": "succeeded",
        "final_position": stored.value.target_position_m,
        "completed_at": utcnow(),
        "simulation_runtime": metrics,
    }
    if changed == "duration":
        metrics["wall_elapsed_ms"] = 601000
    elif changed == "ticks":
        metrics["simulation_elapsed_seconds"] = 20
    elif changed == "error":
        payload["error"] = {"code": "tracking_failed", "message": "No verified grasp"}
    elif changed == "clock":
        payload["completed_at"] = stored.value.created_at - timedelta(seconds=1)
    else:
        payload["final_position"] = (99, 99, 99)
    with pytest.raises((Problem, ValidationError)):
        service._apply_reference(ACTOR, stored, SimulationEpisodeExecution.model_validate(payload))
    assert service.get(ACTOR, "reference_collection", body.request_id).value.status == "running"


def test_cancel_ack_and_stale_poll_never_revive_a_terminal_reference():
    service, project, body, _, _ = reference_service()
    initial = service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    calls = []

    def cancel(*_):
        calls.append(True)
        return initial.value.execution

    service.factory.bridge.cancel_simulation_episode = cancel
    pending = service.cancel_reference_collection(ACTOR, body.request_id)
    assert pending.value.status == "cancelling"
    service.cancel_reference_collection(ACTOR, body.request_id)
    assert calls == [True]
    terminal = service._apply_reference(
        ACTOR, pending, initial.value.execution.model_copy(update={"status": "cancelled"})
    )
    assert (
        service._apply_reference(ACTOR, terminal, initial.value.execution).value.status
        == "cancelled"
    )


def test_reference_command_and_terminal_state_are_immutable_in_real_cosmos_adapter():
    from tests.test_learning_cosmos import store as cosmos_store

    service, project, body, _, _ = reference_service()
    started = service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    cosmos = cosmos_store()
    stored = cosmos.put_learning(ACTOR.owner_key, started.value, None)
    changed = stored.value.model_copy(
        update={"command": stored.value.command.model_copy(update={"authorization_id": uuid4()})}
    )
    with pytest.raises(Problem):
        cosmos.put_learning(ACTOR.owner_key, changed, stored.etag)
    stopped = cosmos.put_learning(
        ACTOR.owner_key, stored.value.model_copy(update={"status": "cancelled"}), stored.etag
    )
    with pytest.raises(Problem):
        cosmos.put_learning(
            ACTOR.owner_key, stopped.value.model_copy(update={"status": "running"}), stopped.etag
        )


def bootstrap_project(*, missing_opt_in=False):
    from types import SimpleNamespace

    from apps.api.learning_models import TrainingParent
    from apps.api.models import SaveEnvironment
    from tests.test_learning_service import setup
    from tests.test_paused_environment import paused_document

    service, _, jobs, _ = setup()
    value = paused_project_payload()
    value.update(
        project_kind="bootstrap", baseline_release_id=None, pretrained_artifact_id=str(uuid4())
    )
    for index, case in enumerate([*value["teaching_cases"], *value["evaluation_plan"]["cases"]]):
        content = paused_document()
        content["environment_id"] = case["environment_id"]
        content["scene"]["seed"] = case["seed"]
        content["execution"].update(
            record_demonstration=True, demonstration_split=case.get("split", "test")
        )
        if missing_opt_in and index == 1:
            content.pop("learning_execution")
        previous = service.store.get_environment(ACTOR.owner_key, case["environment_id"])
        saved = service.factory.save_environment(
            ACTOR,
            SaveEnvironment(
                document_json=json.dumps(content),
                expected_revision=previous.value.revision if previous else None,
            ),
        )
        case["revision"] = saved.revision
    value["revision"] = value["teaching_cases"][0]["revision"]
    request = CreateProject.model_validate(value)
    parent = TrainingParent(
        id=request.pretrained_artifact_id,
        artifact_id=uuid4(),
        owner_key=ACTOR.owner_key,
        tenant_id=ACTOR.tenant_id,
        actor_id=ACTOR.object_id,
        fingerprint="a" * 64,
        created_at=utcnow(),
        updated_at=utcnow(),
        policy_type="smolvla",
        model_sha256="b" * 64,
        processor_sha256="c" * 64,
        source_commit="d" * 40,
        model_revision="e" * 40,
        registered_by=ACTOR.object_id,
        **request.timing_fields(),
    )
    service.catalog = SimpleNamespace(training_parent=lambda *_: parent)
    service.allowed_policy_types = ("smolvla",)
    service.bootstrap_principal_ids = frozenset({ACTOR.object_id})
    service.reference_collections_enabled = True
    return service, request, jobs


def test_reference_phase_can_define_bootstrap_without_inventing_a_quality_passed_p0():
    service, request, jobs = bootstrap_project()
    result = service.create_project(ACTOR, request)
    assert result.value.project_kind == "bootstrap" and result.value.baseline_release_id is None
    assert service.store.list_learning(ACTOR.owner_key, "release") == []
    assert jobs.submissions == []
    assert not service.capabilities(ACTOR)["simulation_learning"]["training_enabled"]


def test_paused_definition_requires_explicit_opt_in_on_every_frozen_case():
    service, request, jobs = bootstrap_project(missing_opt_in=True)
    with pytest.raises(Problem) as failure:
        service.create_project(ACTOR, request)
    assert failure.value.code == "paused_execution_required"
    assert jobs.submissions == []


def test_training_stage_does_not_require_an_already_trained_or_released_p0():
    from apps.api.learning_models import CaptureReceipt, DatasetVersion, StartTraining

    service, request, jobs = bootstrap_project()
    project = service.create_project(ACTOR, request)
    service.paused_training_enabled = True
    case = project.value.teaching_cases[0]
    capture = CaptureReceipt(
        episode_id=uuid4(),
        artifact_id=uuid4(),
        manifest_sha256="f" * 64,
        frame_count=2,
        source="reference_controller",
        seed=case.seed,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        split=case.split,
        task_id=project.value.task_id,
        **project.value.timing_fields(),
    )
    dataset = DatasetVersion(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        tenant_id=ACTOR.tenant_id,
        actor_id=ACTOR.object_id,
        created_at=utcnow(),
        updated_at=utcnow(),
        fingerprint="f" * 64,
        project_id=project.value.id,
        artifact_id=uuid4(),
        manifest_sha256="e" * 64,
        episode_ids=(capture.episode_id,),
        seeds=(capture.seed,),
        captures=(capture,),
        human_teleop_count=0,
        reference_controller_count=1,
        learned_policy_count=0,
        evaluation_plan_sha256=project.value.evaluation_plan.sha256,
        **project.value.timing_fields(),
    )
    service.store.put_learning(ACTOR.owner_key, dataset, None)
    result = service.train(
        ACTOR,
        project.value.id,
        StartTraining(
            request_id=uuid4(),
            dataset_id=dataset.id,
            parent_release_id=None,
            pretrained_artifact_id=request.pretrained_artifact_id,
            policy_type="smolvla",
            optimizer_steps=1,
            paid_approved=True,
            maximum_cost_usd="1",
        ),
        project.etag,
    )
    assert result.value.status == "submitted" and result.value.candidate_id is None
    assert len(jobs.submissions) == 1
    assert service.store.list_learning(ACTOR.owner_key, "release") == []


def test_reference_capture_becomes_dataset_only_after_async_manifest_verification():
    from apps.api.artifact_models import ArtifactResult
    from apps.api.learning_models import CaptureReceipt, CreateDataset, fingerprint
    from apps.api.learning_ports import RuntimeCapture
    from apps.api.models import DemonstrationResult
    from apps.api.simulation_models import SimulationEpisodeExecution
    from tests.test_artifact_operation_api import async_adapter, complete

    service, project, body, _, _ = reference_service()
    started = service.start_reference_collection(ACTOR, project.value.id, body, project.etag)
    result = started.value.execution.model_dump()
    result.update(
        status="succeeded", final_position=started.value.target_position_m, completed_at=utcnow()
    )
    result["simulation_runtime"].update(
        wall_elapsed_ms=100,
        simulation_steps=12,
        simulation_elapsed_seconds=0.2,
        applied_action_count=12,
        reference_route_calls=2,
        phase="stopped",
    )
    service.factory.bridge.simulation_episode = lambda *_: (
        SimulationEpisodeExecution.model_validate(result)
    )
    raw = DemonstrationResult(
        status="uploaded",
        manifest_sha256="f" * 64,
        episode_id=body.request_id,
        frame_count=2,
        manifest_uri=f"https://test.blob.core.windows.net/captures/{body.request_id}/manifest.json",
    )
    service.factory.bridge.capture = lambda *_: RuntimeCapture(
        capture_id=uuid4(),
        command_id=body.request_id,
        epoch=started.value.epoch,
        status="ready",
        receipt=raw,
    )
    ops, registry, calls = async_adapter(service)
    waiting = service.get_reference_collection(ACTOR, body.request_id)
    assert waiting.value.status == "succeeded" and waiting.value.capture_status == "verifying"
    assert waiting.value.capture is None and len(calls) == 1
    case = waiting.value.teaching_case
    receipt = CaptureReceipt(
        episode_id=body.request_id,
        manifest_sha256=raw.manifest_sha256,
        artifact_id=uuid4(),
        frame_count=2,
        source="reference_controller",
        task_id=project.value.task_id,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        seed=case.seed,
        split=case.split,
        **project.value.timing_fields(),
    )
    complete(
        ops,
        registry,
        waiting.value.artifact_operation_id,
        ArtifactResult(
            artifact_id=receipt.artifact_id,
            manifest_sha256=receipt.manifest_sha256,
            capture=receipt,
        ),
    )
    ready = service.get_reference_collection(ACTOR, body.request_id)
    assert (
        ready.value.capture_status == "ready"
        and ready.value.capture.source == "reference_controller"
    )
    seal = CreateDataset(request_id=uuid4(), reference_collection_ids=(body.request_id,))
    pending = service.dataset(ACTOR, project.value.id, seal, project.etag)
    assert pending.value.kind == "artifact_operation" and pending.value.status == "queued"
    complete(
        ops,
        registry,
        pending.value.id,
        ArtifactResult(artifact_id=seal.request_id, manifest_sha256="e" * 64),
    )
    service.get_artifact_operation(ACTOR, pending.value.id)
    dataset = service.get(ACTOR, "dataset", seal.request_id).value
    assert dataset.reference_controller_count == 1 and dataset.human_teleop_count == 0
    assert len(calls) == 2 and dataset.captures == (receipt,)
    legacy = {"request_id": str(uuid4()), "teaching_session_ids": [str(uuid4())]}
    assert fingerprint(CreateDataset.model_validate(legacy).model_dump(mode="json")) == fingerprint(
        legacy
    )


def test_reference_http_cannot_accept_client_grants_or_skip_owner_approval(api):
    from test_api import headers

    from tests.runtime_support import OTHER

    client, _, token = api
    service, project, body, calls, _ = reference_service()
    client.app.state.learning = service
    path = f"/api/learning/projects/{project.value.id}/reference-collections"
    raw = body.model_dump(mode="json")
    assert client.post(path, json=raw).status_code == 401
    assert (
        client.post(
            path,
            json=raw,
            headers={
                "Authorization": f"Bearer {token(oid=str(OTHER.object_id))}",
                "If-Match": project.etag,
            },
        ).status_code
        == 404
    )
    for changed in (
        {"motion_approved": False},
        {"source": "human_teleop"},
        {"authorization_id": str(uuid4())},
        {"wall_expires_at": utcnow().isoformat()},
    ):
        assert (
            client.post(
                path,
                json=raw | changed,
                headers={
                    **headers(token),
                    "If-Match": project.etag,
                },
            ).status_code
            == 422
        )
    assert calls == []
    response = client.post(path, json=raw, headers={**headers(token), "If-Match": project.etag})
    assert response.status_code == 202, response.text
    assert response.json()["item"]["source"] == "reference_controller"
    assert "grant_document_json" not in response.text
    assert (
        response.json()["item"]["command"]["schema"] == "physicalai.simulation-episode-command/v1"
    )
