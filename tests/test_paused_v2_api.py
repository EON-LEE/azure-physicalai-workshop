import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import (
    CaptureReceipt,
    CreateProject,
    LearningProject,
    PausedEvaluationPlan,
    PolicyRelease,
    fingerprint,
)
from apps.api.reference_models import ReferenceAuthorization
from apps.api.simulation_models import (
    PausedRuntimeMetrics,
    ResolvedSimulationAuthorization,
    SimulationEpisodeCommand,
)
from apps.api.simulation_reports import SimulationReport
from tests.runtime_support import ACTOR
from tests.test_paused_learning_api import paused_project_payload
from tests.test_paused_report_projection import project_verified, specimen
from tests.test_paused_wire_contracts import command_payload
from tests.test_reference_collections import envelope

V1 = "franka-position-hold-10hz-paused-v1"
V2 = "franka-position-hold-10hz-paused-v2"


def v2_project():
    value = paused_project_payload()
    value["control_profile_id"] = V2
    value["evaluation_plan"].update(control_profile_id=V2, max_simulation_seconds=60)
    return value


def test_v2_project_declares_its_plan_version_and_does_not_change_wall_or_quality():
    value = v2_project()
    request = CreateProject.model_validate(value)
    saved = LearningProject.create(ACTOR, request)
    assert saved.control_profile_id == saved.evaluation_plan.control_profile_id == V2
    assert saved.evaluation_plan.max_simulation_seconds == 60
    assert saved.evaluation_plan.max_wall_seconds == 600
    assert saved.evaluation_plan.max_policy_wall_ms == 2000
    assert saved.evaluation_plan.minimum_success_rate == 0.9
    assert saved.evaluation_plan.minimum_absolute_improvement == 0.05
    assert saved.budget == CreateProject.model_validate(paused_project_payload()).budget
    assert saved.real_time_admission is False


def test_v1_plan_fingerprint_and_serialized_fields_remain_original():
    old = paused_project_payload()["evaluation_plan"]
    plan = PausedEvaluationPlan.model_validate(old)
    assert plan.model_dump(mode="json") == old
    assert plan.sha256 == fingerprint(old)
    assert "control_profile_id" not in plan.model_dump(mode="json")
    with pytest.raises(ValidationError):
        PausedEvaluationPlan.model_validate(old | {"max_simulation_seconds": 60})


@pytest.mark.parametrize(
    "change",
    [
        {"control_profile_id": V1},
        {"control_profile_id": V2, "max_simulation_seconds": 61},
        {"control_profile_id": V2, "max_wall_seconds": 601},
        {"control_profile_id": V2, "minimum_success_rate": 0.89},
        {"control_profile_id": V2, "minimum_absolute_improvement": 0.04},
    ],
)
def test_v2_plan_rejects_crossed_profiles_or_relaxed_unchanged_budgets(change):
    value = v2_project()
    value["evaluation_plan"].update(change)
    with pytest.raises(ValidationError):
        CreateProject.model_validate(value)


@pytest.mark.parametrize("steps", [6, 1800, 1806, 3600])
def test_explicit_command_v2_admits_only_bounded_multiple_six_ticks(steps):
    value = command_payload() | {"profile_id": V2, "max_simulation_steps": steps}
    parsed = SimulationEpisodeCommand.model_validate(value)
    assert parsed.max_simulation_steps == steps
    assert parsed.profile_id == V2 and parsed.real_time_admission is False
    assert "deadline" not in parsed.model_dump(mode="json")


@pytest.mark.parametrize("profile,steps", [(V1, 1806), (V1, 3600), (V2, 3606), (V2, 3601)])
def test_command_ceiling_is_not_independently_widened(profile, steps):
    with pytest.raises(ValidationError):
        SimulationEpisodeCommand.model_validate(
            command_payload()
            | {
                "profile_id": profile,
                "max_simulation_steps": steps,
            }
        )


def metrics(profile_id=V2, steps=3600):
    return {
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "display_label": "NON_REALTIME_SIMULATION",
        "profile_id": profile_id,
        "control_profile_sha256": "a" * 64,
        "controller": "reference_controller",
        "phase": "stopped",
        "wall_elapsed_ms": 110000,
        "simulation_steps": steps,
        "simulation_elapsed_seconds": steps / 60,
        "policy_predict_calls": 0,
        "applied_action_count": steps,
        "applied_model_sha256": None,
        "reference_route_calls": steps // 6,
    }


def test_actual_simulation_time_is_exactly_derived_from_verified_ticks():
    assert PausedRuntimeMetrics.model_validate(metrics()).simulation_elapsed_seconds == 60
    assert PausedRuntimeMetrics.model_validate(metrics(V1, 1800)).simulation_elapsed_seconds == 30
    for changed in (
        metrics() | {"profile_id": V1},
        metrics(V1, 1800) | {"simulation_elapsed_seconds": 30.0000015646},
        metrics() | {"simulation_elapsed_seconds": 59.9},
    ):
        with pytest.raises(ValidationError):
            PausedRuntimeMetrics.model_validate(changed)


@pytest.mark.parametrize("profile,steps", [(V1, 1800), (V2, 3600)])
@pytest.mark.parametrize("controller", ["reference_controller", "learned"])
def test_actual_runtime_command_and_metrics_roundtrip_without_version_relabeling(
    profile, steps, controller
):
    from apps.api.simulation_models import SimulationEpisodeExecution
    from simulation import paused_contracts as runtime

    payload = command_payload() | {"profile_id": profile, "max_simulation_steps": steps}
    observed = metrics(profile, steps) | {"phase": "applying"}
    if controller == "learned":
        payload.update(
            controller="learned",
            authorization_kind="evaluation_grant",
            policy_type="smolvla",
            model_sha256="f" * 64,
        )
        observed.update(
            controller="learned",
            policy_predict_calls=steps // 6,
            applied_model_sha256="f" * 64,
            reference_route_calls=0,
        )
    request = SimulationEpisodeCommand.model_validate(payload)
    wire = request.model_dump(mode="json", by_alias=True)
    native = runtime.SimulationEpisodeCommand.model_validate(wire)
    assert native.model_dump(mode="json", by_alias=True) == wire
    assert SimulationEpisodeCommand.model_validate(native.model_dump(by_alias=True)) == request
    result = runtime.SimulationEpisodeExecution(
        command_id=native.command_id,
        status="running",
        simulation_runtime=runtime.PausedRuntimeMetrics.model_validate(observed),
    )
    response = result.model_dump(mode="json", by_alias=True)
    decoded = SimulationEpisodeExecution.model_validate(response)
    assert decoded.model_dump(mode="json", by_alias=True) == response
    assert decoded.simulation_runtime.simulation_elapsed_seconds == steps / 60


@pytest.mark.parametrize("profile,steps", [(V1, 1800), (V2, 3600)])
def test_original_operator_grant_roundtrips_actual_runtime_class_and_retains_raw_bytes(
    profile, steps
):
    from simulation.paused_configuration import PausedOperatorGrant

    _, _, value = envelope()
    if profile == V2:
        value["operator_grant"]["authorization"].update(
            profile_id=profile, max_simulation_steps=steps
        )
    value["grant_document_json"] = json.dumps(value["operator_grant"], indent=4) + "\n"
    value["runtime_catalog_record_sha256"] = hashlib.sha256(
        value["grant_document_json"].encode()
    ).hexdigest()
    catalog = ReferenceAuthorization.model_validate(value)
    native = PausedOperatorGrant.model_validate_json(value["grant_document_json"])
    assert native.model_dump(mode="json", by_alias=True) == catalog.operator_grant.model_dump(
        mode="json", by_alias=True
    )
    assert catalog.grant_document_json == value["grant_document_json"]
    authority = catalog.operator_grant.authorization
    assert authority.profile_id == native.authorization.profile_id == profile
    assert ("profile_id" in authority.model_dump(mode="json")) == (profile == V2)
    assert native.authorization.max_simulation_steps == steps


def test_authority_defaults_only_to_legacy_v1_and_v2_grant_must_be_explicit():
    project, case, value = envelope()
    old = value["operator_grant"]["authorization"]
    assert ResolvedSimulationAuthorization.model_validate(old).profile_id == V1
    with pytest.raises(ValidationError):
        ResolvedSimulationAuthorization.model_validate(old | {"max_simulation_steps": 3600})
    value["operator_grant"]["authorization"] = old | {
        "profile_id": V2,
        "max_simulation_steps": 3600,
    }
    value["grant_document_json"] = json.dumps(value["operator_grant"], indent=2) + "\n"
    value["runtime_catalog_record_sha256"] = hashlib.sha256(
        value["grant_document_json"].encode()
    ).hexdigest()
    typed = ReferenceAuthorization.model_validate(value)
    with pytest.raises(Problem):
        typed.authorize(ACTOR, project, case)
    paused = CreateProject.model_validate(v2_project())
    updated = project.model_copy(
        update={
            "control_profile_id": V2,
            "evaluation_plan": paused.evaluation_plan,
        }
    )
    assert typed.authorize(ACTOR, updated, case).profile_id == V2
    assert typed.runtime_catalog_record_sha256 == value["runtime_catalog_record_sha256"]


@pytest.mark.parametrize("profile", [V1, V2])
@pytest.mark.parametrize("local", [False, True])
def test_baseline_resolution_keeps_exact_profile_on_catalog_and_local_paths(profile, local):
    from apps.api.learning_service import LearningService
    from tests.learning_api_support import learning_setup

    factory, store, jobs, artifacts, catalog, _ = learning_setup()
    project = CreateProject.model_validate(
        v2_project() if profile == V2 else paused_project_payload()
    )
    catalog.record = PolicyRelease.model_validate(
        catalog.record.model_dump() | {"policy_type": "smolvla", **project.timing_fields()}
    )
    if local:
        store.put_learning(ACTOR.owner_key, catalog.record, None)
    service = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, allowed_policy_types=("smolvla",)
    )
    assert (
        service._baseline(
            ACTOR,
            catalog.record.id,
            execution_timing="paused_simulation",
            control_profile_id=profile,
        )
        == catalog.record
    )
    with pytest.raises(Problem) as failure:
        service._baseline(
            ACTOR,
            catalog.record.id,
            execution_timing="paused_simulation",
            control_profile_id=V1 if profile == V2 else V2,
        )
    assert failure.value.code == "policy_type_mismatch"
    if profile == V2:
        with pytest.raises(Problem):
            service._baseline(ACTOR, catalog.record.id, execution_timing="paused_simulation")
    assert jobs.submissions == []


def test_compact_report_uses_declared_profile_for_each_trial_ceiling():
    spec, _, native, output = specimen()
    report = project_verified(spec, native, output).model_dump(mode="json")
    extended = deepcopy(report)
    extended["control_profile_id"] = V2
    for row in extended["trials"]:
        row.update(
            applied_action_count=2100, simulation_duration_ms=35000, policy_predict_calls=350
        )
        for name in ("policy", "observation", "hold", "interval"):
            row["phase_wall_ms"][name]["samples"] = 350
    for summary in extended["latency_wall_ms"].values():
        summary["samples"] = 7000
    extended["total_simulation_duration_ms"] = 40 * 35000
    parsed = SimulationReport.model_validate(extended)
    assert parsed.control_profile_id == V2
    assert len(parsed.trials) == 40 and parsed.real_time_admission is False
    with pytest.raises(ValidationError):
        SimulationReport.model_validate(extended | {"control_profile_id": V1})
    assert SimulationReport.model_validate(report).control_profile_id == V1


def test_native_profile_hash_and_capture_counts_cannot_be_relabelled_between_versions():
    from learning.paused import PausedControlProfile

    first = PausedControlProfile("a" * 64)
    second = PausedControlProfile("a" * 64, profile_id=V2, max_simulation_steps=3600)
    assert first.sha256 != second.sha256
    project = CreateProject.model_validate(v2_project() | {"control_profile_sha256": second.sha256})
    case = project.teaching_cases[0]
    from uuid import uuid4

    raw = {
        "episode_id": uuid4(),
        "artifact_id": uuid4(),
        "manifest_sha256": "e" * 64,
        "frame_count": 600,
        "source": "reference_controller",
        "seed": case.seed,
        "task_id": project.task_id,
        "case_id": case.case_id,
        "environment_id": case.environment_id,
        "revision": case.revision,
        "split": case.split,
        **project.timing_fields(),
    }
    assert CaptureReceipt.model_validate(raw).frame_count == 600
    for changed in ({"frame_count": 601}, {"control_profile_id": V1}):
        with pytest.raises(ValidationError):
            CaptureReceipt.model_validate(raw | changed)


def test_reference_dispatch_honors_v2_scene_profile_and_lower_case_ceiling():
    from types import SimpleNamespace

    from apps.api.models import SaveEnvironment
    from apps.api.simulation_models import SimulationEpisodeExecution
    from tests.test_reference_collections import reference_service

    service, stored, body, calls, authority = reference_service()
    environment = service.factory.environment(ACTOR, stored.value.environment_id).value
    document = deepcopy(environment.document)
    document["learning_execution"].update(
        schema="physicalai.paused-simulation/v2",
        profile_id=V2,
        max_simulation_seconds=45,
        max_wall_seconds=120,
    )
    saved = service.factory.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(document),
            expected_revision=environment.revision,
        ),
    )
    plan = PausedEvaluationPlan.model_validate(
        stored.value.evaluation_plan.model_dump()
        | {"control_profile_id": V2, "max_simulation_seconds": 60}
    )
    project = stored.value.model_copy(
        update={
            "control_profile_id": V2,
            "evaluation_plan": plan,
            "revision": saved.revision,
            "teaching_cases": tuple(
                case.model_copy(update={"revision": saved.revision})
                for case in stored.value.teaching_cases
            ),
        }
    )
    saved_project = service.store.put_learning(ACTOR.owner_key, project, stored.etag)
    service.factory.activate(ACTOR, saved.environment_id, saved.revision)
    raw = authority.model_dump(mode="json", by_alias=True)
    raw["operator_grant"]["authorization"].update(
        profile_id=V2, max_simulation_steps=3600, revision=saved.revision
    )
    raw["grant_document_json"] = json.dumps(raw["operator_grant"])
    raw["runtime_catalog_record_sha256"] = hashlib.sha256(
        raw["grant_document_json"].encode()
    ).hexdigest()
    catalog = ReferenceAuthorization.model_validate(raw)
    service.catalog = SimpleNamespace(reference_authorization=lambda *_: catalog)

    def dispatch(owner, command):
        calls.append(command)
        return SimulationEpisodeExecution(
            command_id=command.command_id,
            status="queued",
            simulation_runtime=PausedRuntimeMetrics.model_validate(
                metrics(V2, 0)
                | {
                    "control_profile_sha256": project.control_profile_sha256,
                    "wall_elapsed_ms": 0,
                }
            ),
        )

    service.factory.bridge.dispatch_simulation_episode = dispatch
    result = service.start_reference_collection(ACTOR, project.id, body, saved_project.etag)
    assert len(calls) == 1
    assert calls[0].profile_id == V2 and calls[0].max_simulation_steps == 2700
    assert (calls[0].wall_expires_at - result.value.created_at).total_seconds() <= 120
    assert result.value.execution.simulation_runtime.profile_id == V2
    assert service.factory.bridge.dispatches == 0
    assert saved.document["execution"]["max_step_seconds"] == 30
