"""CPU admission checks for explicit non-real-time scene authority."""

import json
from dataclasses import replace

import pytest
from runtime_support import ACTOR, document, service
from test_paused_environment import paused_document

from apps.api.errors import Problem
from apps.api.models import SaveEnvironment
from simulation.extensions import SceneRegistry, can_reset_in_place


def scene(document_json):
    record = service().save_environment(
        ACTOR, SaveEnvironment(document_json=json.dumps(document_json))
    )
    return SceneRegistry(load_installed=False).build(record), record


def test_paused_opt_in_is_decoded_independently_of_legacy_wall_limit():
    spec, record = scene(paused_document())
    authority = spec.require_paused_authority()
    assert authority.execution_timing == "paused_simulation"
    assert authority.profile_id == "franka-position-hold-10hz-paused-v1"
    assert authority.max_wall_seconds == 600
    assert authority.max_simulation_steps == 1800
    assert authority.real_time_admission is False
    assert record.document["execution"]["max_step_seconds"] == 30


def test_missing_explicit_opt_in_never_uses_the_legacy_30_second_command_as_600_seconds():
    spec, _ = scene(document())
    assert spec.learning_execution is None
    with pytest.raises(Problem, match="Explicit"):
        spec.require_paused_authority()


def test_approved_lower_scene_budgets_are_not_replaced_by_profile_maxima():
    value = paused_document()
    value["learning_execution"].update(max_simulation_seconds=5, max_wall_seconds=20)
    spec, _ = scene(value)
    authority = spec.require_paused_authority()
    authority.validate_request(wall_seconds=19.5, simulation_steps=294)
    with pytest.raises(Problem) as wall_failure:
        authority.validate_request(wall_seconds=21, simulation_steps=294)
    assert wall_failure.value.code == "paused_wall_budget"
    assert wall_failure.value.status == 409
    with pytest.raises(Problem) as simulation_failure:
        authority.validate_request(wall_seconds=19.5, simulation_steps=306)
    assert simulation_failure.value.code == "paused_simulation_budget"
    assert simulation_failure.value.status == 409
    assert authority.max_wall_seconds == 20 and authority.max_simulation_steps == 300


@pytest.mark.parametrize("steps", [0, 1, 7, 1806, True])
def test_nonaligned_or_unapproved_physics_budget_cannot_be_admitted(steps):
    spec, _ = scene(paused_document())
    with pytest.raises(Problem):
        spec.require_paused_authority().validate_request(wall_seconds=30, simulation_steps=steps)


@pytest.mark.parametrize("seconds", [0, -1, 601, float("nan"), float("inf"), True])
def test_invalid_wall_budget_cannot_be_replaced_with_a_success_default(seconds):
    spec, _ = scene(paused_document())
    with pytest.raises(Problem):
        spec.require_paused_authority().validate_request(
            wall_seconds=seconds,
            simulation_steps=1800,
        )


def test_paused_budget_change_requires_a_new_saved_revision_and_scene_binding():
    before_doc = paused_document()
    before, before_record = scene(before_doc)
    after_doc = paused_document()
    after_doc["learning_execution"]["max_wall_seconds"] = 400
    after, after_record = scene(after_doc)
    assert before_record.revision != after_record.revision
    assert not can_reset_in_place(before, after)
    assert can_reset_in_place(before, replace(before, seed=before.seed + 2))


def test_replay_is_not_physical_authority_for_paused_simulation():
    value = paused_document()
    value["execution"]["mode"] = "replay"
    with pytest.raises(Problem, match="live"):
        scene(value)
