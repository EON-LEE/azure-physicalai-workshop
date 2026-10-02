import copy

import pytest

from contracts import validate_environment as validation
from tests.runtime_support import document


def paused_document():
    value = document()
    value["scene"]["template_id"] = "inspection-cell-learning-v1"
    value["learning_execution"] = {
        "schema": "physicalai.paused-simulation/v1",
        "execution_timing": "paused_simulation",
        "profile_id": "franka-position-hold-10hz-paused-v1",
        "max_simulation_seconds": 30,
        "max_wall_seconds": 600,
    }
    return value


def test_explicit_paused_environment_is_valid_without_changing_legacy_wall_limit():
    value = paused_document()
    assert validation.validate_environment(value) == []
    assert value["execution"]["max_step_seconds"] == 30
    assert value["learning_execution"]["max_wall_seconds"] == 600


@pytest.mark.parametrize("simulation,wall", [(1, 1), (10, 200), (30, 600)])
def test_customer_can_request_lower_explicit_paused_budgets(simulation, wall):
    value = paused_document()
    value["learning_execution"].update(max_simulation_seconds=simulation, max_wall_seconds=wall)
    assert validation.validate_environment(value) == []
    assert validation.validate_paused_learning_environment(value) == []


@pytest.mark.parametrize(
    "missing",
    [
        "schema",
        "execution_timing",
        "profile_id",
        "max_simulation_seconds",
        "max_wall_seconds",
    ],
)
def test_every_paused_field_is_explicit_and_required(missing):
    value = paused_document()
    value["learning_execution"].pop(missing)
    assert validation.validate_environment(value)


@pytest.mark.parametrize(
    "changed",
    [
        {"schema": "physicalai.paused-simulation/v2"},
        {"execution_timing": "realtime"},
        {"profile_id": "franka-position-hold-10hz-v1"},
        {"max_simulation_seconds": 31},
        {"max_simulation_seconds": 0},
        {"max_simulation_seconds": 1.5},
        {"max_simulation_seconds": True},
        {"max_wall_seconds": 601},
        {"max_wall_seconds": 0},
        {"max_wall_seconds": True},
        {"real_time_admission": True},
        {"timing_mode": "paused_simulation"},
        {"policy_operation_ms": 99999},
    ],
)
def test_unknown_aliases_unapproved_modes_and_excess_budgets_are_not_authority(changed):
    value = paused_document()
    value["learning_execution"].update(changed)
    assert validation.validate_environment(value)


@pytest.mark.parametrize("template", ["inspection-cell-v1", "another-customer-template"])
def test_only_reviewed_learning_template_accepts_paused_opt_in(template):
    value = paused_document()
    value["scene"]["template_id"] = template
    assert validation.validate_environment(value)
    value.pop("learning_execution")
    assert validation.validate_environment(value) == []


def test_absence_keeps_legacy_valid_but_never_authorizes_paused_execution():
    value = document()
    original = copy.deepcopy(value)
    assert validation.validate_environment(value) == []
    issues = validation.validate_paused_learning_environment(value)
    assert issues and issues[0].code == "paused_execution_required"
    assert value == original


def test_replay_cannot_authorize_a_live_paused_episode():
    value = paused_document()
    value["execution"]["mode"] = "replay"
    issues = validation.validate_paused_learning_environment(value)
    assert any(issue.code == "paused_live_required" for issue in issues)


def test_schema_defines_no_implicit_paused_defaults():
    properties = validation.SCHEMA["properties"]["learning_execution"]["properties"]
    assert all("default" not in item for item in properties.values())


def paused_v2_document():
    value = paused_document()
    value["learning_execution"].update(
        schema="physicalai.paused-simulation/v2",
        profile_id="franka-position-hold-10hz-paused-v2",
        max_simulation_seconds=60,
    )
    return value


@pytest.mark.parametrize("simulation,wall", [(1, 1), (30, 600), (31, 200), (60, 600)])
def test_explicit_v2_allows_only_its_versioned_sixty_sim_second_budget(simulation, wall):
    value = paused_v2_document()
    value["learning_execution"].update(max_simulation_seconds=simulation, max_wall_seconds=wall)
    original = copy.deepcopy(value)
    assert validation.validate_environment(value) == []
    assert validation.validate_paused_learning_environment(value) == []
    assert value == original
    assert value["execution"]["max_step_seconds"] == 30


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "physicalai.paused-simulation/v1"},
        {"profile_id": "franka-position-hold-10hz-paused-v1"},
        {"schema": "physicalai.paused-simulation/v3"},
        {"max_simulation_seconds": 61},
        {"max_simulation_seconds": 0},
        {"max_simulation_seconds": True},
        {"max_simulation_seconds": 30.5},
        {"max_wall_seconds": 601},
        {"max_wall_seconds": True},
        {"max_simulation_steps": 3600},
        {"real_time_admission": True},
    ],
)
def test_v2_rejects_version_mixing_or_unapproved_wall_and_simulation_caps(changes):
    value = paused_v2_document()
    value["learning_execution"].update(changes)
    assert validation.validate_paused_learning_environment(value)


@pytest.mark.parametrize("simulation", [1, 30])
def test_versions_cannot_be_mixed_even_when_the_chosen_simulation_budget_fits_both(simulation):
    for schema, profile in (
        ("physicalai.paused-simulation/v1", "franka-position-hold-10hz-paused-v2"),
        ("physicalai.paused-simulation/v2", "franka-position-hold-10hz-paused-v1"),
    ):
        value = paused_document()
        value["learning_execution"].update(
            schema=schema, profile_id=profile, max_simulation_seconds=simulation
        )
        assert validation.validate_environment(value)


@pytest.mark.parametrize(
    "missing",
    [
        "schema",
        "execution_timing",
        "profile_id",
        "max_simulation_seconds",
        "max_wall_seconds",
    ],
)
def test_v2_never_synthesizes_missing_version_or_budget_fields(missing):
    value = paused_v2_document()
    value["learning_execution"].pop(missing)
    assert validation.validate_environment(value)


def test_v2_remains_learning_template_only_and_does_not_retroactively_widen_v1():
    value = paused_v2_document()
    value["scene"]["template_id"] = "inspection-cell-v1"
    assert validation.validate_environment(value)
    legacy = paused_document()
    legacy["learning_execution"]["max_simulation_seconds"] = 60
    assert validation.validate_environment(legacy)
    assert validation.validate_environment(paused_document()) == []
