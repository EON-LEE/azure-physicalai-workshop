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
