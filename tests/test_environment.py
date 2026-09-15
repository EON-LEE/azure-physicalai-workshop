from __future__ import annotations

import copy

import pytest
from conftest import cases_for, change_document
from jsonschema import Draft202012Validator

from contracts.validate_environment import SCHEMA, validate_environment


@pytest.mark.parametrize("case", cases_for("environment"), ids=lambda case: case["id"])
def test_environment_cases(case, make_document):
    document = make_document(case)
    before = copy.deepcopy(document)
    issues = validate_environment(document)
    assert (not issues) is case["expected_valid"], issues
    assert set(case["expected_codes"]) <= {issue.code for issue in issues}
    assert document == before


@pytest.mark.parametrize("case", cases_for("finite"), ids=lambda case: case["id"])
def test_non_finite_numbers(case, make_document):
    document = change_document(
        make_document({}),
        [{"op": "set", "path": case["path"], "value": float(case["number"])}],
    )
    issues = validate_environment(document)
    assert len(issues) == 1
    assert issues[0].code == "non_finite"
    assert issues[0].path == "/" + "/".join(str(part) for part in case["path"])


def test_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


def test_errors_include_the_invalid_station_path(make_document):
    document = change_document(
        make_document({}), [{"op": "set", "path": ["stations", 2, "position_m", 0], "value": 20}]
    )
    issues = validate_environment(document)
    assert [(issue.path, issue.code) for issue in issues] == [
        ("/stations/2/position_m", "station_bounds")
    ]
