import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from agents.learning_coach import TOOL_NAME, agent_definition, extract_proposal
from apps.api.errors import Problem
from apps.api.learning_models import CoachProposal
from tests.test_learning_coach import context

UNSUPPORTED = {
    "default",
    "format",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "patternProperties",
    "unevaluatedProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "minItems",
    "maxItems",
    "uniqueItems",
}


def schemas(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from schemas(child)


def wire_tool():
    # Exercise the actual installed Foundry SDK serializer, not a hand-built test tool.
    definition = agent_definition("configured-model")
    wire = json.loads(json.dumps(definition.as_dict()))
    assert wire["kind"] == "prompt"
    assert len(wire["tools"]) == 1
    return wire["tools"][0]


def nullable_proposal(owner):
    return {
        "project_id": str(owner.project_id),
        "action": "review_demonstrations",
        "summary": "허가된 데이터와 다음 검증 단계를 검토합니다.",
        "dataset_id": None,
        "optimizer_steps": None,
        "selected_release_id": None,
    }


def response(body):
    return SimpleNamespace(
        id="test-only-foundry-schema-response",
        output=[
            SimpleNamespace(
                type="function_call",
                name=TOOL_NAME,
                arguments=json.dumps(body),
            )
        ],
    )


def test_serialized_coach_tool_uses_the_documented_azure_strict_subset():
    tool = wire_tool()
    assert tool["name"] == TOOL_NAME
    assert tool["strict"] is True
    schema = tool["parameters"]
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == set(CoachProposal.model_fields)
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    for node in schemas(schema):
        assert not UNSUPPORTED.intersection(node), (
            "Do not send Pydantic-only validation/default keywords."
        )
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


@pytest.mark.parametrize("omitted", list(CoachProposal.model_fields))
def test_every_wire_property_is_required_even_when_its_value_can_be_null(omitted):
    body = nullable_proposal(context())
    validator = Draft202012Validator(wire_tool()["parameters"])
    assert validator.is_valid(body)
    del body[omitted]
    assert not validator.is_valid(body)


def test_explicit_nulls_work_on_the_wire_and_real_scoped_proposal_parser():
    owner = context()
    body = nullable_proposal(owner)
    Draft202012Validator(wire_tool()["parameters"]).validate(body)
    parsed, response_id = extract_proposal(response(body), owner)
    assert parsed.dataset_id is None and parsed.optimizer_steps is None
    assert parsed.selected_release_id is None
    assert response_id == "test-only-foundry-schema-response"
    assert (
        wire_tool()["parameters"]["properties"]["action"]["enum"]
        == (CoachProposal.model_json_schema()["properties"]["action"]["enum"])
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        pytest.param("project_id", "not-a-uuid", id="invalid-project-uuid"),
        pytest.param("dataset_id", "not-a-uuid", id="invalid-dataset-uuid"),
        pytest.param("optimizer_steps", 0, id="zero-optimizer-steps"),
        pytest.param("optimizer_steps", 100001, id="excessive-optimizer-steps"),
        pytest.param("summary", "", id="empty-summary"),
        pytest.param("summary", "   ", id="blank-summary"),
        pytest.param("summary", "x" * 2001, id="oversized-summary"),
    ],
)
def test_service_subset_does_not_weaken_local_pydantic_response_validation(field, invalid):
    owner = context()
    body = {**nullable_proposal(owner), field: invalid}
    # These server-side constraints intentionally remain out of the service tool schema.
    assert Draft202012Validator(wire_tool()["parameters"]).is_valid(body)
    with pytest.raises(Problem) as failure:
        extract_proposal(response(body), owner)
    assert failure.value.code == "invalid_learning_proposal"


def test_strict_tool_does_not_grant_paid_motion_or_policy_promotion_fields():
    body = nullable_proposal(context())
    validator = Draft202012Validator(wire_tool()["parameters"])
    for key in ("paid_approved", "motion_approved", "release_approved", "model_uri", "joints"):
        assert not validator.is_valid({**body, key: True})
