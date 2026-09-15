import json
from types import SimpleNamespace

import pytest

from agents.inspection import TOOL_NAME, agent_definition, extract_decision
from apps.api.errors import Problem


def response(arguments=None, name=TOOL_NAME, calls=1):
    arguments = arguments or json.dumps(
        {
            "classification": "rejected",
            "object_id": "part-001",
            "summary": "Visible test defect.",
        }
    )
    return SimpleNamespace(
        id="actual-shaped-response-id",
        output=[SimpleNamespace(type="function_call", name=name, arguments=arguments)] * calls,
    )


def test_definition_uses_the_installed_foundry_sdk_and_one_non_motion_tool():
    definition = agent_definition("configured-model")
    assert definition.kind == "prompt"
    assert len(definition.tools) == 1
    assert definition.tools[0].name == TOOL_NAME
    assert definition.tools[0].strict is True


def test_valid_inspection_has_a_response_id():
    decision, response_id = extract_decision(response())
    assert decision.classification == "rejected"
    assert response_id == "actual-shaped-response-id"


@pytest.mark.parametrize(
    "item",
    [
        response(calls=0),
        response(calls=2),
        response(name="execute_python"),
        response(arguments="{invalid"),
        response(arguments='{"classification":"accepted"}'),
        response(
            arguments=json.dumps(
                {
                    "classification": "accepted",
                    "object_id": "part-001",
                    "summary": "OK",
                    "execute": True,
                }
            )
        ),
    ],
)
def test_missing_multiple_or_invalid_tools_never_become_a_successful_inspection(item):
    with pytest.raises(Problem):
        extract_decision(item)
