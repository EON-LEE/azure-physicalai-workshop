from dataclasses import replace

import pytest

from learning.checks.fixtures import JOINTS, SCOPE, frame
from learning.checks.groot_fixtures import PROFILE, TASK
from learning.common import ContractError
from learning.inference import ControlContext, PolicyObservation


def values():
    sample = frame(0)
    observation = PolicyObservation(
        SCOPE,
        "customer-line",
        "f" * 64,
        "episode-1",
        sample.captured_at_utc,
        sample.monotonic_ns,
        sample.physics_step,
        sample.joint_positions,
        sample.images,
    )
    context = ControlContext(
        SCOPE,
        "customer-line",
        "f" * 64,
        "episode-1",
        "epoch-1",
        "command-1",
        TASK.goal_id,
        True,
        True,
        2_000_000_000,
    )
    return observation, context


def test_roundtrip_strict_ipc_binds_every_control_authority():
    from learning.gr00t.ipc import make_request, make_response, validate_request, validate_response

    observation, context = values()
    request = make_request(
        observation,
        context,
        model_sha256="a" * 64,
        profile=PROFILE,
        task=TASK,
        request_id="request-1",
        sequence=0,
    )
    validated = validate_request(
        request, model_sha256="a" * 64, scope=SCOPE, profile=PROFILE, task=TASK
    )
    assert validated[0] == observation and validated[1] == context
    response = make_response(request, (JOINTS,) * 16, inference_latency_ms=40.0)
    assert validate_response(response, request) == (JOINTS,) * 16
    assert response["policy_type"] == "gr00t_n1_5"
    assert "defect" not in request and "object_pose" not in request


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "b" * 64),
        ("request_id", "replayed"),
        ("sequence", 1),
        ("observation_sha256", "c" * 64),
        ("context_sha256", "d" * 64),
        ("control_profile_sha256", "e" * 64),
        ("task_sha256", "f" * 64),
        ("policy_type", "act"),
    ],
)
def test_ipc_rejects_swapped_or_replayed_predictions(field, value):
    from learning.gr00t.ipc import make_request, make_response, validate_response

    observation, context = values()
    request = make_request(
        observation,
        context,
        model_sha256="a" * 64,
        profile=PROFILE,
        task=TASK,
        request_id="request-1",
        sequence=0,
    )
    response = make_response(request, (JOINTS,) * 16, inference_latency_ms=10.0)
    response[field] = value
    with pytest.raises(ContractError):
        validate_response(response, request)


def test_ipc_rejects_unapproved_goal_and_unsafe_action():
    from learning.gr00t.ipc import make_request, make_response, validate_request

    observation, context = values()
    request = make_request(
        observation,
        replace(context, destination_id="unapproved"),
        model_sha256="a" * 64,
        profile=PROFILE,
        task=TASK,
        request_id="r",
        sequence=0,
    )
    with pytest.raises(ContractError):
        validate_request(request, model_sha256="a" * 64, scope=SCOPE, profile=PROFILE, task=TASK)
    with pytest.raises(ContractError):
        make_response(request, ((float("nan"),) * 9,) * 16, inference_latency_ms=10.0)
