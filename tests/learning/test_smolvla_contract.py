from dataclasses import asdict
from pathlib import Path

import pytest

from learning.checks.fixtures import JOINTS, SCOPE, frame
from learning.checks.groot_fixtures import PROFILE, TASK
from learning.common import ContractError
from learning.inference import ControlContext, PolicyObservation


def base_config():
    return {
        "type": "smolvla",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "push_to_hub": True,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
    }


def test_franka_adaptation_uses_nine_real_features_two_cameras_and_no_aloha_conversion():
    from learning.smolvla.adaptation import adapted_config

    original = base_config()
    result = adapted_config(original, backbone_path=Path("/approved/backbone"))
    assert original["input_features"]["observation.state"]["shape"] == [6]
    assert result["input_features"]["observation.state"]["shape"] == [9]
    assert result["output_features"]["action"]["shape"] == [9]
    assert set(result["input_features"]) == {
        "observation.state",
        "observation.images.inspection",
        "observation.images.overview",
    }
    assert result["empty_cameras"] == 0
    assert result["n_action_steps"] == 1 and result["chunk_size"] == 50
    assert result["max_state_dim"] == result["max_action_dim"] == 32
    assert result["push_to_hub"] is False
    assert result["adapt_to_pi_aloha"] is False and result["use_delta_joint_actions_aloha"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"type": "act"},
        {"type": "gr00t_n1_5"},
        {"max_action_dim": 6},
        {"adapt_to_pi_aloha": True},
        {"use_delta_joint_actions_aloha": True},
    ],
)
def test_smol_initialization_never_aliases_a_different_policy_or_units(change):
    from learning.smolvla.adaptation import adapted_config

    with pytest.raises(ContractError):
        adapted_config({**base_config(), **change}, backbone_path=Path("/approved/backbone"))


def test_smol_model_and_backbone_have_distinct_exact_apache_identity():
    from learning.smolvla import ACTION_HORIZON, POLICY_TYPE, UPSTREAM

    assert POLICY_TYPE == "smolvla" and ACTION_HORIZON == 50
    assert UPSTREAM["source_commit"] == "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
    assert UPSTREAM["model_revision"] == "d9f33c94a60fb382c90dea2164c96845bd955e28"
    assert UPSTREAM["backbone_revision"] == "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    assert UPSTREAM["model_license"] == UPSTREAM["backbone_license"] == "Apache-2.0"


def test_explicit_smol_ipc_does_not_masquerade_as_a_gr00t_chunk():
    from learning.smolvla.ipc import make_request, make_response, validate_response

    sample = frame(0)
    observation = PolicyObservation(
        SCOPE,
        "line",
        "a" * 64,
        "episode",
        sample.captured_at_utc,
        sample.monotonic_ns,
        sample.physics_step,
        sample.joint_positions,
        sample.images,
    )
    context = ControlContext(
        SCOPE,
        "line",
        "a" * 64,
        "episode",
        "epoch",
        "command",
        TASK.goal_id,
        True,
        True,
        2_000_000_000,
    )
    request = make_request(
        observation,
        context,
        model_sha256="b" * 64,
        profile=PROFILE,
        task=TASK,
        request_id="request",
        sequence=0,
    )
    response = make_response(request, (JOINTS,) * 50, inference_latency_ms=10)
    assert request["policy_type"] == response["policy_type"] == "smolvla"
    assert request["schema"] == "physicalai.smolvla-request/v1"
    assert validate_response(response, request) == (JOINTS,) * 50
    with pytest.raises(ContractError):
        make_response(request, (JOINTS,) * 16, inference_latency_ms=10)
    assert asdict(PROFILE)["control_hz"] == 10
