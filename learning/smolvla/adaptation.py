from __future__ import annotations

import copy
from pathlib import Path

from learning.common import require
from learning.contract import CAMERAS
from learning.gr00t.artifacts import _no_dynamic_configuration
from learning.smolvla import ACTION_HORIZON, BACKBONE_ID, POLICY_TYPE

IMAGE_SIZE = 256


def adapted_config(original: dict, *, backbone_path: Path) -> dict:
    _no_dynamic_configuration(original)
    require(
        original.get("type") == POLICY_TYPE
        and original.get("chunk_size") == ACTION_HORIZON
        and original.get("max_state_dim") == original.get("max_action_dim") == 32,
        "Not the pinned SmolVLA architecture",
    )
    require(
        not original.get("adapt_to_pi_aloha") and not original.get("use_delta_joint_actions_aloha"),
        "Aloha angular/delta transformations cannot represent Franka metres/radians",
    )
    require(backbone_path.is_absolute(), "Backbone must be an approved local absolute path")
    require(
        original.get("vlm_model_name") == BACKBONE_ID
        or Path(original["vlm_model_name"]).is_absolute(),
        "Unapproved SmolVLM backbone",
    )
    result = copy.deepcopy(original)
    result.update(
        {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [9]},
                **{
                    f"observation.images.{name}": {
                        "type": "VISUAL",
                        "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
                    }
                    for name in CAMERAS
                },
            },
            "output_features": {"action": {"type": "ACTION", "shape": [9]}},
            "n_obs_steps": 1,
            "n_action_steps": 1,
            "empty_cameras": 0,
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
            "push_to_hub": False,
            "repo_id": None,
            "pretrained_path": None,
            "vlm_model_name": str(backbone_path),
            "load_vlm_weights": True,
            "rtc_config": None,
            "compile_model": False,
        }
    )
    return result


def validate_franka_config(config: dict) -> None:
    _no_dynamic_configuration(config)
    require(
        config.get("type") == POLICY_TYPE
        and config.get("input_features", {}).get("observation.state", {}).get("shape") == [9]
        and config.get("output_features") == {"action": {"type": "ACTION", "shape": [9]}}
        and set(config["input_features"])
        == {"observation.state", *(f"observation.images.{c}" for c in CAMERAS)}
        and config.get("chunk_size") == ACTION_HORIZON
        and config.get("n_action_steps") == 1
        and config.get("empty_cameras") == 0
        and config.get("max_state_dim") == config.get("max_action_dim") == 32
        and config.get("push_to_hub") is False
        and not config.get("adapt_to_pi_aloha")
        and not config.get("use_delta_joint_actions_aloha")
        and config.get("rtc_config") is None,
        "Checkpoint is not actually adapted to the approved nine-joint/two-camera SmolVLA contract",
    )
    for camera in CAMERAS:
        require(
            config["input_features"][f"observation.images.{camera}"]
            == {"type": "VISUAL", "shape": [3, IMAGE_SIZE, IMAGE_SIZE]},
            "Wrong real camera tensor shape",
        )
