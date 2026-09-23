"""Separate N1.7 source/API compatibility contract. Weight use is license-blocked."""

from __future__ import annotations

from io import BytesIO

from learning.common import keys, require, vector
from learning.contract import DemonstrationSource, bounded_joints
from learning.gr00t.artifacts import _no_dynamic_configuration
from learning.gr00t.licensing import require_commercial_model
from learning.inference import PolicyObservation

POLICY_TYPE = "gr00t_n1_7"
SOURCE_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"
MODEL_ID = "nvidia/GR00T-N1.7-3B"
MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BACKBONE_ID = "nvidia/Cosmos-Reason2-2B"
BACKBONE_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
ACTION_HORIZON = 40
PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.7.1"
TRANSFORMERS_VERSION = "4.57.3"
CUDA_VERSION = "12.8"


def modality_config() -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": ["inspection", "overview"]},
        "state": {"delta_indices": [0], "modality_keys": ["arm", "fingers"]},
        "action": {
            "delta_indices": list(range(ACTION_HORIZON)),
            "modality_keys": ["arm", "fingers"],
            "action_configs": [
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": key}
                for key in ("arm", "fingers")
            ],
        },
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
    }


def validate_generation(config: dict, processor: dict) -> None:
    _no_dynamic_configuration(config)
    _no_dynamic_configuration(processor)
    require(
        config.get("model_type") == "Gr00tN1d7"
        and config.get("architectures") == ["Gr00tN1d7"]
        and config.get("action_horizon") == ACTION_HORIZON
        and config.get("max_state_dim") == config.get("max_action_dim") == 132
        and config.get("model_name") == BACKBONE_ID
        and processor.get("processor_class") == "Gr00tN1d7Processor",
        "Wrong N1.7 generation, actual pinned horizon, backbone or processor",
    )


def observation_for_backend(observation: PolicyObservation, task: DemonstrationSource) -> dict:
    import numpy as np
    from PIL import Image

    task.validate()
    joints = bounded_joints(observation.joint_positions, "N1.7 observed joints")
    images = {}
    for name in ("inspection", "overview"):
        with Image.open(BytesIO(observation.images[name].png)) as image:
            images[name] = np.asarray(image.convert("RGB"), dtype=np.uint8)[None, None, ...]
    return {
        "video": images,
        "state": {
            "arm": np.asarray([[joints[:7]]], dtype=np.float32),
            "fingers": np.asarray([[joints[7:]]], dtype=np.float32),
        },
        "language": {"annotation.human.task_description": [[task.instruction]]},
    }


def physical_actions(response: object) -> tuple[tuple[float, ...], ...]:
    require(
        isinstance(response, tuple) and len(response) == 2, "N1.7 returns (action, info), not N1.5"
    )
    action, info = response
    require(isinstance(info, dict), "Invalid N1.7 auxiliary output")
    keys(action, {"arm", "fingers"}, "N1.7 physical actions")
    values = {
        name: value.tolist() if hasattr(value, "tolist") else value
        for name, value in action.items()
    }
    require(
        all(isinstance(value, list) and len(value) == 1 for value in values.values()),
        "N1.7 requires one explicit batch",
    )
    arm, fingers = values["arm"][0], values["fingers"][0]
    require(len(arm) == len(fingers) == ACTION_HORIZON, "N1.7 actual checkpoint horizon mismatch")
    return tuple(
        bounded_joints((*vector(a, 7, "N1.7 arm"), *vector(f, 2, "N1.7 fingers")), "N1.7 target")
        for a, f in zip(arm, fingers, strict=True)
    )


def authorize_weight_use() -> None:
    require_commercial_model(POLICY_TYPE, MODEL_REVISION)
