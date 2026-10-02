"""Installed upstream API/preprocessing check; no vendor weights, optimizer or quality claim."""

import inspect
import json
from importlib.metadata import version
from types import SimpleNamespace

from learning.checks.fixtures import JOINTS
from learning.offline import enforce_offline


def run() -> dict:
    enforce_offline()
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    assert version("lerobot") == "0.4.4" and version("transformers") == "4.57.3"
    config = SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
            **{
                f"observation.images.{name}": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 256, 256),
                )
                for name in ("inspection", "overview")
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
        device="cpu",
        n_action_steps=1,
        empty_cameras=0,
        push_to_hub=False,
    )
    config.validate_features()
    context = SimpleNamespace(config=config)
    measured = torch.tensor([JOINTS], dtype=torch.float32)
    padded = SmolVLAPolicy.prepare_state(context, {"observation.state": measured})
    assert padded.shape == (1, 32) and torch.equal(padded[:, :9], measured)
    assert (padded[:, 9:] == 0).all()
    batch = {
        f"observation.images.{name}": torch.zeros((1, 3, 256, 256))
        for name in ("inspection", "overview")
    }
    images, masks = SmolVLAPolicy.prepare_images(context, batch)
    assert len(images) == len(masks) == 2 and all(mask.all() for mask in masks)
    return {
        "policy_type": "smolvla",
        "lerobot": version("lerobot"),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "actual_api": str(inspect.signature(SmolVLAPolicy.predict_action_chunk)),
        "physical_state_dimensions": 9,
        "model_internal_padded_dimensions": 32,
        "actual_input_camera_count": 2,
        "fabricated_observation_cameras": 0,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "test_only_synthetic_inputs": True,
        "vendor_weights_loaded": False,
        "optimizer_steps": 0,
        "learning_quality_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
