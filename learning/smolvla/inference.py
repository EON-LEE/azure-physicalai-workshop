from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from learning.common import ContractError, read_json, require
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.gr00t.inference import serve
from learning.inference import PolicyObservation
from learning.offline import require_lerobot
from learning.smolvla import ACTION_HORIZON, POLICY_TYPE
from learning.smolvla.artifacts import validate_backbone, validate_model

if TYPE_CHECKING:
    from learning.paused.contract import PausedControlProfile


class LocalSmolVLAPolicy:
    policy_type = POLICY_TYPE
    fps, physics_hz, chunk_size, n_action_steps = 10, 60, ACTION_HORIZON, 1

    def __init__(
        self,
        root: Path,
        *,
        backbone_root: Path,
        scope: Scope,
        model_sha256: str,
        expected_control_profile_sha256: str,
    ) -> None:
        metadata = validate_model(root, expected_scope=scope, expected_model_sha256=model_sha256)
        self._load(
            root,
            backbone_root=backbone_root,
            scope=scope,
            model_sha256=model_sha256,
            expected_control_profile_sha256=expected_control_profile_sha256,
            metadata=metadata,
            profile=ControlProfile(**metadata["control_profile"]),
        )

    def _load(
        self,
        root: Path,
        *,
        backbone_root: Path,
        scope: Scope,
        model_sha256: str,
        expected_control_profile_sha256: str,
        metadata: dict,
        profile: ControlProfile | PausedControlProfile,
    ) -> None:
        self.metadata, self.profile = metadata, profile
        require(
            self.profile.sha256 == expected_control_profile_sha256, "Unapproved Smol servo profile"
        )
        self.task = DemonstrationSource(kind="reference_controller", **self.metadata["task"])
        backbone = validate_backbone(
            backbone_root,
            scope=scope,
            expected_sha256=self.metadata["backbone_manifest_sha256"],
        )
        self.scope, self.model_sha256 = scope, model_sha256
        require_lerobot()
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        require(torch.cuda.is_available(), "Actual CUDA is required; no CPU/reference fallback")
        checkpoint = root / "checkpoint"
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        require(isinstance(config, SmolVLAConfig), "Not an actual SmolVLA checkpoint")
        config.vlm_model_name, config.device = str(backbone), "cuda"
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {"tokenizer_name": str(backbone)},
            },
        )

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_chunk(self, observation: PolicyObservation):
        import numpy as np
        import torch
        from PIL import Image

        require(observation.scope == self.scope, "Smol policy observation owner mismatch")
        inputs = {
            "observation.state": torch.tensor(observation.joint_positions, dtype=torch.float32),
            "task": self.task.instruction,
        }
        for name in ("inspection", "overview"):
            with Image.open(BytesIO(observation.images[name].png)) as image:
                pixels = np.array(
                    image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            inputs[f"observation.images.{name}"] = torch.from_numpy(pixels).permute(2, 0, 1) / 255
        with torch.inference_mode():
            actions = self.postprocessor(
                self.policy.predict_action_chunk(self.preprocessor(inputs))
            )
        require(
            tuple(actions.shape) == (1, ACTION_HORIZON, 9),
            "Wrong actual Smol physical action shape",
        )
        return tuple(tuple(action) for action in actions.detach().cpu()[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment-selected private SmolVLA policy process."
    )
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--backbone-root", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--allowed-client-uid", type=int)
    args = parser.parse_args()
    binding = read_json(args.binding)
    try:
        policy = LocalSmolVLAPolicy(
            args.model_root,
            backbone_root=args.backbone_root,
            scope=Scope(**binding["scope"]),
            model_sha256=args.model_sha256,
            expected_control_profile_sha256=ControlProfile(**binding["control_profile"]).sha256,
        )
        serve(policy, args.socket_path, allowed_client_uid=args.allowed_client_uid)
    except (ContractError, OSError) as exc:
        raise SystemExit(f"SmolVLA policy process stopped: {exc}") from exc


if __name__ == "__main__":
    main()
