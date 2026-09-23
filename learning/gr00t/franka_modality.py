"""Fixed N1.5 modality; no caller-selected Python data-config or processor."""

from dataclasses import asdict

from learning.contract import ControlProfile
from learning.gr00t import ACTION_HORIZON


def config_spec() -> dict:
    profile = asdict(ControlProfile(servo_profile_sha256="0" * 64))
    profile["servo_profile_sha256"] = None
    return {
        "embodiment_tag": "new_embodiment",
        "video": ["video.inspection", "video.overview"],
        "state": ["state.arm", "state.fingers"],
        "action": ["action.arm", "action.fingers"],
        "language": ["annotation.human.task_description"],
        "observation_indices": [0],
        "action_indices": list(range(ACTION_HORIZON)),
        "control_profile": profile,
    }


class FrankaDataConfig:
    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        spec = config_spec()
        return {
            name: ModalityConfig(
                delta_indices=spec["action_indices"] if name == "action" else [0],
                modality_keys=spec[name],
            )
            for name in ("video", "state", "action", "language")
        }

    def transform(self):
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
        from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
        from gr00t.model.transforms import GR00TTransform

        spec = config_spec()
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=spec["video"]),
                VideoResize(apply_to=spec["video"], height=224, width=224, interpolation="linear"),
                VideoToNumpy(apply_to=spec["video"]),
                StateActionToTensor(apply_to=spec["state"]),
                StateActionTransform(
                    apply_to=spec["state"],
                    normalization_modes={key: "min_max" for key in spec["state"]},
                ),
                StateActionToTensor(apply_to=spec["action"]),
                StateActionTransform(
                    apply_to=spec["action"],
                    normalization_modes={key: "min_max" for key in spec["action"]},
                ),
                ConcatTransform(
                    video_concat_order=spec["video"],
                    state_concat_order=spec["state"],
                    action_concat_order=spec["action"],
                ),
                GR00TTransform(
                    state_horizon=1,
                    action_horizon=ACTION_HORIZON,
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
        )
