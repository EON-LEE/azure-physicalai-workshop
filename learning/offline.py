from __future__ import annotations

import os
from importlib.metadata import version

from learning import LEROBOT_VERSION
from learning.common import require

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "DO_NOT_TRACK": "1",
}


def enforce_offline() -> None:
    os.environ.update(OFFLINE_ENV)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        require(not os.environ.get(name), f"{name} must not be present in the learning process")


def require_lerobot() -> None:
    enforce_offline()
    require(version("lerobot") == LEROBOT_VERSION, f"Install locked LeRobot {LEROBOT_VERSION}")
