"""Closed paused profile selection; v2 never widens a default or historical v1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from learning.common import read_json, require
from learning.paused import CONTROL_PROFILE_ID, CONTROL_PROFILE_V2_ID, PausedControlProfile

PausedProfileId = Literal[
    "franka-position-hold-10hz-paused-v1", "franka-position-hold-10hz-paused-v2"
]


def profile_step_limit(profile_id: str) -> int:
    if profile_id == CONTROL_PROFILE_ID:
        return 1800
    if profile_id == CONTROL_PROFILE_V2_ID:
        return 3600
    raise ValueError("An explicitly supported paused profile ID is required.")


def paused_profile(servo_sha256: str, profile_id: str = CONTROL_PROFILE_ID) -> PausedControlProfile:
    value = PausedControlProfile(
        servo_sha256, profile_id=profile_id, max_simulation_steps=profile_step_limit(profile_id)
    )
    value.validate()
    return value


def read_paused_attempt(path: Path, *, expected_profile_id: str) -> dict:
    profile_step_limit(expected_profile_id)
    limit = (8 if expected_profile_id == CONTROL_PROFILE_V2_ID else 4) * 1024**2
    value = read_json(path, max_bytes=limit)
    require(
        value.get("execution_timing") == "paused_simulation"
        and value.get("real_time_admission") is False
        and isinstance(value.get("control_profile"), dict),
        "An explicit paused report profile is required",
    )
    profile = PausedControlProfile(**value["control_profile"])
    profile.validate()
    require(
        profile.profile_id == expected_profile_id
        and value.get("control_profile_sha256") == profile.sha256,
        "The private report profile differs from the approved version",
    )
    return value
