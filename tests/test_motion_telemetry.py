from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.models import MotionTelemetry, SimulationStatus


def test_legacy_simulator_status_has_no_required_telemetry():
    status = SimulationStatus(status="unavailable")
    assert status.motion is None
    assert SimulationStatus.model_validate_json(status.model_dump_json()) == status


@pytest.mark.parametrize(
    "phase",
    [
        "idle",
        "approaching",
        "grasping",
        "lifting",
        "inspection_station",
        "transporting",
        "releasing",
        "returning",
        "complete",
        "stopped",
    ],
)
def test_motion_telemetry_round_trip(phase):
    status = SimulationStatus(
        status="ready",
        motion=MotionTelemetry(
            command_id=uuid4(),
            phase=phase,
            object_position=(0.35, 0.25, 0.2),
            target_station_id="accepted",
        ),
    )
    assert SimulationStatus.model_validate_json(status.model_dump_json()) == status


def test_idle_telemetry_allows_no_command_or_target():
    assert (
        MotionTelemetry(
            command_id=None,
            phase="idle",
            object_position=(0.35, 0.25, 0.2),
            target_station_id=None,
        ).command_id
        is None
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"phase": "pretend_complete"},
        {"object_position": [float("nan"), 0, 0]},
        {"object_position": [0, 0]},
        {"command_id": "not-a-command"},
        {"target_station_id": "../private"},
        {"private_instruction": "must not be accepted"},
    ],
)
def test_malformed_telemetry_fails_closed(updates):
    payload = {
        "command_id": None,
        "phase": "idle",
        "object_position": [0, 0, 0],
        "target_station_id": None,
        **updates,
    }
    with pytest.raises(ValidationError):
        MotionTelemetry.model_validate(payload)
