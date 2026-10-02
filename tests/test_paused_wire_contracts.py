"""Closed CPU-only paused bridge schema; this is not runtime or policy admission."""

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.models import utcnow
from simulation.paused_contracts import (
    PausedRuntimeMetrics,
    ResolvedSimulationAuthorization,
    SimulationEpisodeCommand,
    SimulationEpisodeExecution,
)


def command_payload():
    return {
        "schema": "physicalai.simulation-episode-command/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "profile_id": "franka-position-hold-10hz-paused-v1",
        "command_id": str(uuid4()),
        "environment_id": "paused-case",
        "revision": "a" * 64,
        "epoch": str(uuid4()),
        "state_revision": 1,
        "observation_id": str(uuid4()),
        "object_id": "part-001",
        "target_station_id": "rejected",
        "task": {"task_id": "place-part", "instruction": "Place the part.", "goal_id": "rejected"},
        "wall_expires_at": (utcnow() + timedelta(seconds=600)).isoformat(),
        "max_simulation_steps": 1800,
        "controller": "reference_controller",
        "authorization_kind": "reference_collection",
        "authorization_id": str(uuid4()),
    }


def test_closed_command_roundtrip_has_one_clock_vocabulary_and_no_legacy_deadline():
    value = SimulationEpisodeCommand.model_validate(command_payload())
    wire = value.model_dump(mode="json", by_alias=True)
    assert wire["schema"] == "physicalai.simulation-episode-command/v1"
    assert wire["real_time_admission"] is False
    assert "deadline" not in wire and "timing_mode" not in wire
    assert SimulationEpisodeCommand.model_validate(wire) == value
    assert SimulationEpisodeCommand.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"real_time_admission": True},
        {"real_time_admission": 0},
        {"max_simulation_steps": 1801},
        {"deadline": (utcnow() + timedelta(seconds=600)).isoformat()},
        {"joint_positions": [0] * 9},
        {"controller": "learned"},
    ],
)
def test_wire_cannot_invent_runtime_admission_or_raw_actuator_authority(change):
    with pytest.raises(ValidationError):
        SimulationEpisodeCommand.model_validate(command_payload() | change)


def test_non_realtime_execution_keeps_actual_metrics_out_of_legacy_policy_runtime():
    command = SimulationEpisodeCommand.model_validate(command_payload())
    value = SimulationEpisodeExecution(
        command_id=command.command_id,
        status="queued",
        simulation_runtime=PausedRuntimeMetrics(
            control_profile_sha256="b" * 64,
            controller="reference_controller",
        ),
    )
    wire = value.model_dump(mode="json")
    assert wire["policy_runtime"] is None
    assert wire["simulation_runtime"]["real_time_admission"] is False
    assert wire["simulation_runtime"]["display_label"] == "NON_REALTIME_SIMULATION"


def test_resolved_authority_contains_reviewed_purpose_and_frozen_plan_not_just_uuid():
    command = SimulationEpisodeCommand.model_validate(command_payload())
    record = ResolvedSimulationAuthorization(
        authorization_id=command.authorization_id,
        authorization_kind=command.authorization_kind,
        owner="a" * 64,
        environment_id=command.environment_id,
        revision=command.revision,
        controller=command.controller,
        task=command.task,
        control_profile_sha256="b" * 64,
        wall_expires_at=command.wall_expires_at,
        max_episode_wall_seconds=600,
        max_simulation_steps=1800,
        purpose="integration",
        criteria_sha256="c" * 64,
        frozen_plan_sha256="d" * 64,
    )
    assert record.model_dump(mode="json")["purpose"] == "integration"
    with pytest.raises(ValidationError):
        record.owner = "e" * 64
