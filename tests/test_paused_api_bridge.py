from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from apps.api.bridge_client import AzureSimulatorBridge
from apps.api.models import MotionCommand
from tests.runtime_support import ACTOR
from tests.test_paused_wire_contracts import command_payload


def api_models():
    from apps.api import simulation_models

    return simulation_models


def test_api_command_roundtrips_the_committed_runtime_wire_without_clock_aliases():
    from simulation.paused_contracts import SimulationEpisodeCommand

    payload = command_payload()
    parsed = api_models().SimulationEpisodeCommand.model_validate(payload)
    wire = parsed.model_dump(mode="json", by_alias=True)
    assert (
        SimulationEpisodeCommand.model_validate(wire).model_dump(mode="json", by_alias=True) == wire
    )
    assert "deadline" not in wire and "execution_mode" not in wire
    with pytest.raises(ValidationError):
        MotionCommand.model_validate(wire)


@pytest.mark.parametrize(
    "change",
    [
        {"real_time_admission": True},
        {"real_time_admission": 0},
        {"timing_mode": "paused_simulation"},
        {"max_simulation_steps": 7},
        {"max_simulation_steps": 1806},
        {"controller": "learned"},
        {"policy_type": "smolvla", "model_sha256": "b" * 64},
    ],
)
def test_api_episode_has_explicit_new_mode_and_no_reference_model_masquerade(change):
    with pytest.raises(ValidationError):
        api_models().SimulationEpisodeCommand.model_validate(command_payload() | change)


def runtime_receipt(command):
    from simulation.paused_contracts import PausedRuntimeMetrics, SimulationEpisodeExecution

    return SimulationEpisodeExecution(
        command_id=command.command_id,
        status="running",
        simulation_runtime=PausedRuntimeMetrics(
            control_profile_sha256="b" * 64,
            controller="reference_controller",
            phase="observing",
            wall_elapsed_ms=2000,
            simulation_steps=6,
            simulation_elapsed_seconds=0.1,
            applied_action_count=6,
            reference_route_calls=1,
        ),
    ).model_dump(mode="json")


def test_api_preserves_real_wall_and_simulation_evidence_from_runtime():
    command = api_models().SimulationEpisodeCommand.model_validate(command_payload())
    wire = runtime_receipt(command)
    received = api_models().SimulationEpisodeExecution.model_validate(wire)
    assert received.model_dump(mode="json") == wire
    assert received.policy_runtime is None
    assert received.simulation_runtime.wall_elapsed_ms == 2000
    assert received.simulation_runtime.simulation_elapsed_seconds == 0.1
    assert received.simulation_runtime.real_time_admission is False
    missing = {
        **wire,
        "simulation_runtime": {
            key: value
            for key, value in wire["simulation_runtime"].items()
            if key != "wall_elapsed_ms"
        },
    }
    with pytest.raises(ValidationError):
        api_models().SimulationEpisodeExecution.model_validate(missing)


def test_bridge_uses_separate_private_episode_routes_and_never_retries_motion():
    command = api_models().SimulationEpisodeCommand.model_validate(command_payload())
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=runtime_receipt(command))

    bridge = AzureSimulatorBridge(
        "https://test-controller.internal",
        SimpleNamespace(get_token=lambda _: SimpleNamespace(token="test-only-api-mi")),
        "api://test-controller/.default",
        transport=httpx.MockTransport(handler),
    )
    result = bridge.dispatch_simulation_episode(ACTOR.owner_key, command)
    assert result.simulation_runtime.execution_timing == "paused_simulation"
    assert (
        bridge.simulation_episode(ACTOR.owner_key, command.command_id).command_id
        == command.command_id
    )
    assert (
        bridge.cancel_simulation_episode(ACTOR.owner_key, command.command_id).command_id
        == command.command_id
    )
    assert [(call.method, call.url.path) for call in calls] == [
        ("POST", "/v1/simulation-episodes"),
        ("GET", f"/v1/simulation-episodes/{command.command_id}"),
        ("POST", f"/v1/simulation-episodes/{command.command_id}/cancel"),
    ]
    assert all(call.headers["X-Environment-Owner"] == ACTOR.owner_key for call in calls)
    bridge.close()

    def lost(request):
        calls.append(request)
        raise httpx.RemoteProtocolError("Test-only uncertain episode POST", request=request)

    bridge = AzureSimulatorBridge(
        "https://test-controller.internal",
        SimpleNamespace(get_token=lambda _: SimpleNamespace(token="test-only-api-mi")),
        "api://test-controller/.default",
        transport=httpx.MockTransport(lost),
    )
    from apps.api.errors import Problem

    with pytest.raises(Problem):
        bridge.dispatch_simulation_episode(ACTOR.owner_key, command)
    assert len(calls) == 4
    bridge.close()


def test_resolved_authority_requires_original_owner_and_frozen_condition_bindings():
    from simulation.paused_contracts import ResolvedSimulationAuthorization

    command = api_models().SimulationEpisodeCommand.model_validate(command_payload())
    native = ResolvedSimulationAuthorization(
        authorization_id=command.authorization_id,
        authorization_kind=command.authorization_kind,
        owner=ACTOR.owner_key,
        environment_id=command.environment_id,
        revision=command.revision,
        controller=command.controller,
        task=command.task.model_dump(),
        control_profile_sha256="b" * 64,
        wall_expires_at=command.wall_expires_at,
        max_episode_wall_seconds=600,
        max_simulation_steps=1800,
        purpose="integration",
        criteria_sha256="c" * 64,
        frozen_plan_sha256="d" * 64,
    )
    parsed = api_models().ResolvedSimulationAuthorization.model_validate(
        native.model_dump(mode="json")
    )
    assert parsed.authorization_id == UUID(str(command.authorization_id))
    with pytest.raises(ValidationError):
        api_models().ResolvedSimulationAuthorization.model_validate(
            {
                "authorization_id": str(command.authorization_id),
            }
        )
