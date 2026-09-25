"""Explicit versioned CPU budgets; old failures/data are never re-admitted as v2."""

import json
from dataclasses import asdict, replace

import pytest
from runtime_support import ACTOR
from test_paused_control import state
from test_paused_dispatch import paused_core as paused_core
from test_paused_environment import paused_document, paused_v2_document
from test_paused_scene import scene
from test_paused_wire_contracts import command_payload

from apps.api.errors import Problem
from apps.api.models import utcnow
from learning.paused import CONTROL_PROFILE_ID, CONTROL_PROFILE_V2_ID
from simulation.core import SimulationCore
from simulation.extensions import PausedSceneAuthority, SceneRegistry
from simulation.paused_contracts import (
    PausedRuntimeMetrics,
    ResolvedSimulationAuthorization,
    SimulationEpisodeCommand,
)
from simulation.paused_control import PausedEpisode
from simulation.paused_profiles import paused_profile, read_paused_attempt


def permission(core, request):
    original = core.paused_authorizer.authorize(core.owner, request, core.paused_profile)
    return original.model_dump()


def test_v2_scene_has_an_explicit_sixty_second_budget_without_changing_legacy_wall_limit():
    spec, environment = scene(paused_v2_document())
    assert spec.require_paused_authority().max_simulation_steps == 3600
    assert environment.document["execution"]["max_step_seconds"] == 30
    assert spec.learning_execution.profile_id == CONTROL_PROFILE_V2_ID
    old, old_environment = scene(paused_document())
    assert old.require_paused_authority().max_simulation_steps == 1800
    assert old_environment.revision != environment.revision


@pytest.mark.parametrize(
    "schema,profile",
    [
        ("physicalai.paused-simulation/v1", CONTROL_PROFILE_V2_ID),
        ("physicalai.paused-simulation/v2", CONTROL_PROFILE_ID),
    ],
)
def test_scene_authority_rejects_cross_version_pairs_even_for_a_small_budget(schema, profile):
    with pytest.raises(ValueError):
        PausedSceneAuthority(schema, "paused_simulation", profile, 5, 600)


def test_new_wire_metrics_and_grant_allow_3600_only_with_explicit_v2(paused_core):
    core, _, original = paused_core
    payload = command_payload() | {
        "profile_id": CONTROL_PROFILE_V2_ID,
        "max_simulation_steps": 3600,
    }
    command = SimulationEpisodeCommand.model_validate(payload)
    assert command.max_simulation_steps == 3600
    for profile_id in (CONTROL_PROFILE_ID, "unknown"):
        with pytest.raises(ValueError):
            SimulationEpisodeCommand.model_validate(payload | {"profile_id": profile_id})
    data = permission(core, original)
    permit = ResolvedSimulationAuthorization.model_validate(
        data | {"profile_id": CONTROL_PROFILE_V2_ID, "max_simulation_steps": 3600}
    )
    assert permit.model_dump(mode="json")["profile_id"] == CONTROL_PROFILE_V2_ID
    assert "profile_id" not in ResolvedSimulationAuthorization.model_validate(data).model_dump()
    with pytest.raises(ValueError):
        ResolvedSimulationAuthorization.model_validate(data | {"max_simulation_steps": 3600})
    metrics = PausedRuntimeMetrics(
        profile_id=CONTROL_PROFILE_V2_ID,
        control_profile_sha256="a" * 64,
        controller="reference_controller",
        simulation_steps=3600,
        simulation_elapsed_seconds=60,
    )
    assert metrics.simulation_elapsed_seconds == 60
    with pytest.raises(ValueError):
        PausedRuntimeMetrics.model_validate(
            metrics.model_dump() | {"profile_id": CONTROL_PROFILE_ID}
        )


def test_pure_driver_requires_a_trusted_v2_profile_not_a_larger_generic_limit():
    arguments = {
        "wall_deadline_ns": 601_000_000_000,
        "started_ns": 1_000_000_000,
        "max_simulation_steps": 3600,
        "authorized": lambda: True,
        "clock_ns": lambda: 1_000_000_000,
    }
    with pytest.raises(ValueError):
        PausedEpisode(state(), **arguments)
    with pytest.raises(ValueError):
        PausedEpisode(state(), profile=paused_profile("a" * 64), **arguments)
    value = PausedEpisode(
        state(), profile=paused_profile("a" * 64, CONTROL_PROFILE_V2_ID), **arguments
    )
    assert value.max_simulation_steps == 3600


def test_v2_core_requires_matching_scene_command_resolved_authority_and_metrics(paused_core):
    from datetime import timedelta
    from uuid import uuid4

    from runtime_support import PNG

    original_core, _, original = paused_core
    document = paused_v2_document()
    document["execution"].update(record_demonstration=True, demonstration_split="test")
    _, environment = scene(document)
    profile = paused_profile("a" * 64, CONTROL_PROFILE_V2_ID)
    core = SimulationCore(SceneRegistry(load_installed=False), paused_profile=profile)
    core.activate(ACTOR.owner_key, environment)
    core.next_action()
    for camera in ("inspection", "overview"):
        core.publish_frame(camera, PNG, core.spec.part_position, 0, epoch=core.epoch)
    observation = core.observe(
        ACTOR.owner_key, environment.environment_id, environment.revision, "inspection"
    )
    request = SimulationEpisodeCommand.model_validate(
        command_payload()
        | {
            "command_id": str(uuid4()),
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "epoch": str(core.epoch),
            "state_revision": core.state_revision,
            "observation_id": str(observation.observation_id),
            "object_id": observation.object_id,
            "profile_id": CONTROL_PROFILE_V2_ID,
            "max_simulation_steps": 3600,
            "wall_expires_at": (utcnow() + timedelta(seconds=590)).isoformat(),
        }
    )
    permit = ResolvedSimulationAuthorization.model_validate(
        permission(original_core, original)
        | {
            "authorization_id": request.authorization_id,
            "environment_id": environment.environment_id,
            "revision": environment.revision,
            "task": request.task,
            "profile_id": CONTROL_PROFILE_V2_ID,
            "control_profile_sha256": profile.sha256,
            "max_simulation_steps": 3600,
            "wall_expires_at": request.wall_expires_at,
        }
    )

    class Authority:
        value = permit

        def authorize(self, *args):
            return self.value

    authority = Authority()
    core.paused_authorizer = authority
    authority.value = permit.model_copy(
        update={"profile_id": CONTROL_PROFILE_ID, "max_simulation_steps": 1800}
    )
    with pytest.raises(Problem, match="match"):
        core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert not core.pending and core.active_command is None
    authority.value = permit
    result = core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert result.simulation_runtime.profile_id == CONTROL_PROFILE_V2_ID
    assert core.next_action().request.max_simulation_steps == 3600
    core.begin_motion(request.command_id)
    from types import SimpleNamespace

    from simulation.paused_runtime import PausedReferenceRuntime

    initial = replace(state(), epoch=core.epoch, physics_step=80, world_time=80 / 60)
    hardware = SimpleNamespace(
        prepare_paused_reference=lambda *args: None,
        frozen_physics_state=lambda epoch: initial,
        _measured_tcp=lambda: (0.35, 0.25, 0.38),
    )
    driver = PausedReferenceRuntime(core, request, hardware)
    assert driver.episode.max_simulation_steps == 3600
    assert driver.metrics().profile_id == CONTROL_PROFILE_V2_ID


@pytest.mark.parametrize(
    "profile_id,megabytes,accepted",
    [
        (CONTROL_PROFILE_ID, 5, False),
        (CONTROL_PROFILE_V2_ID, 5, True),
        (CONTROL_PROFILE_V2_ID, 9, False),
    ],
)
def test_only_explicit_v2_private_attempts_have_an_eight_mib_read_budget(
    tmp_path, profile_id, megabytes, accepted
):
    path = tmp_path / "attempt.json"
    profile = paused_profile("a" * 64, profile_id)
    path.write_text(
        json.dumps(
            {
                "schema": "physicalai.paused-reference-attempt/v1",
                "execution_timing": "paused_simulation",
                "real_time_admission": False,
                "control_profile": asdict(profile),
                "control_profile_sha256": profile.sha256,
                "private_trace": "x" * (megabytes * 1024 * 1024),
            }
        )
    )
    if accepted:
        assert (
            read_paused_attempt(path, expected_profile_id=profile_id)["control_profile"][
                "profile_id"
            ]
            == profile_id
        )
        with pytest.raises(ValueError):
            read_paused_attempt(path, expected_profile_id=CONTROL_PROFILE_ID)
    else:
        with pytest.raises(ValueError):
            read_paused_attempt(path, expected_profile_id=profile_id)
