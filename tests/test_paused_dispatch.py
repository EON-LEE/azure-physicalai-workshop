"""CPU wire and authority tests for a new mode, never a relaxed real-time command."""

import json
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from runtime_support import ACTOR, OTHER, PNG, service
from test_paused_environment import paused_document

from apps.api.errors import Problem
from apps.api.models import MotionCommand, SaveEnvironment, utcnow
from learning.paused import PausedControlProfile
from simulation.core import SimulationCore
from simulation.extensions import SceneRegistry
from simulation.paused_contracts import ResolvedSimulationAuthorization, SimulationEpisodeCommand


@pytest.fixture
def paused_core():
    document = paused_document()
    document["execution"].update(record_demonstration=True, demonstration_split="test")
    record = service().save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(document)))
    core = SimulationCore(SceneRegistry(load_installed=False))
    core.paused_profile = PausedControlProfile("a" * 64)
    core.activate(ACTOR.owner_key, record)
    core.next_action()
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (0.35, 0.25, 0.2), 5, epoch=core.epoch)
    observation = core.observe(
        ACTOR.owner_key, record.environment_id, record.revision, "inspection"
    )
    request = SimulationEpisodeCommand(
        schema="physicalai.simulation-episode-command/v1",
        execution_timing="paused_simulation",
        real_time_admission=False,
        profile_id="franka-position-hold-10hz-paused-v1",
        command_id=uuid4(),
        environment_id=record.environment_id,
        revision=record.revision,
        epoch=core.epoch,
        state_revision=core.state_revision,
        observation_id=observation.observation_id,
        object_id=observation.object_id,
        target_station_id="rejected",
        task={
            "task_id": "place-part",
            "instruction": "Place the part in the tray.",
            "goal_id": "rejected",
        },
        wall_expires_at=utcnow() + timedelta(seconds=599),
        max_simulation_steps=1800,
        controller="reference_controller",
        authorization_kind="reference_collection",
        authorization_id=uuid4(),
    )

    class Authority:
        def authorize(self, owner, command, profile):
            if (
                owner != ACTOR.owner_key
                or command.authorization_id != request.authorization_id
                or command.authorization_kind != request.authorization_kind
                or command.controller != request.controller
                or command.task != request.task
                or command.environment_id != request.environment_id
                or command.revision != request.revision
            ):
                raise Problem(
                    403, "paused_authorization_missing", "No matching approved authority."
                )
            return ResolvedSimulationAuthorization(
                authorization_id=request.authorization_id,
                authorization_kind=request.authorization_kind,
                owner=ACTOR.owner_key,
                environment_id=request.environment_id,
                revision=request.revision,
                controller=request.controller,
                task=request.task,
                control_profile_sha256=profile.sha256,
                wall_expires_at=request.wall_expires_at,
                max_episode_wall_seconds=600,
                max_simulation_steps=1800,
                purpose="integration",
                criteria_sha256="c" * 64,
                frozen_plan_sha256="d" * 64,
            )

    core.paused_authorizer = Authority()
    return core, record, request


def test_new_paused_command_uses_explicit_scene_wall_budget_not_legacy_deadline(paused_core):
    core, record, request = paused_core
    result = core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert result.status == "queued"
    assert result.simulation_runtime.execution_timing == "paused_simulation"
    assert result.simulation_runtime.real_time_admission is False
    assert result.simulation_runtime.simulation_steps == 0
    assert core.deadlines[core.active_command] == request.wall_expires_at
    assert record.document["execution"]["max_step_seconds"] == 30
    assert core.next_action().request == request
    assert core.dispatch_simulation_episode(ACTOR.owner_key, request) == result
    assert core.next_action() is None


def test_paused_admission_is_disabled_without_a_separate_installed_profile(paused_core):
    core, _, request = paused_core
    core.paused_profile = None
    with pytest.raises(Problem, match="profile"):
        core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert core.active_command is None


def test_an_opaque_uuid_is_not_reference_collection_authority_by_itself(paused_core):
    core, _, request = paused_core
    core.paused_authorizer = None
    with pytest.raises(Problem, match="authority"):
        core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert core.active_command is None


def test_reference_collection_must_resolve_its_exact_approved_authorization_record(paused_core):
    core, _, request = paused_core
    with pytest.raises(Problem, match="authority"):
        core.dispatch_simulation_episode(
            ACTOR.owner_key, request.model_copy(update={"authorization_id": uuid4()})
        )
    assert core.active_command is None


def test_misbound_resolver_record_cannot_authorize_the_new_episode(paused_core):
    from types import SimpleNamespace

    core, _, request = paused_core
    result = core.paused_authorizer.authorize(ACTOR.owner_key, request, core.paused_profile)
    core.paused_authorizer = SimpleNamespace(
        authorize=lambda *args: result.model_copy(update={"owner": OTHER.owner_key})
    )
    with pytest.raises(Problem, match="authority"):
        core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert core.active_command is None


@pytest.mark.parametrize(
    "changed",
    [
        {"epoch": uuid4()},
        {"revision": "f" * 64},
        {"max_simulation_steps": 7},
        {"wall_expires_at": None},
    ],
)
def test_context_and_budgets_cannot_be_changed_to_gain_extra_paused_authority(paused_core, changed):
    core, _, request = paused_core
    if "wall_expires_at" in changed:
        changed = {"wall_expires_at": utcnow() + timedelta(seconds=601)}
    with pytest.raises((Problem, ValidationError)):
        core.dispatch_simulation_episode(ACTOR.owner_key, request.model_copy(update=changed))
    assert core.active_command is None


def test_owner_cancel_and_request_id_tombstones_still_win_before_dispatch(paused_core):
    core, _, request = paused_core
    core.cancel(ACTOR.owner_key, request.command_id)
    result = core.dispatch_simulation_episode(ACTOR.owner_key, request)
    assert result.status == "cancelled"
    assert core.active_command is None
    assert core.next_action() is None
    with pytest.raises(Problem):
        core.dispatch_simulation_episode(OTHER.owner_key, request)


def test_new_simulation_mode_cannot_be_selected_by_an_old_motion_command(paused_core):
    core, record, request = paused_core
    command = MotionCommand(
        **request.model_dump(
            include={
                "command_id",
                "environment_id",
                "revision",
                "epoch",
                "state_revision",
                "observation_id",
                "object_id",
                "target_station_id",
            }
        ),
        deadline=request.wall_expires_at,
    )
    with pytest.raises(Problem, match="deadline"):
        core.dispatch(ACTOR.owner_key, command)
    assert record.document["execution"]["max_step_seconds"] == 30


def test_unknown_policy_paths_raw_joints_and_real_time_claims_fail_wire_validation(paused_core):
    _, _, request = paused_core
    for fields in (
        {"joint_positions": [0] * 9},
        {"model_url": "https://example.test/model"},
        {"real_time_admission": True},
        {"real_time_admission": 0},
        {"execution_timing": "realtime"},
        {"max_simulation_steps": True},
    ):
        with pytest.raises(ValidationError):
            SimulationEpisodeCommand.model_validate(request.model_dump(by_alias=True) | fields)
