"""CPU lease/protocol tests; they do not validate human operation or GPU servo timing."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from runtime_support import ACTOR, OTHER, PNG, document, service

from apps.api.errors import Problem
from apps.api.models import SaveEnvironment
from learning.contract import ControlProfile
from simulation.core import SimulationCore
from simulation.extensions import SceneRegistry
from simulation.runtime_contracts import (
    CONTROL_PROFILE_ID,
    TeachingInput,
    TeachingLease,
    TeachingStart,
)


@pytest.fixture
def teaching():
    clock = [datetime(2026, 9, 23, tzinfo=UTC), 1_000_000_000]
    doc = document()
    doc["execution"].update(record_demonstration=True, demonstration_split="train")
    environment = service().save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))
    core = SimulationCore(
        SceneRegistry(load_installed=False),
        control_profile=ControlProfile("a" * 64),
        clock_ns=lambda: clock[1],
        clock_utc=lambda: clock[0],
    )
    core.activate(ACTOR.owner_key, environment)
    core.next_action()
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (0.35, 0.25, 0.2), 5, epoch=core.epoch)
    observation = core.observe(
        ACTOR.owner_key, environment.environment_id, environment.revision, "inspection"
    )
    request = TeachingStart(
        session_id=uuid4(),
        lease_id=uuid4(),
        command_id=uuid4(),
        environment_id=environment.environment_id,
        revision=environment.revision,
        epoch=core.epoch,
        state_revision=core.state_revision,
        observation_id=observation.observation_id,
        object_id=observation.object_id,
        target_station_id="rejected",
        session_expires_at=clock[0] + timedelta(seconds=300),
        control_profile_id=CONTROL_PROFILE_ID,
        task={
            "task_id": "place-part",
            "instruction": "Place the part in the tray.",
            "goal_id": "rejected",
        },
        demonstrator_kind="human_teleop",
    )
    return core, request, clock


def jog(request, clock, **changes):
    values = {
        "lease_id": request.lease_id,
        "epoch": request.epoch,
        "sequence": 1,
        "expires_at": clock[0] + timedelta(milliseconds=250),
        "deadman": True,
        "delta_xyz_m": (0.0, 0.0, 0.01),
        "gripper": "hold",
        "grant_id": uuid4(),
        "grant_expires_at": clock[0] + timedelta(seconds=1),
    }
    return TeachingInput(**(values | changes))


def begin(core, request):
    state = core.start_teaching(ACTOR.owner_key, request)
    assert state.status == "queued"
    action = core.next_action()
    assert action.request == request
    assert core.begin_motion(request.command_id)
    return core.binding(request.command_id)


def test_distinct_teaching_session_has_no_motion_authority_without_fresh_input(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    assert core.teaching_intent(binding) is None
    state = core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    assert state.last_sequence == 1
    intent = core.teaching_intent(binding)
    assert intent.delta_xyz_m == (0, 0, 0.01)
    assert intent.expires_at_monotonic_ns == clock[1] + 250_000_000
    clock[1] += 250_000_000
    assert core.teaching_intent(binding) is None
    assert core.command(ACTOR.owner_key, request.command_id).status == "running"


def test_start_and_input_retries_never_extend_original_authority(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    original = jog(request, clock)
    state = core.teaching_input(ACTOR.owner_key, request.session_id, original)
    expiry = core.teaching_intent(binding).expires_at_monotonic_ns
    clock[0] += timedelta(milliseconds=100)
    clock[1] += 100_000_000
    assert (
        core.start_teaching(ACTOR.owner_key, request).session_expires_at
        == request.session_expires_at
    )
    assert core.teaching_input(ACTOR.owner_key, request.session_id, original) == state
    assert core.teaching_intent(binding).expires_at_monotonic_ns == expiry
    with pytest.raises(Problem, match="reused"):
        core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    assert core.next_action() is None


@pytest.mark.parametrize("change", ["owner", "lease", "epoch", "expired", "future", "sequence"])
def test_stale_or_unbound_input_never_grants_motion(teaching, change):
    core, request, clock = teaching
    binding = begin(core, request)
    owner = OTHER.owner_key if change == "owner" else ACTOR.owner_key
    changes = {
        "lease": {"lease_id": uuid4()},
        "epoch": {"epoch": uuid4()},
        "expired": {"expires_at": clock[0]},
        "future": {"expires_at": clock[0] + timedelta(milliseconds=251)},
        "sequence": {"sequence": 2},
    }
    with pytest.raises(Problem):
        core.teaching_input(
            owner, request.session_id, jog(request, clock, **changes.get(change, {}))
        )
    assert core.teaching_intent(binding) is None
    assert core.teaching(ACTOR.owner_key, request.session_id).last_sequence == 0


def test_deadman_release_removes_authority_immediately(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    core.teaching_input(
        ACTOR.owner_key,
        request.session_id,
        jog(request, clock, sequence=2, deadman=False, delta_xyz_m=(0, 0, 0)),
    )
    assert core.teaching_intent(binding) is None


def test_session_deadline_is_monotonic_and_not_renewed_by_input(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    clock[1] += 300_000_000_000
    # Rolling back wall time must not prolong an already admitted lease.
    clock[0] -= timedelta(seconds=10)
    assert core.deadline_expired()
    with pytest.raises(Problem, match="expired"):
        core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    assert core.teaching_intent(binding) is None


def test_finish_is_only_a_request_for_a_real_terminal_hold_boundary(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    lease = TeachingLease(lease_id=request.lease_id, epoch=request.epoch)
    state = core.finish_teaching(ACTOR.owner_key, request.session_id, lease)
    assert state.status == "finishing"
    assert state.execution.status == "running"
    assert core.teaching_intent(binding) is None
    with pytest.raises(Problem, match="finishing"):
        core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    assert core.next_action().binding == binding
    core.finish_teaching(ACTOR.owner_key, request.session_id, lease)
    assert core.next_action() is None


def test_cancellation_and_old_epoch_cannot_reacquire_an_active_lease(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    lease = TeachingLease(lease_id=request.lease_id, epoch=request.epoch)
    assert core.cancel_teaching(ACTOR.owner_key, request.session_id, lease).status == "cancelling"
    assert core.teaching_intent(binding) is None
    core.finish("cancelled", None, binding=binding)
    core.pending.clear()
    core.activate(ACTOR.owner_key, core.environment)
    with pytest.raises(Problem):
        core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    assert core.active_command is None


@pytest.mark.parametrize("seconds", [0, 301])
def test_session_budget_is_validated_independently_of_reference_deadline(teaching, seconds):
    core, request, clock = teaching
    bad = request.model_copy(update={"session_expires_at": clock[0] + timedelta(seconds=seconds)})
    with pytest.raises(Problem):
        core.start_teaching(ACTOR.owner_key, bad)
    assert core.active_command is None


def test_missing_deployment_profile_is_explicit_not_reference_fallback(teaching):
    core, request, _ = teaching
    core.control_profile = None
    with pytest.raises(Problem, match="profile"):
        core.start_teaching(ACTOR.owner_key, request)
    assert core.next_action() is None


@pytest.mark.parametrize(
    "changes",
    [
        {"delta_xyz_m": (0.01, 0.01, 0)},
        {"delta_xyz_m": (float("nan"), 0, 0)},
        {"delta_xyz_m": (True, 0, 0)},
        {"deadman": False},
        {"sequence": True},
        {"joint_positions": [0] * 9},
    ],
)
def test_teaching_wire_cannot_accept_oversized_nonfinite_or_raw_joint_motion(teaching, changes):
    _, request, clock = teaching
    with pytest.raises(ValidationError):
        jog(request, clock, **changes)


def test_delayed_first_admission_cannot_get_fresh_motion_from_an_expired_grant(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    delayed = jog(request, clock)
    clock[0] += timedelta(seconds=2)
    clock[1] += 2_000_000_000
    delayed = delayed.model_copy(update={"expires_at": clock[0] + timedelta(milliseconds=250)})
    with pytest.raises(Problem, match="grant"):
        core.teaching_input(ACTOR.owner_key, request.session_id, delayed)
    assert core.teaching_intent(binding) is None


def test_release_preempts_a_never_seen_lower_sequence_motion_request(teaching):
    core, request, clock = teaching
    binding = begin(core, request)
    delayed = jog(request, clock)
    released = jog(
        request,
        clock,
        sequence=2,
        deadman=False,
        delta_xyz_m=(0, 0, 0),
        grant_id=None,
        grant_expires_at=None,
    )
    core.teaching_input(ACTOR.owner_key, request.session_id, released)
    with pytest.raises(Problem, match="sequence"):
        core.teaching_input(ACTOR.owner_key, request.session_id, delayed)
    assert core.teaching_intent(binding) is None
    assert core.teaching(ACTOR.owner_key, request.session_id).last_sequence == 2


def test_grant_is_consumed_once_and_cannot_authorize_a_different_sequence(teaching):
    core, request, clock = teaching
    begin(core, request)
    first = jog(request, clock)
    core.teaching_input(ACTOR.owner_key, request.session_id, first)
    second = first.model_copy(update={"sequence": 2})
    with pytest.raises(Problem, match="grant"):
        core.teaching_input(ACTOR.owner_key, request.session_id, second)
