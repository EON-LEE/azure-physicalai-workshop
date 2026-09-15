import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from runtime_support import ACTOR, OTHER, PNG, document, service

from apps.api.errors import Problem
from apps.api.models import Execution, SaveEnvironment, StartRun, utcnow


@pytest.fixture
def prepared():
    backend = service()
    environment = backend.save_environment(
        ACTOR, SaveEnvironment(document_json=json.dumps(document()))
    )
    backend.activate(ACTOR, environment.environment_id, environment.revision)
    return backend, environment


def request(environment, request_id=None):
    return StartRun(
        request_id=request_id or uuid4(),
        environment_id=environment.environment_id,
        revision=environment.revision,
        instruction="Inspect and quarantine a defective part.",
    )


def test_planning_has_evidence_but_never_dispatches_motion(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    assert run.status == "awaiting_approval"
    assert backend.bridge.dispatches == 0
    assert backend.evidence(ACTOR, run.id) == PNG
    assert not {"evidence", "request_fingerprint", "environment_document"} & run.public().keys()


def test_same_request_is_idempotent_and_changed_input_conflicts(prepared):
    backend, environment = prepared
    body = request(environment)
    first = backend.start(ACTOR, body)
    assert backend.start(ACTOR, body).id == first.id
    assert backend.planner.calls == 1
    with pytest.raises(Problem, match="new request ID"):
        backend.start(ACTOR, body.model_copy(update={"instruction": "Different task"}))


def test_concurrent_same_request_has_one_planning_call(prepared):
    backend, environment = prepared
    body = request(environment)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: backend.start(ACTOR, body), range(4)))
    assert {result.id for result in results} == {body.request_id}
    assert backend.planner.calls == 1
    assert backend.bridge.dispatches == 0


def test_explicit_approval_dispatches_once_and_ack_is_not_success(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    with pytest.raises(Problem, match="exact displayed"):
        backend.approve(ACTOR, run.id, "wrong-response")
    assert backend.bridge.dispatches == 0
    assert backend.approve(ACTOR, run.id, run.plan.model_response_id).status == "running"
    assert backend.approve(ACTOR, run.id, run.plan.model_response_id).status == "running"
    assert backend.get_run(ACTOR, run.id).status == "running"
    assert backend.bridge.dispatches == 1


def test_expired_approval_never_dispatches(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    stored = backend.store.get_run(ACTOR.owner_key, run.id)
    stored.value.plan.expires_at = utcnow() - timedelta(seconds=1)
    backend.store.put_run(ACTOR.owner_key, stored.value, stored.etag)
    with pytest.raises(Problem, match="expired"):
        backend.approve(ACTOR, run.id, run.plan.model_response_id)
    assert backend.bridge.dispatches == 0


def test_concurrent_approval_cannot_dispatch_twice(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))

    def approve(_):
        try:
            return backend.approve(ACTOR, run.id, run.plan.model_response_id).status
        except Problem as exc:
            assert exc.status == 409
            return "conflict"

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(approve, range(4)))
    assert backend.bridge.dispatches == 1


@pytest.mark.parametrize("field", ["epoch", "state_revision"])
def test_scene_changes_invalidate_approval(prepared, field):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    setattr(backend.bridge, field, uuid4() if field == "epoch" else 2)
    with pytest.raises(Problem, match="changed"):
        backend.approve(ACTOR, run.id, run.plan.model_response_id)
    assert backend.bridge.dispatches == 0


@pytest.mark.parametrize("offset", [-3, 2])
def test_stale_or_future_observation_never_starts_motion(prepared, offset):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    backend.bridge.captured_at = utcnow() + timedelta(seconds=offset)
    with pytest.raises(Problem, match="fresh"):
        backend.approve(ACTOR, run.id, run.plan.model_response_id)
    assert backend.bridge.dispatches == 0


def test_cancellation_during_planning_cannot_be_overwritten(prepared):
    backend, environment = prepared
    body = request(environment)
    backend.planner.after_inspect = lambda: backend.cancel(ACTOR, body.request_id)
    assert backend.start(ACTOR, body).status == "cancelled"
    assert backend.bridge.dispatches == 0


def test_cancel_after_dispatch_waits_for_stop_confirmation(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    assert backend.cancel(ACTOR, run.id).status == "cancelling"
    assert backend.get_run(ACTOR, run.id).status == "cancelling"
    backend.bridge.results[(ACTOR.owner_key, run.id)] = Execution(
        command_id=run.id, status="cancelled", completed_at=utcnow()
    )
    assert backend.get_run(ACTOR, run.id).status == "cancelled"
    backend.bridge.results[(ACTOR.owner_key, run.id)] = Execution(
        command_id=run.id,
        status="succeeded",
        final_position=(0.5, -0.4, 0.2),
        completed_at=utcnow(),
    )
    assert backend.get_run(ACTOR, run.id).status == "cancelled"


@pytest.mark.parametrize(
    ("position", "expected"),
    [((0.5, -0.4, 0.2), "succeeded"), ((8, 8, 8), "failed"), (None, "failed")],
)
def test_success_requires_matching_physical_destination(prepared, position, expected):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    backend.bridge.results[(ACTOR.owner_key, run.id)] = Execution(
        command_id=run.id, status="succeeded", final_position=position, completed_at=utcnow()
    )
    assert backend.get_run(ACTOR, run.id).status == expected


def test_uncertain_dispatch_is_reconciled_without_resending(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    backend.bridge.dispatch_error = Problem(503, "network", "Test disconnect.")
    uncertain = backend.approve(ACTOR, run.id, run.plan.model_response_id)
    assert uncertain.status == "running"
    assert uncertain.error.code == "dispatch_unconfirmed"
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    backend.get_run(ACTOR, run.id)
    assert backend.bridge.dispatches == 1


def test_other_owner_cannot_read_runs_or_images(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    for operation in (
        lambda: backend.get_run(OTHER, run.id),
        lambda: backend.evidence(OTHER, run.id),
    ):
        with pytest.raises(Problem) as error:
            operation()
        assert error.value.status == 404


def test_saved_edits_do_not_mutate_the_snapshot_of_an_approved_run(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    edited = document()
    edited["display_name"] = "A later revision"
    backend.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(edited),
            expected_revision=environment.revision,
        ),
    )
    assert backend.approve(ACTOR, run.id, run.plan.model_response_id).status == "running"
    assert run.environment_document["display_name"] != edited["display_name"]


def test_model_error_is_visible_not_a_default_decision(prepared):
    backend, environment = prepared
    backend.planner.error = Problem(
        503, "model_unavailable", "No real model response.", retryable=True
    )
    run = backend.start(ACTOR, request(environment))
    assert run.status == "failed" and run.plan is None
    assert run.error.code == "model_unavailable"
    assert backend.bridge.dispatches == 0


def test_missing_mode_and_replay_cannot_activate_a_live_scene():
    backend = service()
    replay = document()
    replay["execution"]["mode"] = "replay"
    environment = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(replay)))
    with pytest.raises(Problem, match="Replay"):
        backend.activate(ACTOR, environment.environment_id, environment.revision)


def test_duplicate_json_is_rejected_before_reserialization():
    backend = service()
    with pytest.raises(Problem, match="Duplicate JSON key"):
        backend.save_environment(ACTOR, SaveEnvironment(document_json='{"id":1,"id":2}'))


def test_observation_checksum_mismatch_is_an_error(prepared):
    backend, environment = prepared
    run = backend.start(ACTOR, request(environment))
    backend.artifacts.items[run.evidence.blob_name] = b"changed"
    with pytest.raises(Problem, match="checksum"):
        backend.evidence(ACTOR, run.id)
