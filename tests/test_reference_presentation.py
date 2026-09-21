import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from presentation_support import prepare
from runtime_support import ACTOR, document

from agents.inspection import FoundryInspector
from apps.api.errors import Problem
from apps.api.models import SaveEnvironment
from apps.api.presentation import INSTRUCTION, cycle_run_id
from scripts import run_reference_demo as presentation


@pytest.fixture
def prepared(monkeypatch):
    return prepare(monkeypatch)


def run(prepared, cycles=2, maximum_seconds=60):
    backend, settings, _ = prepared
    return presentation.run_cycles(
        backend,
        settings,
        cycles=cycles,
        maximum_seconds=maximum_seconds,
        authorized=True,
    )


def record(prepared):
    backend, settings, _ = prepared
    return backend.store.get_presentation(
        ACTOR.owner_key, settings.public_demo_presentation_id
    ).value


def test_paired_cycles_complete_and_retries_do_not_dispatch_or_renew(prepared):
    first = run(prepared)
    assert first["successes"] == 2 and first["status"] == "completed"
    assert prepared[0].bridge.dispatches == 2
    stored = record(prepared)
    normal = prepared[0].store.get_run(ACTOR.owner_key, cycle_run_id(stored, 1)).value
    defect = prepared[0].store.get_run(ACTOR.owner_key, cycle_run_id(stored, 2)).value
    assert normal.environment_document["scene"]["seed"] == 42
    assert defect.environment_document["scene"]["seed"] == 43
    assert normal.evidence.epoch != defect.evidence.epoch
    assert normal.plan.classification == "accepted"
    assert defect.plan.classification == "rejected"
    assert normal.instruction == defect.instruction == INSTRUCTION
    assert prepared[0].planner.instructions == [INSTRUCTION, INSTRUCTION]
    second = run(prepared)
    assert second["reused"] is True and second["successes"] == 2
    assert second["expires_at"] == first["expires_at"]
    assert prepared[0].bridge.dispatches == 2


@pytest.mark.parametrize("cycles,seconds", [(0, 60), (1001, 60), (1, 29), (1, 21601), (True, 60)])
def test_invalid_authorization_bounds_never_dispatch(prepared, cycles, seconds):
    with pytest.raises(ValueError, match="bounded"):
        run(prepared, cycles, seconds)
    assert prepared[0].bridge.dispatches == 0


def test_1000_cycle_authorization_is_bounded_by_existing_hard_lifetime(prepared):
    backend, settings, _ = prepared
    runner = presentation.Runner(backend, settings, 1000, 21600)
    assert runner.claim()
    assert runner.record.total_cycles == 1000
    assert (runner.record.expires_at - runner.record.started_at).total_seconds() == 21600


def test_programmatic_entry_requires_explicit_authorization(prepared):
    with pytest.raises(ValueError, match="Explicit operator"):
        presentation.run_cycles(prepared[0], prepared[1], cycles=1, maximum_seconds=60)


def test_missing_paired_configuration_is_not_legacy_auto_motion(prepared):
    backend, settings, _ = prepared
    settings = settings.model_copy(update={"public_demo_defect_revision": None})
    with pytest.raises(Problem, match="pinned reference pair"):
        presentation.run_cycles(backend, settings, cycles=1, maximum_seconds=60, authorized=True)
    assert backend.bridge.dispatches == 0


def test_three_physical_failures_stop_the_presentation(prepared):
    prepared[2].complete = "failed"
    report = run(prepared, cycles=10)
    assert len(report["cycles"]) == 3
    assert report["successes"] == 0
    assert report["stop_reason"] == "repeated_failures"
    assert all(c["result"]["inspection_correct"] for c in report["cycles"])
    assert not any(c["result"]["physical_success"] for c in report["cycles"])


def test_inspection_mismatch_preserves_original_decision_cancels_before_motion(prepared):
    prepared[0].planner.wrong = True
    report = run(prepared, cycles=10)
    assert len(report["cycles"]) == 3 and report["status"] == "failed"
    assert prepared[0].bridge.dispatches == 0
    assert not any(c["result"]["inspection_correct"] for c in report["cycles"])
    stored = record(prepared)
    for cycle in range(1, 4):
        original = prepared[0].store.get_run(ACTOR.owner_key, cycle_run_id(stored, cycle)).value
        assert original.plan.classification == ("rejected" if cycle % 2 else "accepted")
        assert original.execution is None
        assert original.status == "cancelled"
        assert original.evidence is not None


def test_expired_duration_after_inference_cannot_approve_motion(prepared):
    backend, _, clock = prepared
    backend.planner.after_inspect = lambda: clock.sleep(31)
    report = run(prepared, cycles=1, maximum_seconds=30)
    assert backend.bridge.dispatches == 0
    assert report["cycles"][0]["result"]["status"] == "cancelled"
    assert report["stop_reason"] == "time_limit"


def test_nonterminating_execution_requests_cancellation_and_never_claims_success(prepared):
    prepared[2].complete = None
    report = run(prepared, cycles=1, maximum_seconds=30)
    assert prepared[0].bridge.cancellations == 1
    assert report["successes"] == 0 and report["status"] == "stopped"
    assert report["cycles"][0]["result"]["status"] == "cancelled"


def test_cancellation_ack_is_not_terminal_evidence(prepared):
    prepared[2].complete = "running"
    original_sleep = prepared[2].sleep

    def no_cancel_completion(duration):
        original_sleep(duration)
        for result in prepared[0].bridge.results.values():
            if result.status == "cancelled":
                result.status = "cancelling"

    prepared[2].sleep = no_cancel_completion
    report = run(prepared, cycles=1, maximum_seconds=30)
    assert report["status"] == "failed"
    assert report["stop_reason"] == "cancellation_unconfirmed"
    assert report["successes"] == 0 and report["cycles"] == []


@pytest.mark.parametrize("mutation", ["geometry", "seed", "revision"])
def test_pinned_but_unapproved_document_is_rejected(prepared, mutation):
    backend, settings, _ = prepared
    changed = document()
    if mutation == "geometry":
        changed["stations"][0]["position_m"][0] = 0.3
    elif mutation == "seed":
        changed["scene"]["seed"] = 43
    else:
        changed["display_name"] = "Unreviewed changed revision"
    saved = backend.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(changed), expected_revision=settings.public_demo_revision
        ),
    )
    if mutation != "revision":
        prepared = (
            backend,
            settings.model_copy(update={"public_demo_revision": saved.revision}),
            None,
        )
    with pytest.raises(Problem):
        run(prepared)
    assert backend.bridge.dispatches == 0


def test_changed_duration_or_cycle_count_cannot_restart_claim(prepared):
    run(prepared)
    for cycles, seconds in ((3, 60), (2, 120)):
        with pytest.raises(Problem, match="original authorization"):
            run(prepared, cycles, seconds)


def test_concurrent_or_crashed_runner_cannot_reclaim_same_id(prepared):
    runner = presentation.Runner(prepared[0], prepared[1], 2, 60)
    assert runner.claim()
    prepared[2].sleep(100)
    with pytest.raises(Problem, match="restarting cannot renew"):
        run(prepared)
    assert prepared[0].bridge.dispatches == prepared[0].planner.calls == 0


def test_changed_claim_during_inference_prevents_approval(prepared):
    backend, settings, _ = prepared

    def edit_claim():
        stored = backend.store.get_presentation(
            ACTOR.owner_key, settings.public_demo_presentation_id
        )
        backend.store.put_presentation(ACTOR.owner_key, stored.value, stored.etag)

    backend.planner.after_inspect = edit_claim
    with pytest.raises(Problem, match="Concurrent test write"):
        run(prepared, cycles=1)
    assert backend.bridge.dispatches == 0


def test_storage_outage_never_falls_back_to_memory_or_motion(prepared):
    backend, _, _ = prepared
    backend.store.put_presentation = Mock(
        side_effect=Problem(503, "storage_unavailable", "private backend information")
    )
    with pytest.raises(Problem):
        run(prepared)
    assert backend.bridge.dispatches == backend.planner.calls == 0


def test_missing_new_epoch_never_plans_after_activation(prepared):
    backend, _, _ = prepared
    backend.activate(
        ACTOR, prepared[1].public_demo_environment_id, prepared[1].public_demo_revision
    )
    prior_epoch = backend.bridge.epoch
    original = backend.bridge.activate

    def unchanged(owner, environment):
        result = original(owner, environment)
        backend.bridge.epoch = prior_epoch
        return result

    backend.bridge.activate = unchanged
    report = run(prepared, cycles=1, maximum_seconds=30)
    assert report["status"] == "failed"
    assert backend.planner.calls == 0


def test_epoch_change_after_planning_never_approves(prepared):
    backend, _, _ = prepared
    backend.planner.after_inspect = lambda: setattr(backend.bridge, "epoch", uuid4())
    report = run(prepared, cycles=1)
    assert report["stop_reason"] == "scene_changed"
    assert backend.bridge.dispatches == 0
    assert report["successes"] == 0


def test_epoch_change_during_motion_is_not_accepted_as_completion(prepared):
    backend, _, clock = prepared
    clock.on_sleep = lambda: setattr(backend.bridge, "epoch", uuid4())
    report = run(prepared, cycles=1)
    assert report["stop_reason"] == "scene_changed"
    assert report["status"] == "failed"
    assert report["successes"] == 0


def test_real_foundry_adapter_sends_only_image_and_generic_metadata_for_both_seeds(prepared):
    backend, settings, _ = prepared
    inspector = FoundryInspector.__new__(FoundryInspector)
    inspector.client = Mock()
    inspector.agent_name, inspector.agent_version = "test-agent", "1"
    inspector.client.responses.create.return_value = SimpleNamespace(
        id="test-response",
        output=[
            SimpleNamespace(
                type="function_call",
                name="record_inspection",
                arguments=json.dumps(
                    {
                        "classification": "accepted",
                        "object_id": "part-001",
                        "summary": "Test-only",
                    }
                ),
            )
        ],
    )
    for environment_id, revision in (
        (settings.public_demo_environment_id, settings.public_demo_revision),
        (settings.public_demo_defect_environment_id, settings.public_demo_defect_revision),
    ):
        backend.activate(ACTOR, environment_id, revision)
        observation = backend.bridge.observe(ACTOR.owner_key, environment_id, revision)
        inspector.inspect(INSTRUCTION, observation)
        content = inspector.client.responses.create.call_args.kwargs["input"][0]["content"]
        metadata = json.loads(content[0]["text"])
        assert set(metadata) == {
            "task",
            "object_id",
            "observation_id",
            "captured_at",
            "data_origin",
        }
        assert metadata["task"] == INSTRUCTION
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64," + observation.image_base64,
        }
        assert "seed" not in content[0]["text"]
        assert environment_id not in content[0]["text"]
