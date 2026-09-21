import json

import pytest
from runtime_support import ACTOR, document, service

from apps.api.models import Execution, SaveEnvironment, utcnow
from scripts import run_reference_demo as presentation


class Clock:
    def __init__(self, backend, complete="succeeded"):
        self.backend = backend
        self.value = 0.0
        self.complete = complete

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration
        if self.complete is None:
            return
        for key, result in list(self.backend.bridge.results.items()):
            if result.status == "queued":
                target = next(
                    station["position_m"]
                    for station in document()["stations"]
                    if station["role"] == "rejected"
                )
                self.backend.bridge.results[key] = Execution(
                    command_id=result.command_id,
                    status=self.complete,
                    final_position=tuple(target) if self.complete == "succeeded" else None,
                    completed_at=utcnow(),
                )


@pytest.fixture
def prepared(monkeypatch):
    backend = service()
    record = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(document())))
    clock = Clock(backend)
    monkeypatch.setattr(presentation, "time", clock)
    return backend, record, clock


def run(prepared, cycles=2, maximum_seconds=60):
    backend, record, _ = prepared
    return presentation.run_cycles(
        backend,
        ACTOR,
        record.environment_id,
        record.revision,
        "test-only-presentation",
        cycles,
        maximum_seconds,
    )


def test_explicit_cycles_complete_and_retries_do_not_dispatch_twice(prepared):
    first = run(prepared)
    assert first["successes"] == 2
    assert first["anonymous_control"] is False
    assert prepared[0].bridge.dispatches == 2
    second = run(prepared)
    assert second["successes"] == 2
    assert all(cycle["reused"] for cycle in second["cycles"])
    assert prepared[0].bridge.dispatches == 2


@pytest.mark.parametrize("cycles,seconds", [(0, 60), (101, 60), (1, 29), (1, 21601)])
def test_invalid_authorization_bounds_never_dispatch(prepared, cycles, seconds):
    with pytest.raises(ValueError, match="bounded"):
        run(prepared, cycles, seconds)
    assert prepared[0].bridge.dispatches == 0


def test_three_physical_failures_stop_the_presentation(prepared):
    prepared[2].complete = "failed"
    report = run(prepared, cycles=10)
    assert len(report["cycles"]) == 3
    assert report["successes"] == 0
    assert report["stop_reason"] == "repeated_physical_failures"


def test_expired_duration_after_inference_cannot_approve_motion(prepared):
    backend, _, clock = prepared
    backend.planner.after_inspect = lambda: clock.sleep(31)
    report = run(prepared, cycles=1, maximum_seconds=30)
    assert backend.bridge.dispatches == 0
    assert report["cycles"][0]["status"] == "cancelled"
    assert report["stop_reason"] == "presentation_time_limit"


def test_nonterminating_execution_requests_cancellation_and_fails(prepared):
    prepared[2].complete = None
    with pytest.raises(TimeoutError, match="cancellation was requested"):
        run(prepared, cycles=1, maximum_seconds=30)
    assert prepared[0].bridge.cancellations == 1


def test_an_unpublished_customer_scene_cannot_be_automatically_approved(prepared):
    backend, record, _ = prepared
    changed = document()
    changed["scene"]["seed"] = 43
    saved = backend.save_environment(
        ACTOR,
        SaveEnvironment(document_json=json.dumps(changed), expected_revision=record.revision),
    )
    with pytest.raises(ValueError, match="approved synthetic reference"):
        presentation.run_cycles(
            backend, ACTOR, saved.environment_id, saved.revision, "test-only", 1, 60
        )
    assert backend.bridge.dispatches == 0
