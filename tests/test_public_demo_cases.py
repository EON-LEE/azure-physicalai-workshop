from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from presentation_support import prepare
from runtime_support import ACTOR, PNG

from apps.api import service as service_module
from apps.api.errors import Problem
from apps.api.main import create_app
from apps.api.public_demo import PublicDemo
from scripts.run_reference_demo import run_cycles


def recorded(monkeypatch, *, wrong=False):
    backend, settings, clock = prepare(monkeypatch)
    event_type = service_module.Event

    def event(**values):
        values.setdefault("at", clock.now())
        return event_type(**values)

    monkeypatch.setattr(service_module, "Event", event)
    backend.planner.wrong = wrong
    report = run_cycles(backend, settings, cycles=2, maximum_seconds=60, authorized=True)
    return backend, settings, clock, report


def test_cases_are_recorded_approved_runs_without_live_calls_or_history_scan(monkeypatch):
    backend, settings, clock, _ = recorded(monkeypatch)
    clock.sleep(100)
    backend.bridge.status = Mock(side_effect=AssertionError("No live runtime dependency"))
    backend.store.list_runs = Mock(side_effect=AssertionError("No private history scan"))
    backend.store.list_environments = Mock(side_effect=AssertionError("No customer list"))
    backend.planner.inspect = Mock(side_effect=AssertionError("No inference"))
    with TestClient(create_app(settings, backend)) as client:
        response = client.get("/api/demo/cases")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        payload = response.json()
        assert payload["source"] == "recorded_reference_runs"
        assert {case["kind"] for case in payload["cases"]} == {"normal_route", "defect_route"}
        assert [case["cycle"] for case in payload["cases"]] == [1, 2]
        for case in payload["cases"]:
            assert case["result"]["physical_success"] is True
            assert case["motion_authorized"] is True
            assert 0 <= case["physical_duration_seconds"] <= 30
            image = client.get(
                case["image_url"],
                params={
                    "presentation_id": payload["presentation_id"],
                    "observation_id": case["observation_id"],
                },
            )
            assert image.status_code == 200 and image.content == PNG
            assert image.headers["x-frame-id"] == case["observation_id"]
            assert image.headers["x-presentation-id"] == payload["presentation_id"]
            assert "x-physics-steps" not in image.headers
        for private in (ACTOR.owner_key, str(ACTOR.object_id), "blob_name", "model_response_id"):
            assert private not in response.text
        assert client.post("/api/demo/cases").status_code == 405
        assert client.post("/api/demo/cases/evidence").status_code == 405


def test_wrong_inspection_is_a_withheld_case_not_a_physical_success(monkeypatch):
    backend, settings, _, _ = recorded(monkeypatch, wrong=True)
    result = PublicDemo(settings, backend).cases()
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["kind"] == "withheld"
    assert case["scenario"] == "normal" and case["classification"] == "rejected"
    assert case["motion_authorized"] is False
    assert case["physical_duration_seconds"] is None
    assert case["result"]["inspection_correct"] is False
    assert case["result"]["physical_success"] is False


def test_cases_are_empty_before_publication_or_first_measured_result(monkeypatch):
    backend, settings, _ = prepare(monkeypatch)
    assert PublicDemo(settings, backend).cases()["cases"] == []
    settings = settings.model_copy(update={"public_demo_publish_live": False})
    backend.store.get_presentation = Mock(side_effect=AssertionError("Publication disabled"))
    assert PublicDemo(settings, backend).cases()["presentation_id"] is None


@pytest.mark.parametrize("mutation", ["fingerprint", "artifact", "outcome", "missing"])
def test_historical_cases_validate_full_run_scope_before_exposing_details(monkeypatch, mutation):
    backend, settings, _, _ = recorded(monkeypatch)
    public = PublicDemo(settings, backend)
    _, _, record, _ = public._scope()
    outcome = record.outcomes[0]
    stored = backend.store.get_run(ACTOR.owner_key, outcome.run_id)
    run = stored.value.model_copy(deep=True)
    run.plan.summary = "PRIVATE CONTENT"
    if mutation == "fingerprint":
        run.request_fingerprint = "f" * 64
    elif mutation == "artifact":
        run.evidence.blob_name = "another-owner/private.png"
    elif mutation == "outcome":
        run.execution.final_position = (0.8, 0.8, 0.2)
    else:
        get_run = backend.store.get_run
        backend.store.get_run = lambda owner, rid: (
            None if rid == outcome.run_id else get_run(owner, rid)
        )
    if mutation != "missing":
        backend.store.put_run(ACTOR.owner_key, run, stored.etag)
    with TestClient(create_app(settings, backend)) as client:
        response = client.get("/api/demo/cases")
        assert response.status_code == 503
        assert "PRIVATE" not in response.text


def test_unselected_private_observation_and_old_publication_are_not_readable(monkeypatch):
    backend, settings, _, _ = recorded(monkeypatch)
    public = PublicDemo(settings, backend)
    for presentation_id in ("private-presentation", settings.public_demo_presentation_id):
        with pytest.raises(Problem) as error:
            public.case_evidence(presentation_id, uuid4())
        assert error.value.status == 409


def test_explicit_recorded_publication_survives_a_new_live_window_without_adopting_private_data(
    monkeypatch,
):
    backend, settings, _, _ = recorded(monkeypatch)
    old_id = settings.public_demo_presentation_id
    settings = settings.model_copy(
        update={
            "public_demo_presentation_id": "next-live-window",
            "public_demo_cases_presentation_id": old_id,
        }
    )
    public = PublicDemo(settings, backend)
    payload = public.cases()
    assert payload["presentation_id"] == old_id
    assert len(payload["cases"]) == 2
    assert public.snapshot()["presentation"] is None
    assert public.snapshot()["recorded_cases_presentation_id"] == old_id
    observation_id = UUID(payload["cases"][0]["observation_id"])
    assert public.case_evidence(old_id, observation_id)[0] == PNG
    settings.public_demo_cases_presentation_id = "not-authorized"
    with pytest.raises(Problem) as error:
        public.case_evidence(old_id, observation_id)
    assert error.value.status == 409


def test_recorded_image_rechecks_publication_after_blob_read(monkeypatch):
    backend, settings, _, _ = recorded(monkeypatch)
    public = PublicDemo(settings, backend)
    payload = public.cases()
    observation = payload["cases"][0]["observation_id"]
    read = backend.artifacts.get

    def changed(name):
        image = read(name)
        settings.public_demo_presentation_id = "another-presentation"
        return image

    backend.artifacts.get = changed
    with pytest.raises(Problem) as error:
        public.case_evidence(payload["presentation_id"], UUID(observation))
    assert error.value.status in {409, 503}
