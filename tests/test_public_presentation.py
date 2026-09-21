import json
from datetime import timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from azure.core import MatchConditions
from fastapi.testclient import TestClient
from presentation_support import prepare
from pydantic import ValidationError
from runtime_support import ACTOR, PNG

from apps.api.azure_data import CosmosStore
from apps.api.errors import Problem
from apps.api.main import create_app
from apps.api.models import MotionTelemetry, SaveEnvironment, StartRun
from apps.api.presentation import INSTRUCTION, cycle_run_id, result_for
from apps.api.public_demo import PublicDemo
from apps.api.settings import Settings
from scripts.run_reference_demo import Runner, run_cycles


@pytest.fixture
def prepared(monkeypatch):
    return prepare(monkeypatch)


def planned(prepared):
    backend, settings, _ = prepared
    runner = Runner(backend, settings, 2, 60)
    runner.claim()
    backend.activate(ACTOR, settings.public_demo_environment_id, settings.public_demo_revision)
    run_id = cycle_run_id(runner.record, 1)
    runner.persist(status="inspecting", run_id=run_id, scene_epoch=backend.bridge.epoch)
    run = backend.start(
        ACTOR,
        StartRun(
            request_id=run_id,
            environment_id=settings.public_demo_environment_id,
            revision=settings.public_demo_revision,
            instruction=INSTRUCTION,
        ),
    )
    runner.persist(status="awaiting_motion")
    return runner, run


def completed(prepared):
    return run_cycles(prepared[0], prepared[1], cycles=2, maximum_seconds=60, authorized=True)


def test_not_started_is_null_but_storage_failure_is_not_successful_empty(prepared):
    backend, settings, _ = prepared
    public = PublicDemo(settings, backend)
    assert public.snapshot()["presentation"] is None
    backend.store.get_presentation = Mock(side_effect=Problem(503, "db", "PRIVATE DATA"))
    with TestClient(create_app(settings, backend)) as client:
        response = client.get("/api/demo")
        assert response.status_code == 503
        assert "PRIVATE DATA" not in response.text
        assert "presentation" not in response.json()


def test_current_presentation_is_curated_no_history_or_operator_fields(prepared):
    completed(prepared)
    backend, settings, _ = prepared
    backend.store.list_runs = Mock(side_effect=AssertionError("Never scan private histories"))
    backend.store.list_environments = Mock(side_effect=AssertionError("Never scan customer data"))
    with TestClient(create_app(settings, backend)) as client:
        response = client.get("/api/demo")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        payload = response.json()
        assert payload["presentation"]["scenario"] == "surface_defect"
        assert payload["presentation"]["status"] == "completed"
        assert payload["presentation"]["counts"] == {
            "attempted": 2,
            "succeeded": 2,
            "failed": 0,
            "inspected_correctly": 2,
            "physically_completed": 2,
        }
        assert payload["presentation"]["result"]["physical_success"] is True
        assert payload["presentation"]["result"]["inspection_correct"] is True
        assert payload["presentation"]["motion"]["phase"] is None
        for secret in (
            ACTOR.owner_key,
            str(ACTOR.object_id),
            "blob_name",
            "model_response_id",
            "request_fingerprint",
            "runner_id",
        ):
            assert secret not in response.text
        assert client.get("/api/runs").status_code == 401
        assert client.post("/api/demo", json={"cycles": 1000}).status_code == 405
        assert client.post("/api/demo/evidence").status_code == 405


def test_current_original_evidence_is_not_a_live_recapture(prepared):
    runner, run = planned(prepared)
    backend, settings, clock = prepared
    clock.sleep(5)
    with TestClient(create_app(settings, backend)) as client:
        response = client.get(
            "/api/demo/evidence",
            params={
                "observation_id": str(run.evidence.observation_id),
            },
        )
        assert response.status_code == 200 and response.content == PNG
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-frame-id"] == str(run.evidence.observation_id)
        assert response.headers["x-captured-at"] == run.evidence.captured_at.isoformat()
        assert response.headers["x-captured-at"] != clock.now().isoformat()
        assert client.get("/api/demo/evidence").status_code == 422
        mismatch = client.get("/api/demo/evidence", params={"observation_id": str(uuid4())})
        assert mismatch.status_code == 409
    assert runner.record.run_id == run.id


def test_epoch_pinned_frames_have_correct_headers_and_no_cache(prepared):
    _, run = planned(prepared)
    backend, settings, _ = prepared
    with TestClient(create_app(settings, backend)) as client:
        response = client.get(
            "/api/demo/frame",
            params={
                "camera": "inspection",
                "epoch": str(run.evidence.epoch),
            },
        )
        assert response.status_code == 200 and response.content == PNG
        assert response.headers["x-scene-epoch"] == str(run.evidence.epoch)
        assert response.headers["x-frame-id"]
        assert response.headers["x-captured-at"]
        assert response.headers["x-physics-steps"] == "42"
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/api/demo/frame", params={"epoch": str(uuid4())}).status_code == 409
        assert client.get("/api/demo/frame", params={"epoch": "bad"}).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_key", "f" * 64),
        ("id", "another-presentation"),
        ("normal_revision", "f" * 64),
        ("defect_revision", "f" * 64),
        ("normal_environment_id", "private-environment"),
        ("defect_environment_id", "private-defect"),
        ("run_id", uuid4()),
    ],
)
def test_persisted_publication_scope_cannot_select_private_data(prepared, field, value):
    runner, _ = planned(prepared)
    backend, settings, _ = prepared
    record = runner.record.model_copy(update={field: value})
    # Deliberately place corrupt generic JSON under the pinned document key.
    key = (ACTOR.owner_key, "presentation", runner.record.id)
    backend.store.items[key] = runner.stored.model_copy(update={"value": record})
    with TestClient(create_app(settings, backend)) as client:
        response = client.get("/api/demo")
        assert response.status_code == 503
        assert "private-environment" not in response.text
        assert "private-defect" not in response.text
        assert (
            client.get("/api/demo/evidence", params={"observation_id": str(uuid4())}).status_code
            == 503
        )


@pytest.mark.parametrize(
    "mutation", ["instruction", "fingerprint", "document", "observation", "path"]
)
def test_generic_or_private_run_is_rejected_before_summary_or_pixels(prepared, mutation):
    _, run = planned(prepared)
    backend, settings, _ = prepared
    stored = backend.store.get_run(ACTOR.owner_key, run.id)
    changed = stored.value.model_copy(deep=True)
    changed.plan.summary = "PRIVATE OPERATOR SUMMARY"
    if mutation == "instruction":
        changed.instruction = "PRIVATE PROMPT"
    elif mutation == "fingerprint":
        changed.request_fingerprint = "f" * 64
    elif mutation == "document":
        changed.environment_document["display_name"] = "PRIVATE TITLE"
    elif mutation == "observation":
        changed.plan.observation_id = uuid4()
    else:
        changed.evidence.blob_name = "another-owner/private.png"
    backend.store.put_run(ACTOR.owner_key, changed, stored.etag)
    with TestClient(create_app(settings, backend)) as client:
        for path in (
            "/api/demo",
            f"/api/demo/evidence?observation_id={run.evidence.observation_id}",
        ):
            response = client.get(path)
            assert response.status_code == 503
            assert "PRIVATE" not in response.text


def test_evidence_race_rejects_previous_cycle_in_flight(prepared):
    runner, run = planned(prepared)
    backend, settings, _ = prepared
    read = backend.artifacts.get

    def switch(name):
        image = read(name)
        runner.persist(cycle=2, status="preparing", run_id=None, scene_epoch=None)
        return image

    backend.artifacts.get = switch
    with pytest.raises(Problem) as failure:
        PublicDemo(settings, backend).evidence(run.evidence.observation_id)
    assert failure.value.status == 409


def test_live_frame_race_rejects_reset_epoch_even_for_same_revision(prepared):
    planned(prepared)
    backend, settings, _ = prepared
    observe = backend.bridge.observe

    def reset_after_capture(*args):
        observation = observe(*args)
        backend.bridge.epoch = uuid4()
        return observation

    backend.bridge.observe = reset_after_capture
    with pytest.raises(Problem) as failure:
        PublicDemo(settings, backend).frame("overview")
    assert failure.value.status in (409, 503)


def test_new_cycle_clears_decision_and_result_without_relabeling_history(prepared):
    runner, run = planned(prepared)
    backend, settings, clock = prepared
    run = backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = backend.get_run(ACTOR, run.id)
    runner.outcome(run)
    runner.persist(cycle=2, status="preparing", run_id=None, scene_epoch=None)
    payload = PublicDemo(settings, backend).snapshot()["presentation"]
    assert payload["scenario"] == "surface_defect"
    assert payload["decision"] is payload["motion"] is payload["result"] is None
    assert payload["run_id"] is payload["scene_epoch"] is None
    assert payload["counts"]["succeeded"] == 1


def test_motion_uses_only_matching_real_telemetry(prepared):
    runner, run = planned(prepared)
    backend, settings, _ = prepared
    run = backend.approve(ACTOR, run.id, run.plan.model_response_id)
    runner.persist(status="moving")
    status = backend.bridge.status
    telemetry = MotionTelemetry(
        command_id=run.id,
        phase="grasping",
        object_position=(0.35, 0.25, 0.2),
        target_station_id="accepted",
    )
    backend.bridge.status = lambda owner: status(owner).model_copy(update={"motion": telemetry})
    payload = PublicDemo(settings, backend).snapshot()["presentation"]
    assert payload["motion"]["phase"] == "grasping"
    assert payload["motion"]["part_position_m"] == [0.35, 0.25, 0.2]
    telemetry.command_id = uuid4()
    payload = PublicDemo(settings, backend).snapshot()["presentation"]
    assert payload["motion"] is payload["decision"] is None


@pytest.mark.parametrize("terminal", [False, True])
def test_same_epoch_private_command_blocks_frames_and_current_decision(prepared, terminal):
    backend, settings, _ = prepared
    if terminal:
        completed(prepared)
        record = backend.store.get_presentation(
            ACTOR.owner_key, settings.public_demo_presentation_id
        ).value
        run = backend.store.get_run(ACTOR.owner_key, record.run_id).value
    else:
        _, run = planned(prepared)
    status = backend.bridge.status
    backend.bridge.status = lambda owner: status(owner).model_copy(
        update={
            "motion": MotionTelemetry(
                command_id=uuid4(),
                phase="transporting",
                object_position=(0.4, 0.1, 0.3),
                target_station_id=run.plan.target_station_id,
            ),
        }
    )
    backend.bridge.observe = Mock(side_effect=AssertionError("Never capture a private command"))
    with TestClient(create_app(settings, backend)) as client:
        snapshot = client.get("/api/demo")
        assert snapshot.status_code == 200
        payload = snapshot.json()
        assert payload["simulation"]["live_available"] is False
        assert payload["presentation"]["decision"] is payload["presentation"]["motion"] is None
        frame = client.get("/api/demo/frame", params={"epoch": str(run.evidence.epoch)})
        assert frame.status_code == 409
        assert frame.json()["error"]["code"] == "public_command_changed"
        if terminal:
            assert payload["presentation"]["result"]["status"] == "succeeded"
            assert payload["presentation"]["counts"]["succeeded"] == 2
        else:
            assert payload["presentation"]["status"] == "stopped"


def test_same_epoch_private_command_race_discards_captured_frame_and_decision(prepared):
    _, run = planned(prepared)
    backend, settings, _ = prepared
    status, observe = backend.bridge.status, backend.bridge.observe

    def private_command_after_capture(*args):
        observation = observe(*args)
        backend.bridge.status = lambda owner: status(owner).model_copy(
            update={
                "motion": MotionTelemetry(
                    command_id=uuid4(),
                    phase="grasping",
                    object_position=(0.35, 0.25, 0.2),
                    target_station_id="accepted",
                ),
            }
        )
        return observation

    backend.bridge.observe = private_command_after_capture
    public = PublicDemo(settings, backend)
    payload = public.snapshot()
    assert payload["simulation"]["live_available"] is False
    assert payload["presentation"]["decision"] is payload["presentation"]["motion"] is None
    backend.bridge.status = status
    with pytest.raises(Problem) as failure:
        public.frame("overview", run.evidence.epoch)
    assert failure.value.code == "public_command_changed"


@pytest.mark.parametrize("terminal", [False, True])
def test_idle_before_dispatch_and_matching_completed_command_remain_public(prepared, terminal):
    backend, settings, _ = prepared
    if terminal:
        completed(prepared)
        record = backend.store.get_presentation(
            ACTOR.owner_key, settings.public_demo_presentation_id
        ).value
        run = backend.store.get_run(ACTOR.owner_key, record.run_id).value
    else:
        _, run = planned(prepared)
    status = backend.bridge.status
    backend.bridge.status = lambda owner: status(owner).model_copy(
        update={
            "motion": MotionTelemetry(
                command_id=run.id if terminal else None,
                phase="complete" if terminal else "idle",
                object_position=(0.35, 0.25, 0.2),
                target_station_id=run.plan.target_station_id if terminal else None,
            ),
        }
    )
    public = PublicDemo(settings, backend)
    payload = public.snapshot()
    assert payload["simulation"]["live_available"] is True
    assert payload["presentation"]["decision"]["observation_id"] == str(run.evidence.observation_id)
    assert public.frame("overview", run.evidence.epoch)[0] == PNG
    if terminal:
        assert payload["presentation"]["motion"]["phase"] == "complete"


@pytest.mark.parametrize("failure", ["expired", "heartbeat", "camera", "backend"])
def test_expired_stale_or_disconnected_motion_does_not_look_running(prepared, failure):
    runner, run = planned(prepared)
    backend, settings, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    runner.persist(status="moving")
    if failure == "expired":
        clock.sleep(61)
    elif failure == "heartbeat":
        clock.sleep(11)
    elif failure == "camera":
        backend.bridge.captured_at = clock.now() - timedelta(seconds=5)
    else:
        backend.bridge.status_error = Problem(503, "bridge", "PRIVATE BACKEND")
    payload = PublicDemo(settings, backend).snapshot()
    assert payload["presentation"]["status"] == "stopped"
    assert payload["presentation"]["motion"] is None
    if failure in {"camera", "backend"}:
        assert payload["simulation"]["live_available"] is False
    assert "PRIVATE BACKEND" not in json.dumps(payload)


def test_recorded_outcome_survives_camera_outage_without_claiming_live(prepared):
    completed(prepared)
    backend, settings, clock = prepared
    backend.bridge.captured_at = clock.now() - timedelta(seconds=5)
    payload = PublicDemo(settings, backend).snapshot()
    assert payload["simulation"]["live_available"] is False
    assert payload["presentation"]["result"]["status"] == "succeeded"
    assert payload["presentation"]["motion"]["phase"] is None


def test_counts_distinguish_inspection_and_physical_completion(prepared):
    backend, settings, clock = prepared
    clock.complete = "failed"
    completed(prepared)
    payload = PublicDemo(settings, backend).snapshot()["presentation"]
    assert payload["counts"] == {
        "attempted": 2,
        "succeeded": 0,
        "failed": 2,
        "inspected_correctly": 2,
        "physically_completed": 0,
    }
    assert payload["result"]["physical_success"] is False
    assert payload["result"]["inspection_correct"] is True


def test_correct_pose_to_wrong_inspection_target_is_not_task_success(prepared):
    runner, run = planned(prepared)
    backend, _, clock = prepared
    stored = backend.store.get_run(ACTOR.owner_key, run.id)
    stored.value.plan.classification = "rejected"
    stored.value.plan.target_station_id = "rejected"
    backend.store.put_run(ACTOR.owner_key, stored.value, stored.etag)
    run = backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = backend.get_run(ACTOR, run.id)
    result = result_for(runner.record, run)
    assert result.physical_success is True
    assert result.inspection_correct is False
    assert result.status == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "public_demo_defect_environment_id",
        "public_demo_defect_revision",
        "public_demo_presentation_id",
        "public_demo_owner_id",
    ],
)
def test_partial_public_pair_configuration_fails_at_startup(prepared, field):
    values = prepared[1].model_dump()
    values[field] = None
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


def test_both_documents_must_remain_exactly_canonical(prepared):
    planned(prepared)
    backend, settings, _ = prepared
    defect = backend.environment(ACTOR, settings.public_demo_defect_environment_id).value
    changed = defect.document.copy()
    changed["scene"] = {**changed["scene"], "seed": 42}
    revised = backend.save_environment(
        ACTOR,
        SaveEnvironment(document_json=json.dumps(changed), expected_revision=defect.revision),
    )
    settings.public_demo_defect_revision = revised.revision
    with pytest.raises(Problem):
        PublicDemo(settings, backend).snapshot()


def test_cosmos_presentation_is_point_read_and_conditional_write_not_history_scan(prepared):
    runner, _ = planned(prepared)
    store = CosmosStore.__new__(CosmosStore)
    store.container = Mock()
    record = runner.record
    store.container.read_item.return_value = {
        "value": record.model_dump(mode="json"),
        "_etag": "v1",
    }
    read = store.get_presentation(ACTOR.owner_key, record.id)
    store.container.read_item.assert_called_once_with(
        item=f"presentation:{record.id}",
        partition_key=ACTOR.owner_key,
    )
    assert read.value == record and read.etag == "v1"
    store.container.replace_item.return_value = {"_etag": "v2"}
    store.put_presentation(ACTOR.owner_key, record, "v1")
    kwargs = store.container.replace_item.call_args.kwargs
    assert kwargs["etag"] == "v1"
    assert kwargs["match_condition"] == MatchConditions.IfNotModified
    assert kwargs["body"]["kind"] == "presentation"
    assert kwargs["body"]["owner_key"] == ACTOR.owner_key
    store.container.query_items.assert_not_called()
    with pytest.raises(Problem):
        store.put_presentation("different-owner", record, None)
