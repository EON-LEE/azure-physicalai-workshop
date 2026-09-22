from datetime import timedelta
from uuid import uuid4

import pytest
from presentation_support import prepare
from runtime_support import ACTOR, PNG
from test_public_presentation import planned

from apps.api.errors import Problem
from apps.api.models import MotionTelemetry
from apps.api.public_demo import PublicDemo
from scripts.run_reference_demo import run_cycles
from simulation import core as core_module
from simulation.core import LoadScene, SimulationCore, StartMotion
from simulation.extensions import SceneRegistry


def run_with_core(monkeypatch, *, render_after_finish=True, failure=None, cycles=2):
    backend, settings, clock = prepare(monkeypatch)
    core = SimulationCore(SceneRegistry(load_installed=False))
    monkeypatch.setattr(core_module, "utcnow", clock.now)
    backend.bridge = core
    observe = core.observe
    core.observe = lambda owner, environment_id, revision, camera="inspection": observe(
        owner,
        environment_id,
        revision,
        camera,
    )
    public = PublicDemo(settings, backend)
    loading_snapshots = []
    completions = []

    def render(position):
        for camera in ("overview", "inspection"):
            core.publish_frame(camera, PNG, position, core.physics_steps + 1, epoch=core.epoch)

    def sleep(duration):
        clock.value += duration
        action = core.next_action()
        if isinstance(action, LoadScene):
            render((0.35, 0.25, 0.2))
        elif isinstance(action, StartMotion):
            assert core.begin_motion(action.command.command_id)
            target = next(
                item["position_m"]
                for item in core.environment.document["stations"]
                if item["id"] == action.command.target_station_id
            )
            core.finish("succeeded", tuple(target))
            assert core.status(ACTOR.owner_key).status == "loading"
            assert core.frames == {}
            completions.append(clock.monotonic())
            loading_snapshots.append(public.snapshot())
            if failure == "epoch":
                core.epoch = uuid4()
            elif failure == "unavailable":
                core.fail_scene("Test unavailable", epoch=core.epoch)
        elif not core.ready and core.environment and render_after_finish and not failure:
            render((0.22, -0.38, 0.2))

    clock.sleep = sleep
    report = run_cycles(backend, settings, cycles=cycles, maximum_seconds=60, authorized=True)
    return report, loading_snapshots, clock, completions


def test_actual_core_finish_loading_reconciles_and_waits_for_new_frames(monkeypatch):
    report, snapshots, _, completions = run_with_core(monkeypatch)
    assert report["status"] == "completed"
    assert report["successes"] == 2
    assert len(completions) == 2
    for snapshot in snapshots:
        assert snapshot["simulation"]["status"] == "loading"
        assert snapshot["simulation"]["live_available"] is False
        assert snapshot["simulation"]["frame_url"] is None
        assert snapshot["presentation"]["status"] == "moving"
        assert snapshot["presentation"]["motion"]["phase"] is None
    assert all(cycle["result"]["physical_success"] for cycle in report["cycles"])


def test_missing_post_completion_frames_is_bounded_and_does_not_advance(monkeypatch):
    report, _, clock, completions = run_with_core(monkeypatch, render_after_finish=False)
    assert report["status"] == "failed"
    assert len(report["cycles"]) == report["successes"] == 1
    assert len(completions) == 1
    assert clock.monotonic() - completions[0] <= 5.01


@pytest.mark.parametrize("failure", ["epoch", "unavailable"])
def test_terminal_result_does_not_override_changed_or_unavailable_scene(monkeypatch, failure):
    report, _, _, completions = run_with_core(monkeypatch, failure=failure)
    assert report["status"] == "failed"
    assert report["stop_reason"] == "scene_changed"
    assert report["successes"] == 0
    assert len(completions) == 1


def test_loading_keeps_command_fence_and_stale_heartbeat_guards(monkeypatch):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, settings, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    runner.persist(status="moving")
    status = backend.bridge.status
    telemetry = MotionTelemetry(
        command_id=run.id,
        phase="complete",
        object_position=(0.42, -0.22, 0.2),
        target_station_id="accepted",
    )
    backend.bridge.status = lambda owner: status(owner).model_copy(
        update={"status": "loading", "motion": telemetry}
    )
    public = PublicDemo(settings, backend)
    assert public.snapshot()["presentation"]["status"] == "moving"
    with pytest.raises(Problem):
        public.frame("overview", run.evidence.epoch)
    telemetry.command_id = uuid4()
    payload = public.snapshot()
    assert payload["presentation"]["status"] == "stopped"
    assert payload["presentation"]["decision"] is None
    telemetry.command_id = run.id
    clock.sleep(11)
    assert public.snapshot()["presentation"]["status"] == "stopped"


@pytest.mark.parametrize("delay", [6, 61])
def test_completion_frame_fetch_cannot_cross_render_or_authorization_deadline(monkeypatch, delay):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, _, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = runner.checked_run(backend.get_run(ACTOR, run.id))
    runner.outcome(run)
    frame = backend.frame

    def slow_frame(*args):
        image, observation = frame(*args)
        clock.value += delay
        return image, observation

    backend.frame = slow_frame
    with pytest.raises(Problem) as error:
        runner.wait_for_completion_frames(run)
    assert error.value.code in {"completion_frames_timeout", "presentation_expired"}


def test_fresh_but_precompletion_frame_cannot_advance(monkeypatch):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, _, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = runner.checked_run(backend.get_run(ACTOR, run.id))
    runner.outcome(run)
    frame = backend.frame

    def old_frame(*args):
        image, observation = frame(*args)
        return image, observation.model_copy(
            update={"captured_at": run.execution.completed_at - timedelta(milliseconds=1)}
        )

    backend.frame = old_frame
    with pytest.raises(Problem, match="Fresh completion frames"):
        runner.wait_for_completion_frames(run)


@pytest.mark.parametrize("camera", ["overview", "inspection"])
def test_completion_camera_recovered_before_status_reread_is_retried(monkeypatch, camera):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, _, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = runner.checked_run(backend.get_run(ACTOR, run.id))
    frame = backend.frame
    refreshed = False
    captures = []

    def recovering_frame(*args):
        nonlocal refreshed
        if args[-1] == camera and not refreshed:
            refreshed = True
            raise Problem(503, "camera_not_ready", "The camera recovered before status reread.")
        result = frame(*args)
        captures.append(args[-1])
        return result

    backend.frame = recovering_frame
    runner.wait_for_completion_frames(run)
    assert refreshed
    assert captures[-2:] == ["overview", "inspection"]
    assert clock.monotonic() < 5.5


def test_repeated_completion_camera_races_keep_the_original_deadline(monkeypatch):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, _, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = runner.checked_run(backend.get_run(ACTOR, run.id))
    started = clock.monotonic()

    def unavailable_frame(*args):
        raise Problem(503, "camera_not_ready", "No completion frame.")

    backend.frame = unavailable_frame
    with pytest.raises(Problem) as error:
        runner.wait_for_completion_frames(run)
    assert error.value.code == "completion_frames_timeout"
    assert clock.monotonic() - started <= 5.01


def test_completion_camera_refresh_cannot_retry_a_changed_epoch(monkeypatch):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, _, clock = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    clock.sleep(0.5)
    run = runner.checked_run(backend.get_run(ACTOR, run.id))
    calls = 0

    def changed_frame(*args):
        nonlocal calls
        calls += 1
        backend.bridge.epoch = uuid4()
        raise Problem(503, "camera_not_ready", "A different scene is loading.")

    backend.frame = changed_frame
    with pytest.raises(Problem) as error:
        runner.wait_for_completion_frames(run)
    assert error.value.code == "scene_changed"
    assert calls == 1


@pytest.mark.parametrize("throws", [False, True])
def test_public_capture_race_to_loading_never_publishes_pixels_or_false_stop(monkeypatch, throws):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    backend, settings, _ = prepared
    backend.approve(ACTOR, run.id, run.plan.model_response_id)
    runner.persist(status="moving")
    status, frame = backend.bridge.status, backend.frame

    def refreshing_frame(*args):
        result = frame(*args)
        backend.bridge.status = lambda owner: status(owner).model_copy(update={"status": "loading"})
        if throws:
            raise Problem(503, "camera_not_ready", "Waiting for render")
        return result

    backend.frame = refreshing_frame
    snapshot = PublicDemo(settings, backend).snapshot()
    assert snapshot["simulation"]["live_available"] is False
    assert snapshot["simulation"]["frame_url"] is None
    assert snapshot["simulation"]["status"] == "loading"
    assert snapshot["presentation"]["status"] == "moving"
