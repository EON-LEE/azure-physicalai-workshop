"""Single-use, operator-authorized paired presentation. No anonymous inference or motion."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import timedelta
from uuid import uuid4

from apps.api.errors import Problem
from apps.api.main import azure_service
from apps.api.models import (
    TERMINAL,
    PresentationOutcome,
    PresentationRecord,
    RunRecord,
    SimulationStatus,
    StartRun,
    utcnow,
)
from apps.api.presentation import (
    INSTRUCTION,
    cycle_run_id,
    inspection_correct,
    publication,
    result_for,
    slot,
    validate_record,
    validate_run,
    validate_run_identity,
)
from apps.api.settings import Settings

log = logging.getLogger(__name__)


class Runner:
    def __init__(self, service, settings: Settings, cycles: int, maximum_seconds: int):
        self.service, self.settings = service, settings
        self.actor, self.environments = publication(settings, service, paired=True)
        self.cycles, self.maximum_seconds = cycles, maximum_seconds
        self.stored = None
        self.monotonic_deadline = time.monotonic() + maximum_seconds

    @property
    def record(self) -> PresentationRecord:
        return self.stored.value

    def remaining(self) -> float:
        return min(
            self.monotonic_deadline - time.monotonic(),
            (self.record.expires_at - utcnow()).total_seconds(),
        )

    def persist(self, **changes) -> None:
        record = PresentationRecord.model_validate(
            self.record.model_copy(update={**changes, "updated_at": utcnow()}).model_dump()
        )
        validate_record(record, self.settings, self.actor)
        self.stored = self.service.store.put_presentation(
            self.actor.owner_key,
            record,
            self.stored.etag,
        )

    def guard(self) -> None:
        if self.remaining() <= 0:
            raise Problem(409, "presentation_expired", "The presentation authorization expired.")
        publication(self.settings, self.service, paired=True)
        persisted = self.service.store.get_presentation(self.actor.owner_key, self.record.id)
        if (
            persisted is None
            or persisted.etag != self.stored.etag
            or persisted.value.runner_id != self.record.runner_id
            or persisted.value.expires_at != self.record.expires_at
        ):
            raise Problem(409, "presentation_claim_changed", "The presentation claim changed.")

    def summary(self, *, reused: bool = False) -> dict:
        record = self.record
        return {
            "kind": "actual_reference_presentation",
            "presentation_id": record.id,
            "status": record.status,
            "started_at": record.started_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "total_cycles": record.total_cycles,
            "stop_reason": record.stop_reason,
            "cycles": [outcome.model_dump(mode="json") for outcome in record.outcomes],
            "successes": sum(o.result.status == "succeeded" for o in record.outcomes),
            "anonymous_control": False,
            "reused": reused,
        }

    def claim(self) -> bool:
        previous = self.service.store.get_presentation(
            self.actor.owner_key,
            self.settings.public_demo_presentation_id,
        )
        if previous is not None:
            validate_record(previous.value, self.settings, self.actor)
            if (
                previous.value.total_cycles != self.cycles
                or (previous.value.expires_at - previous.value.started_at).total_seconds()
                != self.maximum_seconds
            ):
                raise Problem(
                    409,
                    "authorization_changed",
                    "A presentation ID cannot change its original authorization.",
                )
            if previous.value.status not in {"completed", "stopped", "failed"}:
                raise Problem(
                    409,
                    "presentation_already_claimed",
                    "Reconcile the existing presentation; restarting cannot renew it.",
                )
            self.stored = previous
            return False
        started = utcnow()
        normal, defect = self.environments
        record = PresentationRecord(
            id=self.settings.public_demo_presentation_id,
            owner_key=self.actor.owner_key,
            normal_environment_id=normal.environment_id,
            normal_revision=normal.revision,
            defect_environment_id=defect.environment_id,
            defect_revision=defect.revision,
            runner_id=uuid4(),
            started_at=started,
            expires_at=started + timedelta(seconds=self.maximum_seconds),
            updated_at=started,
            total_cycles=self.cycles,
            cycle=1,
            status="preparing",
        )
        self.stored = self.service.store.put_presentation(self.actor.owner_key, record, None)
        return True

    def checked_run(self, run: RunRecord) -> RunRecord:
        validate_run(self.record, run, self.environments[(self.record.cycle - 1) % 2])
        return run

    def guard_scene(self, *, allow_loading: bool = False) -> SimulationStatus:
        environment_id, revision, _ = slot(self.record)
        status = self.service.runtime(self.actor)
        if (
            status.status not in ({"ready", "loading"} if allow_loading else {"ready"})
            or status.environment_id != environment_id
            or status.revision != revision
            or status.epoch != self.record.scene_epoch
        ):
            raise Problem(409, "scene_changed", "The authorized scene is no longer ready.")
        if (
            status.motion is not None
            and status.motion.command_id is not None
            and status.motion.command_id != self.record.run_id
        ):
            raise Problem(409, "scene_changed", "The authorized command changed.")
        return status

    def wait_for_completion_frames(self, run: RunRecord) -> None:
        deadline = time.monotonic() + min(5, max(0, self.remaining()))
        environment_id, revision, _ = slot(self.record)
        while time.monotonic() < deadline:
            self.guard()
            status = self.guard_scene(allow_loading=True)
            if status.status == "ready":
                fresh = True
                for camera in ("overview", "inspection"):
                    try:
                        _, observation = self.service.frame(
                            self.actor,
                            environment_id,
                            revision,
                            camera,
                        )
                    except Problem as exc:
                        if exc.code != "camera_not_ready":
                            raise
                        self.guard_scene(allow_loading=True)
                        fresh = False
                        break
                    if observation.epoch != self.record.scene_epoch:
                        raise Problem(409, "scene_changed", "The completion frame epoch changed.")
                    fresh = fresh and observation.captured_at >= run.execution.completed_at
                latest = self.guard_scene(allow_loading=True)
                self.guard()
                if fresh and latest.status == "ready" and time.monotonic() < deadline:
                    return
            self.persist()
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        raise Problem(503, "completion_frames_timeout", "Fresh completion frames are unavailable.")

    def cancel(self, run: RunRecord, *, rejected_evidence: bool = False) -> RunRecord:
        def checked(result: RunRecord) -> RunRecord:
            validator = validate_run_identity if rejected_evidence else validate_run
            validator(self.record, result, self.environments[(self.record.cycle - 1) % 2])
            return result

        checked(run)
        if run.status in TERMINAL:
            return run
        run = checked(self.service.cancel(self.actor, run.id))
        deadline = time.monotonic() + 10
        while run.status not in TERMINAL and time.monotonic() < deadline:
            self.persist(status="stopped", stop_reason="time_limit")
            time.sleep(0.5)
            run = checked(self.service.get_run(self.actor, run.id))
        if run.status not in TERMINAL:
            raise Problem(
                503,
                "cancellation_unconfirmed",
                "No simulator terminal event confirmed cancellation.",
            )
        return run

    def outcome(self, run: RunRecord) -> None:
        outcome = PresentationOutcome(
            cycle=self.record.cycle,
            run_id=run.id,
            result=result_for(self.record, run),
        )
        self.persist(outcomes=[*self.record.outcomes, outcome])

    def cycle(self) -> None:
        self.guard()
        environment_id, revision, _ = slot(self.record)
        run_id = cycle_run_id(self.record, self.record.cycle)
        if self.service.store.get_run(self.actor.owner_key, run_id) is not None:
            raise Problem(
                409,
                "presentation_run_exists",
                "A prior run cannot be adopted into a new presentation.",
            )
        previous_epoch = self.service.runtime(self.actor).epoch
        self.service.activate(self.actor, environment_id, revision)
        ready_deadline = time.monotonic() + min(300, self.remaining())
        while time.monotonic() < ready_deadline:
            self.guard()
            status = self.service.runtime(self.actor)
            if (
                status.status == "ready"
                and status.environment_id == environment_id
                and status.revision == revision
                and status.epoch is not None
                and status.epoch != previous_epoch
            ):
                break
            if status.status == "unavailable":
                raise Problem(
                    503, "simulator_unavailable", "The reference simulator is unavailable."
                )
            self.persist(status="preparing")
            time.sleep(1)
        else:
            raise Problem(503, "activation_timeout", "No new ready scene epoch was observed.")
        self.persist(status="inspecting", scene_epoch=status.epoch, run_id=run_id)
        self.guard()
        run = self.checked_run(
            self.service.start(
                self.actor,
                StartRun(
                    request_id=run_id,
                    environment_id=environment_id,
                    revision=revision,
                    instruction=INSTRUCTION,
                ),
            )
        )
        if run.status == "awaiting_approval" and run.plan is not None:
            self.persist(status="awaiting_motion")
            if inspection_correct(self.record, run) is not True:
                run = self.cancel(run)
            elif self.remaining() <= 0:
                run = self.cancel(run)
                self.persist(status="stopped", stop_reason="time_limit")
            else:
                # Recheck the durable authorization after inference, immediately before approval.
                self.guard()
                self.persist(status="moving")
                self.guard()
                self.guard_scene()
                run = self.checked_run(
                    self.service.approve(
                        self.actor,
                        run.id,
                        run.plan.model_response_id,
                    )
                )
        deadline = time.monotonic() + min(35, max(0, self.remaining()))
        while run.status not in TERMINAL and time.monotonic() < deadline:
            self.guard()
            self.guard_scene(allow_loading=True)
            self.persist(status="moving" if run.execution is not None else "inspecting")
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
            run = self.checked_run(self.service.get_run(self.actor, run.id))
            self.guard_scene(allow_loading=True)
        if run.status not in TERMINAL:
            run = self.cancel(run)
            self.persist(status="stopped", stop_reason="time_limit")
        self.guard_scene(allow_loading=True)
        self.outcome(run)
        if run.status == "succeeded" and self.record.outcomes[-1].result.physical_success:
            self.wait_for_completion_frames(run)

    def run(self) -> dict:
        if not self.claim():
            return self.summary(reused=True)
        failures = 0
        try:
            for cycle in range(1, self.cycles + 1):
                if self.remaining() <= 0:
                    self.persist(status="stopped", stop_reason="time_limit")
                    break
                self.persist(cycle=cycle, status="preparing", scene_epoch=None, run_id=None)
                self.cycle()
                failures += self.record.outcomes[-1].result.status != "succeeded"
                if self.record.status == "stopped":
                    break
                if failures >= 3:
                    self.persist(status="failed", stop_reason="repeated_failures")
                    break
                self.persist(status="completed" if cycle == self.cycles else "preparing")
                if cycle != self.cycles:
                    time.sleep(min(5, max(0, self.remaining())))
        except (Problem, TimeoutError) as exc:
            # Never expose dependency messages, prompts or arbitrary operator records publicly.
            log.warning(
                "Reference presentation stopped (%s, code=%s)",
                type(exc).__name__,
                exc.code if isinstance(exc, Problem) else "timeout",
            )
            reason = "dependency_unavailable"
            if isinstance(exc, Problem):
                if exc.code == "presentation_expired":
                    reason = "time_limit"
                elif exc.code == "scene_changed":
                    reason = "scene_changed"
                elif exc.code == "cancellation_unconfirmed":
                    reason = "cancellation_unconfirmed"
            # Stop the owned claim before reading evidence that may have caused this failure.
            self.persist(
                status="stopped" if reason == "time_limit" else "failed", stop_reason=reason
            )
            if self.record.run_id is not None:
                try:
                    saved = self.service.store.get_run(self.actor.owner_key, self.record.run_id)
                    if saved is not None:
                        run = self.cancel(saved.value, rejected_evidence=True)
                        self.checked_run(run)
                        if reason != "scene_changed" and (
                            not self.record.outcomes
                            or self.record.outcomes[-1].cycle != self.record.cycle
                        ):
                            self.outcome(run)
                except Problem:
                    log.warning("Reference presentation cleanup evidence remains unverified")
                    reason = "cancellation_unconfirmed"
            self.persist(
                status="stopped" if reason == "time_limit" else "failed", stop_reason=reason
            )
        return self.summary()


def run_cycles(
    service,
    settings: Settings,
    *,
    cycles: int,
    maximum_seconds: int,
    authorized: bool = False,
) -> dict:
    if not authorized:
        raise ValueError("Explicit operator authorization is required.")
    if (
        type(cycles) is not int
        or type(maximum_seconds) is not int
        or not 1 <= cycles <= 1000
        or not 30 <= maximum_seconds <= 21600
    ):
        raise ValueError("Presentation cycles and duration must be explicitly bounded.")
    return Runner(service, settings, cycles, maximum_seconds).run()


def main() -> None:
    if os.environ.get("ALLOW_REFERENCE_PRESENTATION") != "true":
        raise RuntimeError("Explicit operator authorization is required for the presentation.")
    settings = Settings()
    if os.environ.get("PRESENTATION_ID", settings.public_demo_presentation_id) != (
        settings.public_demo_presentation_id
    ):
        raise ValueError("PRESENTATION_ID must match the pinned PUBLIC_DEMO_PRESENTATION_ID.")
    cycles = int(os.environ.get("PRESENTATION_CYCLES", "20"))
    seconds = int(os.environ["PRESENTATION_MAX_SECONDS"])
    service, resources = azure_service(settings)
    try:
        report = run_cycles(
            service, settings, cycles=cycles, maximum_seconds=seconds, authorized=True
        )
        print(json.dumps(report), flush=True)
        if report["status"] != "completed" or report["successes"] != cycles:
            raise SystemExit(2)
    finally:
        for resource in resources:
            resource.close()


if __name__ == "__main__":
    main()
