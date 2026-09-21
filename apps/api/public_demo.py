"""Curated anonymous viewing, never anonymous identity, inference, or control."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.models import (
    EnvironmentRecord,
    Observation,
    PresentationCounts,
    PresentationDecision,
    PresentationMotion,
    PresentationRecord,
    Principal,
    PublicPresentation,
    RunRecord,
    SimulationStatus,
    utcnow,
)
from apps.api.presentation import (
    ACTIVE,
    INSTRUCTION,
    REFERENCE,
    inspection_correct,
    is_reference_scene,
    publication,
    result_for,
    slot,
    validate_record,
    validate_run,
)
from apps.api.service import FactoryService
from apps.api.settings import Settings

EVIDENCE = json.loads(
    Path(__file__).with_name("public_demo_evidence.json").read_text(encoding="utf-8")
)

# Re-export the existing helper for operator tooling that imports it here.
__all__ = ["PublicDemo", "is_reference_scene"]


class PublicDemo:
    def __init__(self, settings: Settings, service: FactoryService) -> None:
        self.settings = settings
        self.service = service
        self._lock = threading.RLock()

    def _scope(
        self,
    ) -> tuple[Principal, list[EnvironmentRecord], PresentationRecord | None, RunRecord | None]:
        paired = self.settings.public_demo_presentation_id is not None
        actor, environments = publication(self.settings, self.service, paired=paired)
        record = None
        run = None
        if paired:
            stored = self.service.store.get_presentation(
                actor.owner_key, self.settings.public_demo_presentation_id
            )
            if stored is not None:
                record = PresentationRecord.model_validate(stored.value.model_dump())
                validate_record(record, self.settings, actor)
                if record.run_id is not None:
                    saved_run = self.service.store.get_run(actor.owner_key, record.run_id)
                    # A preparing/inspecting record can precede the first run write.
                    if saved_run is None:
                        if record.status not in {"preparing", "inspecting", "failed", "stopped"}:
                            raise Problem(
                                503, "presentation_run_missing", "Published run is unavailable."
                            )
                    else:
                        run = saved_run.value
                        validate_run(record, run, environments[(record.cycle - 1) % 2])
                        if record.outcomes and record.outcomes[-1].cycle == record.cycle:
                            if record.outcomes[-1].result != result_for(record, run):
                                raise Problem(
                                    503,
                                    "presentation_result_changed",
                                    "Published result does not match physical evidence.",
                                )
        return actor, environments, record, run

    @staticmethod
    def _binding(record: PresentationRecord | None) -> tuple | None:
        if record is None:
            return None
        return record.id, record.cycle, record.scene_epoch, record.run_id

    def _current(
        self,
        actor: Principal,
        environments: list[EnvironmentRecord],
        record: PresentationRecord | None,
    ) -> tuple[EnvironmentRecord, SimulationStatus]:
        if self.settings.public_demo_presentation_id and record is None:
            raise Problem(
                503, "presentation_not_started", "The reference presentation has not started."
            )
        environment = environments[(record.cycle - 1) % 2] if record else environments[0]
        status = self.service.runtime(actor)
        if (
            status.status not in {"ready", "loading"}
            or status.environment_id != environment.environment_id
            or status.revision != environment.revision
            or status.epoch is None
            or (record is not None and status.epoch != record.scene_epoch)
        ):
            raise Problem(503, "public_scene_unavailable", "The published scene is not ready.")
        if (
            record is not None
            and status.motion is not None
            and status.motion.command_id is not None
            and status.motion.command_id != record.run_id
        ):
            raise Problem(
                409, "public_command_changed", "The published command changed; reload its state."
            )
        if status.status == "loading":
            raise Problem(503, "public_frame_refresh", "Waiting for fresh published camera frames.")
        return environment, status

    def _frame(
        self,
        camera: Literal["overview", "inspection"],
        epoch: UUID | None,
        scope: tuple[
            Principal, list[EnvironmentRecord], PresentationRecord | None, RunRecord | None
        ],
    ) -> tuple[bytes, Observation, SimulationStatus]:
        actor, environments, record, _ = scope
        environment, status = self._current(actor, environments, record)
        if epoch is not None and epoch != status.epoch:
            raise Problem(
                409, "public_epoch_changed", "The published scene changed; reload its state."
            )
        try:
            image, observation = self.service.frame(
                actor,
                environment.environment_id,
                environment.revision,
                camera,
            )
        except Problem as exc:
            if exc.code == "camera_not_ready":
                self._current(actor, environments, record)
            raise
        _, latest_environments, latest_record, _ = self._scope()
        _, latest_status = self._current(actor, latest_environments, latest_record)
        if (
            observation.epoch != status.epoch
            or latest_status.epoch != status.epoch
            or self._binding(record) != self._binding(latest_record)
        ):
            raise Problem(
                409, "public_epoch_changed", "The published scene changed; reload its state."
            )
        return image, observation, latest_status

    def _presentation(
        self,
        record: PresentationRecord,
        run: RunRecord | None,
        telemetry: SimulationStatus | None,
        live: bool,
        refreshing: bool = False,
    ) -> dict:
        decision = None
        motion = None
        current_result = None
        if record.outcomes and record.outcomes[-1].cycle == record.cycle:
            current_result = record.outcomes[-1].result
        if run is not None and run.plan is not None and run.evidence is not None:
            decision = PresentationDecision(
                classification=run.plan.classification,
                summary=run.plan.summary,
                target_station_id=run.plan.target_station_id,
                observation_id=run.evidence.observation_id,
                captured_at=run.evidence.captured_at,
            )
            if run.execution is not None:
                actual = telemetry.motion if telemetry is not None else None
                matches = (
                    live
                    and actual is not None
                    and actual.command_id == run.execution.command_id
                    and actual.target_station_id == run.plan.target_station_id
                )
                target = next(
                    item["position_m"]
                    for item in run.environment_document["stations"]
                    if item["id"] == run.plan.target_station_id
                )
                motion = PresentationMotion(
                    status=run.execution.status,
                    phase=actual.phase if matches else None,
                    part_position_m=actual.object_position if matches else None,
                    target_position_m=target,
                )
        status = record.status
        age = (utcnow() - record.updated_at).total_seconds()
        # Model calls can take 120 seconds; motion must heartbeat at least every 10 seconds.
        stale_limit = 150 if status == "inspecting" else 10
        if status in ACTIVE and (
            utcnow() >= record.expires_at
            or age < 0
            or age > stale_limit
            or (status in {"awaiting_motion", "moving"} and not live and not refreshing)
        ):
            status = "stopped"
        if (
            status in {"stopped", "failed"}
            and motion is not None
            and motion.status
            in {
                "queued",
                "running",
                "cancelling",
            }
        ):
            # No terminal simulator event was received; do not invent one to stop an animation.
            motion = None
        outcomes = record.outcomes
        current_counted = bool(outcomes and outcomes[-1].cycle == record.cycle)
        correct = sum(o.result.inspection_correct is True for o in outcomes)
        if run is not None and not current_counted and inspection_correct(record, run) is True:
            correct += 1
        return PublicPresentation(
            id=record.id,
            status=status,
            cycle=record.cycle,
            total_cycles=record.total_cycles,
            scenario=slot(record)[2],
            instruction=INSTRUCTION,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            scene_epoch=record.scene_epoch,
            run_id=record.run_id,
            decision=decision,
            motion=motion,
            result=current_result,
            counts=PresentationCounts(
                attempted=record.cycle,
                succeeded=sum(o.result.status == "succeeded" for o in outcomes),
                failed=sum(o.result.status != "succeeded" for o in outcomes),
                inspected_correctly=correct,
                physically_completed=sum(o.result.physical_success for o in outcomes),
            ),
        ).model_dump(mode="json")

    def snapshot(self) -> dict:
        with self._lock:
            simulation = {
                "status": "not_published",
                "live_available": False,
                "message_code": "live_not_published",
                "frame_url": None,
            }
            record = run = telemetry = None
            command_changed = False
            refreshing = False
            if self.settings.public_demo_publish_live:
                try:
                    scope = self._scope()
                    _, _, record, run = scope
                except (Problem, ValidationError) as exc:
                    if self.settings.public_demo_presentation_id:
                        raise Problem(
                            503,
                            "public_state_unavailable",
                            "The published presentation state is unavailable.",
                        ) from exc
                    scope = None
                try:
                    if scope is None:
                        raise Problem(503, "public_state_unavailable", "Publication unavailable.")
                    _, _, telemetry = self._frame("overview", None, scope)
                    simulation = {
                        "status": "ready",
                        "live_available": True,
                        "message_code": "live_ready",
                        "frame_url": "/api/demo/frame",
                    }
                except (Problem, ValidationError) as exc:
                    command_changed = (
                        isinstance(exc, Problem) and exc.code == "public_command_changed"
                    )
                    refreshing = isinstance(exc, Problem) and exc.code == "public_frame_refresh"
                    simulation = {
                        "status": "loading"
                        if refreshing or (record and record.status == "preparing")
                        else "unavailable",
                        "live_available": False,
                        "message_code": "live_unavailable",
                        "frame_url": None,
                    }
            presentation = (
                self._presentation(record, run, telemetry, simulation["live_available"], refreshing)
                if record
                else None
            )
            if command_changed and presentation is not None:
                presentation["decision"] = None
                presentation["motion"] = None
            verified_at = self.service.agent_probe_verified_at
            return {
                "api_version": "public-demo-v1",
                "access": "public_read_only",
                "deployment": "azure",
                "mode": "live" if simulation["live_available"] else "reference",
                "observed_at": utcnow().isoformat(),
                "scene": {
                    "id": "inspection-cell-v1",
                    "name": "Inspection and sorting cell",
                    "length_unit": "m",
                    "stations": deepcopy(REFERENCE["stations"]),
                    "robot": "Franka reference arm",
                    "data_origin": "synthetic_reference_configuration",
                },
                "simulation": simulation,
                "presentation": presentation,
                "agent": {
                    "provider": "microsoft_foundry",
                    "connectivity": "verified" if verified_at is not None else "configured",
                    "verified_at": verified_at.isoformat() if verified_at is not None else None,
                    "verification_scope": "connectivity_only",
                },
                "learning": deepcopy(EVIDENCE["learning_cpu"]),
                "capabilities": {
                    "anonymous_control": False,
                    "anonymous_editing": False,
                    "public_live_video": bool(simulation["live_available"]),
                },
            }

    def frame(
        self,
        camera: Literal["overview", "inspection"],
        epoch: UUID | None = None,
    ) -> tuple[bytes, Observation]:
        with self._lock:
            try:
                image, observation, _ = self._frame(camera, epoch, self._scope())
                return image, observation
            except (Problem, ValidationError) as exc:
                if isinstance(exc, Problem) and exc.code in {
                    "public_epoch_changed",
                    "public_command_changed",
                }:
                    raise
                raise Problem(
                    503,
                    "public_live_unavailable",
                    "The published live camera is unavailable. No substitute frame was returned.",
                ) from exc

    def evidence(self, observation_id: UUID) -> tuple[bytes, RunRecord]:
        with self._lock:
            try:
                actor, _, record, run = self._scope()
                if record is None or run is None or run.plan is None or run.evidence is None:
                    raise Problem(
                        503, "public_evidence_unavailable", "Published evidence unavailable."
                    )
                if run.evidence.observation_id != observation_id:
                    raise Problem(
                        409,
                        "public_observation_changed",
                        "The published observation changed; reload its state.",
                    )
                image = self.service.evidence(actor, run.id)
                _, _, latest, latest_run = self._scope()
                if (
                    self._binding(record) != self._binding(latest)
                    or latest_run is None
                    or latest_run.evidence != run.evidence
                ):
                    raise Problem(
                        409,
                        "public_observation_changed",
                        "The published observation changed; reload its state.",
                    )
                return image, run
            except (Problem, ValidationError) as exc:
                if isinstance(exc, Problem) and exc.code == "public_observation_changed":
                    raise
                raise Problem(
                    503, "public_evidence_unavailable", "The published observation is unavailable."
                ) from exc
