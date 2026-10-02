"""Shared publication scope and physical-result predicates; no anonymous write path."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from apps.api.errors import Problem
from apps.api.models import (
    EnvironmentRecord,
    Plan,
    PresentationRecord,
    PresentationResult,
    Principal,
    RunRecord,
    StartRun,
    utcnow,
)
from apps.api.service import FactoryService, content_hash
from apps.api.settings import Settings
from contracts.validate_environment import validate_environment

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[2] / "examples" / "inspection-cell.json").read_text(
        encoding="utf-8"
    )
)
INSTRUCTION = (
    "Inspect the synthetic part and sort it using the configured accepted or rejected station."
)
ACTIVE = frozenset({"preparing", "inspecting", "awaiting_motion", "moving"})


def is_reference_scene(document: dict, seed: int = 42) -> bool:
    if seed not in (42, 43) or validate_environment(document):
        return False
    expected = deepcopy(REFERENCE)
    expected["scene"]["seed"] = seed
    expected["environment_id"] = document["environment_id"]
    expected["display_name"] = document["display_name"]
    return document == expected


def publication(
    settings: Settings,
    service: FactoryService,
    *,
    paired: bool = False,
) -> tuple[Principal, list[EnvironmentRecord]]:
    if not settings.public_demo_publish_live or not all(
        (
            settings.public_demo_owner_id,
            settings.public_demo_environment_id,
            settings.public_demo_revision,
        )
    ):
        raise Problem(503, "live_not_published", "Live viewing has not been published.")
    pairs = [(settings.public_demo_environment_id, settings.public_demo_revision, 42)]
    if paired:
        if (
            not all(
                (
                    settings.public_demo_presentation_id,
                    settings.public_demo_defect_environment_id,
                    settings.public_demo_defect_revision,
                )
            )
            or settings.public_demo_environment_id == settings.public_demo_defect_environment_id
        ):
            raise Problem(
                503, "presentation_not_configured", "A pinned reference pair is required."
            )
        pairs.append(
            (
                settings.public_demo_defect_environment_id,
                settings.public_demo_defect_revision,
                43,
            )
        )
    actor = Principal(tenant_id=settings.entra_tenant_id, object_id=settings.public_demo_owner_id)
    records = []
    for environment_id, revision, seed in pairs:
        record = service.environment(actor, environment_id).value
        if (
            record.environment_id != environment_id
            or record.revision != revision
            or content_hash(record.document) != revision
            or record.document["environment_id"] != environment_id
            or not is_reference_scene(record.document, seed)
        ):
            raise Problem(503, "publication_changed", "The approved reference publication changed.")
        records.append(record)
    return actor, records


def cycle_run_id(record: PresentationRecord, cycle: int) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        (
            f"physicalai-presentation-v2:{record.owner_key}:{record.id}:"
            f"{record.normal_environment_id}:{record.normal_revision}:"
            f"{record.defect_environment_id}:{record.defect_revision}:{cycle}"
        ),
    )


def slot(record: PresentationRecord, cycle: int | None = None) -> tuple[str, str, str]:
    if (cycle or record.cycle) % 2:
        return record.normal_environment_id, record.normal_revision, "normal"
    return record.defect_environment_id, record.defect_revision, "surface_defect"


def validate_record(record: PresentationRecord, settings: Settings, actor: Principal) -> None:
    expected = (
        settings.public_demo_presentation_id,
        actor.owner_key,
        settings.public_demo_environment_id,
        settings.public_demo_revision,
        settings.public_demo_defect_environment_id,
        settings.public_demo_defect_revision,
    )
    actual = (
        record.id,
        record.owner_key,
        record.normal_environment_id,
        record.normal_revision,
        record.defect_environment_id,
        record.defect_revision,
    )
    if actual != expected or (
        record.run_id is not None and record.run_id != cycle_run_id(record, record.cycle)
    ):
        raise Problem(503, "presentation_scope", "Published presentation scope is invalid.")
    if record.status in {"inspecting", "awaiting_motion", "moving"} and (
        record.run_id is None or record.scene_epoch is None
    ):
        raise Problem(503, "presentation_scope", "Published cycle has no scene binding.")
    if record.status == "completed" and len(record.outcomes) != record.total_cycles:
        raise Problem(503, "presentation_scope", "Published completion has incomplete evidence.")
    for outcome in record.outcomes:
        if outcome.run_id != cycle_run_id(record, outcome.cycle) or not (
            record.started_at <= outcome.result.completed_at <= record.updated_at
        ):
            raise Problem(503, "presentation_scope", "Published outcome scope is invalid.")


def validate_run_identity(
    record: PresentationRecord, run: RunRecord, environment: EnvironmentRecord
) -> None:
    request = StartRun(
        request_id=cycle_run_id(record, record.cycle),
        environment_id=environment.environment_id,
        revision=environment.revision,
        instruction=INSTRUCTION,
    )
    if (
        run.execution_mode != "inspection"
        or run.policy is not None
        or (run.plan is not None and not isinstance(run.plan, Plan))
        or run.id != record.run_id
        or run.id != request.request_id
        or run.environment_id != environment.environment_id
        or run.revision != environment.revision
        or run.environment_document != environment.document
        or run.instruction != INSTRUCTION
        or run.request_fingerprint != content_hash(request.fingerprint_document())
        or not record.started_at <= run.created_at < record.expires_at
        or run.updated_at < run.created_at
    ):
        raise Problem(503, "presentation_run_scope", "Published run scope is invalid.")
    if run.execution is not None and run.execution.command_id != run.id:
        raise Problem(503, "presentation_command_scope", "Published command scope is invalid.")


def validate_run(
    record: PresentationRecord, run: RunRecord, environment: EnvironmentRecord
) -> None:
    validate_run_identity(record, run, environment)
    if run.plan is not None:
        evidence = run.evidence
        plan = run.plan
        target = environment.document["workflow"][
            "accept_station" if plan.classification == "accepted" else "reject_station"
        ]
        oldest_capture = run.created_at - timedelta(
            milliseconds=min(environment.document["execution"]["max_observation_age_ms"], 2000)
        )
        if (
            evidence is None
            or plan.epoch != record.scene_epoch
            or evidence.epoch != record.scene_epoch
            or plan.observation_id != evidence.observation_id
            or plan.object_id != evidence.object_id
            or plan.target_station_id != target
            or evidence.blob_name != f"{record.owner_key}/{run.id}/{evidence.observation_id}.png"
            or not oldest_capture <= evidence.captured_at <= run.updated_at
        ):
            raise Problem(
                503, "presentation_evidence_scope", "Published observation scope is invalid."
            )


def inspection_correct(record: PresentationRecord, run: RunRecord) -> bool | None:
    if not isinstance(run.plan, Plan) or run.execution_mode != "inspection":
        return None
    expected = "accepted" if slot(record)[2] == "normal" else "rejected"
    return run.plan.classification == expected


def physical_success(run: RunRecord) -> bool:
    execution = run.execution
    if (
        run.status != "succeeded"
        or execution is None
        or execution.status != "succeeded"
        or execution.command_id != run.id
        or execution.final_position is None
        or execution.completed_at is None
        or run.command_deadline is None
        or run.plan is None
        or execution.error is not None
        or run.error is not None
        or execution.completed_at > utcnow()
        or execution.completed_at < run.created_at
        or execution.completed_at < run.command_deadline - timedelta(seconds=30)
        or execution.completed_at > run.command_deadline
    ):
        return False
    target = next(
        item["position_m"]
        for item in run.environment_document["stations"]
        if item["id"] == run.plan.target_station_id
    )
    return all(abs(a - b) <= 0.04 for a, b in zip(execution.final_position, target, strict=True))


def result_for(record: PresentationRecord, run: RunRecord) -> PresentationResult:
    correct = inspection_correct(record, run)
    physical = physical_success(run)
    status = (
        "succeeded"
        if physical and correct
        else "failed"
        if correct is False
        else run.status
        if run.status in {"cancelled", "timed_out"}
        else "failed"
    )
    execution = run.execution
    completed = execution.completed_at if execution and execution.completed_at else run.updated_at
    return PresentationResult(
        status=status,
        physical_success=physical,
        inspection_correct=correct,
        final_position_m=execution.final_position if execution else None,
        completed_at=completed,
        message=(
            "Inspection and physical sorting completed."
            if status == "succeeded"
            else "Inspection disagreed with the reference evaluation; motion was not authorized."
            if correct is False and execution is None
            else "The authorized presentation time expired."
            if status == "timed_out"
            else "The reference cycle was cancelled."
            if status == "cancelled"
            else "Physical completion was not verified."
        ),
    )
