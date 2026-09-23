from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from io import BytesIO
from typing import Literal
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from apps.api.errors import Problem
from apps.api.models import (
    TERMINAL,
    Activation,
    EnvironmentRecord,
    Event,
    Evidence,
    Execution,
    MotionCommand,
    Observation,
    Plan,
    Principal,
    RunError,
    RunRecord,
    SaveEnvironment,
    SimulationStatus,
    StartRun,
    Stored,
    utcnow,
)
from apps.api.ports import Artifacts, Bridge, Planner, PolicyAuthorizer, Store
from contracts.validate_environment import parse_document, validate_environment


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def png_bytes(observation: Observation) -> bytes:
    try:
        image = base64.b64decode(observation.image_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise Problem(
            502, "invalid_camera_data", "The simulator returned invalid image encoding."
        ) from exc
    if not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 5 * 1024 * 1024:
        raise Problem(502, "invalid_camera_data", "The simulator must provide a PNG below 5 MiB.")
    try:
        with Image.open(BytesIO(image)) as decoded:
            if decoded.format != "PNG" or decoded.width * decoded.height > 8_000_000:
                raise ValueError("Camera image exceeds the supported resolution.")
            decoded.verify()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as exc:
        raise Problem(
            502, "invalid_camera_data", "The simulator image cannot be verified."
        ) from exc
    return image


def check_fresh(observation: Observation, max_age_ms: int) -> None:
    age_ms = (utcnow() - observation.captured_at).total_seconds() * 1000
    if age_ms < -500 or age_ms > max_age_ms:
        raise Problem(
            409, "stale_observation", "A fresh, clock-consistent simulator observation is required."
        )


class FactoryService:
    def __init__(
        self,
        store: Store,
        artifacts: Artifacts,
        planner: Planner,
        bridge: Bridge,
        approval_ttl_seconds: int = 300,
        *,
        agent_probe_verified_at: datetime | None = None,
        policies: PolicyAuthorizer | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.planner = planner
        self.bridge = bridge
        self.approval_ttl_seconds = approval_ttl_seconds
        self.agent_probe_verified_at = agent_probe_verified_at
        self.policies = policies

    def environment(self, actor: Principal, environment_id: str) -> Stored[EnvironmentRecord]:
        record = self.store.get_environment(actor.owner_key, environment_id)
        if record is None:
            raise Problem(404, "environment_missing", "Environment not found for this user.")
        return record

    def save_environment(self, actor: Principal, request: SaveEnvironment) -> EnvironmentRecord:
        try:
            document = parse_document(request.document_json)
        except ValueError as exc:
            raise Problem(422, "invalid_environment", str(exc)) from exc
        issues = validate_environment(document)
        if issues:
            raise Problem(
                422,
                "invalid_environment",
                "Correct the customer environment configuration.",
                [asdict(issue) for issue in issues],
            )
        assert isinstance(document, dict)
        current = self.store.get_environment(actor.owner_key, document["environment_id"])
        if current is None and request.expected_revision is not None:
            raise Problem(
                409, "revision_conflict", "This environment does not have the expected revision."
            )
        if current is not None and current.value.revision != request.expected_revision:
            raise Problem(409, "revision_conflict", "Reload the current environment before saving.")
        revision = content_hash(document)
        if current is not None and current.value.revision == revision:
            return current.value
        now = utcnow()
        record = EnvironmentRecord(
            environment_id=document["environment_id"],
            display_name=document["display_name"],
            revision=revision,
            document=document,
            created_at=current.value.created_at if current else now,
            updated_at=now,
        )
        return self.store.put_environment(
            actor.owner_key, record, current.etag if current else None
        ).value

    def _live_environment(
        self, actor: Principal, environment_id: str, revision: str
    ) -> EnvironmentRecord:
        environment = self.environment(actor, environment_id).value
        if environment.revision != revision:
            raise Problem(
                409, "revision_conflict", "The requested environment revision is not current."
            )
        if environment.document["execution"]["mode"] != "live":
            raise Problem(
                409, "replay_not_live", "Replay configuration cannot drive the live simulator."
            )
        return environment

    def activate(self, actor: Principal, environment_id: str, revision: str) -> Activation:
        environment = self._live_environment(actor, environment_id, revision)
        return self.bridge.activate(actor.owner_key, environment)

    def runtime(self, actor: Principal) -> SimulationStatus:
        try:
            return self.bridge.status(actor.owner_key)
        except Problem as exc:
            return SimulationStatus(status="unavailable", message=exc.message)

    def frame(
        self,
        actor: Principal,
        environment_id: str,
        revision: str,
        camera: Literal["overview", "inspection"],
    ) -> tuple[bytes, Observation]:
        environment = self._live_environment(actor, environment_id, revision)
        observation = self.bridge.observe(actor.owner_key, environment_id, revision, camera)
        self._check_scene(observation, environment_id, revision)
        check_fresh(observation, environment.document["execution"]["max_observation_age_ms"])
        return png_bytes(observation), observation

    @staticmethod
    def _check_scene(observation: Observation, environment_id: str, revision: str) -> None:
        if observation.environment_id != environment_id or observation.revision != revision:
            raise Problem(
                409, "scene_changed", "The simulator is not running the requested scene revision."
            )

    def _run(self, actor: Principal, run_id: UUID) -> Stored[RunRecord]:
        stored = self.store.get_run(actor.owner_key, run_id)
        if stored is None:
            raise Problem(404, "run_missing", "Run not found for this user.")
        return stored

    def _save(self, actor: Principal, stored: Stored[RunRecord], run: RunRecord) -> RunRecord:
        run.updated_at = utcnow()
        return self.store.put_run(actor.owner_key, run, stored.etag).value

    def start(self, actor: Principal, request: StartRun) -> RunRecord:
        fingerprint = content_hash(request.fingerprint_document())
        existing = self.store.get_run(actor.owner_key, request.request_id)
        if existing is not None:
            if existing.value.request_fingerprint != fingerprint:
                raise Problem(409, "request_id_reused", "Use a new request ID for changed input.")
            return self.get_run(actor, request.request_id)
        environment = self._live_environment(actor, request.environment_id, request.revision)
        policy = None
        if request.policy_release_id is not None:
            if self.policies is None:
                raise Problem(
                    503, "learning_disabled", "Released learned execution is not configured."
                )
            policy = self.policies.resolve_for_run(
                actor, request.policy_release_id, request.environment_id, request.revision
            )
        now = utcnow()
        run = RunRecord(
            id=request.request_id,
            environment_id=environment.environment_id,
            revision=environment.revision,
            instruction=request.instruction,
            status="planning",
            created_at=now,
            updated_at=now,
            request_fingerprint=fingerprint,
            environment_document=environment.document,
            events=[Event(kind="planning", message="Waiting for a real simulator observation.")],
            policy=policy,
        )
        try:
            stored = self.store.put_run(actor.owner_key, run, None)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            winner = self._run(actor, run.id).value
            if winner.request_fingerprint != fingerprint:
                raise Problem(
                    409, "request_id_reused", "Use a new request ID for changed input."
                ) from exc
            return winner
        run = run.model_copy(deep=True)
        try:
            observation = self.bridge.observe(actor.owner_key, run.environment_id, run.revision)
            self._check_scene(observation, run.environment_id, run.revision)
            check_fresh(observation, environment.document["execution"]["max_observation_age_ms"])
            image = png_bytes(observation)
            blob_name = f"{actor.owner_key}/{run.id}/{observation.observation_id}.png"
            self.artifacts.put(blob_name, image)
            run.evidence = Evidence(
                observation_id=observation.observation_id,
                epoch=observation.epoch,
                state_revision=observation.state_revision,
                captured_at=observation.captured_at,
                object_id=observation.object_id,
                object_position=observation.object_position,
                blob_name=blob_name,
                sha256=hashlib.sha256(image).hexdigest(),
            )
            decision, response_id = self.planner.inspect(request.instruction, observation)
            if decision.object_id != observation.object_id:
                raise Problem(
                    422, "wrong_object", "The inspection does not refer to the observed part."
                )
            workflow = environment.document["workflow"]
            target = workflow[
                "accept_station" if decision.classification == "accepted" else "reject_station"
            ]
            if policy is not None and policy.goal_station_id != target:
                raise Problem(
                    409,
                    "policy_goal_mismatch",
                    "The inspection destination is outside this reviewed policy's task.",
                )
            run.plan = Plan(
                **decision.model_dump(),
                target_station_id=target,
                observation_id=observation.observation_id,
                epoch=observation.epoch,
                state_revision=observation.state_revision,
                model_response_id=response_id,
                expires_at=utcnow() + timedelta(seconds=self.approval_ttl_seconds),
            )
            run.status = "awaiting_approval"
            run.events.append(
                Event(kind="awaiting_approval", message="Inspection recorded; no motion executed.")
            )
        except Problem as exc:
            run.status = "failed"
            run.error = RunError(code=exc.code, message=exc.message, retryable=exc.retryable)
            run.events.append(Event(kind="failed", message=exc.message))
        try:
            return self._save(actor, stored, run)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self._run(actor, run.id).value

    def approve(self, actor: Principal, run_id: UUID, response_id: str) -> RunRecord:
        stored = self._run(actor, run_id)
        run = stored.value.model_copy(deep=True)
        if run.plan is None or response_id != run.plan.model_response_id:
            raise Problem(
                409, "approval_mismatch", "Approval must refer to the exact displayed inspection."
            )
        if run.status in TERMINAL or run.status in ("running", "cancelling"):
            return run
        if run.status != "awaiting_approval" or run.evidence is None:
            raise Problem(
                409, "not_awaiting_approval", "This run cannot be approved in its current state."
            )
        if utcnow() >= run.plan.expires_at:
            raise Problem(
                409, "approval_expired", "This inspection approval expired. Request a new plan."
            )
        if content_hash(run.environment_document) != run.revision:
            raise Problem(
                409,
                "revision_corrupted",
                "The saved run configuration no longer matches its revision.",
            )
        observation = self.bridge.observe(actor.owner_key, run.environment_id, run.revision)
        self._check_scene(observation, run.environment_id, run.revision)
        execution_settings = run.environment_document["execution"]
        check_fresh(observation, execution_settings["max_observation_age_ms"])
        if (
            observation.epoch != run.evidence.epoch
            or observation.state_revision != run.evidence.state_revision
            or observation.object_id != run.evidence.object_id
            or any(
                abs(a - b) > 0.02
                for a, b in zip(
                    observation.object_position, run.evidence.object_position, strict=True
                )
            )
        ):
            raise Problem(
                409, "scene_changed", "The part or scene changed. Request a new inspection."
            )
        deadline = utcnow() + timedelta(seconds=execution_settings["max_step_seconds"])
        command = MotionCommand(
            command_id=run.id,
            environment_id=run.environment_id,
            revision=run.revision,
            epoch=observation.epoch,
            state_revision=observation.state_revision,
            observation_id=observation.observation_id,
            object_id=observation.object_id,
            target_station_id=run.plan.target_station_id,
            deadline=deadline,
        )
        run.status = "running"
        run.command_deadline = deadline
        run.execution = Execution(command_id=run.id, status="queued")
        run.events.append(
            Event(kind="approved", message="Approved inspection; reserving one simulator command.")
        )
        reserved = self.store.put_run(actor.owner_key, run, stored.etag)
        run = reserved.value.model_copy(deep=True)
        try:
            result = (
                self.bridge.dispatch_policy(actor.owner_key, command, run.policy)
                if run.policy is not None
                else self.bridge.dispatch(actor.owner_key, command)
            )
            self._apply_execution(run, result)
        except Problem as exc:
            if exc.status < 500:
                run.status = "failed"
            run.error = RunError(
                code="dispatch_unconfirmed" if exc.status >= 500 else exc.code,
                message=exc.message,
                retryable=False,
            )
            run.events.append(
                Event(
                    kind="dispatch_unconfirmed" if exc.status >= 500 else "failed",
                    message="Do not resubmit motion. Reconcile the recorded command ID.",
                )
            )
        try:
            return self._save(actor, reserved, run)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self._run(actor, run.id).value

    @staticmethod
    def _apply_execution(run: RunRecord, execution: Execution) -> None:
        if execution.command_id != run.id:
            raise Problem(
                502, "wrong_command_result", "Simulator result belongs to a different command."
            )
        if run.policy is not None:
            runtime = execution.policy_runtime
            if runtime is None or (
                runtime.policy_release_id != run.policy.policy_release_id
                or runtime.policy_type != run.policy.policy_type
                or runtime.control_profile_id != run.policy.control_profile_id
                or runtime.reference_route_calls != 0
                or (
                    runtime.applied_model_sha is not None
                    and runtime.applied_model_sha != run.policy.model_sha256
                )
                or (
                    execution.status == "succeeded"
                    and (
                        runtime.applied_model_sha != run.policy.model_sha256
                        or runtime.policy_predict_calls == 0
                        or runtime.applied_action_count == 0
                    )
                )
            ):
                raise Problem(
                    502,
                    "learned_execution_unverified",
                    "The command did not provide matching learned-policy action evidence.",
                )
        if execution.status == "succeeded" and (
            execution.final_position is None or execution.completed_at is None
        ):
            raise Problem(
                502, "missing_physical_evidence", "Physical completion evidence is missing."
            )
        if execution.status == "succeeded":
            if run.plan is None:
                raise Problem(502, "missing_plan", "Completion has no approved destination.")
            target = next(
                item["position_m"]
                for item in run.environment_document["stations"]
                if item["id"] == run.plan.target_station_id
            )
            if any(
                abs(a - b) > 0.04 for a, b in zip(execution.final_position, target, strict=True)
            ):
                raise Problem(
                    502,
                    "wrong_destination",
                    "The observed part is outside the approved tray volume.",
                )
            if run.command_deadline is not None and execution.completed_at > run.command_deadline:
                raise Problem(
                    502,
                    "late_completion",
                    "Physical completion occurred after the command deadline.",
                )
        run.execution = execution
        if execution.status in TERMINAL:
            if run.status in TERMINAL:
                return
            run.status = execution.status
            run.error = execution.error
            run.events.append(
                Event(kind=execution.status, message="Simulator terminal result received.")
            )
            if execution.demonstration is not None:
                run.events.append(
                    Event(
                        kind="demonstration",
                        message=(
                            "Owner-scoped demonstration uploaded to Azure."
                            if execution.demonstration.status == "uploaded"
                            else "Demonstration was not published; inspect simulator logs."
                        ),
                    )
                )

    def get_run(self, actor: Principal, run_id: UUID) -> RunRecord:
        stored = self._run(actor, run_id)
        run = stored.value.model_copy(deep=True)
        if run.status in TERMINAL or run.status == "awaiting_approval":
            return run
        if run.status == "planning":
            if (utcnow() - run.created_at).total_seconds() > 150:
                run.status = "timed_out"
                run.error = RunError(
                    code="planning_expired",
                    message="Planning did not complete; no motion was authorized.",
                )
                return self._save(actor, stored, run)
            return run
        try:
            execution = self.bridge.command(actor.owner_key, run.id)
        except Problem as exc:
            if exc.status != 404:
                raise
            if run.command_deadline is not None and utcnow() > run.command_deadline:
                run.status = "timed_out"
                run.error = RunError(
                    code="command_unconfirmed",
                    message=(
                        "Command deadline expired without completion evidence; "
                        "inspect the simulator before restarting."
                    ),
                )
                return self._save(actor, stored, run)
            return run
        try:
            self._apply_execution(run, execution)
        except Problem as exc:
            run.status = "failed"
            run.error = RunError(code=exc.code, message=exc.message)
            run.events.append(Event(kind="failed", message=exc.message))
        if run.model_dump() == stored.value.model_dump():
            return run
        try:
            return self._save(actor, stored, run)
        except Problem as exc:
            if exc.code != "revision_conflict":
                raise
            return self._run(actor, run.id).value

    def cancel(self, actor: Principal, run_id: UUID) -> RunRecord:
        stored = self._run(actor, run_id)
        run = stored.value.model_copy(deep=True)
        if run.status in TERMINAL:
            return run
        if run.status in ("planning", "awaiting_approval"):
            run.status = "cancelled"
            run.events.append(Event(kind="cancelled", message="Cancelled before motion approval."))
            return self._save(actor, stored, run)
        run.status = "cancelling"
        run.events.append(
            Event(kind="cancelling", message="Waiting for simulator stop confirmation.")
        )
        saved = self.store.put_run(actor.owner_key, run, stored.etag)
        result = self.bridge.cancel(actor.owner_key, run.id)
        run = saved.value.model_copy(deep=True)
        self._apply_execution(run, result)
        return self._save(actor, saved, run)

    def evidence(self, actor: Principal, run_id: UUID) -> bytes:
        run = self._run(actor, run_id).value
        if run.evidence is None:
            raise Problem(404, "observation_missing", "No observation was captured for this run.")
        image = self.artifacts.get(run.evidence.blob_name)
        if hashlib.sha256(image).hexdigest() != run.evidence.sha256:
            raise Problem(
                502, "observation_corrupted", "Stored observation checksum does not match."
            )
        return image
