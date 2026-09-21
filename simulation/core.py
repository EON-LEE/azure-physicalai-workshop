from __future__ import annotations

import base64
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from apps.api.errors import Problem
from apps.api.models import (
    TERMINAL,
    Activation,
    EnvironmentRecord,
    Execution,
    MotionCommand,
    MotionPhase,
    MotionTelemetry,
    Observation,
    RunError,
    SimulationStatus,
    utcnow,
)
from apps.api.service import check_fresh, content_hash
from simulation.extensions import SceneRegistry, SceneSpec


@dataclass(frozen=True)
class LoadScene:
    environment: EnvironmentRecord
    spec: SceneSpec
    epoch: UUID


@dataclass(frozen=True)
class StartMotion:
    command: MotionCommand
    target_id: str


@dataclass(frozen=True)
class StopMotion:
    command_id: UUID


class SimulationCore:
    """Thread-safe protocol state; all physics execution stays in the Isaac main thread."""

    def __init__(self, registry: SceneRegistry, max_commands: int = 2048) -> None:
        self.registry = registry
        self.lock = threading.RLock()
        self.owner: str | None = None
        self.environment: EnvironmentRecord | None = None
        self.spec: SceneSpec | None = None
        self.epoch = uuid4()
        self.state_revision = 0
        self.physics_steps = 0
        self.ready = False
        self.error: str | None = None
        self.last_activity = utcnow()
        self.frames: dict[str, Observation] = {}
        self.observations: OrderedDict[UUID, Observation] = OrderedDict()
        self.commands: dict[tuple[str, UUID], Execution] = {}
        self.fingerprints: dict[tuple[str, UUID], str | None] = {}
        self.deadlines: dict[tuple[str, UUID], datetime] = {}
        self.pending: deque[LoadScene | StartMotion | StopMotion] = deque()
        self.active_command: tuple[str, UUID] | None = None
        self.motion: MotionTelemetry | None = None
        self.max_commands = max_commands

    def _owned(self, owner: str) -> None:
        if not self.owner or self.environment is None:
            raise Problem(409, "scene_not_active", "Activate a customer environment first.")
        if owner != self.owner:
            raise Problem(409, "simulator_occupied", "The simulator is leased to another user.")
        self.last_activity = utcnow()

    def status(self, owner: str) -> SimulationStatus:
        with self.lock:
            if self.owner is not None and self.owner != owner:
                return SimulationStatus(
                    status="occupied", message="The simulator is leased to another user."
                )
            if self.environment is None:
                return SimulationStatus(status="unavailable", message="No environment is active.")
            self.last_activity = utcnow()
            stale_message = None
            if self.ready:
                try:
                    for frame in self.frames.values():
                        check_fresh(
                            frame, self.environment.document["execution"]["max_observation_age_ms"]
                        )
                except Problem as exc:
                    stale_message = exc.message
            return SimulationStatus(
                status="unavailable"
                if (self.error or stale_message)
                else ("ready" if self.ready else "loading"),
                environment_id=self.environment.environment_id,
                revision=self.environment.revision,
                epoch=self.epoch,
                physics_steps=self.physics_steps,
                message=self.error or stale_message,
                motion=self.motion.model_copy(deep=True) if self.motion else None,
            )

    def activate(self, owner: str, environment: EnvironmentRecord) -> Activation:
        spec = self.registry.build(environment)
        with self.lock:
            if self.active_command is not None:
                raise Problem(
                    409, "simulator_busy", "Wait for the active command to finish or cancel it."
                )
            if self.owner is not None and self.owner != owner:
                if (utcnow() - self.last_activity).total_seconds() < 600:
                    raise Problem(
                        409, "simulator_occupied", "The simulator is leased to another user."
                    )
            if self.pending:
                raise Problem(409, "scene_loading", "A scene transition is already in progress.")
            self.owner, self.environment, self.spec = owner, environment, spec
            self.epoch = uuid4()
            self.state_revision += 1
            self.ready, self.error = False, None
            self.physics_steps = 0
            self.motion = None
            self.frames.clear()
            self.observations.clear()
            self.last_activity = utcnow()
            self.pending.append(LoadScene(environment, spec, self.epoch))
            return Activation(
                activation_id=uuid4(),
                environment_id=environment.environment_id,
                revision=environment.revision,
            )

    def next_action(self):
        with self.lock:
            return self.pending.popleft() if self.pending else None

    def publish_motion(
        self,
        *,
        epoch: UUID,
        phase: MotionPhase,
        object_position: tuple[float, float, float],
        target_station_id: str | None,
    ) -> None:
        with self.lock:
            if self.environment is None or self.error or epoch != self.epoch:
                return
            previous_command = (
                self.motion.command_id
                if self.motion is not None and phase in {"complete", "stopped"}
                else None
            )
            self.motion = MotionTelemetry(
                command_id=self.active_command[1] if self.active_command else previous_command,
                phase=phase,
                object_position=object_position,
                target_station_id=target_station_id,
            )

    def publish_frame(
        self,
        camera: Literal["overview", "inspection"],
        png: bytes,
        object_position: tuple[float, float, float],
        physics_steps: int,
        *,
        epoch: UUID,
    ) -> None:
        with self.lock:
            if self.environment is None or self.error or epoch != self.epoch:
                return
            observation = Observation(
                observation_id=uuid4(),
                environment_id=self.environment.environment_id,
                revision=self.environment.revision,
                epoch=self.epoch,
                state_revision=self.state_revision,
                captured_at=utcnow(),
                physics_steps=physics_steps,
                object_id="part-001",
                object_position=object_position,
                camera=camera,
                image_base64=base64.b64encode(png).decode("ascii"),
            )
            self.physics_steps = physics_steps
            self.frames[camera] = observation
            self.observations[observation.observation_id] = observation
            while len(self.observations) > 32:
                self.observations.popitem(last=False)
            self.ready = len(self.frames) == 2

    def observe(self, owner: str, environment_id: str, revision: str, camera: str) -> Observation:
        with self.lock:
            self._owned(owner)
            if (
                self.environment.environment_id != environment_id
                or self.environment.revision != revision
            ):
                raise Problem(409, "scene_changed", "The requested scene revision is not active.")
            if not self.ready or camera not in self.frames:
                raise Problem(
                    503,
                    "camera_not_ready",
                    self.error or "Waiting for fresh rendered camera frames.",
                )
            observation = self.frames[camera]
            check_fresh(
                observation, self.environment.document["execution"]["max_observation_age_ms"]
            )
            return observation.model_copy(deep=True)

    def dispatch(self, owner: str, command: MotionCommand) -> Execution:
        fingerprint = content_hash(command.model_dump(mode="json"))
        key = (owner, command.command_id)
        with self.lock:
            if key in self.commands:
                previous = self.fingerprints[key]
                if previous is not None and previous != fingerprint:
                    raise Problem(
                        409, "command_id_reused", "A command ID cannot be reused for changed input."
                    )
                return self.commands[key].model_copy(deep=True)
            self._owned(owner)
            if len(self.commands) >= self.max_commands:
                raise Problem(
                    409,
                    "command_capacity",
                    "Command retention capacity reached; rotate the simulator instance.",
                )
            if self.active_command is not None or self.pending:
                raise Problem(409, "simulator_busy", "Another operation is in progress.")
            if not self.ready or self.spec is None:
                raise Problem(503, "simulator_not_ready", "Live rendering is not ready.")
            if (
                command.environment_id != self.environment.environment_id
                or command.revision != self.environment.revision
                or command.epoch != self.epoch
                or command.state_revision != self.state_revision
            ):
                raise Problem(
                    409, "scene_changed", "The command is based on a different world state."
                )
            observation = self.observations.get(command.observation_id)
            if observation is None or command.object_id != observation.object_id:
                raise Problem(
                    409,
                    "unknown_observation",
                    "The command does not refer to a retained observation.",
                )
            check_fresh(
                observation, self.environment.document["execution"]["max_observation_age_ms"]
            )
            remaining = (command.deadline - utcnow()).total_seconds()
            if (
                remaining <= 0
                or remaining > self.environment.document["execution"]["max_step_seconds"] + 1
            ):
                raise Problem(
                    409,
                    "invalid_deadline",
                    "The command deadline is expired or exceeds the configured limit.",
                )
            if command.target_station_id not in (self.spec.accepted_id, self.spec.rejected_id):
                raise Problem(
                    422,
                    "forbidden_target",
                    "Only the configured accepted/rejected tray may be selected.",
                )
            result = Execution(command_id=command.command_id, status="queued")
            self.commands[key] = result
            self.fingerprints[key] = fingerprint
            self.deadlines[key] = command.deadline
            self.active_command = key
            self.pending.append(StartMotion(command, command.target_station_id))
            return result.model_copy(deep=True)

    def command(self, owner: str, command_id: UUID) -> Execution:
        with self.lock:
            result = self.commands.get((owner, command_id))
            if result is None:
                raise Problem(404, "command_missing", "Command not found for this user.")
            return result.model_copy(deep=True)

    def cancel(self, owner: str, command_id: UUID) -> Execution:
        key = (owner, command_id)
        with self.lock:
            if key not in self.commands:
                if len(self.commands) >= self.max_commands:
                    raise Problem(409, "command_capacity", "Command retention capacity reached.")
                # Cancellation can arrive between API reservation and the dispatch HTTP request.
                self.commands[key] = Execution(
                    command_id=command_id,
                    status="cancelled",
                    completed_at=utcnow(),
                    error=RunError(
                        code="cancelled_before_dispatch",
                        message="Cancellation recorded before any motion was accepted.",
                    ),
                )
                self.fingerprints[key] = None
                return self.commands[key].model_copy(deep=True)
            result = self.command(owner, command_id)
            if result.status in TERMINAL or result.status == "cancelling":
                return result
            self.commands[key] = result.model_copy(update={"status": "cancelling"})
            self.pending.appendleft(StopMotion(command_id))
            return self.commands[key].model_copy(deep=True)

    def begin_motion(self, command_id: UUID) -> bool:
        with self.lock:
            if self.active_command is None or self.active_command[1] != command_id:
                return False
            key = self.active_command
            if self.commands[key].status == "cancelling":
                return False
            if utcnow() >= self.deadlines[key]:
                self.finish("timed_out", None, "Command expired before execution.")
                return False
            self.commands[key] = self.commands[key].model_copy(update={"status": "running"})
            return True

    def deadline_expired(self) -> bool:
        with self.lock:
            return (
                self.active_command is not None and utcnow() >= self.deadlines[self.active_command]
            )

    def should_stop(self, command_id: UUID) -> bool:
        with self.lock:
            return (
                self.active_command is not None
                and self.active_command[1] == command_id
                and self.commands[self.active_command].status == "cancelling"
            )

    def finish(
        self,
        status: Literal["succeeded", "failed", "cancelled", "timed_out"],
        final_position: tuple[float, float, float] | None,
        message: str | None = None,
        *,
        completed_at: datetime | None = None,
        demonstration: dict | None = None,
    ) -> None:
        with self.lock:
            key = self.active_command
            if key is None:
                return
            error = (
                None
                if status == "succeeded"
                else RunError(
                    code=f"simulation_{status}", message=message or f"Simulator command {status}."
                )
            )
            self.commands[key] = Execution(
                command_id=key[1],
                status=status,
                final_position=final_position,
                completed_at=completed_at or utcnow(),
                error=error,
                demonstration=demonstration,
            )
            self.active_command = None
            if self.motion is not None:
                self.motion = self.motion.model_copy(
                    update={
                        "command_id": key[1],
                        "phase": "complete" if status == "succeeded" else "stopped",
                        "object_position": (
                            final_position
                            if final_position is not None
                            else self.motion.object_position
                        ),
                    }
                )
            self.state_revision += 1
            self.frames.clear()
            self.observations.clear()
            self.ready = False
            if status != "succeeded":
                self.error = message or "Reactivate the environment after the stopped command."

    def fail_scene(self, message: str, *, epoch: UUID) -> None:
        with self.lock:
            if epoch != self.epoch:
                return
            self.error = message
            self.ready = False
            self.frames.clear()
            self.observations.clear()
            self.finish("failed", None, message)
