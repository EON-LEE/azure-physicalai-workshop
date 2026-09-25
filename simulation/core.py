from __future__ import annotations

import base64
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from apps.api.errors import Problem
from apps.api.models import (
    TERMINAL,
    Activation,
    DemonstrationResult,
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
from learning.contract import ControlProfile, Scope
from learning.inference import ControlContext
from learning.paused import PausedControlProfile
from simulation.extensions import SceneRegistry, SceneSpec
from simulation.paused_contracts import (
    PausedEpisodeAuthorizer,
    PausedRuntimeMetrics,
    ResolvedSimulationAuthorization,
    SimulationEpisodeCommand,
    SimulationEpisodeExecution,
)
from simulation.policy_executor import PolicyProvider
from simulation.runtime_contracts import (
    CaptureBinding,
    CapturePhase,
    CaptureReceipt,
    CaptureStatus,
    CommandBinding,
    PolicyCommand,
    PolicyExecution,
    PolicyRuntime,
    TeachingInput,
    TeachingIntent,
    TeachingLease,
    TeachingStart,
    TeachingState,
)
from simulation.teaching import TeachingSession


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


@dataclass(frozen=True)
class StartTeaching:
    request: TeachingStart


@dataclass(frozen=True)
class FinishTeaching:
    binding: CommandBinding


@dataclass(frozen=True)
class StartPolicy:
    request: PolicyCommand


@dataclass(frozen=True)
class StartSimulationEpisode:
    request: SimulationEpisodeCommand


class SimulationCore:
    """Thread-safe protocol state; all physics execution stays in the Isaac main thread."""

    def __init__(
        self,
        registry: SceneRegistry,
        max_commands: int = 2048,
        *,
        control_profile: ControlProfile | None = None,
        paused_profile: PausedControlProfile | None = None,
        paused_authorizer: PausedEpisodeAuthorizer | None = None,
        policy_provider: PolicyProvider | None = None,
        tenant_id: str | None = None,
        capture_status_reader: Callable[[str, UUID], CaptureStatus | None] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        clock_utc: Callable[[], datetime] | None = None,
    ) -> None:
        self.registry = registry
        self.control_profile = control_profile
        self.paused_profile, self.paused_authorizer = paused_profile, paused_authorizer
        self.policy_provider, self.tenant_id = policy_provider, tenant_id
        self.capture_status_reader = capture_status_reader
        self.clock_ns = clock_ns
        self.clock_utc = clock_utc if clock_utc is not None else lambda: utcnow()
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
        self.monotonic_deadlines: dict[tuple[str, UUID], int] = {}
        self.pending: deque[
            LoadScene
            | StartMotion
            | StopMotion
            | StartTeaching
            | FinishTeaching
            | StartPolicy
            | StartSimulationEpisode
        ] = deque()
        self.active_command: tuple[str, UUID] | None = None
        self.motion: MotionTelemetry | None = None
        self.max_commands = max_commands
        self.capture_bindings: dict[tuple[str, UUID], CaptureBinding] = {}
        self.captures: dict[tuple[str, UUID], CaptureStatus] = {}
        self.teaching_sessions: dict[tuple[str, UUID], TeachingSession] = {}
        self.teaching_by_command: dict[tuple[str, UUID], tuple[str, UUID]] = {}
        self.policy_commands: dict[tuple[str, UUID], PolicyCommand] = {}
        self.simulation_commands: dict[tuple[str, UUID], SimulationEpisodeCommand] = {}
        self.simulation_authorizations: dict[tuple[str, UUID], ResolvedSimulationAuthorization] = {}
        self.command_started_ns: dict[tuple[str, UUID], int] = {}

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
        return self._dispatch(
            owner,
            command,
            fingerprint=fingerprint,
            action=StartMotion(command, command.target_station_id),
            deadline=command.deadline,
        )

    def _dispatch(
        self,
        owner: str,
        command: MotionCommand | TeachingStart | SimulationEpisodeCommand,
        *,
        fingerprint: str,
        action: StartMotion | StartTeaching | StartPolicy | StartSimulationEpisode,
        deadline: datetime,
        teaching: bool = False,
    ) -> Execution:
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
            remaining = (deadline - self.clock_utc()).total_seconds()
            if isinstance(command, SimulationEpisodeCommand):
                authority = self.spec.require_paused_authority()
                authority.validate_request(
                    wall_seconds=remaining, simulation_steps=command.max_simulation_steps
                )
                maximum = authority.max_wall_seconds
            else:
                maximum = (
                    300
                    if teaching
                    else min(30, self.environment.document["execution"]["max_step_seconds"])
                )
            if remaining <= 0 or remaining > maximum:
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
            self.deadlines[key] = deadline
            started_ns = self.clock_ns()
            self.command_started_ns[key] = started_ns
            self.monotonic_deadlines[key] = started_ns + int(remaining * 1_000_000_000)
            self.active_command = key
            self.pending.append(action)
            return result.model_copy(deep=True)

    def dispatch_simulation_episode(
        self, owner: str, request: SimulationEpisodeCommand
    ) -> Execution:
        request = SimulationEpisodeCommand.model_validate(
            request.model_dump(mode="json", by_alias=True)
        )
        fingerprint = content_hash(request.model_dump(mode="json", by_alias=True))
        key = (owner, request.command_id)
        with self.lock:
            if key in self.commands:
                previous = self.fingerprints[key]
                if previous is not None and previous != fingerprint:
                    raise Problem(
                        409, "command_id_reused", "A simulation command ID cannot be reused."
                    )
                return self.commands[key].model_copy(deep=True)
            self._owned(owner)
            if self.paused_profile is None or self.paused_profile.profile_id != request.profile_id:
                raise Problem(
                    503,
                    "paused_profile_unavailable",
                    "A separately installed non-real-time simulation profile is required.",
                )
            self.paused_profile.validate()
            if self.spec is None:
                raise Problem(503, "simulator_not_ready", "The reviewed scene is not ready.")
            self.spec.require_paused_authority()
            if not self.spec.record_demonstration:
                raise Problem(
                    409, "capture_not_approved", "The paused episode requires approved capture."
                )
            if self.paused_authorizer is None:
                raise Problem(
                    503,
                    "paused_authority_unavailable",
                    "No trusted paused episode authority resolver is installed.",
                )
            resolved = self.paused_authorizer.authorize(owner, request, self.paused_profile)
            if not isinstance(resolved, ResolvedSimulationAuthorization):
                raise Problem(
                    503, "invalid_paused_authority", "No validated authority record returned."
                )
            resolved = ResolvedSimulationAuthorization.model_validate(resolved.model_dump())
            if (
                resolved.authorization_id != request.authorization_id
                or resolved.authorization_kind != request.authorization_kind
                or resolved.owner != owner
                or resolved.environment_id != request.environment_id
                or resolved.revision != request.revision
                or resolved.controller != request.controller
                or resolved.task != request.task
                or resolved.control_profile_sha256 != self.paused_profile.sha256
                or resolved.policy_type != request.policy_type
                or resolved.model_sha256 != request.model_sha256
                or request.wall_expires_at > resolved.wall_expires_at
                or (request.wall_expires_at - self.clock_utc()).total_seconds()
                > resolved.max_episode_wall_seconds
                or request.max_simulation_steps > resolved.max_simulation_steps
            ):
                raise Problem(
                    403, "paused_authority_mismatch", "Resolved authority does not match request."
                )
            result = self._dispatch(
                owner,
                request,
                fingerprint=fingerprint,
                action=StartSimulationEpisode(request),
                deadline=request.wall_expires_at,
            )
            self.simulation_commands[key] = request
            self.simulation_authorizations[key] = resolved.model_copy(deep=True)
            self.commands[key] = SimulationEpisodeExecution(
                **result.model_dump(exclude={"policy_runtime", "simulation_runtime"}),
                simulation_runtime=PausedRuntimeMetrics(
                    control_profile_sha256=self.paused_profile.sha256,
                    controller=request.controller,
                ),
            )
            return self.commands[key].model_copy(deep=True)

    def simulation_episode(self, owner: str, command_id: UUID) -> Execution:
        with self.lock:
            if (owner, command_id) not in self.simulation_commands:
                raise Problem(404, "simulation_episode_missing", "Simulation episode not found.")
            return self.command(owner, command_id)

    def publish_simulation_metrics(
        self, binding: CommandBinding, metrics: PausedRuntimeMetrics
    ) -> bool:
        with self.lock:
            if not self.matches(binding):
                return False
            key = (binding.owner, binding.command_id)
            request = self.simulation_commands.get(key)
            if request is None:
                return False
            checked = PausedRuntimeMetrics.model_validate(metrics.model_dump())
            previous = self.commands[key].simulation_runtime
            if (
                checked.controller != request.controller
                or checked.control_profile_sha256 != previous.control_profile_sha256
                or checked.applied_model_sha256 not in (None, request.model_sha256)
                or checked.simulation_steps < previous.simulation_steps
                or checked.applied_action_count < previous.applied_action_count
                or checked.policy_predict_calls < previous.policy_predict_calls
                or checked.wall_elapsed_ms < previous.wall_elapsed_ms
            ):
                raise ValueError("Paused runtime evidence changed binding or moved backwards.")
            self.commands[key] = self.commands[key].model_copy(
                update={"simulation_runtime": checked}
            )
            return True

    def dispatch_policy(self, owner: str, request: PolicyCommand) -> Execution:
        fingerprint = content_hash(request.model_dump(mode="json"))
        key = (owner, request.command.command_id)
        with self.lock:
            if key in self.commands:
                previous = self.fingerprints[key]
                if previous is not None and previous != fingerprint:
                    raise Problem(409, "command_id_reused", "A policy command ID cannot be reused.")
                return self.commands[key].model_copy(deep=True)
            if self.control_profile is None:
                raise Problem(
                    503, "control_profile_unverified", "No verified policy servo profile."
                )
            if self.policy_provider is None or self.tenant_id is None:
                raise Problem(503, "policy_unavailable", "No deployment-approved policy provider.")
            self.control_profile.validate()
            self._owned(owner)
            self.policy_provider.authorize(owner, request, self.control_profile)
            result = self._dispatch(
                owner,
                request.command,
                fingerprint=fingerprint,
                action=StartPolicy(request),
                deadline=request.command.deadline,
            )
            self.policy_commands[key] = request.model_copy(deep=True)
            self.commands[key] = PolicyExecution(
                **result.model_dump(exclude={"policy_runtime"}),
                policy_runtime=PolicyRuntime(
                    policy_release_id=request.policy_release_id, policy_type=request.policy_type
                ),
            )
            return self.commands[key].model_copy(deep=True)

    def policy_context(self, binding: CommandBinding) -> ControlContext:
        with self.lock:
            key = (binding.owner, binding.command_id)
            request = self.policy_commands.get(key)
            if request is None or self.tenant_id is None:
                raise RuntimeError("There is no approved policy context for this command.")
            scope = Scope(self.tenant_id, binding.owner)
            scope.validate()
            return ControlContext(
                scope,
                binding.environment_id,
                binding.revision,
                str(binding.command_id),
                str(binding.epoch),
                str(binding.command_id),
                request.command.target_station_id,
                True,
                self.actuation_allowed(binding),
                self.monotonic_deadlines[key],
            )

    def publish_policy_metrics(self, binding: CommandBinding, metrics: PolicyRuntime) -> bool:
        with self.lock:
            if not self.matches(binding):
                return False
            key = (binding.owner, binding.command_id)
            request = self.policy_commands.get(key)
            if request is None:
                return False
            metrics = PolicyRuntime.model_validate(metrics.model_dump())
            if (
                metrics.policy_release_id != request.policy_release_id
                or metrics.policy_type != request.policy_type
                or metrics.applied_model_sha not in (None, request.model_sha256)
            ):
                raise ValueError("Applied policy metrics do not belong to the approved release.")
            previous = self.commands[key].policy_runtime
            if (
                metrics.policy_predict_calls < previous.policy_predict_calls
                or metrics.applied_action_count < previous.applied_action_count
            ):
                raise ValueError("Policy application counters cannot move backwards.")
            self.commands[key] = self.commands[key].model_copy(
                update={"policy_runtime": metrics.model_copy(deep=True)}
            )
            return True

    def apply_guarded(self, binding: CommandBinding, submit: Callable[[], None]) -> None:
        with self.lock:
            if not self.actuation_allowed(binding):
                raise RuntimeError("The bound motion command is no longer active.")
            submit()

    def start_teaching(self, owner: str, request: TeachingStart) -> TeachingState:
        with self.lock:
            key = (owner, request.session_id)
            if key in self.teaching_sessions:
                if self.teaching_sessions[key].request != request:
                    raise Problem(409, "session_reused", "A teaching session ID cannot be reused.")
                return self.teaching(owner, request.session_id)
            if (
                self.control_profile is None
                or self.control_profile.profile_id != request.control_profile_id
            ):
                raise Problem(
                    503,
                    "control_profile_unverified",
                    "The teaching control profile is not enabled.",
                )
            self.control_profile.validate()
            self._owned(owner)
            if not self.spec.record_demonstration:
                raise Problem(
                    409,
                    "capture_not_approved",
                    "Teaching requires an approved recorded scene and dataset split.",
                )
            if request.split != self.spec.demonstration_split:
                raise Problem(
                    409,
                    "teaching_split_changed",
                    "The teaching split must match the immutable approved environment.",
                )
            self._dispatch(
                owner,
                request,
                fingerprint=content_hash(request.model_dump(mode="json")),
                action=StartTeaching(request),
                deadline=request.session_expires_at,
                teaching=True,
            )
            self.teaching_sessions[key] = TeachingSession(request.model_copy(deep=True))
            self.teaching_by_command[(owner, request.command_id)] = key
            return self.teaching(owner, request.session_id)

    def _teaching(
        self, owner: str, session_id: UUID, lease: TeachingLease | None = None
    ) -> TeachingSession:
        session = self.teaching_sessions.get((owner, session_id))
        if session is None:
            raise Problem(404, "teaching_missing", "Teaching session not found for this user.")
        if lease is not None and (
            lease.lease_id != session.request.lease_id or lease.epoch != session.request.epoch
        ):
            raise Problem(409, "teaching_lease_changed", "Teaching lease or epoch changed.")
        return session

    def teaching(self, owner: str, session_id: UUID) -> TeachingState:
        with self.lock:
            session = self._teaching(owner, session_id)
            request = session.request
            result = self.commands[(owner, request.command_id)]
            return TeachingState(
                session_id=request.session_id,
                lease_id=request.lease_id,
                command_id=request.command_id,
                epoch=request.epoch,
                control_profile_id=request.control_profile_id,
                demonstrator_kind=request.demonstrator_kind,
                session_expires_at=request.session_expires_at,
                status=(
                    "finishing"
                    if session.finishing and result.status == "running"
                    else result.status
                ),
                last_sequence=session.last_input.sequence if session.last_input else 0,
                input_expires_at=session.last_input.expires_at if session.last_input else None,
                execution=result.model_copy(deep=True),
                capture=self.captures.get((owner, request.command_id)),
            )

    def teaching_input(self, owner: str, session_id: UUID, request: TeachingInput) -> TeachingState:
        with self.lock:
            session = self._teaching(owner, session_id, request)
            key = (owner, session.request.command_id)
            if (
                key != self.active_command
                or session.request.epoch != self.epoch
                or self.commands[key].status != "running"
            ):
                raise Problem(409, "teaching_not_running", "Teaching motion is not active.")
            if self.deadline_expired():
                raise Problem(409, "teaching_expired", "The teaching session has expired.")
            if session.finishing:
                raise Problem(409, "teaching_finishing", "The teaching session is finishing.")
            session.admit(request, self.clock_utc(), self.clock_ns())
            return self.teaching(owner, session_id)

    def teaching_intent(self, binding: CommandBinding) -> TeachingIntent | None:
        with self.lock:
            if not self.actuation_allowed(binding):
                return None
            key = self.teaching_by_command.get((binding.owner, binding.command_id))
            if key is None:
                return None
            return self.teaching_sessions[key].intent(self.clock_ns())

    def teaching_hold_allowed(self, binding: CommandBinding, sequence: int) -> bool:
        with self.lock:
            if not self.actuation_allowed(binding):
                return False
            key = self.teaching_by_command.get((binding.owner, binding.command_id))
            if key is None:
                return False
            session = self.teaching_sessions[key]
            return (
                session.last_input is not None
                and session.last_input.deadman
                and session.last_input.sequence >= sequence
                and self.clock_ns() < session.input_deadline_ns
            )

    def finish_teaching(self, owner: str, session_id: UUID, lease: TeachingLease) -> TeachingState:
        with self.lock:
            session = self._teaching(owner, session_id, lease)
            command_id = session.request.command_id
            if self.commands[(owner, command_id)].status in TERMINAL or session.finishing:
                return self.teaching(owner, session_id)
            binding = self.binding(command_id)
            if binding.epoch != lease.epoch or binding.owner != owner:
                raise Problem(409, "teaching_lease_changed", "Teaching lease or epoch changed.")
            session.finishing = True
            self.pending.append(FinishTeaching(binding))
            return self.teaching(owner, session_id)

    def cancel_teaching(self, owner: str, session_id: UUID, lease: TeachingLease) -> TeachingState:
        with self.lock:
            session = self._teaching(owner, session_id, lease)
            self.cancel(owner, session.request.command_id)
            return self.teaching(owner, session_id)

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
            if self.commands[key].status != "queued":
                return False
            if self.deadline_expired():
                self.finish("timed_out", None, "Command expired before execution.")
                return False
            self.commands[key] = self.commands[key].model_copy(update={"status": "running"})
            return True

    def deadline_expired(self) -> bool:
        with self.lock:
            return self.active_command is not None and (
                self.clock_utc() >= self.deadlines[self.active_command]
                or self.clock_ns() >= self.monotonic_deadlines[self.active_command]
            )

    def should_stop(self, command_id: UUID) -> bool:
        with self.lock:
            return (
                self.active_command is not None
                and self.active_command[1] == command_id
                and self.commands[self.active_command].status == "cancelling"
            )

    def begin_capture(self, command_id: UUID) -> CaptureBinding:
        with self.lock:
            key = self.active_command
            if (
                key is None
                or key[1] != command_id
                or self.commands[key].status != "running"
                or self.environment is None
                or self.spec is None
                or not self.spec.record_demonstration
            ):
                raise Problem(409, "capture_not_approved", "Capture requires approved motion.")
            if key in self.capture_bindings:
                raise Problem(409, "capture_exists", "The command already has a capture.")
            binding = CaptureBinding(
                key[0],
                self.environment.environment_id,
                self.environment.revision,
                self.epoch,
                command_id,
                uuid4(),
            )
            self.capture_bindings[key] = binding
            self.captures[key] = CaptureStatus(
                capture_id=binding.capture_id,
                command_id=command_id,
                epoch=binding.epoch,
                status="recording",
            )
            return binding

    def binding(self, command_id: UUID) -> CommandBinding:
        with self.lock:
            if self.active_command is None or self.active_command[1] != command_id:
                raise Problem(409, "command_not_active", "The command is no longer active.")
            return CommandBinding(
                self.active_command[0],
                self.environment.environment_id,
                self.environment.revision,
                self.epoch,
                command_id,
            )

    def matches(self, binding: CommandBinding) -> bool:
        with self.lock:
            return (
                self.active_command == (binding.owner, binding.command_id)
                and self.owner == binding.owner
                and self.epoch == binding.epoch
                and self.environment is not None
                and self.environment.environment_id == binding.environment_id
                and self.environment.revision == binding.revision
            )

    def actuation_allowed(self, binding: CommandBinding) -> bool:
        with self.lock:
            return (
                self.matches(binding)
                and self.commands[self.active_command].status == "running"
                and self.error is None
                and not self.deadline_expired()
            )

    def capture(self, owner: str, command_id: UUID) -> CaptureStatus:
        with self.lock:
            state = self.captures.get((owner, command_id))
        if state is None and self.capture_status_reader is not None:
            state = self.capture_status_reader(owner, command_id)
        with self.lock:
            if state is None:
                raise Problem(404, "capture_missing", "Capture not found for this user.")
            return state.model_copy(deep=True)

    def publish_capture(
        self,
        binding: CaptureBinding,
        status: CapturePhase,
        *,
        receipt: CaptureReceipt | DemonstrationResult | None = None,
        message: str | None = None,
    ) -> bool:
        with self.lock:
            key = (binding.owner, binding.command_id)
            if self.capture_bindings.get(key) != binding:
                return False
            previous = self.captures[key]
            if previous.status in {"ready", "invalid"}:
                return False
            order = ("recording", "finalizing", "uploading", "ready", "invalid")
            if order.index(status) < order.index(previous.status):
                return False
            result = self.commands[key]
            demonstration = None
            if status == "ready":
                if result.status not in TERMINAL or receipt is None:
                    raise ValueError("Capture publication requires an independent physical result.")
                demonstration = (
                    DemonstrationResult(status="uploaded", **asdict(receipt))
                    if isinstance(receipt, CaptureReceipt)
                    else DemonstrationResult.model_validate(receipt.model_dump())
                )
                if (
                    demonstration.status != "uploaded"
                    or demonstration.episode_id != binding.command_id
                ):
                    raise ValueError("Capture receipt is not bound to the completed command.")
            elif status == "invalid":
                if not message:
                    raise ValueError("Invalid capture requires an explicit reason.")
                demonstration = DemonstrationResult(status="failed", message=message)
            self.captures[key] = CaptureStatus(
                capture_id=binding.capture_id,
                command_id=binding.command_id,
                epoch=binding.epoch,
                status=status,
                receipt=demonstration if status == "ready" else None,
                message=message,
            )
            if result.status in TERMINAL and demonstration is not None:
                self.commands[key] = result.model_copy(update={"demonstration": demonstration})
            return True

    def finish(
        self,
        status: Literal["succeeded", "failed", "cancelled", "timed_out"],
        final_position: tuple[float, float, float] | None,
        message: str | None = None,
        *,
        completed_at: datetime | None = None,
        demonstration: dict | None = None,
        binding: CommandBinding | None = None,
    ) -> None:
        with self.lock:
            key = self.active_command
            if key is None or (binding is not None and not self.matches(binding)):
                return
            completed_at = completed_at or self.clock_utc()
            if status == "succeeded":
                if self.commands[key].status == "cancelling":
                    status, message = "cancelled", "Cancellation won the completion race."
                elif (
                    completed_at >= self.deadlines[key]
                    or self.clock_ns() >= self.monotonic_deadlines[key]
                ):
                    status, message = "timed_out", "Completion exceeded the command deadline."
            capture = self.captures.get(key)
            if demonstration is None and capture is not None and capture.status == "invalid":
                demonstration = {"status": "failed", "message": capture.message}
            error = (
                None
                if status == "succeeded"
                else RunError(
                    code=f"simulation_{status}", message=message or f"Simulator command {status}."
                )
            )
            terminal = Execution(
                command_id=key[1],
                status=status,
                final_position=final_position,
                completed_at=completed_at,
                error=error,
                demonstration=demonstration,
            )
            if key in self.policy_commands:
                terminal = PolicyExecution(
                    **terminal.model_dump(exclude={"policy_runtime"}),
                    policy_runtime=self.commands[key].policy_runtime,
                )
            elif key in self.simulation_commands:
                terminal = SimulationEpisodeExecution(
                    **terminal.model_dump(exclude={"policy_runtime", "simulation_runtime"}),
                    simulation_runtime=self.commands[key].simulation_runtime.model_copy(
                        update={"phase": "stopped"}
                    ),
                )
            self.commands[key] = terminal
            teaching_key = self.teaching_by_command.get(key)
            if teaching_key is not None:
                self.teaching_sessions[teaching_key].used_grants.clear()
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
