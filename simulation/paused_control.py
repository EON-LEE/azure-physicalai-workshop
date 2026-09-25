"""Explicit non-real-time simulation timing; never a real-time admission fallback."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID, uuid4

from learning.common import finite, integer, vector
from learning.contract import AppliedControl, bounded_joints
from simulation.control import validate_position_target
from simulation.motion import arm_gravity_efforts

PENDING_WALL_NS = 2_000_000_000
INTERVAL_WALL_NS = 5_000_000_000
MAX_EPISODE_WALL_NS = 600_000_000_000
HOLD_STEPS = 6
PHYSICS_DT = 1 / 60


@dataclass(frozen=True)
class FrozenPhysicsState:
    epoch: UUID
    physics_step: int
    world_time: float
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    object_position: tuple[float, float, float]
    object_orientation: tuple[float, float, float, float]
    object_linear_velocity: tuple[float, float, float]
    object_angular_velocity: tuple[float, float, float]

    def validate(self) -> None:
        if any(
            type(value) is not tuple
            for value in (
                self.joint_positions,
                self.joint_velocities,
                self.object_position,
                self.object_orientation,
                self.object_linear_velocity,
                self.object_angular_velocity,
            )
        ):
            raise ValueError("Frozen physics vectors must be immutable tuples.")
        if not isinstance(self.epoch, UUID):
            raise ValueError("A simulator epoch is required for frozen-state evidence.")
        integer(self.physics_step, "actual physics step")
        if finite(self.world_time, "actual simulation time") < 0:
            raise ValueError("Simulation time cannot be negative.")
        bounded_joints(self.joint_positions, "frozen joint positions")
        vector(self.joint_velocities, 9, "frozen joint velocities")
        vector(self.object_position, 3, "frozen object position")
        vector(self.object_orientation, 4, "frozen object orientation")
        vector(self.object_linear_velocity, 3, "frozen object velocity")
        vector(self.object_angular_velocity, 3, "frozen object angular velocity")


class PausedEpisode:
    """Owns only clock/freeze/tick accounting; the main thread alone touches Isaac."""

    def __init__(
        self,
        initial: FrozenPhysicsState,
        *,
        wall_deadline_ns: int,
        max_simulation_steps: int,
        authorized: Callable[[], bool],
        clock_ns: Callable[[], int],
        started_ns: int | None = None,
    ) -> None:
        initial.validate()
        now_ns = integer(clock_ns(), "current wall clock", 1)
        started_ns = now_ns if started_ns is None else integer(started_ns, "episode wall start", 1)
        integer(wall_deadline_ns, "episode wall deadline", 1)
        integer(max_simulation_steps, "episode simulation step budget", HOLD_STEPS, 1800)
        if (
            not started_ns <= now_ns < wall_deadline_ns
            or not 0 < wall_deadline_ns - started_ns <= MAX_EPISODE_WALL_NS
            or max_simulation_steps % HOLD_STEPS
        ):
            raise ValueError(
                "Paused episodes require bounded independent wall and simulation budgets."
            )
        self.initial = initial
        self.wall_deadline_ns = wall_deadline_ns
        self.max_simulation_steps = max_simulation_steps
        self.authorized, self.clock_ns = authorized, clock_ns
        self.started_ns = started_ns
        self.last_seen_ns = now_ns
        self.phase = "idle"
        self.current = initial
        self.frozen: FrozenPhysicsState | None = None
        self.freeze_id: UUID | None = None
        self.interval_deadline_ns = wall_deadline_ns
        self.operation_deadline_ns = wall_deadline_ns
        self.interval_started_ns = self.observation_ready_ns = self.policy_ready_ns = 0
        self.tick_started_ns: int | None = None
        self.targets: tuple[float, ...] | None = None
        self.previous_targets: tuple[float, ...] | None = None
        self.controls: list[AppliedControl] = []
        self.intervals: list[dict] = []
        self.failure: str | None = None
        self.failure_phase: str | None = None
        self.stop_reason: str | None = None

    def fail(self, message: str) -> None:
        if self.failure is None:
            self.failure = message
            self.failure_phase = self.phase
        self.phase = "stopped"

    def _fail(self, message: str) -> None:
        self.fail(message)
        raise RuntimeError(message)

    def _guard(self) -> int:
        now = integer(self.clock_ns(), "actual monotonic wall clock", 1)
        if self.phase == "stopped":
            raise RuntimeError(self.failure or "The paused episode is stopped.")
        if now < self.last_seen_ns:
            self._fail("The actual wall clock moved backwards.")
        self.last_seen_ns = now
        if self.authorized() is not True:
            self._fail(
                "Current owner, epoch, command, model or lease authority is no longer active."
            )
        if now >= self.wall_deadline_ns:
            self._fail("The original wall deadline expired independently of simulation time.")
        if self.phase != "idle" and (
            now >= self.interval_deadline_ns or now >= self.operation_deadline_ns
        ):
            self._fail("The original paused-operation or whole-interval deadline expired.")
        return now

    def _frozen(self, state: FrozenPhysicsState) -> None:
        state.validate()
        if state != self.frozen:
            self._fail("The actual physics state changed while it was required to remain frozen.")

    def _expect(self, phase: str) -> None:
        if self.phase != phase:
            self._fail(f"Expected paused phase {phase}; received {self.phase}.")

    def _binding(self, freeze_id: UUID) -> None:
        if freeze_id != self.freeze_id:
            self._fail("The asynchronous result belongs to a different freeze.")

    def begin_observation(self, state: FrozenPhysicsState) -> UUID:
        now = self._guard()
        self._expect("idle")
        state.validate()
        if state != self.current:
            self._fail("Physics moved between frozen intervals without an authorized control tick.")
        if state.physics_step - self.initial.physics_step + HOLD_STEPS > self.max_simulation_steps:
            self._fail("The approved simulation-step budget is exhausted.")
        self.frozen = state
        self.freeze_id = uuid4()
        self.interval_started_ns = now
        self.interval_deadline_ns = min(now + INTERVAL_WALL_NS, self.wall_deadline_ns)
        self.operation_deadline_ns = min(now + PENDING_WALL_NS, self.interval_deadline_ns)
        self.controls = []
        self.tick_started_ns = None
        self.phase = "observing"
        return self.freeze_id

    def waiting(self, state: FrozenPhysicsState) -> None:
        self._guard()
        if self.phase not in {"observing", "predicting"}:
            self._fail("Only pending observation or policy operations may wait while frozen.")
        self._frozen(state)

    def observation_ready(self, freeze_id: UUID, state: FrozenPhysicsState) -> None:
        now = self._guard()
        self._expect("observing")
        self._binding(freeze_id)
        self._frozen(state)
        self.observation_ready_ns = now
        self.operation_deadline_ns = min(now + PENDING_WALL_NS, self.interval_deadline_ns)
        self.phase = "predicting"

    def policy_ready(
        self, freeze_id: UUID, state: FrozenPhysicsState, targets: tuple[float, ...]
    ) -> None:
        now = self._guard()
        self._expect("predicting")
        self._binding(freeze_id)
        self._frozen(state)
        try:
            self.targets = validate_position_target(
                state.joint_positions, targets, self.previous_targets
            )
        except ValueError as exc:
            self._fail(str(exc))
        self.policy_ready_ns = now
        self.operation_deadline_ns = min(now + PENDING_WALL_NS, self.interval_deadline_ns)
        self.phase = "applying"

    def before_tick(self, state: FrozenPhysicsState) -> None:
        now = self._guard()
        self._expect("applying")
        state.validate()
        if self.tick_started_ns is not None:
            self._fail("The preceding actual physics tick has not been accounted for.")
        if state != self.current:
            self._fail("Actual physics changed before the next authorized tick.")
        if len(self.controls) >= HOLD_STEPS:
            self._fail("An interval cannot apply more than six physical ticks.")
        self.tick_started_ns = now

    def after_tick(self, state: FrozenPhysicsState, applied: AppliedControl) -> bool:
        self._expect("applying")
        state.validate()
        if self.tick_started_ns is None:
            self._fail("No guarded physical tick was authorized.")
        now = integer(self.clock_ns(), "actual tick completion time", 1)
        if (
            state.epoch != self.initial.epoch
            or state.physics_step != self.current.physics_step + 1
            or abs(state.world_time - self.current.world_time - PHYSICS_DT) > 1e-6
        ):
            self._fail("A paused action must advance exactly one actual 60 Hz physics tick.")
        if (
            applied.physics_step != state.physics_step
            or not self.tick_started_ns <= applied.monotonic_ns <= now
            or applied.commanded_joint_targets != self.targets
            or vector(applied.commanded_joint_velocities, 9, "applied velocity targets")
            != (0.0,) * 9
            or arm_gravity_efforts(applied.gravity_efforts) != applied.gravity_efforts
        ):
            self._fail("Actual actuation differs from the approved held position/gravity profile.")
        self.current = state
        self.controls.append(applied)
        blocked_ns = now - self.tick_started_ns
        self.tick_started_ns = None
        if blocked_ns > PENDING_WALL_NS:
            self._fail("A synchronous physics call exceeded the hard heartbeat wall budget.")
        self._guard()
        if len(self.controls) != HOLD_STEPS:
            return False
        self.intervals.append(
            {
                "freeze_id": str(self.freeze_id),
                "observation_physics_step": self.frozen.physics_step,
                "completed_physics_step": state.physics_step,
                "observation_wall_ms": (self.observation_ready_ns - self.interval_started_ns) / 1e6,
                "policy_wall_ms": (self.policy_ready_ns - self.observation_ready_ns) / 1e6,
                "hold_wall_ms": (now - self.policy_ready_ns) / 1e6,
                "interval_wall_ms": (now - self.interval_started_ns) / 1e6,
                "simulated_seconds": state.world_time - self.frozen.world_time,
                "applied_controls": tuple(self.controls),
            }
        )
        self.previous_targets = self.targets
        self.phase = "idle"
        return True

    def cancel(self, reason: str = "The paused episode was cancelled.") -> None:
        self.phase = "stopped"
        if self.stop_reason is None:
            self.stop_reason = reason

    def metrics(self) -> dict:
        return {
            "execution_timing": "paused_simulation",
            "real_time_admission": False,
            "public_label": "NON_REALTIME_SIMULATION",
            "phase": self.phase,
            "wall_elapsed_ms": (self.clock_ns() - self.started_ns) / 1e6,
            "simulation_steps": self.current.physics_step - self.initial.physics_step,
            "simulation_elapsed_seconds": self.current.world_time - self.initial.world_time,
            "intervals": deepcopy(self.intervals),
            "failure": self.failure,
            "failure_phase": self.failure_phase,
            "stop_reason": self.stop_reason,
        }
