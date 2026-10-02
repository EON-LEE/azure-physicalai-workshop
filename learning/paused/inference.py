from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Protocol

from learning.common import integer, require, sha256
from learning.contract import DemonstrationSource, Scope, bounded_joints, validate_joint_tracking
from learning.paused.contract import (
    CONTEXT_SCHEMA,
    EXECUTION_TIMING,
    FrozenPolicyObservation,
    InitialFrozenPublication,
    PausedControlContext,
    PausedControlProfile,
    PausedJointCommand,
)


class PausedChunkPolicy(Protocol):
    policy_type: str
    execution_timing: str
    real_time_admission: bool
    scope: Scope
    model_sha256: str
    profile: PausedControlProfile
    task: DemonstrationSource
    chunk_size: int
    n_action_steps: int

    def reset(self) -> None: ...

    def predict_chunk(
        self, observation: FrozenPolicyObservation, context: PausedControlContext
    ) -> Sequence[Sequence[float]]: ...


def _episode_binding(context: PausedControlContext) -> tuple:
    return (
        context.scope,
        context.environment_id,
        context.revision,
        context.episode_id,
        context.epoch,
        context.command_id,
        context.destination_id,
        context.model_sha256,
        context.control_profile_sha256,
        context.episode_started_ns,
        context.wall_deadline_ns,
        context.episode_initial_physics_step,
        context.simulation_step_deadline,
    )


class PausedGuardedPolicyAdapter:
    """One in-flight prediction on the caller's worker; no SDK, actuator or implicit approval."""

    def __init__(
        self,
        policy: PausedChunkPolicy,
        *,
        profile: PausedControlProfile,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        publication_guard: Callable[
            [InitialFrozenPublication, FrozenPolicyObservation, PausedControlContext], bool
        ]
        | None = None,
    ) -> None:
        profile.validate()
        require(
            policy.policy_type == "smolvla"
            and policy.execution_timing == EXECUTION_TIMING
            and policy.real_time_admission is False
            and isinstance(policy.profile, PausedControlProfile)
            and policy.profile.sha256 == profile.sha256,
            "Expected an explicitly bound paused Smol policy, never a real-time fallback",
        )
        require(
            type(policy.chunk_size) is int
            and policy.chunk_size == 50
            and type(policy.n_action_steps) is int
            and policy.n_action_steps == 1,
            "Paused simulation preserves the model horizon but consumes only one fresh action",
        )
        policy.scope.validate()
        policy.task.validate()
        sha256(policy.model_sha256)
        self.policy, self.profile, self.clock_ns = policy, profile, clock_ns
        self.publication_guard = publication_guard
        self._lock = threading.Lock()
        self._generation, self._predict_calls = 0, 0
        self._inflight, self._faulted = False, True
        self._binding = None
        self._previous: FrozenPolicyObservation | None = None
        self._previous_target: tuple[float, ...] | None = None
        self._freezes: set[str] = set()

    @property
    def predict_calls(self) -> int:
        with self._lock:
            return self._predict_calls

    def _policy_binding(self, context: PausedControlContext) -> None:
        require(
            context.schema == CONTEXT_SCHEMA
            and context.execution_timing == EXECUTION_TIMING
            and context.real_time_admission is False
            and context.approved is True
            and context.active is True
            and context.scope == self.policy.scope
            and context.model_sha256 == self.policy.model_sha256
            and context.control_profile_sha256 == self.profile.sha256,
            "Paused owner/model/profile/controller authority does not match",
        )
        require(
            context.destination_id == self.policy.task.goal_id,
            "Context goal differs from the approved model task",
        )
        now = integer(self.clock_ns(), "current wall clock", 1)
        require(
            now < context.operation_deadline_ns <= context.wall_deadline_ns,
            "Original paused operation deadline expired",
        )

    def reset(self, context: PausedControlContext) -> None:
        with self._lock:
            require(not self._inflight, "An in-flight paused prediction must finish before reset")
            self._faulted, self._inflight = True, True
            self._generation += 1
            generation = self._generation
        try:
            self._policy_binding(context)
            self.policy.reset()
            self._policy_binding(context)
            with self._lock:
                require(generation == self._generation, "Paused reset was cancelled")
                self._binding = _episode_binding(context)
                self._previous, self._previous_target = None, None
                self._freezes.clear()
                self._faulted = False
        finally:
            with self._lock:
                self._inflight = False

    def stop(self) -> None:
        # Never wait for a model call or reset a policy from the simulator's cancellation path.
        with self._lock:
            self._generation += 1
            self._faulted = True
            self._binding = None

    def _publication(
        self, observation: FrozenPolicyObservation, context: PausedControlContext
    ) -> None:
        if observation.initial_publication is not None:
            require(
                self.publication_guard is not None
                and self.publication_guard(observation.initial_publication, observation, context)
                is True,
                "Initial publication requires a known current trusted runtime record",
            )

    def step(
        self, observation: FrozenPolicyObservation, context: PausedControlContext
    ) -> PausedJointCommand:
        with self._lock:
            require(not self._inflight, "Only one in-flight paused prediction is allowed")
            require(not self._faulted and self._binding is not None, "Paused policy needs a reset")
            self._faulted, self._inflight = True, True
            generation, binding = self._generation, self._binding
        try:
            started = integer(self.clock_ns(), "actual prediction start", 1)
            context.validate(self.profile, observation, now_ns=started)
            self._policy_binding(context)
            self._publication(observation, context)
            require(_episode_binding(context) == binding, "Original episode authority changed")
            require(
                observation.freeze_id not in self._freezes, "A freeze cannot be predicted twice"
            )
            if self._previous is not None:
                previous = self._previous
                require(
                    observation.physics_step == previous.physics_step + self.profile.hold_steps
                    and observation.control_tick == previous.control_tick + 1
                    and observation.state_revision > previous.state_revision
                    and observation.observation_started_ns > previous.ready_ns
                    and abs(
                        observation.simulation_time - previous.simulation_time - Fraction(1, 10)
                    )
                    <= Fraction(1, 1_000_000_000),
                    "Missing actual six-tick cadence or changed simulation clock",
                )
            with self._lock:
                require(generation == self._generation, "Paused prediction was cancelled")
                self._freezes.add(observation.freeze_id)
                self._predict_calls += 1
            actions = self.policy.predict_chunk(observation, context)
            finished = integer(self.clock_ns(), "actual prediction finish", 1)
            require(finished >= started, "Prediction clock moved backwards")
            context.validate(self.profile, observation, now_ns=finished)
            require(len(actions) == self.policy.chunk_size, "Wrong actual policy chunk length")
            targets = tuple(
                bounded_joints(action, "actual predicted targets") for action in actions
            )
            validate_joint_tracking(
                targets[0],
                observation.joint_positions,
                self._previous_target,
                fps=self.profile.control_sim_hz,
            )
            self._policy_binding(context)
            self._publication(observation, context)
            ready = integer(self.clock_ns(), "validated prediction ready", 1)
            require(
                finished <= ready < context.operation_deadline_ns,
                "Validation exceeded the original policy deadline",
            )
            command = PausedJointCommand(
                targets=targets[0],
                physics_step=observation.physics_step,
                hold_steps=self.profile.hold_steps,
                expires_at_monotonic_ns=min(
                    context.interval_deadline_ns,
                    context.wall_deadline_ns,
                    ready + self.profile.max_hold_wall_ms * 1_000_000,
                ),
                model_sha256=self.policy.model_sha256,
                inference_latency_ms=(ready - context.operation_started_ns) / 1_000_000,
                freeze_id=observation.freeze_id,
                observation_sha256=observation.sha256,
                state_revision=observation.state_revision,
                control_profile_sha256=self.profile.sha256,
                context_sha256=context.sha256,
            )
            with self._lock:
                require(generation == self._generation, "Late paused prediction was cancelled")
                self._previous, self._previous_target = observation, targets[0]
                self._faulted = False
            return command
        finally:
            with self._lock:
                self._inflight = False
