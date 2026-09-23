"""Command-bound policy decisions; only the Isaac caller supplies an actuator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from learning.common import integer, require, sha256
from learning.contract import ControlProfile
from learning.inference import ChunkPolicy, ControlContext, JointCommand, PolicyObservation
from simulation.control import validate_position_target
from simulation.runtime_contracts import PolicyCommand, PolicyRuntime, PolicyType


class CountedChunkPolicy(ChunkPolicy, Protocol):
    predict_calls: int
    metadata: dict


class PolicyPort(Protocol):
    policy: CountedChunkPolicy

    def reset(self, context: ControlContext) -> None: ...

    def step(self, observation: PolicyObservation, context: ControlContext) -> JointCommand: ...

    def stop(self) -> None: ...


class PolicyProvider(Protocol):
    def authorize(self, owner: str, request: PolicyCommand, profile: ControlProfile) -> None: ...

    def create(self, owner: str, request: PolicyCommand, profile: ControlProfile) -> PolicyPort: ...


class PolicyExecutor:
    def __init__(
        self,
        port: PolicyPort,
        *,
        profile: ControlProfile,
        release_id: UUID,
        policy_type: PolicyType,
        expected_model_sha256: str,
        context: Callable[[], ControlContext],
        clock_ns: Callable[[], int],
        apply_guard: Callable[[Callable[[], None]], None],
    ) -> None:
        self.port, self.profile = port, profile
        self.release_id, self.expected_model_sha256 = release_id, expected_model_sha256
        self.policy_type = policy_type
        self.context, self.clock_ns, self.apply_guard = context, clock_ns, apply_guard
        self.initial: ControlContext | None = None
        self.current: JointCommand | None = None
        self.applied_offset = 0
        self.applied_actions = 0
        self.applied_model_sha: str | None = None
        self.predict_baseline = port.policy.predict_calls
        self.faulted = True
        self.started = False

    def start(self) -> None:
        require(self.initial is None, "A policy executor cannot be reused after reset or stop")
        self.profile.validate()
        sha256(self.expected_model_sha256, "approved model digest")
        context = self._check_context()
        self.initial = context
        self.port.reset(context)
        self._check_context()
        self.predict_baseline = integer(self.port.policy.predict_calls, "actual predict count")
        self.started = True
        self.faulted = False

    def _check_context(self) -> ControlContext:
        current = self.context()
        require(current.approved is True and current.active is True, "Policy command is not active")
        require(self.clock_ns() < current.deadline_monotonic_ns, "Policy deadline expired")
        if self.initial is not None:
            require(
                current.binding() == self.initial.binding()
                and current.deadline_monotonic_ns == self.initial.deadline_monotonic_ns,
                "Policy owner, scene, command or immutable deadline changed",
            )
        policy = self.port.policy
        require(
            policy.model_sha256 == self.expected_model_sha256
            and policy.scope == current.scope
            and policy.metadata.get("policy_type") == self.policy_type,
            "Policy model or owner changed",
        )
        require(
            policy.fps == self.profile.control_hz
            and policy.physics_hz == self.profile.physics_hz
            and policy.n_action_steps == 1,
            "Policy cadence or action horizon differs from the approved servo profile",
        )
        return current

    def predict(self, observation: PolicyObservation) -> JointCommand:
        try:
            require(not self.faulted and self.started, "Policy needs a new approved executor")
            require(
                self.current is None or self.applied_offset == self.profile.hold_steps,
                "The previous policy action has not completed its actual hold",
            )
            current = self._check_context()
            command = self.port.step(observation, current)
            self._check_context()
            require(
                command.model_sha256 == self.expected_model_sha256
                and command.physics_step == observation.physics_step
                and command.hold_steps == self.profile.hold_steps,
                "Policy response model, observation or hold binding changed",
            )
            require(
                self.clock_ns() < command.expires_at_monotonic_ns <= current.deadline_monotonic_ns,
                "Policy action expired or exceeded its immutable deadline",
            )
            validate_position_target(
                observation.joint_positions,
                command.targets,
                self.current.targets if self.current is not None else None,
            )
            self.current, self.applied_offset = command, 0
            return command
        except (ValueError, RuntimeError, TypeError, OSError):
            self.stop()
            raise

    def apply(
        self,
        command: JointCommand,
        *,
        physics_step: int,
        actuator: Callable[[tuple[float, ...]], None],
    ) -> None:
        def guarded_submission() -> None:
            require(not self.faulted and self.started, "Policy action was stopped or reset")
            self._check_context()
            require(
                command is self.current
                and self.applied_offset < command.hold_steps
                and physics_step == command.physics_step + self.applied_offset + 1,
                "Stale, repeated or skipped policy actuation tick",
            )
            require(
                self.clock_ns() < command.expires_at_monotonic_ns,
                "Policy action expired before actuator submission",
            )
            actuator(command.targets)
            self.applied_model_sha = command.model_sha256
            self.applied_actions += 1
            self.applied_offset += 1

        try:
            self.apply_guard(guarded_submission)
        except (ValueError, RuntimeError, TypeError, OSError):
            self.stop()
            raise

    def stop(self) -> None:
        self.faulted = True
        self.current = None
        if self.started:
            self.started = False
            self.port.stop()

    def metrics(self) -> PolicyRuntime:
        calls = (
            integer(self.port.policy.predict_calls, "actual predict count") - self.predict_baseline
        )
        require(calls >= 0, "Policy prediction counter moved backwards")
        return PolicyRuntime(
            policy_type=self.policy_type,
            policy_release_id=self.release_id,
            applied_model_sha=self.applied_model_sha,
            policy_predict_calls=calls,
            applied_action_count=self.applied_actions,
        )
