"""Actual paused policy execution; model I/O stays off the Isaac main thread."""

from __future__ import annotations

from dataclasses import dataclass, replace

from apps.api.errors import Problem
from learning.common import finite, integer, require
from learning.contract import Scope
from learning.paused import PausedControlContext, PausedJointCommand
from learning.paused.contract import COMMAND_SCHEMA
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.paused_contracts import PausedRuntimeMetrics
from simulation.paused_runtime import PausedEpisodeRuntime
from simulation.paused_worker import PausedPolicyWorker


@dataclass(frozen=True)
class PredictionResult:
    command: PausedJointCommand
    started_ns: int
    finished_ns: int


class PausedLearnedRuntime(PausedEpisodeRuntime):
    def __init__(self, core, request, hardware, worker: PausedPolicyWorker, recorder=None) -> None:
        require(
            request.controller == "learned"
            and request.policy_type == "smolvla"
            and request.model_sha256 is not None
            and request.authorization_kind in {"evaluation_grant", "policy_release"},
            "A typed, authorized paused learned model is required; no reference fallback",
        )
        require(
            core.paused_policy_provider is not None,
            "The installed paused policy provider is unavailable",
        )
        self.provider = core.paused_policy_provider
        self.worker = worker
        self.port: PausedGuardedPolicyAdapter | None = None
        self.context: PausedControlContext | None = None
        self.command: PausedJointCommand | None = None
        self.started = False
        self.applied_actions = 0
        self.applied_model_sha256 = None
        super().__init__(core, request, hardware, recorder)
        self.authority = core.simulation_authorizations[(self.binding.owner, request.command_id)]
        self.publications = hardware.paused_publications
        self.port = self.provider.create(
            self.binding.owner, request, self.profile, publication_guard=self._publication_guard
        )
        require(
            isinstance(self.port, PausedGuardedPolicyAdapter),
            "A real guarded paused model port is required",
        )
        self.predict_baseline = integer(self.port.predict_calls, "initial model prediction count")
        self._check_live()

    def _prepare_hardware(self) -> None:
        self.hardware.prepare_paused_learned(self.request, self.core)

    def _check_live(self) -> None:
        with self.core.lock:
            key = (self.binding.owner, self.binding.command_id)
            require(
                self.core.actuation_allowed(self.binding)
                and self.core.simulation_commands.get(key) == self.request
                and self.core.simulation_authorizations.get(key) == self.authority
                and self.core.paused_profile == self.profile
                and self.core.paused_policy_provider is self.provider
                and self.core.monotonic_deadlines.get(key) == self.episode.wall_deadline_ns,
                "Paused learned owner, epoch, command, model or original authority changed",
            )
            try:
                permission = self.provider.authorize(self.binding.owner, self.request, self.profile)
            except Problem as exc:
                raise RuntimeError(f"Paused learned authority failed: {exc}") from exc
            require(permission == self.authority, "Installed paused model authority changed")
            policy = self.port.policy
            require(
                policy.model_sha256 == self.request.model_sha256
                and policy.scope == Scope(self.core.tenant_id, self.binding.owner)
                and policy.profile == self.profile
                and policy.policy_type == self.request.policy_type
                and policy.execution_timing == "paused_simulation"
                and policy.real_time_admission is False
                and policy.chunk_size == 50
                and policy.n_action_steps == 1
                and (policy.task.task_id, policy.task.instruction, policy.task.goal_id)
                == (
                    self.request.task.task_id,
                    self.request.task.instruction,
                    self.request.task.goal_id,
                ),
                "The actual paused model port no longer matches its installed binding",
            )

    def _publication_guard(self, proof, observation, context) -> bool:
        # Worker callback: immutable records and protocol locks only, never SDK reads.
        return (
            self.core.actuation_allowed(self.binding)
            and context == self.context
            and self.publications.verifies(proof, observation)
        )

    def _context(self) -> PausedControlContext:
        self._check_live()
        observation, episode = self.observation, self.episode
        return PausedControlContext(
            scope=Scope(self.core.tenant_id, self.binding.owner),
            environment_id=self.request.environment_id,
            revision=self.request.revision,
            episode_id=str(self.request.command_id),
            epoch=str(self.binding.epoch),
            command_id=str(self.binding.command_id),
            destination_id=self.request.task.goal_id,
            approved=True,
            active=True,
            model_sha256=self.request.model_sha256,
            control_profile_sha256=self.profile.sha256,
            freeze_id=str(episode.freeze_id),
            observation_sha256=observation.sha256,
            state_revision=observation.state_revision,
            episode_started_ns=episode.started_ns,
            episode_initial_physics_step=episode.initial.physics_step,
            simulation_step_deadline=episode.initial.physics_step + episode.max_simulation_steps,
            wall_deadline_ns=episode.wall_deadline_ns,
            interval_started_ns=episode.interval_started_ns,
            interval_deadline_ns=episode.interval_deadline_ns,
            operation_started_ns=episode.observation_ready_ns,
            operation_deadline_ns=episode.operation_deadline_ns,
        )

    def _predict(self, state) -> bool:
        self._check_live()
        if self.context is None or self.context.freeze_id != str(self.episode.freeze_id):
            self.context = self._context()
            observation = replace(
                self.observation,
                images={
                    name: replace(image, png=bytes(image.png))
                    for name, image in self.observation.images.items()
                },
            )
            context = self.context
            context.validate(self.profile, observation, now_ns=self.core.clock_ns())

            def predict():
                started_ns = self.core.clock_ns()
                if not self.started:
                    self.port.reset(context)
                    self.started = True
                command = self.port.step(observation, context)
                return PredictionResult(command, started_ns, self.core.clock_ns())

            self.worker.submit(
                self.episode.freeze_id, predict, deadline_ns=context.operation_deadline_ns
            )
            return False
        result = self.worker.poll(self.episode.freeze_id)
        if result is None:
            return False
        self._check_live()
        self.episode.waiting(self.hardware.frozen_physics_state(self.core.epoch))
        require(isinstance(result, PredictionResult), "Invalid paused prediction result type")
        command, context = result.command, self.context
        require(
            isinstance(command, PausedJointCommand)
            and command.schema == COMMAND_SCHEMA
            and command.execution_timing == "paused_simulation"
            and command.real_time_admission is False
            and command.model_sha256 == self.request.model_sha256
            and command.control_profile_sha256 == self.profile.sha256
            and command.freeze_id == context.freeze_id
            and command.observation_sha256 == self.observation.sha256 == context.observation_sha256
            and command.context_sha256 == context.sha256
            and command.physics_step == self.observation.physics_step
            and command.hold_steps == self.profile.hold_steps
            and command.state_revision == self.core.state_revision == context.state_revision,
            "The returned learned command changed model, observation, freeze, or context",
        )
        integer(command.expires_at_monotonic_ns, "original learned command expiry", 1)
        require(
            self.core.clock_ns()
            < command.expires_at_monotonic_ns
            <= min(context.interval_deadline_ns, context.wall_deadline_ns)
            and context.operation_started_ns
            <= result.started_ns
            <= result.finished_ns
            < context.operation_deadline_ns
            and 0
            <= finite(command.inference_latency_ms, "actual paused inference latency")
            <= 2000,
            "The learned result exceeded its original operation or command deadline",
        )
        self.policy_started_ns, self.policy_finished_ns = result.started_ns, result.finished_ns
        self.command = command
        self._accept_targets(command.targets, expires_at_ns=command.expires_at_monotonic_ns)
        return False

    def _before_apply(self) -> None:
        self._check_live()
        require(
            self.command is not None
            and self.core.clock_ns() < self.command.expires_at_monotonic_ns
            and self.command.targets == self.episode.targets
            and self.command.freeze_id == str(self.episode.freeze_id)
            and self.command.physics_step + len(self.episode.controls)
            == self.episode.current.physics_step,
            "The learned target expired, changed, or skipped a physical tick",
        )

    def _after_apply(self) -> None:
        self.applied_actions += 1
        self.applied_model_sha256 = self.command.model_sha256

    def _stop_controller(self) -> None:
        self.worker.cancel()
        if self.port is not None:
            self.port.stop()

    def metrics(self):
        calls = (
            integer(self.port.predict_calls, "actual model predict attempts")
            - self.predict_baseline
        )
        require(calls >= 0, "Paused model prediction count moved backwards")
        return PausedRuntimeMetrics(
            **{
                **self._metric_values(),
                "policy_predict_calls": calls,
                "applied_action_count": self.applied_actions,
                "applied_model_sha256": self.applied_model_sha256,
            }
        )
