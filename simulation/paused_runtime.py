"""Shared main-thread paused episode accounting and the explicit reference controller."""

from __future__ import annotations

from dataclasses import replace

from learning.paused import PausedFrameSample
from simulation.paused_contracts import PausedRuntimeMetrics
from simulation.paused_control import PausedEpisode
from simulation.paused_teacher import PausedReferenceTeacher


class PausedEpisodeRuntime:
    def __init__(self, core, request, hardware, recorder=None) -> None:
        self.core, self.request, self.hardware, self.recorder = core, request, hardware, recorder
        self.profile = core.paused_profile
        if self.profile is None or core.spec is None:
            raise RuntimeError("A separately approved paused scene/profile is required.")
        self.binding = core.binding(request.command_id)
        self._prepare_hardware()
        initial = hardware.frozen_physics_state(core.epoch)
        key = (self.binding.owner, request.command_id)
        self.episode = PausedEpisode(
            initial,
            wall_deadline_ns=core.monotonic_deadlines[key],
            started_ns=core.command_started_ns[key],
            max_simulation_steps=request.max_simulation_steps,
            authorized=lambda: core.actuation_allowed(self.binding),
            clock_ns=core.clock_ns,
        )
        self.observation = None
        self.pending_frame: PausedFrameSample | None = None
        self.hold_started_ns = self.hold_deadline_ns = 0
        self.complete_intervals = 0
        self.reference_calls = 0
        self.policy_started_ns = self.policy_finished_ns = None
        self.done = self.succeeded = False

    def _prepare_hardware(self) -> None:
        raise NotImplementedError("An explicit paused controller implementation is required.")

    def _predict(self, state) -> bool:
        raise NotImplementedError("An explicit paused controller implementation is required.")

    def _before_apply(self) -> None:
        return None

    def _after_apply(self) -> None:
        return None

    def _stop_controller(self) -> None:
        return None

    def _accept_targets(self, targets, *, expires_at_ns=None) -> None:
        self.episode.policy_ready(
            self.episode.freeze_id,
            self.hardware.frozen_physics_state(self.core.epoch),
            targets,
            expires_at_ns=expires_at_ns,
        )
        self.hold_started_ns = self.episode.policy_ready_ns
        self.hold_deadline_ns = self.episode.operation_deadline_ns

    def advance(self) -> bool:
        if self.done:
            return False
        try:
            return self._advance()
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            self.fail(str(exc))
            raise

    def _advance(self) -> bool:
        state = self.hardware.frozen_physics_state(self.core.epoch)
        if self.episode.phase == "idle":
            self.episode.begin_observation(state)
            return False
        if self.episode.phase == "observing":
            self.episode.waiting(state)
            observation = self.hardware.paused_observation(
                self.request, self.core, self.episode, self.complete_intervals
            )
            current = self.hardware.frozen_physics_state(self.core.epoch)
            self.episode.waiting(current)
            if observation is None:
                return False
            observation.validate(self.profile, now_ns=self.core.clock_ns())
            self.observation = observation
            self.episode.observation_ready(self.episode.freeze_id, current)
            return self._predict(current)
        if self.episode.phase == "predicting":
            self.episode.waiting(state)
            return self._predict(state)
        self.episode.before_tick(state)
        applied = None

        def apply():
            nonlocal applied
            self._before_apply()
            applied = self.hardware.apply_paused_tick(self.episode.targets)
            self._after_apply()

        self.core.apply_guarded(self.binding, apply)
        completed = self.episode.after_tick(
            self.hardware.frozen_physics_state(self.core.epoch), applied
        )
        if completed:
            frame = PausedFrameSample(
                observation=self.observation,
                commanded_joint_targets=self.episode.targets,
                applied_controls=tuple(self.episode.controls),
                interval_deadline_ns=self.episode.interval_deadline_ns,
                hold_started_ns=self.hold_started_ns,
                hold_deadline_ns=self.hold_deadline_ns,
                policy_started_ns=self.policy_started_ns,
                policy_finished_ns=self.policy_finished_ns,
            )
            if self.pending_frame is not None and self.recorder is not None:
                self.recorder.append(self.pending_frame)
            self.pending_frame = frame
            self.complete_intervals += 1
            with self.core.lock:
                if not self.core.matches(self.binding):
                    raise RuntimeError("Paused command changed at the completed physical boundary.")
                self.core.state_revision += 1
            if self.hardware.paused_goal_reached():
                self.done = self.succeeded = True
                return True
        return False

    def stop(self) -> None:
        self.done = True
        self._stop_controller()
        if not self.succeeded:
            self.episode.cancel("Paused runtime stop requested.")

    def fail(self, message: str) -> None:
        self.done, self.succeeded = True, False
        self.episode.fail(message)
        self._stop_controller()

    def finish_recording(self, *, truncated: bool) -> None:
        if self.recorder is None:
            return
        if self.episode.controls and len(self.episode.controls) != 6:
            self.recorder.invalidate("Paused capture ended during a partial six-tick interval.")
            raise RuntimeError("A partial held interval cannot be padded into a paused dataset.")
        if self.pending_frame is None:
            self.recorder.invalidate("Paused capture has no actual completed interval.")
            raise RuntimeError("Paused capture has no completed interval to finalize.")
        if (
            self.pending_frame.applied_controls[-1].physics_step
            != self.episode.current.physics_step
        ):
            self.recorder.invalidate(
                "An invalid actual held interval cannot be omitted from capture."
            )
            raise RuntimeError(
                "Paused capture contains unrecorded or invalid actual physics ticks."
            )
        self.recorder.append(
            replace(
                self.pending_frame,
                terminated=not truncated,
                truncated=truncated,
            )
        )
        self.recorder.seal()
        self.pending_frame = None

    def _metric_values(self) -> dict:
        evidence = self.episode.metrics()
        return dict(
            control_profile_sha256=self.profile.sha256,
            controller=self.request.controller,
            phase="stopped" if self.done else evidence["phase"],
            wall_elapsed_ms=evidence["wall_elapsed_ms"],
            simulation_steps=evidence["simulation_steps"],
            simulation_elapsed_seconds=evidence["simulation_elapsed_seconds"],
            applied_action_count=evidence["simulation_steps"],
            reference_route_calls=self.reference_calls,
        )

    def metrics(self) -> PausedRuntimeMetrics:
        return PausedRuntimeMetrics(**self._metric_values())


class PausedReferenceRuntime(PausedEpisodeRuntime):
    def __init__(self, core, request, hardware, recorder=None) -> None:
        if request.controller != "reference_controller":
            raise RuntimeError("A real paused model provider is required; no reference fallback.")
        super().__init__(core, request, hardware, recorder)
        self.teacher = PausedReferenceTeacher(
            core.spec,
            initial=self.episode.initial,
            tcp=hardware._measured_tcp(),
            target_station_id=request.target_station_id,
            wall_deadline_ns=self.episode.wall_deadline_ns,
            clock_ns=core.clock_ns,
            task=request.task,
        )

    def _prepare_hardware(self) -> None:
        self.hardware.prepare_paused_reference(self.request, self.core)

    def _predict(self, state) -> bool:
        phase = self.teacher.route.current.name if not self.teacher.route.done else None
        tcp = self.hardware._measured_tcp()
        planned = self.teacher.target(
            state,
            tcp=tcp,
            finger_gap=sum(state.joint_positions[7:]),
            route_point=self.hardware.paused_reference_route_point(tcp, phase),
        )
        if planned is None:
            if not self.hardware.paused_goal_reached():
                raise RuntimeError("The scripted route ended without measured task success.")
            self.done = self.succeeded = True
            return True
        targets = self.hardware.paused_reference_targets(
            *planned,
            phase=phase,
            control_tick=self.complete_intervals,
        )
        self.reference_calls += 1
        self._accept_targets(targets)
        return False
