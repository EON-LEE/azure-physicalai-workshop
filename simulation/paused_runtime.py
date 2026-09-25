"""Main-thread non-real-time reference episode; no model or real-time fallback."""

from __future__ import annotations

from dataclasses import replace

from learning.paused import PausedFrameSample
from simulation.paused_contracts import PausedRuntimeMetrics
from simulation.paused_control import PausedEpisode
from simulation.paused_teacher import PausedReferenceTeacher


class PausedReferenceRuntime:
    def __init__(self, core, request, hardware, recorder=None) -> None:
        self.core, self.request, self.hardware, self.recorder = core, request, hardware, recorder
        if request.controller != "reference_controller":
            raise RuntimeError("A real paused model provider is required; no reference fallback.")
        self.profile = core.paused_profile
        if self.profile is None or core.spec is None:
            raise RuntimeError("A separately approved paused scene/profile is required.")
        self.binding = core.binding(request.command_id)
        hardware.prepare_paused_reference(request, core)
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
        self.teacher = PausedReferenceTeacher(
            core.spec,
            initial=initial,
            tcp=hardware._measured_tcp(),
            target_station_id=request.target_station_id,
            wall_deadline_ns=core.monotonic_deadlines[key],
            clock_ns=core.clock_ns,
            task=request.task,
        )
        self.observation = None
        self.pending_frame: PausedFrameSample | None = None
        self.hold_started_ns = self.hold_deadline_ns = 0
        self.complete_intervals = 0
        self.reference_calls = 0
        self.done = self.succeeded = False

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
            phase = self.teacher.route.current.name if not self.teacher.route.done else None
            planned = self.teacher.target(
                current,
                tcp=self.hardware._measured_tcp(),
                finger_gap=sum(current.joint_positions[7:]),
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
            self.episode.policy_ready(
                self.episode.freeze_id, self.hardware.frozen_physics_state(self.core.epoch), targets
            )
            self.hold_started_ns = self.episode.policy_ready_ns
            self.hold_deadline_ns = self.episode.operation_deadline_ns
            return False
        self.episode.before_tick(state)
        applied = None

        def apply():
            nonlocal applied
            applied = self.hardware.apply_paused_tick(self.episode.targets)

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
        if not self.succeeded:
            self.episode.cancel("Paused runtime stop requested.")

    def fail(self, message: str) -> None:
        self.done, self.succeeded = True, False
        self.episode.fail(message)

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

    def metrics(self) -> PausedRuntimeMetrics:
        evidence = self.episode.metrics()
        return PausedRuntimeMetrics(
            control_profile_sha256=self.profile.sha256,
            controller="reference_controller",
            phase="stopped" if self.done else evidence["phase"],
            wall_elapsed_ms=evidence["wall_elapsed_ms"],
            simulation_steps=evidence["simulation_steps"],
            simulation_elapsed_seconds=evidence["simulation_elapsed_seconds"],
            applied_action_count=evidence["simulation_steps"],
            reference_route_calls=self.reference_calls,
        )
