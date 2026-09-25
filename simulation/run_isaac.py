"""Azure GPU entry point. CPU tests never import or emulate the Isaac SDK here."""

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from uuid import UUID

import uvicorn
from azure.core.exceptions import AzureError

from learning.paused.capture import PausedEpisodeBudget
from simulation.camera_observation import control_experience_path, sensor_launch_config
from simulation.capture_status import CaptureStatusStore
from simulation.capture_worker import CaptureBackend, CaptureWorker
from simulation.core import (
    FinishTeaching,
    LoadScene,
    SimulationCore,
    StartMotion,
    StartPolicy,
    StartSimulationEpisode,
    StartTeaching,
    StopMotion,
)
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry
from simulation.health import HEARTBEAT
from simulation.http import BridgeSettings, create_bridge_app
from simulation.paused_capture import PausedDemonstration, prepare_paused_capture
from simulation.paused_runtime import PausedReferenceRuntime
from simulation.physics_scheduling import physics_scheduling_readback, require_control_scheduling
from simulation.policy_executor import PolicyExecutor
from simulation.runtime_configuration import load_deployment
from simulation.runtime_contracts import CaptureBinding, CommandBinding

log = logging.getLogger(__name__)


class SimulatorRuntime:
    """One bounded main-thread tick; all capture persistence runs outside this object."""

    def __init__(
        self,
        core: SimulationCore,
        hardware,
        *,
        heartbeat: Path = HEARTBEAT,
        clock: Callable[[], float] = time.monotonic,
        capture_factory: Callable[[CaptureBinding], CaptureBackend] | None = None,
        capture_store: CaptureStatusStore | None = None,
    ) -> None:
        self.core, self.hardware = core, hardware
        self.heartbeat, self.clock = heartbeat, clock
        self.capture_factory = capture_factory
        self.capture_store = capture_store
        self.capture_worker: CaptureWorker | None = None
        self.capture_workers: dict[UUID, CaptureWorker] = {}
        self.pending_start: (
            StartMotion | StartTeaching | StartPolicy | StartSimulationEpisode | None
        ) = None
        self.policy_executor: PolicyExecutor | None = None
        self.binding: CommandBinding | None = None
        self.active_epoch = core.epoch
        self.last_capture = 0.0
        self.last_heartbeat = 0.0

    def _publish_capture(self) -> None:
        for capture_id, worker in list(self.capture_workers.items()):
            update = worker.snapshot()
            self.core.publish_capture(
                update.binding,
                update.state.status,
                receipt=update.state.receipt,
                message=update.state.message,
            )
            if update.state.status == "invalid" and self.core.matches(update.binding):
                raise RuntimeError(update.state.message)
            if not worker.thread.is_alive() and update.state.status in {"ready", "invalid"}:
                self.capture_workers.pop(capture_id)

    def _begin_hardware(self, action, recording) -> None:
        binding = self.binding
        self.hardware.actuation_guard = partial(self.core.apply_guarded, binding)
        if isinstance(action, StartSimulationEpisode):
            driver = PausedReferenceRuntime(self.core, action.request, self.hardware, recording)
            self.hardware.paused_driver = driver
            self.hardware.recording = recording
        elif isinstance(action, StartTeaching):
            self.hardware.start_teaching(action.request, self.core, recording)
            session = self.core.teaching_sessions[(binding.owner, action.request.session_id)]
            if session.finishing:
                self.hardware.request_finish()
        elif isinstance(action, StartPolicy):
            self.hardware.start_learned(action.request, self.core, self.policy_executor, recording)
        else:
            self.hardware.start(action.target_id, recording)

    def _start(
        self, action: StartMotion | StartTeaching | StartPolicy | StartSimulationEpisode
    ) -> None:
        command = (
            action.request
            if isinstance(action, (StartTeaching, StartSimulationEpisode))
            else action.request.command
            if isinstance(action, StartPolicy)
            else action.command
        )
        self.binding = self.core.binding(command.command_id)
        self.policy_executor = None
        if (
            isinstance(action, StartSimulationEpisode)
            and action.request.controller != "reference_controller"
        ):
            raise RuntimeError(
                "A real paused policy provider is unavailable; no scripted fallback."
            )
        if isinstance(action, StartPolicy):
            if self.core.policy_provider is None:
                raise RuntimeError("The approved policy provider is unavailable.")
            port = self.core.policy_provider.create(
                self.binding.owner, action.request, self.core.control_profile
            )
            self.policy_executor = PolicyExecutor(
                port,
                profile=self.core.control_profile,
                release_id=action.request.policy_release_id,
                policy_type=action.request.policy_type,
                expected_model_sha256=action.request.model_sha256,
                context=partial(self.core.policy_context, self.binding),
                clock_ns=self.core.clock_ns,
                apply_guard=partial(self.core.apply_guarded, self.binding),
            )
        if self.core.spec.record_demonstration:
            if len(self.capture_workers) >= 2:
                raise RuntimeError("Both bounded capture workers are persisting earlier episodes.")
            capture_binding = self.core.begin_capture(command.command_id)
            if self.capture_factory is None:
                if isinstance(action, StartSimulationEpisode):
                    initial = self.hardware.frozen_physics_state(self.core.epoch)
                    key = (self.binding.owner, command.command_id)
                    budget = PausedEpisodeBudget(
                        self.core.command_started_ns[key],
                        self.core.monotonic_deadlines[key],
                        initial.physics_step,
                        initial.physics_step + command.max_simulation_steps,
                    )
                    request = prepare_paused_capture(self.core, command.command_id, budget)
                    factory = partial(PausedDemonstration, request)
                else:
                    request = Demonstration.prepare(self.core, command.command_id)
                    factory = partial(Demonstration, request)
            else:
                factory = partial(self.capture_factory, capture_binding)
            self.capture_worker = CaptureWorker(
                capture_binding,
                factory,
                terminal_sink=self.capture_store.put if self.capture_store is not None else None,
            )
            self.capture_workers[capture_binding.capture_id] = self.capture_worker
            self.pending_start = action
        else:
            with self.core.lock:
                if self.core.actuation_allowed(self.binding):
                    self._begin_hardware(action, None)

    def _start_prepared_capture(self) -> None:
        if self.pending_start is None:
            return
        if not self.core.matches(self.binding):
            self.capture_worker.invalidate("Command changed during capture preparation.")
            self.pending_start = None
            return
        with self.core.lock:
            if self.capture_worker.prepared.is_set() and self.core.actuation_allowed(self.binding):
                self._begin_hardware(self.pending_start, self.capture_worker)
                self.pending_start = None

    def _publish_policy_metrics(self) -> None:
        if self.policy_executor is not None and self.binding is not None:
            self.core.publish_policy_metrics(self.binding, self.policy_executor.metrics())
        driver = getattr(self.hardware, "paused_driver", None)
        if self.binding is not None and driver is not None:
            self.core.publish_simulation_metrics(self.binding, driver.metrics())

    def _record_paused_failure(self, message: str) -> None:
        driver = getattr(self.hardware, "paused_driver", None)
        if (
            self.binding is not None
            and driver is not None
            and driver.binding == self.binding
            and self.core.matches(self.binding)
        ):
            driver.fail(message)

    def finish(self, status, message=None) -> None:
        if self.binding is None or not self.core.matches(self.binding):
            return
        completed_at = self.core.clock_utc()
        paused = (self.binding.owner, self.binding.command_id) in self.core.simulation_commands
        with self.core.lock:
            if status == "succeeded":
                result = self.core.command(self.binding.owner, self.binding.command_id)
                if result.status == "cancelling":
                    status, message = "cancelled", "Cancellation won the completion race."
                elif self.core.deadline_expired():
                    status, message = "timed_out", "Completion exceeded the command deadline."
        if paused and status in {"failed", "timed_out"}:
            self._record_paused_failure(message or "The paused episode failed.")
        if status != "succeeded":
            self.hardware.stop()
        position = self.hardware.position()
        recorder = self.hardware.recording
        if recorder is not None:
            self.hardware.stop()
            try:
                self.hardware.finish_recording(truncated=status != "succeeded")
            except (ValueError, RuntimeError, TypeError, OSError):
                log.exception("Capture could not be sealed; physical outcome remains separate")
                recorder.invalidate("Capture terminal observation failed; no dataset published.")
                self.hardware.recording = None
        elif self.pending_start is not None:
            self.capture_worker.invalidate("Motion ended before capture preparation completed.")
        self._publish_policy_metrics()
        self.core.finish(
            status,
            position,
            message,
            completed_at=completed_at,
            binding=self.binding,
        )
        status = self.core.command(self.binding.owner, self.binding.command_id).status
        self.pending_start = None
        self.binding = None
        self.policy_executor = None
        self._publish_capture()
        if status == "succeeded" and not paused:
            self.hardware.world.play()

    def tick(self) -> None:
        try:
            self._publish_capture()
            action = self.core.next_action()
            if isinstance(action, LoadScene):
                self.active_epoch = action.epoch
                self.hardware.scene_epoch = action.epoch
                self.hardware.load(action.spec)
                self.hardware.paused_scene_core = (
                    self.core if action.spec.learning_execution is not None else None
                )
                if self.core.control_profile is not None or (
                    self.core.paused_profile is not None
                    and action.spec.learning_execution is not None
                ):
                    self.hardware.prime_control_profile(on_tick=self.write_heartbeat)
            elif isinstance(action, StopMotion) and self.core.should_stop(action.command_id):
                self.binding = self.core.binding(action.command_id)
                self.hardware.stop()
                self.finish("cancelled", "Simulation stop confirmed.")
            elif isinstance(
                action, (StartMotion, StartTeaching, StartPolicy, StartSimulationEpisode)
            ):
                command = (
                    action.request
                    if isinstance(action, (StartTeaching, StartSimulationEpisode))
                    else action.request.command
                    if isinstance(action, StartPolicy)
                    else action.command
                )
                if self.core.begin_motion(command.command_id):
                    self._start(action)
            elif isinstance(action, FinishTeaching) and self.core.matches(action.binding):
                if self.pending_start is None:
                    self.hardware.request_finish()
            if self.core.deadline_expired():
                self._record_paused_failure("Simulation command deadline expired.")
                self.hardware.stop()
                self.finish("timed_out", "Simulation command deadline expired.")
            self._start_prepared_capture()
            if self.hardware.world is not None:
                completed = self.hardware.advance()
                self._publish_policy_metrics()
                if completed:
                    self.finish("succeeded")
                self.core.publish_motion(
                    epoch=self.active_epoch,
                    phase=self.hardware.motion_phase(),
                    object_position=self.hardware.position(),
                    target_station_id=self.hardware.target,
                )
                if self.clock() - self.last_capture >= 0.2:
                    for camera in ("overview", "inspection"):
                        image = self.hardware.capture(camera)
                        if image is not None:
                            self.core.publish_frame(
                                camera,
                                image,
                                self.hardware.position(),
                                self.hardware.steps,
                                epoch=self.active_epoch,
                            )
                    self.last_capture = self.clock()
        except (RuntimeError, ValueError, TypeError, OSError, AzureError) as exc:
            log.exception("Isaac scene or command failed")
            self._record_paused_failure(str(exc))
            self.hardware.stop()
            self.finish("failed", str(exc))
            self.core.fail_scene(str(exc), epoch=self.active_epoch)
        finally:
            self.write_heartbeat()

    def write_heartbeat(self) -> None:
        if self.clock() - self.last_heartbeat >= 1:
            temporary = self.heartbeat.with_suffix(".new")
            temporary.write_text(str(self.clock()), encoding="ascii")
            temporary.replace(self.heartbeat)
            self.last_heartbeat = self.clock()

    def close(self) -> None:
        self.hardware.stop()
        with self.core.lock:
            if self.binding is None and self.core.active_command is not None:
                self.binding = self.core.binding(self.core.active_command[1])
        self.finish("cancelled", "Simulator is shutting down.")
        for worker in self.capture_workers.values():
            worker.close()
        self._publish_capture()


def create_simulation_app(*, sensor_only: bool = False):
    from isaacsim import SimulationApp

    config = (
        sensor_launch_config() if sensor_only else {"headless": True, "width": 1280, "height": 720}
    )
    simulation_app = (
        SimulationApp(config, experience=str(control_experience_path()))
        if sensor_only
        else SimulationApp(config)
    )
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    try:
        for extension in (
            "isaacsim.core.api",
            "isaacsim.robot.manipulators.examples",
            "isaacsim.sensors.camera",
        ):
            manager.set_extension_enabled_immediate(extension, True)
            if not manager.is_extension_enabled(extension):
                raise RuntimeError(f"Required simulator extension did not load: {extension}")
        if sensor_only:
            import carb

            observed = physics_scheduling_readback(
                carb.settings.get_settings(), phase="application_ready"
            )
            log.info("PHYSICALAI_CPU_PHYSICS_STARTUP %s", json.dumps(observed, sort_keys=True))
            require_control_scheduling(observed)
    except RuntimeError:
        log.exception("Isaac startup failed before control admission")
        try:
            simulation_app.close()
        except SystemExit:
            log.error("Isaac cleanup requested process exit after a rejected startup")
        raise
    return simulation_app


def main() -> None:
    settings = BridgeSettings()
    if (
        not Path(settings.sim_tls_cert_file).is_file()
        or not Path(settings.sim_tls_key_file).is_file()
    ):
        raise RuntimeError(
            "A provisioned TLS certificate and key are required; plaintext is disabled."
        )
    profile, policies = load_deployment()
    simulation_app = create_simulation_app(sensor_only=profile is not None)
    from simulation.isaac_adapter import IsaacWorkcell

    hardware = IsaacWorkcell()
    capture_store = CaptureStatusStore(
        Path(os.environ.get("CAPTURE_STATUS_ROOT", "/data/demonstrations/.capture-status"))
    )
    core = SimulationCore(
        SceneRegistry(),
        control_profile=profile,
        policy_provider=policies,
        tenant_id=str(settings.entra_tenant_id),
        capture_status_reader=capture_store.get,
    )
    app = create_bridge_app(core, settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=settings.bridge_port,
            ssl_certfile=settings.sim_tls_cert_file,
            ssl_keyfile=settings.sim_tls_key_file,
            log_level="info",
        )
    )
    thread = threading.Thread(target=server.run, name="authenticated-bridge", daemon=True)
    thread.start()
    runtime = SimulatorRuntime(core, hardware, capture_store=capture_store)

    try:
        while simulation_app.is_running() and thread.is_alive():
            runtime.tick()
            if hardware.world is None:
                simulation_app.update()
            time.sleep(0.001)
    finally:
        server.should_exit = True
        try:
            runtime.close()
        finally:
            thread.join(timeout=10)
            try:
                simulation_app.close()
            finally:
                HEARTBEAT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
