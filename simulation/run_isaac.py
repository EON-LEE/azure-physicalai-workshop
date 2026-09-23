"""Azure GPU entry point. CPU tests never import or emulate the Isaac SDK here."""

import logging
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import uvicorn
from azure.core.exceptions import AzureError

from apps.api.models import utcnow
from simulation.capture_worker import CaptureBackend, CaptureWorker
from simulation.core import LoadScene, SimulationCore, StartMotion, StopMotion
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry
from simulation.health import HEARTBEAT
from simulation.http import BridgeSettings, create_bridge_app
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
    ) -> None:
        self.core, self.hardware = core, hardware
        self.heartbeat, self.clock = heartbeat, clock
        self.capture_factory = capture_factory
        self.capture_worker: CaptureWorker | None = None
        self.pending_start: StartMotion | None = None
        self.binding: CommandBinding | None = None
        self.active_epoch = core.epoch
        self.last_capture = 0.0
        self.last_heartbeat = 0.0

    def _publish_capture(self) -> None:
        if self.capture_worker is None:
            return
        update = self.capture_worker.snapshot()
        if (
            update.binding.epoch != self.core.epoch
            or update.binding.owner != self.core.owner
        ):
            self.capture_worker.invalidate("Capture lease or epoch changed before publication.")
            return
        self.core.publish_capture(
            update.binding,
            update.state.status,
            receipt=update.state.receipt,
            message=update.state.message,
        )
        if update.state.status == "invalid" and self.core.matches(update.binding):
            raise RuntimeError(update.state.message)

    def _start(self, action: StartMotion) -> None:
        self.binding = self.core.binding(action.command.command_id)
        if self.core.spec.record_demonstration:
            if self.capture_worker is not None and self.capture_worker.thread.is_alive():
                raise RuntimeError("The bounded capture worker is persisting another episode.")
            capture_binding = self.core.begin_capture(action.command.command_id)
            if self.capture_factory is None:
                request = Demonstration.prepare(self.core, action.command.command_id)
                factory = partial(Demonstration, request)
            else:
                factory = partial(self.capture_factory, capture_binding)
            self.capture_worker = CaptureWorker(capture_binding, factory)
            self.pending_start = action
        else:
            with self.core.lock:
                if self.core.actuation_allowed(self.binding):
                    self.hardware.start(action.target_id, None)

    def _start_prepared_capture(self) -> None:
        if self.pending_start is None:
            return
        if not self.core.matches(self.binding):
            self.capture_worker.invalidate("Command changed during capture preparation.")
            self.pending_start = None
            return
        with self.core.lock:
            if (
                self.capture_worker.prepared.is_set()
                and self.core.actuation_allowed(self.binding)
            ):
                self.hardware.start(self.pending_start.target_id, self.capture_worker)
                self.pending_start = None

    def finish(self, status, message=None) -> None:
        if self.binding is None or not self.core.matches(self.binding):
            return
        completed_at = utcnow()
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
        self._publish_capture()
        if status == "succeeded":
            self.hardware.world.play()

    def tick(self) -> None:
        try:
            self._publish_capture()
            action = self.core.next_action()
            if isinstance(action, LoadScene):
                self.active_epoch = action.epoch
                self.hardware.load(action.spec)
            elif isinstance(action, StopMotion) and self.core.should_stop(action.command_id):
                self.binding = self.core.binding(action.command_id)
                self.hardware.stop()
                self.finish("cancelled", "Simulation stop confirmed.")
            elif isinstance(action, StartMotion) and self.core.begin_motion(
                action.command.command_id
            ):
                self._start(action)
            if self.core.deadline_expired():
                self.hardware.stop()
                self.finish("timed_out", "Simulation command deadline expired.")
            self._start_prepared_capture()
            if self.hardware.world is not None:
                completed = self.hardware.advance()
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
            self.hardware.stop()
            self.finish("failed", str(exc))
            self.core.fail_scene(str(exc), epoch=self.active_epoch)
        finally:
            if self.clock() - self.last_heartbeat >= 1:
                temporary = self.heartbeat.with_suffix(".new")
                temporary.write_text(str(self.clock()), encoding="ascii")
                temporary.replace(self.heartbeat)
                self.last_heartbeat = self.clock()

    def close(self) -> None:
        self.hardware.stop()
        self.finish("cancelled", "Simulator is shutting down.")
        if self.capture_worker is not None:
            self.capture_worker.close()
            self._publish_capture()


def main() -> None:
    settings = BridgeSettings()
    if (
        not Path(settings.sim_tls_cert_file).is_file()
        or not Path(settings.sim_tls_key_file).is_file()
    ):
        raise RuntimeError(
            "A provisioned TLS certificate and key are required; plaintext is disabled."
        )
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        "isaacsim.core.api",
        "isaacsim.robot.manipulators.examples",
        "isaacsim.sensors.camera",
    ):
        manager.set_extension_enabled_immediate(extension, True)
        if not manager.is_extension_enabled(extension):
            raise RuntimeError(f"Required simulator extension did not load: {extension}")
    from simulation.isaac_adapter import IsaacWorkcell

    hardware = IsaacWorkcell()
    core = SimulationCore(SceneRegistry())
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
    runtime = SimulatorRuntime(core, hardware)

    try:
        while simulation_app.is_running() and thread.is_alive():
            runtime.tick()
            if hardware.world is None:
                simulation_app.update()
            time.sleep(0.001)
    finally:
        runtime.close()
        server.should_exit = True
        thread.join(timeout=10)
        simulation_app.close()
        HEARTBEAT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
