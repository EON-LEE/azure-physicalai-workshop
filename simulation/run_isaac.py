"""Azure GPU entry point. CPU tests never import or emulate the Isaac SDK here."""

import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import uvicorn
from azure.core.exceptions import AzureError

from apps.api.models import utcnow
from simulation.core import LoadScene, SimulationCore, StartMotion, StopMotion
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry
from simulation.http import BridgeSettings, create_bridge_app

log = logging.getLogger(__name__)


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
    last_capture = 0.0

    def finish(status, message=None):
        completed_at = utcnow()
        position = hardware.position()
        demonstration = None
        if hardware.recording is not None:
            hardware.stop()
            with core.lock:
                core.ready = False
                core.frames.clear()
            try:
                receipt = hardware.finish_recording(truncated=status != "succeeded")
                demonstration = {"status": "uploaded", **asdict(receipt)}
            except (ValueError, OSError, AzureError):
                log.exception("Demonstration was not published; physical outcome is separate")
                hardware.recording = None
                demonstration = {
                    "status": "failed",
                    "message": "Capture failed; no dataset was published.",
                }
        core.finish(
            status, position, message, completed_at=completed_at, demonstration=demonstration
        )
        if status == "succeeded":
            hardware.world.play()

    try:
        while simulation_app.is_running() and thread.is_alive():
            action = core.next_action()
            try:
                if isinstance(action, LoadScene):
                    hardware.load(action.spec)
                elif isinstance(action, StopMotion) and core.should_stop(action.command_id):
                    hardware.stop()
                    finish("cancelled", "Simulation stop confirmed.")
                elif isinstance(action, StartMotion) and core.begin_motion(
                    action.command.command_id
                ):
                    recording = (
                        Demonstration(core, action.command.command_id)
                        if core.spec.record_demonstration
                        else None
                    )
                    hardware.start(action.target_id, recording)
                if core.deadline_expired():
                    hardware.stop()
                    finish("timed_out", "Simulation command deadline expired.")
                if hardware.world is None:
                    simulation_app.update()
                else:
                    completed = hardware.advance()
                    if completed:
                        finish("succeeded")
                    if time.monotonic() - last_capture >= 0.2:
                        for camera in ("overview", "inspection"):
                            image = hardware.capture(camera)
                            if image is not None:
                                core.publish_frame(
                                    camera, image, hardware.position(), hardware.steps
                                )
                        last_capture = time.monotonic()
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                log.exception("Isaac scene or command failed")
                hardware.stop()
                if hardware.recording is not None:
                    finish("failed", "Simulation or capture failed.")
                core.fail_scene(str(exc))
            time.sleep(0.001)
    finally:
        hardware.stop()
        server.should_exit = True
        thread.join(timeout=10)
        simulation_app.close()


if __name__ == "__main__":
    main()
