"""Azure GPU entry point. CPU tests never import or emulate the Isaac SDK here."""

import logging
import threading
import time
from pathlib import Path

import uvicorn

from simulation.core import LoadScene, SimulationCore, StartMotion, StopMotion
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
    try:
        while simulation_app.is_running() and thread.is_alive():
            action = core.next_action()
            try:
                if isinstance(action, LoadScene):
                    hardware.load(action.spec)
                elif isinstance(action, StopMotion):
                    hardware.stop()
                    core.finish("cancelled", hardware.position(), "Simulation stop confirmed.")
                elif isinstance(action, StartMotion) and core.begin_motion(
                    action.command.command_id
                ):
                    hardware.start(action.target_id)
                if core.deadline_expired():
                    hardware.stop()
                    core.finish(
                        "timed_out", hardware.position(), "Simulation command deadline expired."
                    )
                if hardware.world is None:
                    simulation_app.update()
                else:
                    completed = hardware.advance()
                    if completed:
                        core.finish("succeeded", hardware.position())
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
                core.fail_scene(str(exc))
            time.sleep(0.001)
    finally:
        hardware.stop()
        server.should_exit = True
        thread.join(timeout=10)
        simulation_app.close()


if __name__ == "__main__":
    main()
