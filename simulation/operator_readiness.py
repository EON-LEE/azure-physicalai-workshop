"""Read-only bounded startup readiness; unavailable is never motion authority."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite


def wait_for_known_idle(
    fetch: Callable[[], dict],
    *,
    expected_owner: str,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict:
    start = clock()
    if not isfinite(deadline) or not 0 < deadline - start <= 300:
        raise ValueError("Readiness requires its original deadline bounded to 300 seconds.")
    previous = start
    while True:
        now = clock()
        if now < previous or now >= deadline:
            raise RuntimeError(
                "The original startup-readiness deadline expired or clock regressed."
            )
        previous = now
        value = fetch()
        after_read = clock()
        if after_read < now or after_read >= deadline:
            raise RuntimeError("The original startup-readiness deadline expired during the read.")
        if not isinstance(value, dict) or value.get("owner") != expected_owner:
            raise RuntimeError("Malformed readiness response or foreign owner.")
        status = value.get("status_code")
        if status in (401, 403):
            raise RuntimeError("Startup readiness authorization failed; no retry or grant.")
        if status != 200:
            if status not in (502, 503, 504):
                raise RuntimeError("Unexpected readiness HTTP status; refusing motion authority.")
        else:
            runtime = value.get("runtime")
            if not isinstance(runtime, dict):
                raise RuntimeError("Malformed simulator readiness response.")
            simulation = runtime.get("simulation")
            if (
                not isinstance(simulation, dict)
                or not {
                    "status",
                    "environment_id",
                    "motion",
                    "message",
                }
                <= simulation.keys()
            ):
                raise RuntimeError("Malformed simulator readiness state.")
            if value.get("presentation_status") not in ("stopped", "completed", "failed"):
                raise RuntimeError(
                    "A presentation is active or unknown; refusing to stop the simulator."
                )
            if simulation["motion"] is not None or simulation["environment_id"] is not None:
                raise RuntimeError("A command or scene is active or unknown; no idle authority.")
            if (
                simulation["status"] == "unavailable"
                and simulation["message"] == "No environment is active."
            ):
                return value
            if simulation["status"] not in ("unavailable", "loading"):
                raise RuntimeError("Unexpected simulation state; refusing idle authority.")
        left = deadline - clock()
        if left <= 0:
            raise RuntimeError("The original startup-readiness deadline expired.")
        sleep(min(5.0, left))
