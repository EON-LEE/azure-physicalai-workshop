"""Read-only admission evidence for the reviewed CPU PhysX scheduling profile."""

from __future__ import annotations

import json
from typing import Protocol

PHYSICS_THREAD_SETTING = "/persistent/physics/numThreads"
CONTROL_PHYSICS_THREADS = 0


class SettingsReader(Protocol):
    def get(self, path: str) -> object: ...


class PhysicsContextReader(Protocol):
    @property
    def device(self) -> str: ...

    def is_gpu_dynamics_enabled(self) -> bool: ...


def physics_scheduling_readback(
    settings: SettingsReader,
    *,
    phase: str,
    context: PhysicsContextReader | None = None,
) -> dict:
    return {
        "schema": "physicalai.cpu-physics-scheduling/v1",
        "phase": phase,
        "setting": PHYSICS_THREAD_SETTING,
        "requested_num_threads": CONTROL_PHYSICS_THREADS,
        "observed_num_threads": settings.get(PHYSICS_THREAD_SETTING),
        "physics_device": context.device if context is not None else None,
        "gpu_dynamics_enabled": context.is_gpu_dynamics_enabled() if context is not None else None,
    }


def require_control_scheduling(readback: dict) -> None:
    observed = readback.get("observed_num_threads")
    if type(observed) is not int or observed != CONTROL_PHYSICS_THREADS:
        raise RuntimeError(
            "Control CPU physics requires actual numThreads=0; readback: "
            + json.dumps(readback, default=str, sort_keys=True)
        )
    if readback.get("phase") != "application_ready" and (
        readback.get("physics_device") != "cpu" or readback.get("gpu_dynamics_enabled") is not False
    ):
        raise RuntimeError(
            "Main-thread scheduling applies only to the unchanged CPU physics backend; readback: "
            + json.dumps(readback, default=str, sort_keys=True)
        )
