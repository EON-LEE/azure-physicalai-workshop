"""Isaac camera identity and non-advancing observation synchronization."""

from __future__ import annotations

import json
from collections.abc import Callable
from math import isfinite
from numbers import Integral


def _integer(value, *, positive=False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Renderer identity must contain integral native values.")
    number = int(value)
    if not (1 if positive else 0) <= number <= 2**63 - 1:
        raise ValueError("Renderer identity is outside the unchanged raw-contract integer range.")
    return number


def render_identity(value) -> tuple[int, int]:
    if isinstance(value, dict):
        return (
            _integer(value.get("referenceTimeNumerator")),
            _integer(value.get("referenceTimeDenominator"), positive=True),
        )
    return _integer(value), 1


def camera_evidence(world, cameras, *, physics_step: int) -> dict:
    observations = {}
    for name, sensor in cameras.items():
        frame = sensor.get_current_frame()
        identity = frame.get("rendering_frame")
        if isinstance(identity, dict):
            identity = {
                key: int(identity[key])
                if isinstance(identity.get(key), Integral)
                else str(identity.get(key))
                for key in ("referenceTimeNumerator", "referenceTimeDenominator")
            }
        elif isinstance(identity, Integral):
            identity = int(identity)
        rendering_time = frame.get("rendering_time")
        observations[name] = {
            "rendering_time": float(rendering_time) if rendering_time is not None else None,
            "rendering_frame": identity,
        }
    return {
        "world_time": float(world.current_time),
        "world_physics_index": int(world.current_time_step_index),
        "physics_step": physics_step,
        "cameras": observations,
    }


def observation_barrier(
    world,
    cameras,
    *,
    dt: float,
    physics_step: int,
    clock_ns: Callable[[], int],
    deadline_ns: int,
) -> int:
    if set(cameras) != {"inspection", "overview"}:
        raise ValueError("Both actual observation cameras are required.")
    start_time = float(world.current_time)
    start_step = int(world.current_time_step_index)
    for _ in range(2):
        if clock_ns() >= deadline_ns:
            raise RuntimeError("Camera observation exhausted the existing control budget.")
        world.render()
        stamp = clock_ns()
        if (
            float(world.current_time) != start_time
            or int(world.current_time_step_index) != start_step
        ):
            raise RuntimeError("The observational render barrier advanced physics.")
        evidence = camera_evidence(world, cameras, physics_step=physics_step)
        if stamp >= deadline_ns:
            raise RuntimeError(
                "Camera observation exhausted the existing control budget: "
                + json.dumps(evidence, default=str, sort_keys=True)
            )
        synchronized = True
        for sensor in cameras.values():
            sample = sensor.get_current_frame()
            rendering_time = sample.get("rendering_time")
            render_identity(sample.get("rendering_frame"))
            if (
                isinstance(rendering_time, bool)
                or not isinstance(rendering_time, (float, int))
                or not isfinite(rendering_time)
                or abs(rendering_time - start_time) > dt / 2
            ):
                synchronized = False
        if synchronized:
            return stamp
    raise ValueError(
        "Camera observation is not synchronized after a non-advancing render barrier: "
        + json.dumps(evidence, default=str, sort_keys=True)
    )
