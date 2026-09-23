"""Isaac camera identity and non-advancing observation synchronization."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

from simulation.physics_scheduling import CONTROL_PHYSICS_THREADS, PHYSICS_THREAD_SETTING

CONTROL_EXPERIENCE = "isaacsim.exp.base.zero_delay.kit"
CONTROL_EXPERIENCE_SHA256 = "776a905289b9029d760fdc0d9b9d6e6cb96b20a4a5f00763f8f120ea9f7e0b88"


def control_experience_path() -> Path:
    root = os.environ.get("EXP_PATH")
    if not root:
        raise ValueError("The pinned Isaac experience directory is unavailable.")
    path = Path(root) / CONTROL_EXPERIENCE
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("The reviewed zero-delay control experience is unavailable.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_EXPERIENCE_SHA256:
        raise ValueError(
            "The installed zero-delay experience checksum differs from the reviewed SDK."
        )
    return path


def sensor_launch_config() -> dict:
    return {
        "headless": True,
        "width": 320,
        "height": 320,
        "renderer": "RaytracedLighting",
        "disable_viewport_updates": True,
        "extra_args": [f"--{PHYSICS_THREAD_SETTING}={CONTROL_PHYSICS_THREADS}"],
    }


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
            "rendering_time_type": (
                type(rendering_time).__module__ + "." + type(rendering_time).__name__
            ),
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
    published_ns: int | None = None,
    previous_identities: dict[str, tuple[int, int]] | None = None,
    diagnostics: dict | None = None,
) -> int:
    if set(cameras) != {"inspection", "overview"}:
        raise ValueError("Both actual observation cameras are required.")
    start_time = float(world.current_time)
    start_step = int(world.current_time_step_index)
    details = diagnostics if diagnostics is not None else {}
    details.update(
        before=camera_evidence(world, cameras, physics_step=physics_step),
        previous_identities=previous_identities,
        reused_published_frame=False,
        render_calls=0,
    )

    def synchronized_and_new(reasons: dict | None = None) -> bool:
        synchronized = True
        for name, sensor in cameras.items():
            sample = sensor.get_current_frame()
            rendering_time = sample.get("rendering_time")
            identity = render_identity(sample.get("rendering_frame"))
            previous = (previous_identities or {}).get(name)
            rejected = []
            if previous is not None:
                if identity[1] != previous[1] or identity[0] < previous[0]:
                    raise ValueError("Native camera identity rewound or changed its timebase.")
                if identity[0] == previous[0]:
                    synchronized = False
                    rejected.append("native_frame_repeated")
            if (
                isinstance(rendering_time, bool)
                or not isinstance(rendering_time, Real)
                or not isfinite(rendering_time)
                or abs(rendering_time - start_time) > dt / 2
            ):
                synchronized = False
                rejected.append("rendering_time_type_or_alignment")
            if reasons is not None:
                reasons[name] = rejected
        return synchronized

    now = clock_ns()
    details["published_age_ns"] = now - published_ns if published_ns is not None else None
    details["before_eligibility"] = {}
    ready = synchronized_and_new(details["before_eligibility"])
    if now >= deadline_ns:
        raise RuntimeError("Camera observation exhausted the existing control budget.")
    if (
        published_ns is not None
        and 0 < published_ns <= now
        and now - published_ns <= 200_000_000
        and ready
    ):
        details["reused_published_frame"] = True
        return published_ns
    for _ in range(2):
        if clock_ns() >= deadline_ns:
            raise RuntimeError("Camera observation exhausted the existing control budget.")
        world.render()
        details["render_calls"] += 1
        stamp = clock_ns()
        if (
            float(world.current_time) != start_time
            or int(world.current_time_step_index) != start_step
        ):
            raise RuntimeError("The observational render barrier advanced physics.")
        evidence = camera_evidence(world, cameras, physics_step=physics_step)
        details["after"] = evidence
        if stamp >= deadline_ns:
            raise RuntimeError(
                "Camera observation exhausted the existing control budget: "
                + json.dumps(evidence, default=str, sort_keys=True)
            )
        if synchronized_and_new():
            return stamp
    raise ValueError(
        "Camera observation is not synchronized after a non-advancing render barrier: "
        + json.dumps(evidence, default=str, sort_keys=True)
    )
