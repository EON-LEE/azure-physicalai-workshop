"""Isaac camera identity and non-advancing observation synchronization."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

from simulation.physics_scheduling import CONTROL_PHYSICS_THREADS, PHYSICS_THREAD_SETTING

CONTROL_EXPERIENCE = "isaacsim.exp.base.zero_delay.kit"
CONTROL_EXPERIENCE_SHA256 = "776a905289b9029d760fdc0d9b9d6e6cb96b20a4a5f00763f8f120ea9f7e0b88"
PAUSED_RENDER_ATTEMPTS = 8


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
    return _observation_barrier(
        world,
        cameras,
        dt=dt,
        physics_step=physics_step,
        clock_ns=clock_ns,
        deadline_ns=deadline_ns,
        published_ns=published_ns,
        previous_identities=previous_identities,
        diagnostics=diagnostics,
        render_attempts=2,
    )


def paused_observation_barrier(
    world,
    cameras,
    *,
    dt: float,
    physics_step: int,
    clock_ns: Callable[[], int],
    deadline_ns: int,
    previous_identities: dict[str, tuple[int, int]],
    read_native_clocks: Callable[[], dict],
    verify_frozen: Callable[[], None],
    diagnostics: dict,
) -> int:
    initial_clocks = None

    def verify(evidence: dict) -> None:
        nonlocal initial_clocks
        native = evidence["native_clocks"]
        if (
            native.get("multitick_enabled") is False
            or ("physics_steps" in native and native["physics_steps"] != physics_step)
            or evidence["world_physics_index"] != physics_step
            or (
                "simulation_time" in native
                and abs(native["simulation_time"] - evidence["world_time"]) > dt / 2
            )
            or (
                "external_simulation_time" in native
                and abs(native["external_simulation_time"] - evidence["world_time"]) > dt / 2
            )
        ):
            raise RuntimeError("Native simulation clocks disagree with the frozen physical frame.")
        if initial_clocks is not None and native != initial_clocks:
            raise RuntimeError("Native simulation clocks changed during the frozen observation.")
        initial_clocks = native
        verify_frozen()

    return _observation_barrier(
        world,
        cameras,
        dt=dt,
        physics_step=physics_step,
        clock_ns=clock_ns,
        deadline_ns=deadline_ns,
        previous_identities=previous_identities,
        diagnostics=diagnostics,
        render_attempts=PAUSED_RENDER_ATTEMPTS,
        read_native_clocks=read_native_clocks,
        verify_evidence=verify,
    )


def _observation_barrier(
    world,
    cameras,
    *,
    dt: float,
    physics_step: int,
    clock_ns: Callable[[], int],
    deadline_ns: int,
    render_attempts: int,
    published_ns: int | None = None,
    previous_identities: dict[str, tuple[int, int]] | None = None,
    diagnostics: dict | None = None,
    read_native_clocks: Callable[[], dict] | None = None,
    verify_evidence: Callable[[dict], None] | None = None,
) -> int:
    if set(cameras) != {"inspection", "overview"}:
        raise ValueError("Both actual observation cameras are required.")
    start_time = float(world.current_time)
    start_step = int(world.current_time_step_index)
    details = diagnostics if diagnostics is not None else {}

    def snapshot() -> dict:
        evidence = camera_evidence(world, cameras, physics_step=physics_step)
        if read_native_clocks is not None:
            evidence["native_clocks"] = read_native_clocks()
        return evidence

    details.update(
        before=snapshot(),
        previous_identities=previous_identities,
        reused_published_frame=False,
        render_calls=0,
        deadline_ns=deadline_ns,
        max_render_calls=render_attempts,
    )
    if verify_evidence is not None:
        details["attempts"] = []
        verify_evidence(details["before"])

    def synchronized_and_new(reasons: dict | None = None) -> bool:
        synchronized = True
        native_times = []
        for name, sensor in cameras.items():
            sample = sensor.get_current_frame()
            rendering_time = sample.get("rendering_time")
            identity = render_identity(sample.get("rendering_frame"))
            if read_native_clocks is not None:
                native_times.append(Fraction(*identity))
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
            if (
                read_native_clocks is not None
                and abs(float(Fraction(*identity)) - start_time) > dt / 2
            ):
                synchronized = False
                rejected.append("native_simulation_time_alignment")
            if reasons is not None:
                reasons[name] = rejected
        if native_times and len(set(native_times)) != 1:
            synchronized = False
            if reasons is not None:
                for name in cameras:
                    reasons[name].append("native_camera_pair_time_mismatch")
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
    for _ in range(render_attempts):
        if clock_ns() >= deadline_ns:
            raise RuntimeError("Camera observation exhausted the existing control budget.")
        if verify_evidence is not None:
            attempt = {"before": snapshot()}
            details["attempts"].append(attempt)
            verify_evidence(attempt["before"])
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
        evidence = snapshot()
        details["after"] = evidence
        if verify_evidence is not None:
            attempt["after"] = evidence
            verify_evidence(evidence)
            stamp = clock_ns()
        if stamp >= deadline_ns:
            raise RuntimeError(
                "Camera observation exhausted the existing control budget: "
                + json.dumps(evidence, default=str, sort_keys=True)
            )
        reasons = {} if verify_evidence is not None else None
        ready = synchronized_and_new(reasons)
        if verify_evidence is not None:
            attempt["eligibility"] = reasons
        if ready:
            return stamp
    raise ValueError(
        "Camera observation is not synchronized after a non-advancing render barrier: "
        + json.dumps(evidence, default=str, sort_keys=True)
    )
