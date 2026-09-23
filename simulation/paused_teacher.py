"""Scripted non-real-time expert progression bound to actual completed simulation ticks."""

from __future__ import annotations

from collections.abc import Callable
from math import dist

from learning.common import finite
from simulation.extensions import SceneSpec
from simulation.motion import InspectionRoute, move_toward
from simulation.paused_control import FrozenPhysicsState


class PausedReferenceTeacher:
    def __init__(
        self,
        spec: SceneSpec,
        *,
        initial: FrozenPhysicsState,
        tcp,
        target_station_id: str,
        wall_deadline_ns: int,
        clock_ns: Callable[[], int],
    ) -> None:
        self.spec = spec
        self.initial = initial
        self.wall_deadline_ns, self.clock_ns = wall_deadline_ns, clock_ns
        self.route = InspectionRoute(
            tcp,
            initial.object_position,
            spec.station(spec.inspection_id).position,
            spec.station(target_station_id).position,
            speed=min(0.1, spec.requested_speed),
            dt=0.1,
        )
        self.previous = initial
        self.cached = None
        self.grasp_verified = False

    def target(self, state: FrozenPhysicsState, *, tcp, finger_gap):
        if self.clock_ns() >= self.wall_deadline_ns:
            raise RuntimeError("The original automated reference episode wall deadline expired.")
        state.validate()
        gap = finite(finger_gap, "measured finger gap")
        if state.epoch != self.initial.epoch:
            raise RuntimeError("The reference episode changed simulator epoch.")
        if self.route.done:
            return None
        if state.physics_step == self.previous.physics_step:
            if state != self.previous:
                raise RuntimeError("The reference scene changed without an actual physics tick.")
            if self.cached is None:
                waypoint = self.route.current
                self.route.target = move_toward(
                    self.route.target, waypoint.position, self.route.speed * self.route.dt
                )
                self.cached = (self.route.target, waypoint.closed)
            return self.cached
        if (
            state.physics_step - self.previous.physics_step != 6
            or abs(state.world_time - self.previous.world_time - 0.1) > 1e-6
        ):
            raise RuntimeError(
                "Reference progression requires exactly six completed physics ticks."
            )
        self.previous = state
        if self.route.index >= 5 and not self.grasp_verified:
            if not (
                state.object_position[2] >= self.initial.object_position[2] + 0.05
                and dist(tcp, state.object_position) <= 0.09
                and 0.005 <= gap <= 0.055
            ):
                raise RuntimeError("The actual reference lift did not verify a part grasp.")
            self.grasp_verified = True
        self.cached = self.route.next_target(tcp, gap, state.object_position)
        if self.route.done:
            self.cached = None
        return self.cached
