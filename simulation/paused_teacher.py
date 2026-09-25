"""Scripted non-real-time expert progression bound to actual completed simulation ticks."""

from __future__ import annotations

from collections.abc import Callable
from math import dist

from learning.common import finite
from simulation.extensions import SceneSpec
from simulation.motion import InspectionRoute, PickPlaceRoute, move_toward
from simulation.paused_control import FrozenPhysicsState
from simulation.runtime_contracts import TaskDefinition

PICK_PLACE_TASK_ID = "manufacturing-part-placement-v1"
PICK_PLACE_INSTRUCTION = (
    "Pick up the synthetic part from the source platform and place it in the quarantine tray."
)


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
        task: TaskDefinition | None = None,
    ) -> None:
        self.spec = spec
        self.initial = initial
        self.wall_deadline_ns, self.clock_ns = wall_deadline_ns, clock_ns
        if task is not None and task.task_id == PICK_PLACE_TASK_ID:
            if (
                task.instruction != PICK_PLACE_INSTRUCTION
                or task.goal_id != target_station_id
                or target_station_id != spec.rejected_id
            ):
                raise ValueError("The direct reference route requires the exact approved task.")
            self.route = PickPlaceRoute(
                tcp,
                initial.object_position,
                spec.station(target_station_id).position,
                tuple(station.position for station in spec.stations),
                speed=min(0.1, spec.requested_speed),
                dt=0.1,
            )
        else:
            self.route = InspectionRoute(
                tcp,
                initial.object_position,
                spec.station(spec.inspection_id).position,
                spec.station(target_station_id).position,
                speed=min(0.1, spec.requested_speed),
                dt=0.1,
            )
        self.lift_index = next(
            index for index, point in enumerate(self.route.points) if point.name == "lift-part"
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
        if self.route.index > self.lift_index and not self.grasp_verified:
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
