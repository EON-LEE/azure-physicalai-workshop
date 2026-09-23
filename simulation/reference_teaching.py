"""Explicit scripted expert for bootstrap collection, never a learned-policy fallback."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import dist
from uuid import uuid4

from learning.common import require
from simulation.extensions import SceneSpec
from simulation.motion import InspectionRoute, move_toward
from simulation.runtime_contracts import TeachingInput, TeachingStart


class ReferenceTeacher:
    def __init__(self, request: TeachingStart, spec: SceneSpec, *, tcp, part) -> None:
        require(
            request.demonstrator_kind == "reference_controller",
            "A scripted expert must be labelled reference_controller, never human input",
        )
        self.request = request
        self.sequence = 0
        self.initial_part = part
        self.grasp_verified = False
        self.route = InspectionRoute(
            tcp,
            part,
            spec.station(spec.inspection_id).position,
            spec.station(request.target_station_id).position,
            speed=min(0.1, spec.requested_speed),
            dt=0.1,
        )

    @property
    def done(self) -> bool:
        return self.route.done

    def next_input(self, *, now: datetime, tcp, finger_gap: float, part) -> TeachingInput | None:
        remaining = (self.request.session_expires_at - now).total_seconds()
        require(
            0 < remaining <= 30, "The fixed reference collection deadline expired or exceeds 30s"
        )
        if self.done:
            return None
        if self.route.index >= 5 and not self.grasp_verified:
            require(
                part[2] >= self.initial_part[2] + 0.05
                and dist(tcp, part) <= 0.09
                and 0.005 <= finger_gap <= 0.055,
                "Measured lift did not verify a real part grasp",
            )
            self.grasp_verified = True
        target, closed = self.route.next_target(tcp, finger_gap, part)
        target = move_toward(tcp, target, 0.01 - 1e-12)
        self.sequence += 1
        expiry = min(now + timedelta(milliseconds=250), self.request.session_expires_at)
        return TeachingInput(
            lease_id=self.request.lease_id,
            epoch=self.request.epoch,
            sequence=self.sequence,
            expires_at=expiry,
            deadman=True,
            delta_xyz_m=tuple(b - a for a, b in zip(tcp, target, strict=True)),
            gripper="close" if closed else "open",
            grant_id=uuid4(),
            grant_expires_at=min(now + timedelta(seconds=1), self.request.session_expires_at),
        )
