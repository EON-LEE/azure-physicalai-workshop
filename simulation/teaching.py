"""Short-lived Cartesian intent; a teaching lease by itself grants no movement."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from apps.api.errors import Problem
from apps.api.service import content_hash
from simulation.runtime_contracts import TeachingInput, TeachingIntent, TeachingStart


@dataclass
class TeachingSession:
    request: TeachingStart
    last_input: TeachingInput | None = None
    input_fingerprint: str | None = None
    input_deadline_ns: int = 0
    finishing: bool = False
    used_grants: set[UUID] = field(default_factory=set)

    def admit(self, request: TeachingInput, now: datetime, now_ns: int) -> None:
        fingerprint = content_hash(request.model_dump(mode="json"))
        last_sequence = self.last_input.sequence if self.last_input is not None else 0
        if request.sequence == last_sequence:
            if fingerprint != self.input_fingerprint:
                raise Problem(409, "sequence_reused", "An input sequence cannot be reused.")
            return
        if request.sequence <= last_sequence or (
            request.deadman and request.sequence != last_sequence + 1
        ):
            raise Problem(409, "invalid_sequence", "The next teaching sequence is required.")
        remaining = (request.expires_at - now).total_seconds()
        if request.deadman:
            if not 0 < remaining <= 0.25:
                raise Problem(409, "input_expired", "Jog expiry must be within the next 250 ms.")
            if (
                request.grant_id is None
                or request.grant_expires_at is None
                or not 0 < (request.grant_expires_at - now).total_seconds() <= 1
                or request.expires_at > request.grant_expires_at
                or request.grant_id in self.used_grants
            ):
                raise Problem(
                    409,
                    "control_grant_invalid",
                    "A fresh, unused server control grant is required.",
                )
            self.used_grants.add(request.grant_id)
        self.last_input = request.model_copy(deep=True)
        self.input_fingerprint = fingerprint
        self.input_deadline_ns = now_ns + int(max(0, remaining) * 1_000_000_000)

    def intent(self, now_ns: int) -> TeachingIntent | None:
        if (
            self.finishing
            or self.last_input is None
            or not self.last_input.deadman
            or now_ns >= self.input_deadline_ns
        ):
            return None
        return TeachingIntent(
            self.last_input.sequence,
            self.last_input.delta_xyz_m,
            self.last_input.gripper,
            self.input_deadline_ns,
        )
