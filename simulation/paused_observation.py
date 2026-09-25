"""Runtime-only proof for the bounded first already-published frozen observation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import UUID

from learning.common import canonical, digest, integer, sha256, utc
from learning.contract import Scope
from learning.paused import FrozenCameraSample
from simulation.paused_control import FrozenPhysicsState


@dataclass(frozen=True)
class PausedPublication:
    publication_id: UUID
    scope: Scope
    environment_id: str
    revision: str
    profile_sha256: str
    state_revision: int
    frozen_state: FrozenPhysicsState
    freeze_established_ns: int
    joint_sample_ns: int
    published_ns: int
    captured_at_utc: str
    images: tuple[tuple[str, FrozenCameraSample], ...]

    @property
    def frozen_state_sha256(self) -> str:
        return digest(
            canonical(
                {
                    "publication_id": str(self.publication_id),
                    "scope": asdict(self.scope),
                    "environment_id": self.environment_id,
                    "revision": self.revision,
                    "profile_sha256": self.profile_sha256,
                    "state_revision": self.state_revision,
                    "frozen_state": {
                        **asdict(self.frozen_state),
                        "epoch": str(self.frozen_state.epoch),
                    },
                    "freeze_established_ns": self.freeze_established_ns,
                }
            )
        )

    @property
    def record_sha256(self) -> str:
        return digest(canonical(self.metadata()))

    def metadata(self) -> dict:
        return {
            "publication_id": str(self.publication_id),
            "scope": asdict(self.scope),
            "environment_id": self.environment_id,
            "revision": self.revision,
            "profile_sha256": self.profile_sha256,
            "state_revision": self.state_revision,
            "epoch": str(self.frozen_state.epoch),
            "physics_step": self.frozen_state.physics_step,
            "joint_positions": self.frozen_state.joint_positions,
            "freeze_established_ns": self.freeze_established_ns,
            "joint_sample_ns": self.joint_sample_ns,
            "published_ns": self.published_ns,
            "captured_at_utc": self.captured_at_utc,
            "images": {name: image.metadata() for name, image in self.images},
        }

    def private_evidence(self) -> dict:
        return {
            "record": self.metadata(),
            "record_sha256": self.record_sha256,
            "private_frozen_state": {
                **asdict(self.frozen_state),
                "epoch": str(self.frozen_state.epoch),
            },
            "private_frozen_state_sha256": self.frozen_state_sha256,
        }


class PausedPublicationCache:
    def __init__(self) -> None:
        self.publication: PausedPublication | None = None
        self.consumed = False

    def record(self, publication: PausedPublication) -> None:
        publication.scope.validate()
        publication.frozen_state.validate()
        sha256(publication.revision, "published environment revision")
        sha256(publication.profile_sha256, "published paused profile")
        for value in (
            publication.freeze_established_ns,
            publication.joint_sample_ns,
            publication.published_ns,
        ):
            integer(value, "original publication clock", 1)
        if (
            not isinstance(publication.publication_id, UUID)
            or not publication.freeze_established_ns
            <= publication.joint_sample_ns
            <= publication.published_ns
        ):
            raise ValueError("Frozen state must be established before actual publication.")
        if self.publication is not None and (
            publication.publication_id == self.publication.publication_id
            or (
                publication.frozen_state.epoch == self.publication.frozen_state.epoch
                and publication.frozen_state.physics_step
                <= self.publication.frozen_state.physics_step
            )
        ):
            raise ValueError("An old frozen publication cannot be registered as a new origin.")
        if (
            type(publication.images) is not tuple
            or {name for name, _ in publication.images}
            != {
                "inspection",
                "overview",
            }
            or len(publication.images) != 2
        ):
            raise ValueError("A frozen publication needs two immutable actual camera records.")
        for _, image in publication.images:
            if (
                type(image.png) is not bytes
                or image.physics_step != publication.frozen_state.physics_step
                or not publication.freeze_established_ns
                <= image.monotonic_ns
                <= publication.published_ns
                or abs(float(image.simulation_time) - publication.frozen_state.world_time) > 1 / 120
                or utc(image.captured_at_utc) > utc(publication.captured_at_utc)
            ):
                raise ValueError("The actual camera publication is not bound to the frozen state.")
        self.publication, self.consumed = publication, False

    def take(
        self,
        *,
        scope: Scope,
        environment_id: str,
        revision: str,
        profile_sha256: str,
        state_revision: int,
        state: FrozenPhysicsState,
        now_ns: int,
    ) -> PausedPublication:
        if self.publication is None:
            raise RuntimeError("No contemporaneous frozen publication is available.")
        if self.consumed:
            raise RuntimeError("The original frozen publication was already consumed.")
        value = self.publication
        oldest = min(value.joint_sample_ns, *(image.monotonic_ns for _, image in value.images))
        if not 0 <= now_ns - value.published_ns <= 2_000_000_000 or now_ns - oldest > 2_000_000_000:
            self.consumed = True
            raise RuntimeError("The original frozen publication is future-dated or expired.")
        if (
            scope != value.scope
            or environment_id != value.environment_id
            or revision != value.revision
            or profile_sha256 != value.profile_sha256
            or state_revision != value.state_revision
            or state != value.frozen_state
        ):
            self.consumed = True
            raise RuntimeError("Owner, scene, epoch or physical state changed after publication.")
        self.consumed = True
        return value
