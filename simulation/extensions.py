from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from math import hypot
from pathlib import Path

from apps.api.errors import Problem
from apps.api.models import EnvironmentRecord, Position
from apps.api.service import content_hash
from contracts.validate_environment import (
    validate_environment,
    validate_paused_learning_environment,
)


@dataclass(frozen=True)
class Station:
    id: str
    role: str
    position: Position


@dataclass(frozen=True)
class PausedSceneAuthority:
    schema: str
    execution_timing: str
    profile_id: str
    max_simulation_seconds: int
    max_wall_seconds: int

    def __post_init__(self) -> None:
        if (
            self.schema != "physicalai.paused-simulation/v1"
            or self.execution_timing != "paused_simulation"
            or self.profile_id != "franka-position-hold-10hz-paused-v1"
            or type(self.max_simulation_seconds) is not int
            or not 1 <= self.max_simulation_seconds <= 30
            or type(self.max_wall_seconds) is not int
            or not 1 <= self.max_wall_seconds <= 600
        ):
            raise ValueError("Invalid explicit paused scene authority.")

    @property
    def real_time_admission(self) -> bool:
        return False

    @property
    def max_simulation_steps(self) -> int:
        return self.max_simulation_seconds * 60

    def validate_request(self, *, wall_seconds: float, simulation_steps: int) -> None:
        if type(wall_seconds) not in (int, float) or not 0 < wall_seconds <= self.max_wall_seconds:
            raise Problem(
                409, "paused_wall_budget", "Requested wall budget exceeds the approved scene limit."
            )
        if (
            type(simulation_steps) is not int
            or not 6 <= simulation_steps <= self.max_simulation_steps
            or simulation_steps % 6
        ):
            raise Problem(
                409,
                "paused_simulation_budget",
                "Simulation budget must be six-tick aligned within the approved scene limit.",
            )


@dataclass(frozen=True)
class SceneSpec:
    stations: tuple[Station, ...]
    source_id: str
    inspection_id: str
    accepted_id: str
    rejected_id: str
    seed: int
    defective: bool
    robot_profile: str
    requested_speed: float
    requested_payload: float
    platform_color: tuple[float, float, float]
    record_demonstration: bool = False
    demonstration_split: str | None = None
    builder_id: str = "inspection-cell-v1"
    initial_part_position: Position | None = None
    scene_builder_sha256: str | None = None
    learning_execution: PausedSceneAuthority | None = None

    def require_paused_authority(self) -> PausedSceneAuthority:
        if self.learning_execution is None:
            raise Problem(
                409,
                "paused_execution_required",
                "Explicit saved non-real-time simulation authority is required.",
            )
        return self.learning_execution

    @property
    def part_position(self) -> Position:
        return (
            self.initial_part_position
            if self.initial_part_position is not None
            else self.station(self.source_id).position
        )

    def station(self, station_id: str) -> Station:
        for station in self.stations:
            if station.id == station_id:
                return station
        raise Problem(422, "unknown_station", "The requested station is not in the loaded scene.")


def can_reset_in_place(previous: SceneSpec, current: SceneSpec) -> bool:
    return (
        not previous.record_demonstration
        and not current.record_demonstration
        and replace(
            previous,
            seed=current.seed,
            defective=current.defective,
            initial_part_position=current.initial_part_position,
        )
        == current
    )


class SceneBuilder(ABC):
    api_version = "1"

    @abstractmethod
    def build(self, environment: EnvironmentRecord) -> SceneSpec:
        """Return a declarative scene; only the main Isaac thread creates USD objects."""


class InspectionCell(SceneBuilder):
    platform_color = (0.15, 0.3, 0.45)

    def build(self, environment: EnvironmentRecord) -> SceneSpec:
        doc = environment.document
        workflow = doc["workflow"]
        return SceneSpec(
            stations=tuple(
                Station(id=item["id"], role=item["role"], position=tuple(item["position_m"]))
                for item in doc["stations"]
            ),
            source_id=workflow["source_station"],
            inspection_id=workflow["inspect_station"],
            accepted_id=workflow["accept_station"],
            rejected_id=workflow["reject_station"],
            seed=doc["scene"]["seed"],
            defective=doc["scene"]["seed"] % 2 == 1,
            robot_profile=doc["scene"]["robot_profile"],
            requested_speed=doc["limits"]["max_cartesian_speed_m_s"],
            requested_payload=doc["limits"]["max_payload_kg"],
            platform_color=self.platform_color,
            record_demonstration=doc["execution"].get("record_demonstration", False),
            demonstration_split=doc["execution"].get("demonstration_split"),
            builder_id=doc["scene"]["template_id"],
        )


class CustomerInspectionCell(InspectionCell):
    """A code-level extension can change procedural geometry/style without API edits."""

    platform_color = (0.35, 0.22, 0.4)


class LearningInspectionCell(InspectionCell):
    """A reviewed initial placement distribution, not a live part-position override."""

    def build(self, environment: EnvironmentRecord) -> SceneSpec:
        spec = super().build(environment)
        x, y, z = spec.station(spec.source_id).position
        if any(
            not 0.15 <= hypot(x + dx, y + dy) <= 0.75
            for dx in (-0.02, 0.02)
            for dy in (-0.02, 0.02)
        ):
            raise Problem(
                422,
                "learning_source_margin",
                "The learning source needs a two-centimetre reachable placement margin.",
            )
        seed = hashlib.sha256(f"{spec.builder_id}:{spec.seed}".encode("ascii")).digest()
        offsets = tuple(
            (2 * int.from_bytes(seed[index : index + 4], "big") / (2**32 - 1) - 1) * 0.02
            for index in (0, 4)
        )
        return replace(
            spec, defective=False, initial_part_position=(x + offsets[0], y + offsets[1], z)
        )


class SceneRegistry:
    def __init__(self, load_installed: bool = True) -> None:
        self.builders: dict[str, SceneBuilder] = {
            "inspection-cell-v1": InspectionCell(),
            "inspection-cell-custom-v1": CustomerInspectionCell(),
            "inspection-cell-learning-v1": LearningInspectionCell(),
        }
        if load_installed:
            for point in entry_points(group="physicalai.scenes"):
                builder_class = point.load()
                self.register(point.name, builder_class())

    def register(self, name: str, builder: SceneBuilder) -> None:
        if name in self.builders:
            raise ValueError(f"Duplicate scene registration: {name}")
        if not isinstance(builder, SceneBuilder) or builder.api_version != "1":
            raise ValueError(f"Unsupported scene extension API: {name}")
        self.builders[name] = builder

    def builder_digest(self, name: str) -> str:
        builder = self.builders.get(name)
        if builder is None:
            raise Problem(
                422, "template_not_installed", "The reviewed scene builder is unavailable."
            )
        return hashlib.sha256(Path(inspect.getfile(type(builder))).read_bytes()).hexdigest()

    def build(self, environment: EnvironmentRecord) -> SceneSpec:
        if validate_environment(environment.document):
            raise Problem(
                422, "invalid_environment", "The simulator rejected the environment contract."
            )
        if content_hash(environment.document) != environment.revision:
            raise Problem(
                422, "revision_mismatch", "Environment content does not match its revision."
            )
        if environment.document["execution"]["mode"] != "live":
            raise Problem(409, "replay_not_live", "This simulator accepts live environments only.")
        paused = None
        if "learning_execution" in environment.document:
            if validate_paused_learning_environment(environment.document):
                raise Problem(
                    422, "invalid_paused_environment", "Paused scene authority is invalid."
                )
            paused = PausedSceneAuthority(**environment.document["learning_execution"])
        builder = self.builders.get(environment.document["scene"]["template_id"])
        if builder is None:
            raise Problem(
                422, "template_not_installed", "The reviewed scene template is not installed."
            )
        spec = builder.build(environment)
        if not isinstance(spec, SceneSpec):
            raise Problem(
                422, "invalid_extension", "The scene extension returned an invalid specification."
            )
        spec = replace(
            spec,
            scene_builder_sha256=self.builder_digest(environment.document["scene"]["template_id"]),
            learning_execution=paused,
        )
        if spec.robot_profile != "reference-arm":
            raise Problem(
                422,
                "robot_not_installed",
                "Only the reference Franka profile is installed in this image.",
            )
        if not (0 < spec.requested_speed <= 0.25) or not (0.05 <= spec.requested_payload <= 2):
            raise Problem(
                422, "controller_limit", "Requested limits exceed the reference controller profile."
            )
        for station in spec.stations:
            x, y, z = station.position
            if not (0.15 <= hypot(x, y) <= 0.75 and 0.08 <= z <= 0.6):
                raise Problem(
                    422,
                    "unreachable_station",
                    "A station is outside the reference robot's supported cell envelope.",
                )
        return spec
