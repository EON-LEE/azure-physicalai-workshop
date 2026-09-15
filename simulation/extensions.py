from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import entry_points
from math import hypot

from apps.api.errors import Problem
from apps.api.models import EnvironmentRecord, Position
from apps.api.service import content_hash
from contracts.validate_environment import validate_environment


@dataclass(frozen=True)
class Station:
    id: str
    role: str
    position: Position


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

    def station(self, station_id: str) -> Station:
        for station in self.stations:
            if station.id == station_id:
                return station
        raise Problem(422, "unknown_station", "The requested station is not in the loaded scene.")


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
        )


class CustomerInspectionCell(InspectionCell):
    """A code-level extension can change procedural geometry/style without API edits."""

    platform_color = (0.35, 0.22, 0.4)


class SceneRegistry:
    def __init__(self, load_installed: bool = True) -> None:
        self.builders: dict[str, SceneBuilder] = {
            "inspection-cell-v1": InspectionCell(),
            "inspection-cell-custom-v1": CustomerInspectionCell(),
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
