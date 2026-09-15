from typing import Literal, Protocol
from uuid import UUID

from apps.api.models import (
    Activation,
    Decision,
    EnvironmentRecord,
    Execution,
    MotionCommand,
    Observation,
    RunRecord,
    SimulationStatus,
    Stored,
)


class Store(Protocol):
    def get_environment(
        self, owner: str, environment_id: str
    ) -> Stored[EnvironmentRecord] | None: ...
    def list_environments(self, owner: str) -> list[EnvironmentRecord]: ...
    def put_environment(
        self, owner: str, record: EnvironmentRecord, etag: str | None
    ) -> Stored[EnvironmentRecord]: ...
    def get_run(self, owner: str, run_id: UUID) -> Stored[RunRecord] | None: ...
    def list_runs(self, owner: str) -> list[RunRecord]: ...
    def put_run(self, owner: str, record: RunRecord, etag: str | None) -> Stored[RunRecord]: ...


class Artifacts(Protocol):
    def put(self, name: str, image: bytes) -> None: ...
    def get(self, name: str) -> bytes: ...


class Planner(Protocol):
    def inspect(self, instruction: str, observation: Observation) -> tuple[Decision, str]: ...


class Bridge(Protocol):
    def status(self, owner: str) -> SimulationStatus: ...
    def activate(self, owner: str, environment: EnvironmentRecord) -> Activation: ...
    def observe(
        self,
        owner: str,
        environment_id: str,
        revision: str,
        camera: Literal["overview", "inspection"] = "inspection",
    ) -> Observation: ...
    def dispatch(self, owner: str, command: MotionCommand) -> Execution: ...
    def command(self, owner: str, command_id: UUID) -> Execution: ...
    def cancel(self, owner: str, command_id: UUID) -> Execution: ...
