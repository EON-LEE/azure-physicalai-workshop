import base64
import json
import threading
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image

from apps.api.errors import Problem
from apps.api.models import (
    Activation,
    Decision,
    Execution,
    Observation,
    Principal,
    SimulationStatus,
    Stored,
    utcnow,
)
from apps.api.service import FactoryService
from apps.api.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
_image_buffer = BytesIO()
Image.new("RGB", (2, 2), (32, 128, 64)).save(_image_buffer, format="PNG")
PNG = _image_buffer.getvalue()
TENANT = UUID("11111111-1111-4111-8111-111111111111")
AUDIENCE = UUID("33333333-3333-4333-8333-333333333333")
ACTOR = Principal(tenant_id=TENANT, object_id=UUID("55555555-5555-4555-8555-555555555555"))
OTHER = Principal(tenant_id=TENANT, object_id=UUID("66666666-6666-4666-8666-666666666666"))


def settings():
    return Settings(
        entra_tenant_id=TENANT,
        entra_spa_client_id="22222222-2222-4222-8222-222222222222",
        entra_api_client_id=AUDIENCE,
        azure_client_id="44444444-4444-4444-8444-444444444444",
        foundry_project_endpoint="https://test.services.ai.azure.com/api/projects/test",
        foundry_agent_name="inspection",
        foundry_agent_version="1",
        cosmos_endpoint="https://test.documents.azure.com",
        storage_account_url="https://test.blob.core.windows.net",
        sim_bridge_endpoint="https://sim.example.test",
        web_dist=ROOT / "test-results" / "nonexistent-web-build",
    )


def document():
    return json.loads((ROOT / "examples" / "inspection-cell.json").read_text(encoding="utf-8"))


class MemoryStore:
    """Test-only; this module is excluded from production container images."""

    def __init__(self):
        self.items = {}
        self.lock = threading.RLock()
        self.version = 0

    def _get(self, owner, kind, key):
        with self.lock:
            item = self.items.get((owner, kind, str(key)))
            return item.model_copy(deep=True) if item else None

    def _put(self, owner, kind, key, record, etag):
        with self.lock:
            current = self._get(owner, kind, key)
            if (current is None and etag is not None) or (
                current is not None and current.etag != etag
            ):
                raise Problem(409, "revision_conflict", "Concurrent test write.")
            self.version += 1
            stored = Stored(value=record.model_copy(deep=True), etag=str(self.version))
            self.items[(owner, kind, str(key))] = stored
            return stored.model_copy(deep=True)

    def get_environment(self, owner, environment_id):
        return self._get(owner, "environment", environment_id)

    def put_environment(self, owner, record, etag):
        return self._put(owner, "environment", record.environment_id, record, etag)

    def list_environments(self, owner):
        return [
            value.value.model_copy(deep=True)
            for key, value in self.items.items()
            if key[0] == owner and key[1] == "environment"
        ]

    def get_run(self, owner, run_id):
        return self._get(owner, "run", run_id)

    def put_run(self, owner, record, etag):
        return self._put(owner, "run", record.id, record, etag)

    def list_runs(self, owner):
        return [
            value.value.model_copy(deep=True)
            for key, value in self.items.items()
            if key[0] == owner and key[1] == "run"
        ]


class MemoryArtifacts:
    def __init__(self):
        self.items = {}

    def put(self, name, image):
        self.items[name] = image

    def get(self, name):
        return self.items[name]


class TestPlanner:
    __test__ = False

    def __init__(self):
        self.calls = 0
        self.error = None
        self.after_inspect = None

    def inspect(self, instruction, observation):
        self.calls += 1
        if self.error:
            raise self.error
        if self.after_inspect:
            self.after_inspect()
        return Decision(
            classification="rejected", object_id=observation.object_id, summary="Test-only defect."
        ), "foundry-test-response"


class TestBridge:
    __test__ = False

    def __init__(self):
        self.owner = None
        self.environment = None
        self.epoch = uuid4()
        self.state_revision = 1
        self.dispatches = 0
        self.cancellations = 0
        self.results = {}
        self.captured_at = None
        self.dispatch_error = None
        self.status_error = None

    def status(self, owner):
        if self.status_error:
            raise self.status_error
        if self.owner != owner:
            return SimulationStatus(status="occupied")
        return SimulationStatus(
            status="ready",
            environment_id=self.environment.environment_id,
            revision=self.environment.revision,
            epoch=self.epoch,
            physics_steps=42,
        )

    def activate(self, owner, environment):
        self.owner, self.environment = owner, environment
        return Activation(
            activation_id=uuid4(),
            environment_id=environment.environment_id,
            revision=environment.revision,
        )

    def observe(self, owner, environment_id, revision, camera="inspection"):
        if owner != self.owner or self.environment is None:
            raise Problem(409, "scene_not_active", "No matching active test scene.")
        return Observation(
            observation_id=uuid4(),
            environment_id=environment_id,
            revision=revision,
            epoch=self.epoch,
            state_revision=self.state_revision,
            captured_at=self.captured_at or utcnow(),
            physics_steps=42,
            object_id="part-001",
            object_position=tuple(
                next(
                    station["position_m"]
                    for station in self.environment.document["stations"]
                    if station["id"] == self.environment.document["workflow"]["source_station"]
                )
            ),
            camera=camera,
            image_base64=base64.b64encode(PNG).decode(),
        )

    def dispatch(self, owner, command):
        self.dispatches += 1
        self.results[(owner, command.command_id)] = Execution(
            command_id=command.command_id, status="queued"
        )
        if self.dispatch_error:
            raise self.dispatch_error
        return self.results[(owner, command.command_id)]

    def command(self, owner, command_id):
        if (owner, command_id) not in self.results:
            raise Problem(404, "command_missing", "No test command.")
        return self.results[(owner, command_id)]

    def cancel(self, owner, command_id):
        self.cancellations += 1
        result = self.command(owner, command_id).model_copy(update={"status": "cancelling"})
        self.results[(owner, command_id)] = result
        return result


def service():
    return FactoryService(MemoryStore(), MemoryArtifacts(), TestPlanner(), TestBridge())
