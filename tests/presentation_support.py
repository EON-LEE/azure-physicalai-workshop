"""Explicit reference runner doubles, not live model or simulator evidence."""

import json
from datetime import timedelta
from uuid import uuid4

import runtime_support

from apps.api import presentation, public_demo
from apps.api import service as service_module
from apps.api.models import Decision, Execution, SaveEnvironment, utcnow
from scripts import run_reference_demo as runner


class Clock:
    def __init__(self, backend):
        self.backend = backend
        self.start = utcnow()
        self.value = 0.0
        self.complete = "succeeded"
        self.on_sleep = None

    def monotonic(self):
        return self.value

    def now(self):
        return self.start + timedelta(seconds=self.value)

    def sleep(self, duration):
        self.value += duration
        if self.on_sleep:
            self.on_sleep()
        for key, result in list(self.backend.bridge.results.items()):
            status = "cancelled" if result.status == "cancelling" else self.complete
            if status is None or result.status not in {"queued", "running", "cancelling"}:
                continue
            stored = self.backend.store.get_run(runtime_support.ACTOR.owner_key, result.command_id)
            target_id = stored.value.plan.target_station_id
            target = next(
                station["position_m"]
                for station in stored.value.environment_document["stations"]
                if station["id"] == target_id
            )
            self.backend.bridge.results[key] = Execution(
                command_id=result.command_id,
                status=status,
                final_position=tuple(target) if status == "succeeded" else None,
                completed_at=self.now(),
            )


def prepare(monkeypatch):
    backend = runtime_support.service()
    clock = Clock(backend)
    for module in (runtime_support, service_module, presentation, public_demo, runner):
        monkeypatch.setattr(module, "utcnow", clock.now)
    monkeypatch.setattr(runner, "time", clock)
    normal = runtime_support.document()
    defect = runtime_support.document()
    defect["environment_id"] = "reference-defect"
    defect["scene"]["seed"] = 43
    records = [
        backend.save_environment(
            runtime_support.ACTOR, SaveEnvironment(document_json=json.dumps(doc))
        )
        for doc in (normal, defect)
    ]
    settings = runtime_support.settings().model_copy(
        update={
            "public_demo_publish_live": True,
            "public_demo_owner_id": runtime_support.ACTOR.object_id,
            "public_demo_environment_id": records[0].environment_id,
            "public_demo_revision": records[0].revision,
            "public_demo_defect_environment_id": records[1].environment_id,
            "public_demo_defect_revision": records[1].revision,
            "public_demo_presentation_id": "test-presentation",
        }
    )
    activation = backend.bridge.activate

    def activate(owner, environment):
        backend.bridge.epoch = uuid4()
        return activation(owner, environment)

    backend.bridge.activate = activate
    backend.planner.wrong = False
    backend.planner.instructions = []

    def inspect(instruction, observation):
        backend.planner.calls += 1
        backend.planner.instructions.append(instruction)
        classification = (
            "accepted" if backend.bridge.environment.document["scene"]["seed"] == 42 else "rejected"
        )
        if backend.planner.wrong:
            classification = "rejected" if classification == "accepted" else "accepted"
        if backend.planner.after_inspect:
            backend.planner.after_inspect()
        return Decision(
            classification=classification,
            object_id=observation.object_id,
            summary="Explicit test-double image classification.",
        ), f"test-response-{uuid4()}"

    backend.planner.inspect = inspect
    return backend, settings, clock
