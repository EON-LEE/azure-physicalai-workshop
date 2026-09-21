"""Run a bounded, operator-authorized reference presentation entirely inside Azure."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from apps.api.errors import Problem
from apps.api.main import azure_service
from apps.api.models import TERMINAL, Principal, StartRun
from apps.api.public_demo import is_reference_scene
from apps.api.settings import Settings


def run_cycles(
    service,
    actor: Principal,
    environment_id: str,
    revision: str,
    presentation_id: str,
    cycles: int,
    maximum_seconds: int,
) -> dict:
    if not 1 <= cycles <= 100 or not 30 <= maximum_seconds <= 21600:
        raise ValueError("Presentation cycles and duration must be explicitly bounded.")
    environment = service.environment(actor, environment_id).value
    if environment.revision != revision or not is_reference_scene(environment.document):
        raise ValueError("Only the explicitly approved synthetic reference revision can repeat.")
    started = time.monotonic()
    report = {
        "kind": "actual_reference_presentation",
        "started_at": datetime.now(UTC).isoformat(),
        "presentation_id": presentation_id,
        "cycles": [],
        "anonymous_control": False,
    }
    failures = 0
    for index in range(cycles):
        if time.monotonic() - started >= maximum_seconds:
            report["stop_reason"] = "presentation_time_limit"
            break
        run_id = uuid5(
            NAMESPACE_URL, f"physicalai-presentation:{presentation_id}:{revision}:{index}"
        )
        previous = service.store.get_run(actor.owner_key, run_id)
        if previous is not None:
            result = service.get_run(actor, run_id)
            if result.status not in TERMINAL:
                raise Problem(
                    409, "previous_cycle_active", "Reconcile the prior presentation cycle."
                )
            report["cycles"].append({"id": str(run_id), "status": result.status, "reused": True})
            continue
        service.activate(actor, environment_id, revision)
        ready_deadline = min(started + maximum_seconds, time.monotonic() + 300)
        while time.monotonic() < ready_deadline:
            status = service.runtime(actor)
            if status.status == "ready" and status.revision == revision:
                break
            if status.status == "unavailable":
                raise Problem(
                    503, "simulator_unavailable", status.message or "Simulator unavailable."
                )
            time.sleep(1)
        else:
            raise TimeoutError(
                "Reference scene did not become ready within the presentation limit."
            )
        result = service.start(
            actor,
            StartRun(
                request_id=run_id,
                environment_id=environment_id,
                revision=revision,
                instruction=(
                    "Inspect the synthetic part and sort it using the configured "
                    "accepted or rejected station."
                ),
            ),
        )
        if result.status == "awaiting_approval" and result.plan is not None:
            if time.monotonic() >= started + maximum_seconds:
                result = service.cancel(actor, run_id)
                report["stop_reason"] = "presentation_time_limit"
            else:
                # Approval is the operator's bounded authorization, never an anonymous request.
                result = service.approve(actor, run_id, result.plan.model_response_id)
        cycle_deadline = min(started + maximum_seconds, time.monotonic() + 45)
        while result.status not in TERMINAL and time.monotonic() < cycle_deadline:
            time.sleep(0.5)
            result = service.get_run(actor, run_id)
        if result.status not in TERMINAL:
            result = service.cancel(actor, run_id)
            raise TimeoutError(
                "A presentation cycle did not terminate; cancellation was requested."
            )
        entry = {
            "id": str(run_id),
            "status": result.status,
            "model_response_id": result.plan.model_response_id if result.plan else None,
            "execution": result.execution.model_dump(mode="json") if result.execution else None,
        }
        report["cycles"].append(entry)
        print(json.dumps({"cycle": index + 1, **entry}), flush=True)
        if result.status != "succeeded":
            failures += 1
            if failures >= 3:
                report["stop_reason"] = "repeated_physical_failures"
                break
        time.sleep(2)
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["successes"] = sum(item["status"] == "succeeded" for item in report["cycles"])
    return report


def main() -> None:
    if os.environ.get("ALLOW_REFERENCE_PRESENTATION") != "true":
        raise RuntimeError(
            "Explicit operator authorization is required for a repeating presentation."
        )
    settings = Settings()
    if not settings.public_demo_publish_live or settings.public_demo_owner_id is None:
        raise ValueError("A pinned public reference publication is required.")
    actor = Principal(tenant_id=settings.entra_tenant_id, object_id=settings.public_demo_owner_id)
    service, resources = azure_service(settings)
    try:
        report = run_cycles(
            service,
            actor,
            settings.public_demo_environment_id,
            settings.public_demo_revision,
            os.environ["PRESENTATION_ID"],
            int(os.environ.get("PRESENTATION_CYCLES", "20")),
            int(os.environ.get("PRESENTATION_MAX_SECONDS", "1800")),
        )
        print(json.dumps(report), flush=True)
        if report["successes"] != len(report["cycles"]) or not report["cycles"]:
            raise SystemExit(2)
    finally:
        for resource in resources:
            resource.close()


if __name__ == "__main__":
    main()
