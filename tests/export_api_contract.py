"""Export actual HTTP handler responses using explicitly test-only service dependencies."""

import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from runtime_support import ACTOR, document, service, settings  # noqa: E402

from apps.api.main import create_app  # noqa: E402
from apps.api.models import Execution, utcnow  # noqa: E402


class TestIdentity:
    def user(self, authorization):
        return ACTOR


def main():
    backend = service()
    cases = []
    with TestClient(create_app(settings(), backend, TestIdentity())) as client:

        def record(schema, response):
            response.raise_for_status()
            value = response.json()
            cases.append({"schema": schema, "value": value})
            return value

        record("configSchema", client.get("/api/config"))
        record("templatesSchema", client.get("/api/environment-templates"))
        record("jsonSchemaSchema", client.get("/api/environment-schema"))
        environment = record(
            "environmentSchema",
            client.post(
                "/api/environments",
                json={
                    "document_json": json.dumps(document()),
                    "expected_revision": None,
                },
            ),
        )
        record("environmentsSchema", client.get("/api/environments"))
        record(
            "activationSchema",
            client.post(
                f"/api/environments/{environment['environment_id']}/activate",
                json={"revision": environment["revision"]},
            ),
        )
        record("runtimeSchema", client.get("/api/runtime"))
        run = record(
            "runSchema",
            client.post(
                "/api/runs",
                json={
                    "request_id": str(uuid4()),
                    "environment_id": environment["environment_id"],
                    "revision": environment["revision"],
                    "instruction": "TEST ONLY: inspect this part.",
                },
            ),
        )
        record(
            "runSchema",
            client.post(
                f"/api/runs/{run['id']}/approve",
                json={
                    "plan_response_id": run["plan"]["model_response_id"],
                },
            ),
        )
        stored = next(iter(backend.bridge.results))
        backend.bridge.results[stored] = Execution(
            command_id=stored[1],
            status="succeeded",
            final_position=tuple(document()["stations"][3]["position_m"]),
            completed_at=utcnow(),
        )
        record("runSchema", client.get(f"/api/runs/{run['id']}"))
        record("runsSchema", client.get("/api/runs"))
    output = ROOT / "test-results" / "api-contract.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "scope": "TEST-ONLY API/web wire contract; Azure/GPU execution is not verified",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(cases)} real HTTP response shapes using test-only adapters.")


if __name__ == "__main__":
    main()
