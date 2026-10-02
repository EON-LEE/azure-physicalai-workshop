"""Opt-in live Azure evidence collection; a smoke report is never a release pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import AzureCliCredential, CredentialUnavailableError
from PIL import Image, UnidentifiedImageError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator

from apps.api.models import Activation, EnvironmentRecord, Execution, Plan, SimulationStatus
from apps.api.service import content_hash
from contracts.validate_environment import SCHEMA, parse_document, validate_environment

ROOT = Path(__file__).resolve().parents[1]
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Source = Literal["actual", "fixture", "local"]
Status = Literal["passed", "failed", "blocked", "skipped", "cancelled", "planned"]
Nonblank = Annotated[str, Field(min_length=1, pattern=r"\S")]


def now() -> datetime:
    return datetime.now(UTC)


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.azurecontainerapps\.io", parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an HTTPS Azure Container Apps origin, without credentials or paths.")
    return f"https://{parsed.hostname}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelProvenance(StrictModel):
    deployment: Nonblank
    name: Nonblank
    version: Nonblank


class Provenance(StrictModel):
    """An operator-supplied deployment snapshot, not an inference execution claim."""

    candidate_commit: SHA
    endpoint: str
    image: str = Field(pattern=r"^[a-z0-9.-]+/[a-zA-Z0-9._/-]+@sha256:[a-f0-9]{64}$")
    model: ModelProvenance
    deployment_config_sha256: Digest
    observed_at: AwareDatetime
    source: Source

    _endpoint = field_validator("endpoint")(endpoint)


class Case(StrictModel):
    id: Nonblank
    name: Nonblank
    status: Status
    source: Source
    started_at: AwareDatetime
    finished_at: AwareDatetime
    evidence: dict = Field(default_factory=dict)
    failure: str | None = None


class Gate(StrictModel):
    id: Literal["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    status: Status
    case_ids: list[Nonblank] = Field(min_length=1)


class Report(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    scope: Literal["smoke", "physical", "release_gate"]
    candidate_commit: SHA
    source: Source
    started_at: AwareDatetime
    finished_at: AwareDatetime
    status: Status
    provenance: Provenance
    provenance_origin: Literal["operator_deployment_snapshot"] = "operator_deployment_snapshot"
    cases: list[Case] = Field(min_length=1)
    gates: list[Gate] = Field(default_factory=list)
    release_ready: Literal[False] = False


class Options(StrictModel):
    endpoint: str
    tenant_id: UUID
    api_client_id: UUID
    exercise: bool = False
    episodes: int = Field(default=1, ge=1, le=100)
    request_timeout: float = Field(default=30, gt=0, le=120)
    poll_timeout: float = Field(default=60, gt=0, le=300)
    poll_interval: float = Field(default=1, ge=1, le=10)

    _endpoint = field_validator("endpoint")(endpoint)

    @field_validator("tenant_id", "api_client_id")
    @classmethod
    def nonzero(cls, value: UUID) -> UUID:
        if not value.int:
            raise ValueError("A nonzero Entra application/tenant ID is required.")
        return value


class Token(Protocol):
    token: str


class Credential(Protocol):
    def get_token(self, *scopes: str) -> Token: ...


class CheckError(Exception):
    def __init__(self, code: str, *, blocked: bool = False):
        super().__init__(code)
        self.code, self.blocked = code, blocked


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CheckError(code)


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def resolve_commit(explicit: str | None) -> str:
    value = explicit if explicit is not None else current_commit()
    require(re.fullmatch(r"[a-f0-9]{40}", value) is not None, "invalid_candidate_commit")
    return value


def fresh(timestamp: datetime, reference: datetime, seconds: float) -> bool:
    return 0 <= (reference - timestamp).total_seconds() <= seconds


def write_report(path: Path, value: BaseModel | dict) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Harness:
    def __init__(
        self,
        options: Options,
        provenance: Provenance,
        client: httpx.Client,
        credential: Credential,
        *,
        source: Source,
        clock: Callable[[], datetime] = now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.options, self.provenance = options, provenance
        self.client, self.credential, self.source = client, credential, source
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.cases: list[Case] = []
        self.last_run_id: str | None = None
        self.last_run_terminal = True
        self.seen_commands: set[str] = set()
        self.seen_observations: set[str] = set()
        self.partial_evidence: dict = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
        expected: int = 200,
        body: dict | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        if method != "GET" and not self.options.exercise:
            raise CheckError("exercise_consent_required")
        headers = {}
        if auth:
            try:
                # AzureCliCredential converts /.default to the Azure CLI --resource audience.
                token = self.credential.get_token(
                    f"api://{self.options.api_client_id}/.default"
                ).token
            except (ClientAuthenticationError, CredentialUnavailableError):
                raise CheckError("azure_cli_authentication_failed", blocked=True) from None
            require(bool(token.strip()), "empty_access_token")
            headers["Authorization"] = f"Bearer {token}"
        limit = min(timeout or self.options.request_timeout, self.options.request_timeout)
        deadline = self.monotonic() + limit
        try:
            with self.client.stream(
                method,
                self.options.endpoint + path,
                headers=headers,
                json=body,
                timeout=limit,
                follow_redirects=False,
            ) as response:
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    require(len(chunks) <= 8 * 1024 * 1024, "response_too_large")
                    require(self.monotonic() <= deadline, "response_deadline_exceeded")
                result = httpx.Response(
                    response.status_code, headers=response.headers, content=bytes(chunks)
                )
        except httpx.TimeoutException:
            raise CheckError("http_timeout", blocked=True) from None
        except httpx.RequestError:
            raise CheckError("http_transport_failed", blocked=True) from None
        if result.status_code != expected:
            # Do not retain error bodies, request headers, tokens, or SDK exception strings.
            raise CheckError(
                f"http_status_{result.status_code}_expected_{expected}",
                blocked=result.status_code in (401, 403, 409, 429, 503),
            )
        return result

    def get_json(self, path: str, **kwargs) -> dict:
        result = self.request("GET", path, **kwargs).json()
        require(isinstance(result, dict), "expected_json_object")
        return result

    def case(self, case_id: str, name: str, action: Callable[[], dict]) -> dict | None:
        started = self.clock()
        self.partial_evidence = {}
        try:
            evidence = action()
            require(bool(evidence), "empty_evidence")
        except CheckError as exc:
            self.cases.append(
                Case(
                    id=case_id,
                    name=name,
                    source=self.source,
                    started_at=started,
                    finished_at=self.clock(),
                    status="blocked" if exc.blocked else "failed",
                    failure=exc.code,
                    evidence=self.partial_evidence,
                )
            )
            return None
        except (ValidationError, ValueError, KeyError, TypeError, IndexError):
            self.cases.append(
                Case(
                    id=case_id,
                    name=name,
                    source=self.source,
                    started_at=started,
                    finished_at=self.clock(),
                    status="failed",
                    failure="invalid_response_contract",
                    evidence=self.partial_evidence,
                )
            )
            return None
        self.cases.append(
            Case(
                id=case_id,
                name=name,
                source=self.source,
                started_at=started,
                finished_at=self.clock(),
                status="passed",
                evidence=evidence,
            )
        )
        return evidence

    def bootstrap(self) -> dict:
        config = self.get_json("/api/config", auth=False)
        require(
            config["api_version"] == "v1" and config["deployment"] == "azure",
            "incorrect_api_deployment",
        )
        auth = config["auth"]
        require(UUID(auth["tenant_id"]) == self.options.tenant_id, "tenant_mismatch")
        require(UUID(auth["client_id"]).int != 0, "invalid_spa_client")
        require(
            auth["scope"] == f"api://{self.options.api_client_id}/access_as_user",
            "api_scope_mismatch",
        )
        return {
            "api_version": "v1",
            "tenant_id": auth["tenant_id"],
            "spa_client_id": auth["client_id"],
            "scope": auth["scope"],
        }

    def unauthenticated(self) -> dict:
        response = self.request("GET", "/api/runtime", auth=False, expected=401)
        error = response.json()["error"]
        require(
            all(isinstance(error[k], str) and error[k] for k in ("code", "message")),
            "invalid_401_error",
        )
        require(isinstance(error["details"], list), "invalid_401_details")
        require(
            response.headers.get("www-authenticate", "").lower() == "bearer",
            "missing_bearer_challenge",
        )
        return {"status_code": 401, "bearer_challenge": True}

    def schema(self) -> dict:
        schema = self.get_json("/api/environment-schema")
        require(schema == SCHEMA, "deployed_schema_mismatch")
        return {"schema_sha256": content_hash(schema)}

    def templates(self) -> dict:
        items = self.get_json("/api/environment-templates")["items"]
        require(isinstance(items, list) and bool(items), "empty_templates")
        for item in items:
            require(
                isinstance(item["name"], str) and bool(item["name"].strip()), "unnamed_template"
            )
            require(not validate_environment(item["document"]), "invalid_template")
        return {
            "count": len(items),
            "document_sha256": [content_hash(i["document"]) for i in items],
        }

    def environments(self) -> dict:
        items = self.get_json("/api/environments")["items"]
        require(isinstance(items, list), "invalid_environment_list")
        records = [EnvironmentRecord.model_validate(item) for item in items]
        for record in records:
            require(not validate_environment(record.document), "invalid_saved_environment")
            require(
                record.environment_id == record.document["environment_id"],
                "saved_environment_id_mismatch",
            )
            require(record.revision == content_hash(record.document), "saved_revision_mismatch")
        return {"count": len(records), "revisions": [r.revision for r in records]}

    def runtime(self, **kwargs) -> dict:
        result = self.get_json("/api/runtime", **kwargs)
        require(result["deployment"] == "azure", "non_azure_runtime")
        require(
            result["agent"] == {"provider": "microsoft_foundry", "configured": True},
            "agent_not_configured",
        )
        require(result["storage"] == {"provider": "azure_cosmos_blob"}, "incorrect_storage")
        require(result["release_ready"] is False, "unexpected_release_claim")
        simulation = SimulationStatus.model_validate(result["simulation"])
        if simulation.status == "ready":
            require(
                all(
                    value is not None
                    for value in (
                        simulation.environment_id,
                        simulation.revision,
                        simulation.epoch,
                        simulation.physics_steps,
                    )
                ),
                "ready_runtime_missing_scene",
            )
            require(simulation.physics_steps >= 0, "invalid_physics_steps")
        # A dependency's free-form message is deliberately not persisted.
        return {
            "deployment": "azure",
            "simulation": simulation.model_dump(mode="json", exclude={"message"}),
            "agent_configured_only": True,
            "foundry_execution_verified": False,
        }

    def gpu_ready(self, runtime: dict) -> dict:
        simulation = runtime["simulation"]
        if simulation["status"] != "ready":
            raise CheckError(f"gpu_{simulation['status']}", blocked=True)
        return {"epoch": simulation["epoch"], "physics_steps": simulation["physics_steps"]}

    def poll(self, read: Callable[[float], dict], done: Callable[[dict], bool]) -> dict:
        deadline = self.monotonic() + self.options.poll_timeout
        while self.monotonic() < deadline:
            result = read(deadline - self.monotonic())
            if self.monotonic() <= deadline and done(result):
                return result
            self.sleep(min(self.options.poll_interval, max(0, deadline - self.monotonic())))
        raise CheckError("poll_deadline_exceeded", blocked=True)

    def frame(self, environment_id: str, revision: str, max_age_ms: int) -> dict:
        response = self.request(
            "GET", f"/api/environments/{environment_id}/frame?revision={revision}&camera=overview"
        )
        require(response.headers.get("content-type") == "image/png", "camera_not_png")
        require(response.headers.get("cache-control") == "no-store", "camera_cached")
        captured = datetime.fromisoformat(response.headers["x-captured-at"])
        require(
            captured.tzinfo is not None and fresh(captured, self.clock(), max_age_ms / 1000),
            "stale_camera",
        )
        frame_id = str(UUID(response.headers["x-frame-id"]))
        steps = int(response.headers["x-physics-steps"])
        require(steps > 0, "no_physics_steps")
        try:
            with Image.open(BytesIO(response.content)) as image:
                require(
                    image.format == "PNG" and image.width * image.height <= 8_000_000,
                    "invalid_camera_png",
                )
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
            raise CheckError("invalid_camera_png") from None
        return {
            "frame_id": frame_id,
            "captured_at": captured.isoformat(),
            "physics_steps": steps,
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }

    def episode(self, raw_document: str) -> dict:
        self.last_run_id, self.last_run_terminal = None, True
        document = parse_document(raw_document)
        require(not validate_environment(document), "invalid_customer_environment")
        require(document["execution"]["mode"] == "live", "replay_not_live")
        environment_id, revision = document["environment_id"], content_hash(document)
        self.partial_evidence = {"environment_id": environment_id, "revision": revision}
        existing = self.get_json("/api/environments")["items"]
        records = [EnvironmentRecord.model_validate(item) for item in existing]
        matching = [r for r in records if r.environment_id == environment_id]
        require(len(matching) <= 1, "duplicate_saved_environment")
        saved = EnvironmentRecord.model_validate(
            self.request(
                "POST",
                "/api/environments",
                expected=201,
                body={
                    "document_json": raw_document,
                    "expected_revision": matching[0].revision if matching else None,
                },
            ).json()
        )
        require(
            saved.environment_id == environment_id
            and saved.revision == revision
            and saved.document == document,
            "saved_environment_mismatch",
        )
        activation = Activation.model_validate(
            self.request(
                "POST",
                f"/api/environments/{environment_id}/activate",
                expected=202,
                body={"revision": revision},
            ).json()
        )
        require(
            activation.environment_id == environment_id and activation.revision == revision,
            "activation_mismatch",
        )
        self.partial_evidence["activation_id"] = str(activation.activation_id)
        runtime = self.poll(
            lambda remaining: self.runtime(timeout=remaining),
            lambda r: (
                r["simulation"]["status"] == "ready"
                and r["simulation"]["environment_id"] == environment_id
                and r["simulation"]["revision"] == revision
            ),
        )
        epoch = runtime["simulation"]["epoch"]
        max_age = min(document["execution"]["max_observation_age_ms"], 2000)
        before = self.frame(environment_id, revision, max_age)
        self.partial_evidence.update(epoch=epoch, before=before)
        run_id = str(uuid4())
        # The server may create the run even if its response is lost.
        self.last_run_id, self.last_run_terminal = run_id, False
        self.partial_evidence["run_id"] = run_id
        run = self.request(
            "POST",
            "/api/runs",
            expected=201,
            body={
                "request_id": run_id,
                "environment_id": environment_id,
                "revision": revision,
                "instruction": (
                    "Inspect this part and sort it into the correct accepted or rejected tray."
                ),
            },
        ).json()
        self.check_run(run, run_id, environment_id, revision)
        require(run["status"] == "awaiting_approval", "planning_not_awaiting_approval")
        plan = Plan.model_validate(run["plan"])
        self.partial_evidence.update(
            model_response_id=plan.model_response_id,
            observation_id=str(plan.observation_id),
        )
        require(
            bool(plan.model_response_id.strip()) and str(plan.epoch) == epoch,
            "missing_model_execution_or_wrong_epoch",
        )
        require(plan.expires_at > self.clock(), "expired_plan")
        require(str(plan.observation_id) not in self.seen_observations, "reused_observation")
        self.seen_observations.add(str(plan.observation_id))
        target_id = document["workflow"][
            "accept_station" if plan.classification == "accepted" else "reject_station"
        ]
        require(plan.target_station_id == target_id, "incorrect_target_station")
        target = next(s["position_m"] for s in document["stations"] if s["id"] == target_id)
        approved_at = self.clock()
        self.partial_evidence["approved_at"] = approved_at.isoformat()
        approved = self.request(
            "POST", f"/api/runs/{run_id}/approve", body={"plan_response_id": plan.model_response_id}
        ).json()
        self.check_run(approved, run_id, environment_id, revision)
        dispatched = Execution.model_validate(approved["execution"])
        self.partial_evidence["command_id"] = str(dispatched.command_id)
        require(str(dispatched.command_id) not in self.seen_commands, "reused_command")
        self.seen_commands.add(str(dispatched.command_id))
        final = self.poll(
            lambda remaining: self.get_json(f"/api/runs/{run_id}", timeout=remaining),
            lambda r: r["status"] in {"succeeded", "failed", "cancelled", "timed_out"},
        )
        self.check_run(final, run_id, environment_id, revision)
        self.last_run_terminal = True
        execution = Execution.model_validate(final["execution"])
        self.partial_evidence["execution"] = execution.model_dump(mode="json", exclude={"error"})
        require(
            final["status"] == "succeeded"
            and execution.status == "succeeded"
            and final["error"] is None
            and execution.error is None,
            "motion_not_succeeded",
        )
        require(execution.command_id == dispatched.command_id, "command_changed")
        require(
            execution.completed_at is not None and execution.final_position is not None,
            "missing_completion_or_final_pose",
        )
        require(approved_at <= execution.completed_at <= self.clock(), "invalid_completion_time")
        require(
            (execution.completed_at - approved_at).total_seconds()
            <= min(document["execution"]["max_step_seconds"], 30),
            "motion_deadline_exceeded",
        )
        require(
            all(
                math.isfinite(p) and abs(p - t) <= 0.04
                for p, t in zip(execution.final_position, target, strict=True)
            ),
            "final_pose_outside_target",
        )
        after = self.frame(environment_id, revision, max_age)
        self.partial_evidence["after"] = after
        require(
            after["frame_id"] != before["frame_id"]
            and after["physics_steps"] > before["physics_steps"]
            and datetime.fromisoformat(after["captured_at"])
            > datetime.fromisoformat(before["captured_at"]),
            "no_fresh_physical_progress",
        )
        after_runtime = self.runtime()["simulation"]
        require(
            after_runtime["epoch"] == epoch
            and after_runtime["revision"] == revision
            and after_runtime["environment_id"] == environment_id,
            "scene_changed",
        )
        return {
            "run_id": run_id,
            "activation_id": str(activation.activation_id),
            "environment_id": environment_id,
            "revision": revision,
            "epoch": epoch,
            "model_response_id": plan.model_response_id,
            "observation_id": str(plan.observation_id),
            "target_station_id": target_id,
            "target_position": target,
            "approved_at": approved_at.isoformat(),
            "motion_wall_seconds": (execution.completed_at - approved_at).total_seconds(),
            "execution": execution.model_dump(mode="json"),
            "before": before,
            "after": after,
        }

    @staticmethod
    def check_run(run: dict, run_id: str, environment_id: str, revision: str) -> None:
        require(
            run["id"] == run_id
            and run["environment_id"] == environment_id
            and run["revision"] == revision,
            "run_identity_mismatch",
        )

    def cancel_outstanding(self) -> dict:
        require(self.last_run_id is not None, "no_outstanding_run")
        run_id = self.last_run_id
        self.request("POST", f"/api/runs/{run_id}/cancel")
        final = self.poll(
            lambda remaining: self.get_json(f"/api/runs/{run_id}", timeout=remaining),
            lambda r: r["status"] in {"succeeded", "failed", "cancelled", "timed_out"},
        )
        require(final["id"] == run_id, "cancel_run_identity_mismatch")
        if final.get("execution") is not None:
            execution = Execution.model_validate(final["execution"])
            require(execution.status == final["status"], "cancel_motion_not_terminal")
        self.last_run_terminal = True
        return {
            "run_id": run_id,
            "confirmed_terminal_status": final["status"],
            "purpose": "cleanup_only_not_motion_success",
        }

    def run(self, raw_document: str | None = None) -> Report:
        started = self.clock()
        config = self.case(
            "LIVE-CONFIG", "Azure public bootstrap matches requested identity", self.bootstrap
        )
        self.case("LIVE-UNAUTH-401", "Anonymous runtime access is rejected", self.unauthenticated)
        runtime = None
        if config is not None:
            self.case("LIVE-SCHEMA", "Authenticated deployed configuration schema", self.schema)
            self.case("LIVE-TEMPLATES", "Authenticated valid environment templates", self.templates)
            self.case(
                "LIVE-ENVIRONMENTS", "Authenticated persisted environment list", self.environments
            )
            runtime = self.case(
                "LIVE-RUNTIME", "Authenticated Azure runtime contract", self.runtime
            )
        baseline_ok = all(c.status == "passed" for c in self.cases)
        if self.options.exercise and baseline_ok:

            def customer() -> dict:
                require(raw_document is not None, "customer_document_required")
                document = parse_document(raw_document)
                require(not validate_environment(document), "invalid_customer_environment")
                return {
                    "raw_sha256": hashlib.sha256(raw_document.encode()).hexdigest(),
                    "revision": content_hash(document),
                }

            if self.case("LIVE-CUSTOMER-CONFIG", "Validate explicit customer input", customer):
                successes = 0
                for index in range(self.options.episodes):
                    result = self.case(
                        f"LIVE-EPISODE-{index + 1:03d}",
                        "Actual inspection and physical completion",
                        lambda: self.episode(raw_document),
                    )
                    if result is not None:
                        successes += 1
                    elif not self.last_run_terminal:
                        self.case(
                            "LIVE-CLEANUP",
                            "Confirm outstanding task termination",
                            self.cancel_outstanding,
                        )
                        break
                    elif self.last_run_id is None:
                        # Do not repeatedly hit a missing GPU or failed activation.
                        break

                def threshold() -> dict:
                    attempted = sum(c.id.startswith("LIVE-EPISODE-") for c in self.cases)
                    self.partial_evidence = {
                        "episodes": attempted,
                        "successes": successes,
                        "minimum_episodes": 20,
                        "minimum_successes": 18,
                    }
                    require(
                        attempted >= 20 and successes >= 18 and successes / attempted >= 0.9,
                        "physical_20_episode_threshold_not_met",
                    )
                    return self.partial_evidence

                self.case(
                    "LIVE-PHYSICAL-THRESHOLD",
                    "At least 18/20 physically evidenced successes",
                    threshold,
                )
        elif runtime is not None:
            self.case(
                "LIVE-GPU-READY",
                "GPU scene availability (not a motion test)",
                lambda: self.gpu_ready(runtime),
            )
        statuses = {case.status for case in self.cases}
        status = (
            "failed" if "failed" in statuses else "blocked" if "blocked" in statuses else "passed"
        )
        return Report(
            scope="physical" if self.options.exercise else "smoke",
            candidate_commit=self.provenance.candidate_commit,
            source=self.source,
            started_at=started,
            finished_at=self.clock(),
            status=status,
            provenance=self.provenance,
            cases=self.cases,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--api-client-id", required=True)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--candidate-commit", help="Full host Git SHA when WSL cannot resolve HEAD."
    )
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--poll-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    logging.getLogger("azure").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    try:
        options = Options.model_validate(
            {
                key: value
                for key, value in vars(args).items()
                if key not in {"provenance", "output", "environment", "candidate_commit"}
            }
        )
        provenance = Provenance.model_validate(
            parse_document(args.provenance.read_text(encoding="utf-8"))
        )
        require(provenance.source == "actual", "actual_deployment_snapshot_required")
        require(
            provenance.candidate_commit == resolve_commit(args.candidate_commit),
            "candidate_commit_mismatch",
        )
        require(provenance.endpoint == options.endpoint, "deployment_endpoint_mismatch")
        require(fresh(provenance.observed_at, now(), 86400), "stale_deployment_snapshot")
        require(
            options.exercise or (args.environment is None and args.episodes == 1),
            "exercise_consent_required",
        )
        require(not options.exercise or args.environment is not None, "customer_document_required")
        document = args.environment.read_text(encoding="utf-8") if args.environment else None
        with (
            AzureCliCredential(
                tenant_id=str(options.tenant_id), process_timeout=options.request_timeout
            ) as credential,
            httpx.Client(trust_env=False) as client,
        ):
            report = Harness(options, provenance, client, credential, source="actual").run(document)
        write_report(args.output, report)
    except (OSError, ValueError, ValidationError, CheckError, subprocess.SubprocessError) as exc:
        code = exc.code if isinstance(exc, CheckError) else "invalid_input_or_evidence_io"
        # Invalid preflight input cannot create a valid evidence envelope.
        print(f"Live acceptance blocked: {code}. No release pass.")
        return 2
    print(f"Live acceptance: {report.status}; {len(report.cases)} cases; not a full release.")
    return 0 if report.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
