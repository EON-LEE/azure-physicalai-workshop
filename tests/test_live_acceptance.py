"""All transport/identity/physics responses here are explicit test doubles, never live evidence."""

import copy
import json
import subprocess
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from azure.core.exceptions import ClientAuthenticationError
from PIL import Image
from pydantic import ValidationError

from apps.api.service import content_hash
from contracts.validate_environment import SCHEMA
from scripts import live_acceptance as live

COMMIT = "a" * 40
TENANT = "11111111-1111-4111-8111-111111111111"
API = "33333333-3333-4333-8333-333333333333"
ENDPOINT = "https://unit-test.region.azurecontainerapps.io"
EPOCH = str(uuid4())
START = datetime(2026, 9, 20, 12, tzinfo=UTC)
DOCUMENT = (live.ROOT / "examples" / "inspection-cell.json").read_text(encoding="utf-8")


class Clock:
    def __init__(self):
        self.elapsed = 0

    def now(self):
        return START + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


class CredentialDouble:
    def __init__(self, fail=False):
        self.scopes = []
        self.fail = fail

    def get_token(self, scope):
        self.scopes.append(scope)
        if self.fail:
            raise ClientAuthenticationError("SECRET SDK TOKEN MUST NEVER BE REPORTED")
        return SimpleNamespace(token="TEST-ONLY-TOKEN")


def provenance(clock=None):
    return live.Provenance(
        candidate_commit=COMMIT,
        endpoint=ENDPOINT,
        image=f"unit.azurecr.io/api@sha256:{'b' * 64}",
        model={"deployment": "unit", "name": "unit-model", "version": "1"},
        deployment_config_sha256="c" * 64,
        observed_at=(clock or Clock()).now(),
        source="fixture",
    )


class APIDouble:
    def __init__(self, clock):
        self.clock = clock
        self.requests = []
        self.override = {}
        self.document = json.loads(DOCUMENT)
        self.saved = None
        self.runs = {}
        self.steps = 100
        self.simulation_status = "ready"
        self.final_status = "succeeded"
        self.final_position = [0.5, -0.4, 0.2]
        self.missing_execution_field = None
        self.frozen_camera = False
        self.stale_camera = False
        self.reuse_command = False
        self.command = str(uuid4())
        self.frame_id = str(uuid4())
        image = BytesIO()
        Image.new("RGB", (4, 4), (0, 50, 100)).save(image, format="PNG")
        self.png = image.getvalue()

    def response(self, request):
        self.requests.append(request)
        path = request.url.path
        key = (request.method, path)
        if key in self.override:
            result = self.override[key]
            return result(request) if callable(result) else result
        if path == "/api/config":
            return httpx.Response(
                200,
                json={
                    "api_version": "v1",
                    "deployment": "azure",
                    "auth": {
                        "tenant_id": TENANT,
                        "client_id": TENANT,
                        "scope": f"api://{API}/access_as_user",
                    },
                },
            )
        if "authorization" not in request.headers:
            return httpx.Response(
                401,
                headers={"WWW-Authenticate": "Bearer"},
                json={
                    "error": {
                        "code": "authentication_required",
                        "message": "Sign in.",
                        "details": [],
                    },
                },
            )
        assert request.headers["authorization"] == "Bearer TEST-ONLY-TOKEN"
        if path == "/api/environment-schema":
            return httpx.Response(200, json=SCHEMA)
        if path == "/api/environment-templates":
            return httpx.Response(
                200,
                json={
                    "items": [{"name": "Test double template", "document": self.document}],
                },
            )
        if path == "/api/environments" and request.method == "GET":
            return httpx.Response(200, json={"items": [self.saved] if self.saved else []})
        if path == "/api/environments":
            body = json.loads(request.content)
            assert body["document_json"] == DOCUMENT
            self.saved = {
                "environment_id": self.document["environment_id"],
                "display_name": self.document["display_name"],
                "document": self.document,
                "revision": content_hash(self.document),
                "created_at": self.clock.now().isoformat(),
                "updated_at": self.clock.now().isoformat(),
            }
            return httpx.Response(201, json=self.saved)
        if path.endswith("/activate"):
            return httpx.Response(
                202,
                json={
                    "activation_id": str(uuid4()),
                    "status": "loading",
                    "environment_id": self.document["environment_id"],
                    "revision": content_hash(self.document),
                },
            )
        if path == "/api/runtime":
            return httpx.Response(
                200,
                json={
                    "deployment": "azure",
                    "agent": {"provider": "microsoft_foundry", "configured": True},
                    "storage": {"provider": "azure_cosmos_blob"},
                    "release_ready": False,
                    "simulation": {
                        "backend": "isaac_sim",
                        "status": self.simulation_status,
                        "environment_id": self.document["environment_id"],
                        "revision": content_hash(self.document),
                        "epoch": EPOCH,
                        "physics_steps": self.steps,
                        "message": None,
                    },
                },
            )
        if path.endswith("/frame"):
            if not self.frozen_camera:
                self.steps += 1
                self.frame_id = str(uuid4())
                self.clock.sleep(0.1)
            captured = self.clock.now() - timedelta(seconds=10 if self.stale_camera else 0)
            return httpx.Response(
                200,
                content=self.png,
                headers={
                    "Content-Type": "image/png",
                    "Cache-Control": "no-store",
                    "X-Frame-Id": self.frame_id,
                    "X-Physics-Steps": str(self.steps),
                    "X-Captured-At": captured.isoformat(),
                },
            )
        if path == "/api/runs":
            body = json.loads(request.content)
            run_id = body["request_id"]
            self.runs[run_id] = {
                "id": run_id,
                "environment_id": body["environment_id"],
                "revision": body["revision"],
                "status": "awaiting_approval",
                "error": None,
                "plan": {
                    "classification": "rejected",
                    "object_id": "part-001",
                    "summary": "Test double",
                    "target_station_id": "rejected",
                    "observation_id": str(uuid4()),
                    "epoch": EPOCH,
                    "state_revision": 1,
                    "model_response_id": f"test-{run_id}",
                    "expires_at": (self.clock.now() + timedelta(seconds=300)).isoformat(),
                },
                "execution": None,
            }
            return httpx.Response(201, json=self.runs[run_id])
        run_id = path.split("/")[3]
        run = self.runs[run_id]
        if path.endswith("/approve"):
            command_id = self.command if self.reuse_command else str(uuid4())
            run["status"] = "running"
            run["execution"] = {"command_id": command_id, "status": "running"}
        elif path.endswith("/cancel"):
            run["status"] = "cancelled"
            if run["execution"] is not None:
                run["execution"]["status"] = "cancelled"
        elif run["status"] == "running":
            run["status"] = self.final_status
            run["execution"].update(
                status=self.final_status,
                final_position=self.final_position,
                completed_at=self.clock.now().isoformat(),
            )
            if self.missing_execution_field:
                run["execution"].pop(self.missing_execution_field, None)
        return httpx.Response(200, json=run)


def harness(*, exercise=False, episodes=1):
    clock = Clock()
    api = APIDouble(clock)
    credential = CredentialDouble()
    client = httpx.Client(transport=httpx.MockTransport(api.response))
    runner = live.Harness(
        live.Options(
            endpoint=ENDPOINT,
            tenant_id=TENANT,
            api_client_id=API,
            exercise=exercise,
            episodes=episodes,
            poll_timeout=3,
        ),
        provenance(clock),
        client,
        credential,
        source="fixture",
        clock=clock.now,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return runner, api, credential


def failures(report):
    return [case.failure for case in report.cases if case.status != "passed"]


def test_smoke_is_read_only_and_not_release():
    runner, api, credential = harness()
    report = runner.run()
    assert report.status == "passed"
    assert report.scope == "smoke" and not report.release_ready and not report.gates
    assert all(request.method == "GET" for request in api.requests)
    assert all(case.source == "fixture" for case in report.cases)
    assert set(credential.scopes) == {f"api://{API}/.default"}
    assert "TEST-ONLY-TOKEN" not in report.model_dump_json()
    assert report.cases[-2].evidence["foundry_execution_verified"] is False


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/environments"),
        ("POST", "/api/environments/reference-cell/activate"),
        ("POST", "/api/runs"),
        ("POST", f"/api/runs/{uuid4()}/approve"),
        ("POST", f"/api/runs/{uuid4()}/cancel"),
    ],
)
def test_all_mutations_require_exercise(method, path):
    runner, api, _ = harness()
    with pytest.raises(live.CheckError, match="exercise_consent_required"):
        runner.request(method, path)
    assert not api.requests


@pytest.mark.parametrize(
    "url",
    [
        "http://unit.azurecontainerapps.io",
        "https://localhost",
        "https://127.0.0.1",
        "https://evil.test",
        "https://app.azurecontainerapps.io.evil.test",
        "https://user:password@unit.azurecontainerapps.io",
        "https://unit.azurecontainerapps.io?secret=value",
        "https://unit.azurecontainerapps.io/api",
        "https://unit.azurecontainerapps.io#fragment",
        "https://unit.azurecontainerapps.io:8080",
    ],
)
def test_reject_unsafe_or_non_aca_origin(url):
    with pytest.raises((ValueError, ValidationError)):
        live.Options(endpoint=url, tenant_id=TENANT, api_client_id=API)


def test_bootstrap_identity_mismatch_never_sends_token():
    runner, api, credential = harness()
    api.override[("GET", "/api/config")] = httpx.Response(
        200,
        json={
            "api_version": "v1",
            "deployment": "azure",
            "auth": {"tenant_id": API, "client_id": TENANT, "scope": "wrong"},
        },
    )
    report = runner.run()
    assert "tenant_mismatch" in failures(report)
    assert not credential.scopes


def test_invalid_api_scope_blocks_token():
    runner, api, credential = harness()
    api.override[("GET", "/api/config")] = httpx.Response(
        200,
        json={
            "api_version": "v1",
            "deployment": "azure",
            "auth": {
                "tenant_id": TENANT,
                "client_id": TENANT,
                "scope": "api://wrong/access_as_user",
            },
        },
    )
    assert "api_scope_mismatch" in failures(runner.run())
    assert not credential.scopes


def test_auth_error_is_blocked_and_redacted():
    runner, _, credential = harness()
    credential.fail = True
    report = runner.run()
    assert report.status == "blocked"
    assert "azure_cli_authentication_failed" in failures(report)
    assert "SECRET" not in report.model_dump_json()


@pytest.mark.parametrize("status", [301, 401, 403, 409, 422, 429, 500, 503])
def test_http_error_never_passes_or_leaks_body(status):
    runner, api, _ = harness()
    api.override[("GET", "/api/environment-schema")] = httpx.Response(
        status, json={"secret": "DO NOT LOG"}, headers={"Location": "https://evil.test"}
    )
    report = runner.run()
    assert f"http_status_{status}_expected_200" in failures(report)
    assert "DO NOT LOG" not in report.model_dump_json()
    assert all(
        request.url.host == "unit-test.region.azurecontainerapps.io" for request in api.requests
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("/api/config", {}),
        ("/api/environment-schema", {}),
        ("/api/environment-templates", {"items": []}),
        ("/api/environments", {}),
        ("/api/runtime", {"deployment": "azure"}),
    ],
)
def test_missing_response_fields_and_zero_evidence_fail(path, value):
    runner, api, _ = harness()
    api.override[("GET", path)] = lambda req: (
        httpx.Response(200, json=value)
        if "authorization" in req.headers or path == "/api/config"
        else httpx.Response(
            401,
            headers={"WWW-Authenticate": "Bearer"},
            json={
                "error": {"code": "auth", "message": "auth", "details": []},
            },
        )
    )
    assert runner.run().status == "failed"


def test_no_evidence_action_fails():
    runner, _, _ = harness()
    assert runner.case("EMPTY", "Empty result", lambda: {}) is None
    assert runner.cases[-1].failure == "empty_evidence"


def test_http_timeout_is_bounded_and_redacted():
    runner, api, _ = harness()

    def timeout(request):
        assert request.extensions["timeout"]["read"] == 30
        raise httpx.ReadTimeout("TOKEN", request=request)

    api.override[("GET", "/api/environment-schema")] = timeout
    report = runner.run()
    assert "http_timeout" in failures(report)
    assert "TOKEN" not in report.model_dump_json()


def test_polling_timeout_and_gpu_unavailable_do_not_pass():
    runner, api, _ = harness(exercise=True)
    api.simulation_status = "unavailable"
    report = runner.run(DOCUMENT)
    assert "poll_deadline_exceeded" in failures(report)
    assert not any(request.url.path == "/api/runs" for request in api.requests)
    assert runner.monotonic() <= 3
    assert not report.release_ready
    smoke, api, _ = harness()
    api.simulation_status = "unavailable"
    assert "gpu_unavailable" in failures(smoke.run())


def test_twenty_double_episodes_have_individual_physical_evidence_not_release():
    runner, api, _ = harness(exercise=True, episodes=20)
    report = runner.run(DOCUMENT)
    assert report.status == "passed"
    episodes = [case for case in report.cases if case.id.startswith("LIVE-EPISODE-")]
    assert len(episodes) == 20
    assert len({case.evidence["execution"]["command_id"] for case in episodes}) == 20
    assert report.cases[-1].evidence["successes"] == 20
    assert not report.release_ready and not report.gates
    assert report.source == "fixture"
    assert sum(request.url.path.endswith("/approve") for request in api.requests) == 20


def test_one_success_does_not_satisfy_physical_suite():
    runner, _, _ = harness(exercise=True)
    report = runner.run(DOCUMENT)
    assert "physical_20_episode_threshold_not_met" in failures(report)
    assert any(c.id == "LIVE-EPISODE-001" and c.status == "passed" for c in report.cases)
    assert report.cases[-1].evidence["episodes"] == 1


@pytest.mark.parametrize("unsuccessful,threshold_status", [(2, "passed"), (3, "failed")])
def test_physical_threshold_measures_18_of_20_without_hiding_failures(
    unsuccessful, threshold_status
):
    runner, api, _ = harness(exercise=True, episodes=20)
    original = api.response

    def respond(request):
        api.final_status = "failed" if len(api.runs) <= unsuccessful else "succeeded"
        return original(request)

    runner.client = httpx.Client(transport=httpx.MockTransport(respond))
    report = runner.run(DOCUMENT)
    threshold = report.cases[-1]
    assert threshold.status == threshold_status
    assert threshold.evidence["episodes"] == 20
    assert threshold.evidence["successes"] == 20 - unsuccessful
    assert report.status == "failed"
    assert len([c for c in report.cases if c.failure == "motion_not_succeeded"]) == unsuccessful


@pytest.mark.parametrize("missing", ["final_position", "completed_at", "command_id"])
def test_ack_or_success_label_without_physical_evidence_is_failure(missing):
    runner, api, _ = harness(exercise=True)
    api.missing_execution_field = missing
    report = runner.run(DOCUMENT)
    assert report.status == "failed"
    episode = next(c for c in report.cases if c.id == "LIVE-EPISODE-001")
    assert episode.status == "failed" and episode.evidence["run_id"]


@pytest.mark.parametrize("position", [[0, 0, 0], [0.55, -0.4, 0.2]])
def test_wrong_final_pose_rejected(position):
    runner, api, _ = harness(exercise=True)
    api.final_position = position
    assert "final_pose_outside_target" in failures(runner.run(DOCUMENT))


@pytest.mark.parametrize(
    "flag,error",
    [
        ("stale_camera", "stale_camera"),
        ("frozen_camera", "no_fresh_physical_progress"),
    ],
)
def test_stale_or_frozen_frame_fails(flag, error):
    runner, api, _ = harness(exercise=True)
    setattr(api, flag, True)
    assert error in failures(runner.run(DOCUMENT))


def test_reused_command_rejected_and_cleanup_requires_confirmed_terminal():
    runner, api, _ = harness(exercise=True, episodes=20)
    api.reuse_command = True
    report = runner.run(DOCUMENT)
    assert "reused_command" in failures(report)
    cleanup = next(case for case in report.cases if case.id == "LIVE-CLEANUP")
    assert cleanup.evidence["confirmed_terminal_status"] == "cancelled"
    assert cleanup.evidence["purpose"] == "cleanup_only_not_motion_success"
    assert not report.release_ready


def test_nonterminal_ack_times_out_then_cancels():
    runner, api, _ = harness(exercise=True)
    api.final_status = "running"
    report = runner.run(DOCUMENT)
    assert "poll_deadline_exceeded" in failures(report)
    assert any(r.url.path.endswith("/cancel") for r in api.requests)


def test_invalid_customer_json_is_not_reserialized_or_sent():
    runner, api, _ = harness(exercise=True)
    report = runner.run('{"environment_id":"a","environment_id":"b"}')
    assert report.status == "failed"
    assert all(request.method == "GET" for request in api.requests)


def test_git_override_and_unresolved_head(monkeypatch):
    def broken():
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(live, "current_commit", broken)
    assert live.resolve_commit(COMMIT) == COMMIT
    with pytest.raises(subprocess.SubprocessError):
        live.resolve_commit(None)
    with pytest.raises(live.CheckError, match="invalid_candidate_commit"):
        live.resolve_commit("main")


def test_atomic_report_round_trip(tmp_path):
    runner, _, _ = harness()
    report = runner.run()
    target = tmp_path / "evidence.json"
    live.write_report(target, report)
    assert live.Report.model_validate_json(target.read_text()) == report
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evidence.json"]


def test_report_rejects_empty_cases_and_naive_timestamps():
    runner, _, _ = harness()
    data = runner.run().model_dump(mode="json")
    empty = copy.deepcopy(data)
    empty["cases"] = []
    with pytest.raises(ValidationError):
        live.Report.model_validate(empty)
    data["started_at"] = "2026-09-20T12:00:00"
    with pytest.raises(ValidationError):
        live.Report.model_validate(data)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("fixture", "actual_deployment_snapshot_required"),
        ("stale", "stale_deployment_snapshot"),
        ("commit", "candidate_commit_mismatch"),
        ("endpoint", "deployment_endpoint_mismatch"),
        ("exercise", "customer_document_required"),
        ("episodes", "exercise_consent_required"),
    ],
)
def test_cli_preflight_fails_before_identity_or_network(
    tmp_path, monkeypatch, capsys, mutation, expected
):
    snapshot = provenance().model_dump(mode="json")
    snapshot["source"] = "actual"
    extra = []
    if mutation == "fixture":
        snapshot["source"] = "fixture"
    elif mutation == "stale":
        snapshot["observed_at"] = (START - timedelta(days=2)).isoformat()
    elif mutation == "commit":
        snapshot["candidate_commit"] = "f" * 40
    elif mutation == "endpoint":
        snapshot["endpoint"] = "https://different.azurecontainerapps.io"
    elif mutation == "exercise":
        extra = ["--exercise"]
    elif mutation == "episodes":
        extra = ["--episodes", "20"]
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    def forbidden(**kwargs):
        pytest.fail("No credential acquisition during invalid preflight")

    monkeypatch.setattr(live, "now", lambda: START)
    monkeypatch.setattr(live, "AzureCliCredential", forbidden)
    assert (
        live.main(
            [
                "--endpoint",
                ENDPOINT,
                "--tenant-id",
                TENANT,
                "--api-client-id",
                API,
                "--candidate-commit",
                COMMIT,
                "--provenance",
                str(path),
                "--output",
                str(tmp_path / "report.json"),
                *extra,
            ]
        )
        == 2
    )
    assert expected in capsys.readouterr().out
