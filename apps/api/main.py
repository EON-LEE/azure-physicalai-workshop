import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from apps.api.auth import EntraTokens
from apps.api.errors import Problem
from apps.api.middleware import BodyLimit
from apps.api.models import (
    ActivateEnvironment,
    ApproveRun,
    Principal,
    SaveEnvironment,
    StartRun,
    utcnow,
)
from apps.api.public_demo import PublicDemo
from apps.api.service import FactoryService, check_fresh
from apps.api.settings import Settings
from contracts.validate_environment import SCHEMA

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def azure_service(settings: Settings):
    from azure.identity import ManagedIdentityCredential

    from agents.inspection import FoundryInspector
    from apps.api.azure_data import BlobArtifacts, CosmosStore
    from apps.api.bridge_client import AzureSimulatorBridge

    credential = ManagedIdentityCredential(client_id=str(settings.azure_client_id))
    store = CosmosStore(
        settings.cosmos_endpoint, credential, settings.cosmos_database, settings.cosmos_container
    )
    artifacts = BlobArtifacts(settings.storage_account_url, credential, settings.storage_container)
    version = settings.foundry_agent_version
    ca_pem = settings.sim_bridge_ca_pem
    agent_probe_verified_at = None
    if version is None or ca_pem is None:
        from scripts.bootstrap import BootstrapManifest

        manifest = BootstrapManifest.model_validate_json(
            artifacts.get(settings.runtime_bootstrap_blob)
        )
        if (
            manifest.project_endpoint != settings.foundry_project_endpoint
            or manifest.agent_name != settings.foundry_agent_name
        ):
            raise RuntimeError(
                "Azure bootstrap manifest belongs to a different Foundry deployment."
            )
        version = version or manifest.agent_version
        if (
            manifest.probe_response_id
            and manifest.probe_scope == "agent_connectivity_only_not_physical_inspection"
        ):
            agent_probe_verified_at = manifest.created_at
        if ca_pem is None:
            ca = artifacts.get(settings.sim_ca_blob)
            if hashlib.sha256(ca).hexdigest() != manifest.sim_ca_sha256:
                raise RuntimeError("Simulator CA does not match the Azure bootstrap manifest.")
            ca_pem = ca.decode("ascii")
    planner = FoundryInspector(
        settings.foundry_project_endpoint,
        credential,
        settings.foundry_agent_name,
        version,
        settings.model_timeout_seconds,
    )
    bridge = AzureSimulatorBridge(
        settings.sim_bridge_endpoint,
        credential,
        settings.simulator_scope,
        settings.bridge_timeout_seconds,
        ca_pem,
    )
    return FactoryService(
        store,
        artifacts,
        planner,
        bridge,
        settings.approval_ttl_seconds,
        agent_probe_verified_at=agent_probe_verified_at,
    ), [
        bridge,
        planner,
        artifacts,
        store,
        credential,
    ]


def create_app(
    settings: Settings | None = None,
    service: FactoryService | None = None,
    authorizer: EntraTokens | None = None,
) -> FastAPI:
    configuration = settings or Settings()
    identity = authorizer or EntraTokens(
        configuration.entra_tenant_id, configuration.entra_api_client_id
    )
    if service is None and configuration.applicationinsights_connection_string:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(
            connection_string=configuration.applicationinsights_connection_string,
            instrumentation_options={"azure_sdk": {"enabled": True}},
            enable_live_metrics=False,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = []
        if service is None:
            actual, resources = azure_service(configuration)
            app.state.service = actual
        else:
            app.state.service = service
        app.state.public_demo = PublicDemo(configuration, app.state.service)
        try:
            yield
        finally:
            for resource in resources:
                resource.close()

    app = FastAPI(title="Azure Physical AI", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    @app.exception_handler(Problem)
    async def problem_handler(request: Request, exc: Problem):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
            status_code=exc.status,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Correct the request fields.",
                    "details": [
                        {
                            "path": "/".join(str(x) for x in item["loc"]),
                            "message": item["msg"],
                            "code": item["type"],
                        }
                        for item in exc.errors()
                    ],
                }
            },
            status_code=422,
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self' https://login.microsoftonline.com; "
            "frame-src 'self' https://login.microsoftonline.com; "
            "frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        if (
            request.url.path == "/api/demo"
            and response.status_code == 200
            and configuration.public_demo_presentation_id is None
        ):
            response.headers["Cache-Control"] = "public, max-age=1"
        return response

    def actor(authorization: Annotated[str | None, Header()] = None) -> Principal:
        return identity.user(authorization)

    def factory(request: Request) -> FactoryService:
        return request.app.state.service

    Actor = Annotated[Principal, Depends(actor)]
    Service = Annotated[FactoryService, Depends(factory)]

    @app.get("/api/demo")
    def public_demo(request: Request):
        return request.app.state.public_demo.snapshot()

    @app.get("/api/demo/frame")
    def public_frame(
        request: Request,
        camera: Literal["overview", "inspection"] = "overview",
        epoch: UUID | None = None,
    ):
        image, observation = request.app.state.public_demo.frame(camera, epoch)
        check_fresh(observation, 2000)
        return Response(
            image,
            media_type="image/png",
            headers={
                "X-Frame-Id": str(observation.observation_id),
                "X-Captured-At": observation.captured_at.isoformat(),
                "X-Physics-Steps": str(observation.physics_steps),
                "X-Scene-Epoch": str(observation.epoch),
                "X-Server-Time": utcnow().isoformat(),
            },
        )

    @app.get("/api/demo/evidence")
    def public_evidence(request: Request, observation_id: UUID):
        image, run = request.app.state.public_demo.evidence(observation_id)
        return Response(
            image,
            media_type="image/png",
            headers={
                "X-Frame-Id": str(run.evidence.observation_id),
                "X-Captured-At": run.evidence.captured_at.isoformat(),
            },
        )

    @app.get("/healthz")
    def health(response: Response):
        response.headers["X-Server-Time"] = utcnow().isoformat()
        return {"status": "process_running", "deployment": "azure"}

    @app.get("/api/config")
    def config():
        return {
            "api_version": "v1",
            "deployment": "azure",
            "auth": {
                "tenant_id": str(configuration.entra_tenant_id),
                "client_id": str(configuration.entra_spa_client_id),
                "scope": configuration.delegated_scope,
            },
        }

    @app.get("/api/runtime")
    def runtime(user: Actor, backend: Service):
        return {
            "deployment": "azure",
            "simulation": backend.runtime(user).model_dump(mode="json"),
            "agent": {"provider": "microsoft_foundry", "configured": True},
            "storage": {"provider": "azure_cosmos_blob"},
            "release_ready": False,
        }

    @app.get("/api/environment-schema")
    def environment_schema(user: Actor):
        return SCHEMA

    @app.get("/api/environment-templates")
    def environment_templates(user: Actor):
        items = []
        for filename in ("inspection-cell.json", "compact-cell.json"):
            document = json.loads((ROOT / "examples" / filename).read_text(encoding="utf-8"))
            items.append({"name": document["display_name"], "document": document})
        return {"items": items}

    @app.get("/api/environments")
    def list_environments(user: Actor, backend: Service):
        return {"items": backend.store.list_environments(user.owner_key)}

    @app.post("/api/environments", status_code=201)
    def save_environment(body: SaveEnvironment, user: Actor, backend: Service):
        return backend.save_environment(user, body)

    @app.post("/api/environments/{environment_id}/activate", status_code=202)
    def activate(environment_id: str, body: ActivateEnvironment, user: Actor, backend: Service):
        return backend.activate(user, environment_id, body.revision)

    @app.get("/api/environments/{environment_id}/frame")
    def frame(
        environment_id: str,
        user: Actor,
        backend: Service,
        revision: Annotated[str, Query(pattern=r"^[a-f0-9]{64}$")],
        camera: Literal["overview", "inspection"] = "overview",
    ):
        image, observation = backend.frame(user, environment_id, revision, camera)
        return Response(
            image,
            media_type="image/png",
            headers={
                "X-Frame-Id": str(observation.observation_id),
                "X-Captured-At": observation.captured_at.isoformat(),
                "X-Physics-Steps": str(observation.physics_steps),
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/runs", status_code=201)
    def start_run(body: StartRun, user: Actor, backend: Service):
        return backend.start(user, body).public()

    @app.get("/api/runs")
    def list_runs(user: Actor, backend: Service):
        return {"items": [run.public() for run in backend.store.list_runs(user.owner_key)]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: UUID, user: Actor, backend: Service):
        return backend.get_run(user, run_id).public()

    @app.get("/api/runs/{run_id}/observation")
    def observation(run_id: UUID, user: Actor, backend: Service):
        return Response(backend.evidence(user, run_id), media_type="image/png")

    @app.post("/api/runs/{run_id}/approve")
    def approve(run_id: UUID, body: ApproveRun, user: Actor, backend: Service):
        return backend.approve(user, run_id, body.plan_response_id).public()

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: UUID, user: Actor, backend: Service):
        return backend.cancel(user, run_id).public()

    assets = configuration.web_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        icon = configuration.web_dist / "favicon.svg"
        if not icon.is_file():
            raise Problem(404, "asset_missing", "The requested web asset is not built.")
        return FileResponse(icon, media_type="image/svg+xml")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise Problem(404, "endpoint_missing", "API endpoint not found.")
        index = configuration.web_dist / "index.html"
        if not index.is_file():
            raise Problem(
                503, "frontend_not_built", "Build the web console before serving the application."
            )
        return FileResponse(index)

    return app
