from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from apps.api.auth import EntraTokens
from apps.api.errors import Problem
from apps.api.middleware import BodyLimit
from apps.api.models import EnvironmentRecord, MotionCommand
from simulation.core import SimulationCore
from simulation.paused_contracts import SimulationEpisodeCommand
from simulation.runtime_contracts import PolicyCommand, TeachingInput, TeachingLease, TeachingStart


class BridgeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    entra_tenant_id: UUID
    entra_api_client_id: UUID
    bridge_allowed_principal_ids: list[UUID] = Field(min_length=1)
    bridge_port: int = Field(default=8443, ge=1024, le=65535)
    sim_tls_cert_file: str
    sim_tls_key_file: str


def create_bridge_app(
    core: SimulationCore, settings: BridgeSettings, tokens: EntraTokens | None = None
) -> FastAPI:
    identity = tokens or EntraTokens(settings.entra_tenant_id, settings.entra_api_client_id)
    app = FastAPI(title="Isaac Sim Azure bridge", docs_url=None, redoc_url=None)
    app.add_middleware(BodyLimit)

    @app.exception_handler(Problem)
    async def problem_handler(request: Request, exc: Problem):
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
            status_code=exc.status,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_command",
                    "message": "Command contract validation failed.",
                    "details": [],
                }
            },
            status_code=422,
        )

    def owner(
        authorization: Annotated[str | None, Header()] = None,
        x_environment_owner: Annotated[str | None, Header(pattern=r"^[a-f0-9]{64}$")] = None,
    ) -> str:
        identity.controller(authorization, set(settings.bridge_allowed_principal_ids))
        if not x_environment_owner:
            raise Problem(422, "owner_required", "A scoped environment owner is required.")
        return x_environment_owner

    Owner = Annotated[str, Depends(owner)]

    @app.get("/healthz")
    def health():
        return {"status": "process_running", "runtime": "isaac_sim"}

    @app.get("/v1/status")
    def status(principal: Owner):
        return core.status(principal)

    @app.post("/v1/scene", status_code=202)
    def scene(environment: EnvironmentRecord, principal: Owner):
        return core.activate(principal, environment)

    @app.get("/v1/observation")
    def observation(
        principal: Owner,
        environment_id: str,
        revision: Annotated[str, Query(pattern=r"^[a-f0-9]{64}$")],
        camera: Literal["overview", "inspection"] = "inspection",
    ):
        return core.observe(principal, environment_id, revision, camera)

    @app.post("/v1/commands", status_code=202)
    def dispatch(command: MotionCommand, principal: Owner):
        return core.dispatch(principal, command)

    @app.post("/v1/policy/commands", status_code=202)
    def policy(command: PolicyCommand, principal: Owner):
        return core.dispatch_policy(principal, command)

    @app.post("/v1/simulation-episodes", status_code=202)
    def simulation_episode(command: SimulationEpisodeCommand, principal: Owner):
        return core.dispatch_simulation_episode(principal, command)

    @app.get("/v1/simulation-episodes/{command_id}")
    def simulation_episode_status(command_id: UUID, principal: Owner):
        return core.simulation_episode(principal, command_id)

    @app.post("/v1/simulation-episodes/{command_id}/cancel")
    def cancel_simulation_episode(command_id: UUID, principal: Owner):
        core.simulation_episode(principal, command_id)
        return core.cancel(principal, command_id)

    @app.get("/v1/commands/{command_id}")
    def command(command_id: UUID, principal: Owner):
        return core.command(principal, command_id)

    @app.get("/v1/commands/{command_id}/capture")
    def capture(command_id: UUID, principal: Owner):
        return core.capture(principal, command_id)

    @app.post("/v1/commands/{command_id}/cancel")
    def cancel(command_id: UUID, principal: Owner):
        return core.cancel(principal, command_id)

    @app.post("/v1/teaching", status_code=202)
    def start_teaching(request: TeachingStart, principal: Owner):
        return core.start_teaching(principal, request)

    @app.get("/v1/teaching/{session_id}")
    def teaching(session_id: UUID, principal: Owner):
        return core.teaching(principal, session_id)

    @app.post("/v1/teaching/{session_id}/input")
    def teaching_input(session_id: UUID, request: TeachingInput, principal: Owner):
        return core.teaching_input(principal, session_id, request)

    @app.post("/v1/teaching/{session_id}/finish")
    def finish_teaching(session_id: UUID, request: TeachingLease, principal: Owner):
        return core.finish_teaching(principal, session_id, request)

    @app.post("/v1/teaching/{session_id}/cancel")
    def cancel_teaching(session_id: UUID, request: TeachingLease, principal: Owner):
        return core.cancel_teaching(principal, session_id, request)

    return app
