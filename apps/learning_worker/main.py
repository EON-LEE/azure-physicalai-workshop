from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from apps.api.auth import EntraTokens
from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    BootstrapReport,
    CaptureReceipt,
    LearningProject,
    PairedReport,
    PolicyCandidate,
    TeachingSession,
    TrainingRun,
)
from apps.api.learning_ports import JobSpecification
from apps.api.models import Model, Principal


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEARNING_WORKER_", extra="ignore")
    tenant_id: UUID
    audience: UUID
    managed_identity_client_id: UUID
    allowed_api_principals: frozenset[UUID] = frozenset()
    bootstrap_owner_ids: frozenset[UUID] = frozenset()
    allowed_policy_types: tuple[str, ...] = ()
    registry_account_url: str = Field(pattern=r"^https://[a-z0-9]{3,24}\.blob\.core\.windows\.net$")
    registry_container: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
    capture_account_url: str = Field(pattern=r"^https://[a-z0-9]{3,24}\.blob\.core\.windows\.net$")
    capture_container: str = "demonstrations"


class WorkerEnvelope(Model):
    actor: Principal
    payload: dict


def create_worker(settings: WorkerSettings | None = None, operations=None, identity=None):
    configuration = settings or WorkerSettings()
    authorizer = identity or EntraTokens(configuration.tenant_id, configuration.audience)

    @asynccontextmanager
    async def lifespan(app):
        resources = []
        if operations is None:
            from azure.identity import ManagedIdentityCredential

            from apps.learning_worker.artifacts import VerifiedArtifacts
            from apps.learning_worker.backend import PolicyLearningWorker
            from apps.learning_worker.registry import BlobRegistry

            credential = ManagedIdentityCredential(
                client_id=str(configuration.managed_identity_client_id)
            )
            registry = BlobRegistry(
                configuration.registry_account_url, configuration.registry_container, credential
            )
            artifacts = VerifiedArtifacts(
                registry,
                credential,
                configuration.capture_account_url,
                configuration.capture_container,
                allowed_policy_types=configuration.allowed_policy_types,
            )
            app.state.worker = PolicyLearningWorker(
                registry,
                artifacts,
                configuration.managed_identity_client_id,
                allowed_policy_types=configuration.allowed_policy_types,
            )
            resources = [registry, credential]
        else:
            app.state.worker = operations
        try:
            yield
        finally:
            for resource in resources:
                resource.close()

    app = FastAPI(
        title="Private Azure learning worker", lifespan=lifespan, docs_url=None, redoc_url=None
    )

    @app.exception_handler(Problem)
    async def problem(request, error):
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message}}, status_code=error.status
        )

    @app.exception_handler(ValidationError)
    async def invalid_receipt(request, error):
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_worker_contract",
                    "message": "Worker payload or artifact failed validation.",
                }
            },
            status_code=422,
        )

    def controller(authorization: Annotated[str | None, Header()] = None):
        authorizer.controller(authorization, configuration.allowed_api_principals)

    def scoped(actor, owner):
        if actor.tenant_id != configuration.tenant_id or actor.owner_key != owner:
            raise Problem(403, "worker_scope_mismatch", "Actor and immutable owner scope differ.")
        return actor

    def read_actor(
        tenant_id: UUID,
        actor_id: UUID,
        owner: Annotated[str, Header(alias="X-Environment-Owner")],
    ):
        return scoped(Principal(tenant_id=tenant_id, object_id=actor_id), owner)

    def post_actor(
        body: WorkerEnvelope, owner: Annotated[str, Header(alias="X-Environment-Owner")]
    ):
        scoped(body.actor, owner)
        return body

    @app.get("/healthz")
    def health():
        return {"status": "process_running", "training_verified": False}

    @app.post("/v1/learning/preflight", dependencies=[Depends(controller)])
    def preflight(request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]):
        specification = JobSpecification.model_validate(body.payload)
        request.app.state.worker.preflight(body.actor, specification)
        return {
            "verified": True,
            "owner_key": body.actor.owner_key,
            "sha256": specification.run.specification_sha256,
        }

    @app.post("/v1/learning/jobs/{job_name}", dependencies=[Depends(controller)])
    def submit(
        job_name: str, request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]
    ):
        specification = JobSpecification.model_validate(body.payload)
        if specification.run.backend_job_name != job_name:
            raise Problem(409, "worker_job_mismatch", "URL and immutable job claim differ.")
        return request.app.state.worker.submit(body.actor, specification)

    @app.get("/v1/learning/jobs/{job_name}", dependencies=[Depends(controller)])
    def status(job_name: str, request: Request, actor: Annotated[Principal, Depends(read_actor)]):
        worker = request.app.state.worker
        specification = worker.registry.job(actor, job_name)
        if specification is None:
            raise Problem(404, "worker_job_missing", "No owned job claim exists.")
        result = worker.status(actor, specification.run)
        if result is None:
            raise Problem(404, "worker_job_missing", "No owned Azure job receipt exists.")
        return result

    @app.post("/v1/learning/jobs/{job_name}/cancel", dependencies=[Depends(controller)])
    def cancel(
        job_name: str, request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]
    ):
        worker = request.app.state.worker
        specification = worker.registry.job(body.actor, job_name)
        if specification is None or body.payload.get("id") != str(specification.run.id):
            raise Problem(404, "worker_job_missing", "No matching owned job claim exists.")
        return worker.cancel(body.actor, specification.run)

    @app.get("/v1/learning/releases/{release_id}", dependencies=[Depends(controller)])
    def release(
        release_id: UUID, request: Request, actor: Annotated[Principal, Depends(read_actor)]
    ):
        return request.app.state.worker.registry.release(actor, release_id)

    @app.get("/v1/learning/training-parents/{artifact_id}", dependencies=[Depends(controller)])
    def parent(
        artifact_id: UUID, request: Request, actor: Annotated[Principal, Depends(read_actor)]
    ):
        return request.app.state.worker.registry.training_parent(actor, artifact_id)

    @app.post("/v1/learning/artifacts/capture", dependencies=[Depends(controller)])
    def capture(request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]):
        payload = body.payload
        project, session = (
            LearningProject.model_validate(payload["project"]),
            TeachingSession.model_validate(payload["session"]),
        )
        if (
            project.owner_key != body.actor.owner_key
            or session.owner_key != body.actor.owner_key
            or session.project_id != project.id
        ):
            raise Problem(403, "worker_scope_mismatch", "Capture owner or project differs.")
        return request.app.state.worker.artifacts.verify_capture(
            body.actor, project, session, payload["receipt"]
        )

    @app.post("/v1/learning/artifacts/dataset", dependencies=[Depends(controller)])
    def dataset(request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]):
        project = LearningProject.model_validate(body.payload["project"])
        if project.owner_key != body.actor.owner_key:
            raise Problem(403, "worker_scope_mismatch", "Dataset owner differs.")
        receipts = tuple(CaptureReceipt.model_validate(item) for item in body.payload["captures"])
        artifact_id, digest = request.app.state.worker.artifacts.seal_dataset(
            body.actor, project, UUID(body.payload["dataset_id"]), receipts
        )
        return {
            "artifact_id": str(artifact_id),
            "manifest_sha256": digest,
            "owner_key": body.actor.owner_key,
        }

    @app.post("/v1/learning/artifacts/candidate", dependencies=[Depends(controller)])
    def candidate(request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]):
        payload = body.payload
        project = LearningProject.model_validate(payload["project"])
        run = TrainingRun.model_validate(payload["run"])
        value = PolicyCandidate.model_validate(payload["candidate"])
        if any(record.owner_key != body.actor.owner_key for record in (project, run, value)):
            raise Problem(403, "worker_scope_mismatch", "Candidate owner differs.")
        request.app.state.worker.artifacts.verify_candidate(body.actor, project, run, value)
        return {
            "verified": True,
            "owner_key": body.actor.owner_key,
            "sha256": value.manifest_sha256,
        }

    @app.post("/v1/learning/artifacts/report", dependencies=[Depends(controller)])
    def report(request: Request, body: Annotated[WorkerEnvelope, Depends(post_actor)]):
        from apps.api.learning_models import EvaluationRun

        project = LearningProject.model_validate(body.payload["project"])
        run = EvaluationRun.model_validate(body.payload["run"])
        if project.owner_key != body.actor.owner_key or run.owner_key != body.actor.owner_key:
            raise unavailable("Exact persisted evaluation report")
        model = BootstrapReport if run.comparison_kind == "reference_bootstrap" else PairedReport
        proposed = model.model_validate(body.payload["report"])
        if run.report is not None and run.report != proposed:
            raise Problem(
                409, "worker_report_mismatch", "Report differs from the original evaluation."
            )
        if proposed.evaluation_plan_sha256 != run.evaluation_plan_sha256:
            raise Problem(
                409, "worker_report_mismatch", "Report belongs to a different held-out plan."
            )
        request.app.state.worker.artifacts.verify_report(body.actor, project, run, proposed)
        return {
            "verified": True,
            "owner_key": body.actor.owner_key,
            "sha256": proposed.report_sha256,
        }

    return app
