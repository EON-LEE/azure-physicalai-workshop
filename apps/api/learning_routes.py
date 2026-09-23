from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, Header, Request, Response

from apps.api.learning_models import (
    Approval,
    ArmTeaching,
    CreateDataset,
    CreateProject,
    JogIntent,
    ReleasePolicy,
    StartEvaluation,
    StartTeaching,
    StartTraining,
    TeachingControl,
)
from apps.api.learning_service import LearningService
from apps.api.models import Principal, Stored


def learning_response(stored: Stored, response: Response) -> dict:
    response.headers["ETag"] = stored.etag
    return {"item": stored.value.public(), "etag": stored.etag}


def install_learning_routes(app, actor):
    def learning(request: Request) -> LearningService:
        return request.app.state.learning

    Actor = Annotated[Principal, Depends(actor)]
    Service = Annotated[LearningService, Depends(learning)]
    Match = Annotated[str | None, Header(alias="If-Match")]

    @app.get("/api/learning/capabilities")
    def capabilities(user: Actor, backend: Service):
        return backend.capabilities()

    @app.get("/api/learning/projects")
    def list_projects(user: Actor, backend: Service):
        return {
            "items": [
                {"item": entry.value.public(), "etag": entry.etag}
                for entry in backend.list(user, "project")
            ]
        }

    @app.post("/api/learning/projects", status_code=201)
    def create_project(body: CreateProject, user: Actor, backend: Service, response: Response):
        return learning_response(backend.create_project(user, body), response)

    @app.get("/api/learning/projects/{project_id}")
    def project(project_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get(user, "project", project_id), response)

    @app.get("/api/learning/projects/{project_id}/records")
    def project_records(
        project_id: UUID,
        user: Actor,
        backend: Service,
        kind: Literal["teaching", "dataset", "training", "evaluation", "candidate", "release"],
    ):
        return {
            "items": [
                {"item": entry.value.public(), "etag": entry.etag}
                for entry in backend.list(user, kind, project_id)
            ]
        }

    @app.post("/api/learning/projects/{project_id}/teaching-sessions", status_code=202)
    def teach(
        project_id: UUID,
        body: StartTeaching,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.start_teaching(user, project_id, body, if_match), response)

    @app.get("/api/teaching-sessions/{session_id}")
    def teaching(session_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get_teaching(user, session_id), response)

    @app.post("/api/teaching-sessions/{session_id}/jog")
    def jog(
        session_id: UUID,
        body: JogIntent,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.jog(user, session_id, body, if_match), response)

    @app.post("/api/teaching-sessions/{session_id}/arm", status_code=201)
    def arm(
        session_id: UUID,
        body: ArmTeaching,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.arm(user, session_id, body, if_match), response)

    @app.post("/api/teaching-sessions/{session_id}/finish", status_code=202)
    def finish(
        session_id: UUID,
        body: TeachingControl,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(
            backend.control_teaching(user, session_id, body, if_match, "finish"), response
        )

    @app.post("/api/teaching-sessions/{session_id}/cancel", status_code=202)
    def cancel_teaching(
        session_id: UUID,
        body: TeachingControl,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(
            backend.control_teaching(user, session_id, body, if_match, "cancel"), response
        )

    @app.post("/api/learning/projects/{project_id}/datasets", status_code=201)
    def dataset(
        project_id: UUID,
        body: CreateDataset,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.dataset(user, project_id, body, if_match), response)

    @app.get("/api/learning/datasets/{dataset_id}")
    def get_dataset(dataset_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get(user, "dataset", dataset_id), response)

    @app.post("/api/learning/projects/{project_id}/train", status_code=202)
    def train(
        project_id: UUID,
        body: StartTraining,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.train(user, project_id, body, if_match), response)

    @app.post("/api/learning/projects/{project_id}/evaluate", status_code=202)
    def evaluate(
        project_id: UUID,
        body: StartEvaluation,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.evaluate(user, project_id, body, if_match), response)

    @app.get("/api/learning/jobs/{job_id}")
    def job(job_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get_job(user, job_id), response)

    @app.post("/api/learning/jobs/{job_id}/cancel", status_code=202)
    def cancel(
        job_id: UUID,
        body: Approval,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(
            backend.cancel_job(user, job_id, body.request_id, if_match), response
        )

    @app.get("/api/learning/candidates/{candidate_id}")
    def candidate(candidate_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get(user, "candidate", candidate_id), response)

    @app.post("/api/policy-releases", status_code=201)
    def release(
        body: ReleasePolicy,
        user: Actor,
        backend: Service,
        response: Response,
        if_match: Match = None,
    ):
        return learning_response(backend.release(user, body, if_match), response)

    @app.get("/api/policy-releases/{release_id}")
    def get_release(release_id: UUID, user: Actor, backend: Service, response: Response):
        return learning_response(backend.get(user, "release", release_id), response)

    @app.get("/api/demo/learning")
    def public_learning():
        return {"api_version": "public-learning-v1", "status": "not_published", "publication": None}
