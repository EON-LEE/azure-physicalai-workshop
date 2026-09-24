from __future__ import annotations

import logging
import ssl
from uuid import UUID

import httpx
from azure.core.exceptions import AzureError
from pydantic import ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import CaptureReceipt, PolicyRelease, TrainingParent
from apps.api.learning_ports import BackendJob, JobSpecification
from apps.api.models import Principal

log = logging.getLogger(__name__)


class ManagedLearningGateway:
    """Private MI worker transport; no implicit retries of a submitted job or operation."""

    def __init__(
        self,
        endpoint: str,
        credential,
        scope: str,
        *,
        timeout: float = 20,
        ca_pem: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        context = ssl.create_default_context()
        if ca_pem:
            context.load_verify_locations(cadata=ca_pem)
        self.credential, self.scope = credential, scope
        self.http = httpx.Client(
            base_url=endpoint,
            timeout=timeout,
            verify=context,
            follow_redirects=False,
            transport=transport,
        )

    def _request(self, actor: Principal, method: str, path: str, payload=None):
        try:
            token = self.credential.get_token(self.scope).token
            response = self.http.request(
                method,
                path,
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Environment-Owner": actor.owner_key,
                },
                json=None
                if payload is None
                else {
                    "actor": actor.model_dump(mode="json"),
                    "payload": payload,
                },
                params={"tenant_id": str(actor.tenant_id), "actor_id": str(actor.object_id)}
                if method == "GET"
                else None,
            )
        except (httpx.HTTPError, AzureError) as exc:
            log.exception("Learning worker request failed: %s %s", method, path)
            raise unavailable("Azure learning worker") from exc
        if response.status_code == 404:
            return None
        if response.status_code == 403:
            log.error("Learning worker denied %s %s", method, path)
            raise Problem(403, "worker_forbidden", "The private learning operation was denied.")
        if response.status_code >= 300:
            log.error("Learning worker returned %s for %s %s", response.status_code, method, path)
            raise unavailable("Azure learning worker")
        try:
            value = response.json()
        except ValueError as exc:
            raise unavailable("Azure learning worker protocol") from exc
        if not isinstance(value, dict):
            raise unavailable("Azure learning worker protocol")
        return value

    @staticmethod
    def _parse(model, value):
        if value is None:
            raise unavailable("Required owner-scoped learning artifact")
        try:
            return model.model_validate(value)
        except ValidationError as exc:
            log.exception("Learning worker returned an invalid typed receipt")
            raise unavailable("Azure learning worker receipt") from exc

    @staticmethod
    def _verified(actor, response, digest):
        if response != {"verified": True, "owner_key": actor.owner_key, "sha256": digest}:
            raise unavailable("Owner-scoped artifact verification")

    def preflight(self, actor, specification: JobSpecification):
        value = self._request(
            actor, "POST", "/v1/learning/preflight", specification.model_dump(mode="json")
        )
        self._verified(actor, value, specification.run.specification_sha256)

    def submit(self, actor, specification: JobSpecification) -> BackendJob:
        return self._parse(
            BackendJob,
            self._request(
                actor,
                "POST",
                f"/v1/learning/jobs/{specification.run.backend_job_name}",
                specification.model_dump(mode="json"),
            ),
        )

    def status(self, actor, run):
        result = self._request(actor, "GET", f"/v1/learning/jobs/{run.backend_job_name}")
        return None if result is None else self._parse(BackendJob, result)

    def cancel(self, actor, run):
        return self._parse(
            BackendJob,
            self._request(
                actor,
                "POST",
                f"/v1/learning/jobs/{run.backend_job_name}/cancel",
                run.model_dump(mode="json"),
            ),
        )

    def resolve(self, actor, release_id: UUID):
        result = self._request(actor, "GET", f"/v1/learning/releases/{release_id}")
        if result is None:
            raise Problem(404, "policy_release_missing", "No reviewed release is registered.")
        return self._parse(PolicyRelease, result)

    def training_parent(self, actor, artifact_id: UUID):
        result = self._request(actor, "GET", f"/v1/learning/training-parents/{artifact_id}")
        if result is None:
            raise Problem(
                404, "training_parent_missing", "No approved train-only artifact is registered."
            )
        return self._parse(TrainingParent, result)

    def verify_capture(self, actor, project, session, receipt):
        return self._parse(
            CaptureReceipt,
            self._request(
                actor,
                "POST",
                "/v1/learning/artifacts/capture",
                {
                    "project": project.model_dump(mode="json"),
                    "session": session.model_dump(mode="json"),
                    "receipt": receipt,
                },
            ),
        )

    def seal_dataset(self, actor, project, dataset_id, captures):
        response = self._request(
            actor,
            "POST",
            "/v1/learning/artifacts/dataset",
            {
                "project": project.model_dump(mode="json"),
                "dataset_id": str(dataset_id),
                "captures": [capture.model_dump(mode="json") for capture in captures],
            },
        )
        if response is None or set(response) != {"artifact_id", "manifest_sha256", "owner_key"}:
            raise unavailable("Sealed dataset receipt")
        from pydantic import TypeAdapter

        from apps.api.learning_models import Revision

        try:
            artifact_id = UUID(response["artifact_id"])
            digest = TypeAdapter(Revision).validate_python(response["manifest_sha256"])
        except (ValueError, TypeError, ValidationError) as exc:
            raise unavailable("Sealed dataset receipt") from exc
        if response["owner_key"] != actor.owner_key:
            raise unavailable("Sealed dataset owner")
        return artifact_id, digest

    def verify_candidate(self, actor, project, run, candidate):
        response = self._request(
            actor,
            "POST",
            "/v1/learning/artifacts/candidate",
            {
                "project": project.model_dump(mode="json"),
                "run": run.model_dump(mode="json"),
                "candidate": candidate.model_dump(mode="json"),
            },
        )
        self._verified(actor, response, candidate.manifest_sha256)

    def verify_report(self, actor, project, run, report):
        response = self._request(
            actor,
            "POST",
            "/v1/learning/artifacts/report",
            {
                "project": project.model_dump(mode="json"),
                "run": run.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
            },
        )
        self._verified(actor, response, report.report_sha256)

    def close(self):
        self.http.close()
