from __future__ import annotations

import logging
import ssl
import time
from typing import Literal
from uuid import UUID

import httpx
from azure.core.exceptions import AzureError
from pydantic import ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_ports import RuntimeCapture, TeachingRuntimeState, TeachingStartSpec
from apps.api.models import (
    Activation,
    EnvironmentRecord,
    Execution,
    MotionCommand,
    Observation,
    ReleasedPolicyBinding,
    SimulationStatus,
)

log = logging.getLogger(__name__)


class AzureSimulatorBridge:
    def __init__(
        self,
        endpoint: str,
        credential,
        scope: str,
        timeout: float = 10,
        ca_pem: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.credential = credential
        self.scope = scope
        self.timeout = timeout
        context = ssl.create_default_context()
        if ca_pem:
            context.load_verify_locations(cadata=ca_pem)
        self.http = httpx.Client(
            base_url=endpoint,
            timeout=timeout,
            verify=context,
            transport=transport,
            follow_redirects=False,
            limits=httpx.Limits(keepalive_expiry=2),
        )

    def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        deadline = time.monotonic() + self.timeout
        try:
            return self.http.request(method, path, **kwargs)
        except httpx.RemoteProtocolError:
            remaining = deadline - time.monotonic()
            if method != "GET" or remaining <= 0:
                raise
            log.warning("Retrying one read-only simulator GET after a closed connection: %s", path)
            return self.http.request(method, path, timeout=remaining, **kwargs)

    def _request(self, method: str, path: str, owner: str, **kwargs) -> dict:
        try:
            token = self.credential.get_token(self.scope).token
            response = self._send(
                method,
                path,
                headers={"Authorization": f"Bearer {token}", "X-Environment-Owner": owner},
                **kwargs,
            )
        except (httpx.HTTPError, AzureError) as exc:
            log.exception("Simulator %s %s request failed", method, path)
            raise unavailable("Isaac Sim bridge") from exc
        if response.status_code >= 300:
            structured_error = response.status_code in (404, 409, 422)
            observation_unavailable = (
                response.status_code == 503 and method == "GET" and path == "/v1/observation"
            )
            if structured_error or observation_unavailable:
                try:
                    error = response.json()["error"]
                    code, message = error["code"], error["message"]
                    if not isinstance(code, str) or not isinstance(message, str):
                        raise ValueError("Invalid error shape")
                except (ValueError, KeyError, TypeError) as exc:
                    raise unavailable("Isaac Sim bridge protocol") from exc
                if structured_error or code == "camera_not_ready":
                    raise Problem(response.status_code, code, message)
            log.error("Simulator returned HTTP %s", response.status_code)
            raise unavailable("Isaac Sim bridge")
        try:
            data = response.json()
        except ValueError as exc:
            raise unavailable("Isaac Sim bridge protocol") from exc
        if not isinstance(data, dict):
            raise unavailable("Isaac Sim bridge protocol")
        return data

    def _model(self, model, method: str, path: str, owner: str, **kwargs):
        try:
            return model.model_validate(self._request(method, path, owner, **kwargs))
        except ValidationError as exc:
            log.exception("Invalid simulator response")
            raise unavailable("Isaac Sim bridge protocol") from exc

    def status(self, owner: str) -> SimulationStatus:
        return self._model(SimulationStatus, "GET", "/v1/status", owner)

    def activate(self, owner: str, environment: EnvironmentRecord) -> Activation:
        return self._model(
            Activation, "POST", "/v1/scene", owner, json=environment.model_dump(mode="json")
        )

    def observe(
        self,
        owner: str,
        environment_id: str,
        revision: str,
        camera: Literal["overview", "inspection"] = "inspection",
    ) -> Observation:
        return self._model(
            Observation,
            "GET",
            "/v1/observation",
            owner,
            params={"environment_id": environment_id, "revision": revision, "camera": camera},
        )

    def dispatch(self, owner: str, command: MotionCommand) -> Execution:
        return self._model(
            Execution, "POST", "/v1/commands", owner, json=command.model_dump(mode="json")
        )

    def dispatch_policy(
        self,
        owner: str,
        command: MotionCommand,
        policy: ReleasedPolicyBinding,
    ) -> Execution:
        return self._model(
            Execution,
            "POST",
            "/v1/policy/commands",
            owner,
            json={
                "command": command.model_dump(mode="json"),
                "policy_release_id": str(policy.policy_release_id),
                "model_sha256": policy.model_sha256,
                "control_profile_id": policy.control_profile_id,
                "task": {
                    "task_id": policy.task_id,
                    "instruction": policy.instruction,
                    "goal_id": policy.goal_station_id,
                },
            },
        )

    def start_teaching(self, owner: str, body: TeachingStartSpec) -> TeachingRuntimeState:
        return self._model(
            TeachingRuntimeState,
            "POST",
            "/v1/teaching",
            owner,
            json=body.model_dump(mode="json"),
        )

    def teaching(self, owner: str, session_id: UUID) -> TeachingRuntimeState:
        return self._model(TeachingRuntimeState, "GET", f"/v1/teaching/{session_id}", owner)

    def teaching_input(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState:
        return self._model(
            TeachingRuntimeState,
            "POST",
            f"/v1/teaching/{session_id}/input",
            owner,
            json=body,
        )

    def finish_teaching(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState:
        return self._model(
            TeachingRuntimeState,
            "POST",
            f"/v1/teaching/{session_id}/finish",
            owner,
            json=body,
        )

    def cancel_teaching(self, owner: str, session_id: UUID, body: dict) -> TeachingRuntimeState:
        return self._model(
            TeachingRuntimeState,
            "POST",
            f"/v1/teaching/{session_id}/cancel",
            owner,
            json=body,
        )

    def capture(self, owner: str, command_id: UUID) -> RuntimeCapture:
        return self._model(RuntimeCapture, "GET", f"/v1/commands/{command_id}/capture", owner)

    def command(self, owner: str, command_id: UUID) -> Execution:
        return self._model(Execution, "GET", f"/v1/commands/{command_id}", owner)

    def cancel(self, owner: str, command_id: UUID) -> Execution:
        return self._model(Execution, "POST", f"/v1/commands/{command_id}/cancel", owner)

    def close(self) -> None:
        self.http.close()
