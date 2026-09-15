from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar
from uuid import UUID

from azure.core import MatchConditions
from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from pydantic import BaseModel

from apps.api.errors import Problem, unavailable
from apps.api.models import EnvironmentRecord, RunRecord, Stored

log = logging.getLogger(__name__)
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


def _cosmos(call: Callable[[], T]) -> T:
    try:
        return call()
    except CosmosHttpResponseError as exc:
        if exc.status_code in (409, 412):
            raise Problem(
                409, "revision_conflict", "State changed; reload before retrying."
            ) from exc
        log.exception("Cosmos operation failed")
        raise unavailable("Azure Cosmos DB") from exc
    except AzureError as exc:
        log.exception("Cosmos authentication or transport failed")
        raise unavailable("Azure Cosmos DB") from exc


class CosmosStore:
    def __init__(self, endpoint: str, credential, database: str, container: str) -> None:
        self.client = CosmosClient(
            endpoint, credential=credential, connection_timeout=5, read_timeout=10
        )
        self.container = self.client.get_database_client(database).get_container_client(container)

    def _read(self, owner: str, item_id: str, model: type[M]) -> Stored[M] | None:
        def read():
            try:
                return self.container.read_item(item=item_id, partition_key=owner)
            except CosmosResourceNotFoundError:
                return None

        item = _cosmos(read)
        if item is None:
            return None
        return Stored(value=model.model_validate(item["value"]), etag=item["_etag"])

    def _write(self, owner: str, item_id: str, kind: str, value: M, etag: str | None) -> Stored[M]:
        body = {
            "id": item_id,
            "owner_key": owner,
            "kind": kind,
            "updated_at": value.model_dump(mode="json")["updated_at"],
            "value": value.model_dump(mode="json"),
        }
        if etag is None:
            item = _cosmos(lambda: self.container.create_item(body=body))
        else:
            item = _cosmos(
                lambda: self.container.replace_item(
                    item=item_id,
                    body=body,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            )
        return Stored(value=value, etag=item["_etag"])

    def _list(self, owner: str, kind: str, model: type[M]) -> list[M]:
        items = _cosmos(
            lambda: list(
                self.container.query_items(
                    query="SELECT TOP 50 * FROM c WHERE c.kind = @kind ORDER BY c.updated_at DESC",
                    parameters=[{"name": "@kind", "value": kind}],
                    partition_key=owner,
                )
            )
        )
        return [model.model_validate(item["value"]) for item in items]

    def get_environment(self, owner: str, environment_id: str) -> Stored[EnvironmentRecord] | None:
        return self._read(owner, f"environment:{environment_id}", EnvironmentRecord)

    def list_environments(self, owner: str) -> list[EnvironmentRecord]:
        return self._list(owner, "environment", EnvironmentRecord)

    def put_environment(
        self, owner: str, record: EnvironmentRecord, etag: str | None
    ) -> Stored[EnvironmentRecord]:
        history_id = f"revision:{record.environment_id}:{record.revision}"
        existing = self._read(owner, history_id, EnvironmentRecord)
        if existing is None:
            try:
                self._write(owner, history_id, "environment_revision", record, None)
            except Problem as exc:
                if exc.code != "revision_conflict":
                    raise
                # A competing writer may have already stored this content-addressed revision.
                existing = self._read(owner, history_id, EnvironmentRecord)
                if existing is None or existing.value.document != record.document:
                    raise
        elif existing.value.document != record.document:
            raise Problem(409, "revision_collision", "Stored revision content does not match.")
        return self._write(
            owner, f"environment:{record.environment_id}", "environment", record, etag
        )

    def get_run(self, owner: str, run_id: UUID) -> Stored[RunRecord] | None:
        return self._read(owner, f"run:{run_id}", RunRecord)

    def list_runs(self, owner: str) -> list[RunRecord]:
        return self._list(owner, "run", RunRecord)

    def put_run(self, owner: str, record: RunRecord, etag: str | None) -> Stored[RunRecord]:
        return self._write(owner, f"run:{record.id}", "run", record, etag)

    def close(self) -> None:
        self.client.close()


class BlobArtifacts:
    def __init__(self, account_url: str, credential, container: str) -> None:
        self.client = BlobServiceClient(account_url=account_url, credential=credential)
        self.container = self.client.get_container_client(container)

    def put(self, name: str, image: bytes) -> None:
        try:
            self.container.upload_blob(
                name=name,
                data=image,
                overwrite=False,
                content_settings=ContentSettings(content_type="image/png"),
            )
        except ResourceExistsError as exc:
            raise Problem(
                409, "artifact_exists", "An observation with this name already exists."
            ) from exc
        except AzureError as exc:
            log.exception("Observation upload failed")
            raise unavailable("Azure Blob Storage") from exc

    def get(self, name: str) -> bytes:
        try:
            return self.container.download_blob(name).readall()
        except ResourceNotFoundError as exc:
            raise Problem(
                404, "observation_missing", "The captured observation is missing."
            ) from exc
        except AzureError as exc:
            log.exception("Observation download failed")
            raise unavailable("Azure Blob Storage") from exc

    def close(self) -> None:
        self.client.close()
