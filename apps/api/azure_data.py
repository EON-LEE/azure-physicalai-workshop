from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import TypeVar
from uuid import UUID, uuid4

from azure.core import MatchConditions
from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from pydantic import BaseModel, ValidationError

from apps.api.artifact_models import ArtifactOperationRecord
from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    CoachRecord,
    ControlGrant,
    DatasetVersion,
    EvaluationRun,
    LearningMutation,
    LearningProject,
    LearningRecord,
    PolicyCandidate,
    PolicyRelease,
    TeachingSession,
    TrainingParent,
    TrainingRun,
    transition,
)
from apps.api.models import (
    EnvironmentCursor,
    EnvironmentPage,
    EnvironmentRecord,
    PresentationRecord,
    Principal,
    RunRecord,
    Stored,
    utcnow,
)

log = logging.getLogger(__name__)
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
LEARNING_MODELS = {
    "artifact_operation": ArtifactOperationRecord,
    "project": LearningProject,
    "teaching": TeachingSession,
    "dataset": DatasetVersion,
    "training": TrainingRun,
    "evaluation": EvaluationRun,
    "candidate": PolicyCandidate,
    "release": PolicyRelease,
    "mutation": LearningMutation,
    "control_grant": ControlGrant,
    "training_parent": TrainingParent,
    "coach": CoachRecord,
}
LEARNING_MUTABLE = {
    "artifact_operation": {
        "updated_at",
        "status",
        "phase",
        "result",
        "error_code",
        "message",
        "claim_id",
        "heartbeat_at",
    },
    "teaching": {
        "updated_at",
        "status",
        "last_sequence",
        "last_input_fingerprint",
        "input_expires_at",
        "capture",
        "artifact_operation_id",
        "verification_status",
        "error_code",
        "message",
        "physical_status",
    },
    "training": {
        "updated_at",
        "status",
        "azure_job_id",
        "metrics",
        "candidate_id",
        "error_code",
        "message",
        "job_deadline_utc",
        "backend_status",
        "azure_status",
        "cancellation",
    },
    "evaluation": {
        "updated_at",
        "status",
        "azure_job_id",
        "metrics",
        "report",
        "error_code",
        "message",
        "job_deadline_utc",
        "backend_status",
        "azure_status",
        "cancellation",
    },
    "mutation": {"updated_at", "status", "error_code"},
    "control_grant": {"updated_at", "consumed_by"},
    "coach": {"updated_at", "status", "proposal", "model_response_id", "error_code", "message"},
}


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
        if kind == "environment_cursor":
            body["ttl"] = 600
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

    def page_environments(
        self, owner: str, *, page_size: int | None = None, cursor: UUID | None = None
    ) -> EnvironmentPage:
        now = utcnow()
        created_before, after = now, None
        if cursor is not None:
            previous = self._read(owner, f"environment-cursor:{cursor}", EnvironmentCursor)
            if previous is None or (
                previous.value.id != cursor
                or previous.value.owner_key != owner
                or now >= previous.value.expires_at
                or (page_size is not None and previous.value.page_size != page_size)
            ):
                raise Problem(
                    422,
                    "invalid_environment_cursor",
                    "Cursor is invalid or expired for this owner. Reload the first page.",
                )
            page_size = previous.value.page_size
            created_before, after = (
                previous.value.created_before,
                previous.value.after_environment_id,
            )
        size = 50 if page_size is None else page_size
        if type(size) is not int or not 1 <= size <= 50:
            raise Problem(422, "invalid_page_size", "Environment page size must be 1 through 50.")
        properties = _cosmos(self.container.read)
        if properties.get("defaultTtl") != -1:
            raise Problem(
                503,
                "environment_pagination_unavailable",
                "Cursor TTL requires a verified non-expiring container default before pagination.",
            )
        query = "SELECT TOP @limit * FROM c WHERE c.kind = @kind"
        parameters = [
            {"name": "@kind", "value": "environment"},
            {"name": "@limit", "value": size + 1},
        ]
        if after is not None:
            query += " AND c.id < @after"
            parameters.append({"name": "@after", "value": f"environment:{after}"})
        query += " ORDER BY c.id DESC"
        rows = _cosmos(
            lambda: list(
                self.container.query_items(
                    query=query, parameters=parameters, partition_key=owner, max_item_count=size + 1
                )
            )
        )
        if len(rows) > size + 1:
            raise unavailable("Bounded environment page")
        scanned, items = [], []
        for row in rows[:size]:
            if row.get("owner_key") != owner or row.get("kind") != "environment":
                raise Problem(503, "environment_scope_corrupted", "Environment page scope differs.")
            try:
                value = EnvironmentRecord.model_validate(row["value"])
            except (KeyError, ValidationError) as exc:
                log.exception("Stored environment page failed validation")
                raise unavailable("Stored environment page") from exc
            if row["id"] != f"environment:{value.environment_id}":
                raise Problem(
                    503, "environment_scope_corrupted", "Environment document ID differs."
                )
            scanned.append(value)
            # Compare parsed times after a bounded scan; ISO strings can have different precision.
            if value.created_at <= created_before:
                items.append(value)
        next_cursor = None
        if len(rows) > size:
            record = EnvironmentCursor(
                id=uuid4(),
                owner_key=owner,
                after_environment_id=scanned[-1].environment_id,
                page_size=size,
                created_before=created_before,
                expires_at=now + timedelta(minutes=10),
                updated_at=now,
            )
            self._write(
                owner, f"environment-cursor:{record.id}", "environment_cursor", record, None
            )
            next_cursor = record.id
        return EnvironmentPage(items=items, next_cursor=next_cursor)

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

    def get_presentation(
        self, owner: str, presentation_id: str
    ) -> Stored[PresentationRecord] | None:
        return self._read(owner, f"presentation:{presentation_id}", PresentationRecord)

    def put_presentation(
        self, owner: str, record: PresentationRecord, etag: str | None
    ) -> Stored[PresentationRecord]:
        if record.owner_key != owner:
            raise Problem(403, "presentation_owner", "Presentation owner does not match.")
        return self._write(owner, f"presentation:{record.id}", "presentation", record, etag)

    def list_runs(self, owner: str) -> list[RunRecord]:
        return self._list(owner, "run", RunRecord)

    def put_run(self, owner: str, record: RunRecord, etag: str | None) -> Stored[RunRecord]:
        return self._write(owner, f"run:{record.id}", "run", record, etag)

    def get_learning(self, owner: str, kind: str, resource_id: UUID) -> Stored | None:
        model = LEARNING_MODELS.get(kind)
        if model is None:
            raise Problem(422, "unknown_learning_kind", "Unknown learning resource type.")
        stored = self._read(owner, f"learning:{kind}:{resource_id}", model)
        if stored is not None and stored.value.owner_key != owner:
            raise Problem(503, "learning_scope_corrupted", "Stored learning scope is inconsistent.")
        return stored

    def put_learning(
        self,
        owner: str,
        record: LearningRecord | ArtifactOperationRecord,
        etag: str | None,
    ) -> Stored:
        model = LEARNING_MODELS[record.kind]
        record = model.model_validate(record.model_dump())
        if (
            record.owner_key != owner
            or Principal(tenant_id=record.tenant_id, object_id=record.actor_id).owner_key != owner
        ):
            raise Problem(
                403, "learning_owner_mismatch", "Learning actor and partition must match."
            )
        if etag is not None:
            current = self.get_learning(owner, record.kind, record.id)
            if current is None or current.etag != etag:
                raise Problem(
                    409, "revision_conflict", "Learning state changed; reload before acting."
                )
            allowed = LEARNING_MUTABLE.get(record.kind, set())
            old = current.value.model_dump()
            changed = {key for key, value in record.model_dump().items() if old[key] != value}
            if changed - allowed:
                raise Problem(
                    409, "immutable_learning_record", "Create a new version for changed pins."
                )
            if record.kind in ("training", "evaluation"):
                transition("job", current.value.status, record.status)
                if current.value.azure_job_id and record.azure_job_id != current.value.azure_job_id:
                    raise Problem(
                        409, "immutable_learning_record", "An Azure job ID cannot be replaced."
                    )
                if current.value.job_deadline_utc is not None and (
                    record.job_deadline_utc != current.value.job_deadline_utc
                ):
                    raise Problem(
                        409, "immutable_learning_record", "The original deadline cannot change."
                    )
                previous = current.value.cancellation
                if previous is not None and (
                    record.cancellation is None
                    or any(
                        getattr(previous, field) != getattr(record.cancellation, field)
                        for field in ("request_id", "reason", "requested_at")
                    )
                    or (previous.state != "claimed" and previous != record.cancellation)
                ):
                    raise Problem(
                        409, "immutable_learning_record", "A cancellation claim cannot be renewed."
                    )
            if record.kind == "teaching":
                if current.value.status in ("ready", "cancelled", "invalid", "blocked") and changed:
                    raise Problem(
                        409, "immutable_learning_record", "Terminal teaching cannot be revived."
                    )
                if record.last_sequence < current.value.last_sequence:
                    raise Problem(
                        409, "teaching_sequence", "Teaching input sequence cannot regress."
                    )
            if record.kind == "mutation" and current.value.status != "claimed" and changed:
                raise Problem(
                    409, "immutable_learning_record", "An operation claim cannot be renewed."
                )
            if record.kind == "control_grant" and current.value.consumed_by is not None and changed:
                raise Problem(409, "teaching_grant_consumed", "A control grant cannot be reused.")
            if record.kind == "coach" and current.value.status != "planning" and changed:
                raise Problem(
                    409, "immutable_learning_record", "A completed coach result cannot be replaced."
                )
        return self._write(
            owner, f"learning:{record.kind}:{record.id}", f"learning:{record.kind}", record, etag
        )

    def list_learning(
        self,
        owner: str,
        kind: str,
        project_id: UUID | None = None,
    ) -> list[Stored]:
        model = LEARNING_MODELS.get(kind)
        if model is None or kind in ("mutation", "control_grant"):
            raise Problem(422, "unknown_learning_kind", "Unknown listed learning resource type.")
        query = "SELECT TOP 50 * FROM c WHERE c.kind = @kind"
        parameters = [{"name": "@kind", "value": f"learning:{kind}"}]
        if project_id is not None:
            query += " AND c.value.project_id = @project"
            parameters.append({"name": "@project", "value": str(project_id)})
        query += " ORDER BY c.updated_at DESC"
        items = _cosmos(
            lambda: list(
                self.container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=owner,
                )
            )
        )
        result = []
        for item in items:
            record = model.model_validate(item["value"])
            if record.owner_key != owner:
                raise Problem(
                    503, "learning_scope_corrupted", "Listed learning scope is inconsistent."
                )
            result.append(Stored(value=record, etag=item["_etag"]))
        return result

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
