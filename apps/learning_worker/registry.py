from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

from azure.core import MatchConditions
from azure.core.exceptions import (
    AzureError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobServiceClient, ContentSettings
from pydantic import AwareDatetime, Field, ValidationError

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import Frozen, PolicyRelease, TrainingParent, fingerprint
from apps.api.learning_ports import JobSpecification
from apps.api.models import Model, Principal, Revision, Stored, utcnow

log = logging.getLogger(__name__)


class ReconciliationTarget(Frozen):
    actor_id: UUID
    job_name: str = Field(pattern=r"^learning-[a-f0-9-]+$", max_length=100)
    specification_sha256: Revision
    configuration_sha256: Revision
    job_deadline_utc: AwareDatetime


class ReconciliationHeartbeat(Frozen):
    target: ReconciliationTarget
    worker_client_id: UUID
    observed_at: AwareDatetime


class CancellationClaim(Frozen):
    owner_key: Revision
    job_name: str
    specification_sha256: Revision
    configuration_sha256: Revision
    requested_at: AwareDatetime
    state: Literal["claimed", "acknowledged", "uncertain", "forbidden"] = "claimed"
    error_code: str | None = None


class BlobRegistry:
    def __init__(self, account_url: str, container: str, credential):
        self.client = BlobServiceClient(
            account_url,
            credential=credential,
            connection_timeout=5,
            read_timeout=10,
            retry_total=0,
        )
        self.container = self.client.get_container_client(container)

    @staticmethod
    def key(actor: Principal, suffix: str) -> str:
        path = PurePosixPath(suffix)
        if path.is_absolute() or ".." in path.parts or "\\" in suffix:
            raise Problem(
                422, "invalid_registry_path", "Registry paths must remain owner-relative."
            )
        return f"tenants/{actor.tenant_id}/owners/{actor.owner_key}/learning/{suffix}"

    def get(self, actor: Principal, suffix: str):
        try:
            content = self.container.download_blob(self.key(actor, suffix)).readall()
            if len(content) > 2 * 1024 * 1024:
                raise Problem(503, "registry_document_size", "Registry metadata exceeds its bound.")
            return json.loads(content)
        except ResourceNotFoundError:
            return None
        except (AzureError, ValueError) as exc:
            raise unavailable("Private learning registry") from exc

    def put(self, actor: Principal, suffix: str, value: dict) -> bool:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        try:
            self.container.upload_blob(
                name=self.key(actor, suffix),
                data=payload,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/json"),
            )
            return True
        except ResourceExistsError as exc:
            if self.get(actor, suffix) != value:
                raise Problem(
                    409,
                    "immutable_registry_conflict",
                    "An immutable artifact/claim already differs.",
                ) from exc
            return False
        except AzureError as exc:
            raise unavailable("Private learning registry") from exc

    def approved_plan(self, actor, specification: JobSpecification):
        kind = (
            "train" if specification.run.kind == "training" else specification.run.comparison_kind
        )
        value = self.get(actor, f"projects/{specification.project.id}/{kind}-approval.json")
        if value is None:
            raise unavailable("Operator-reviewed Azure learning plan")
        return value

    def claim_job(self, actor, specification, configuration):
        self.put(
            actor,
            f"jobs/{specification.run.backend_job_name}/configuration.json",
            {
                "specification_sha256": specification.run.specification_sha256,
                "configuration_sha256": fingerprint(configuration),
                "configuration": configuration,
            },
        )
        return self.put(
            actor,
            f"jobs/{specification.run.backend_job_name}/specification.json",
            specification.model_dump(mode="json"),
        )

    def job(self, actor, name):
        value = self.get(actor, f"jobs/{name}/specification.json")
        return None if value is None else JobSpecification.model_validate(value)

    def job_configuration(self, actor, specification):
        value = self.get(actor, f"jobs/{specification.run.backend_job_name}/configuration.json")
        if value is None:
            return None
        if (
            not isinstance(value, dict)
            or set(value) != {"specification_sha256", "configuration_sha256", "configuration"}
            or (
                value["specification_sha256"] != specification.run.specification_sha256
                or not isinstance(value["configuration"], dict)
                or value["configuration_sha256"] != fingerprint(value["configuration"])
            )
        ):
            raise Problem(
                503, "worker_configuration_corrupted", "Frozen job configuration differs."
            )
        return value["configuration"]

    def _read_record(self, actor, suffix, model: type[Model]) -> Stored | None:
        try:
            download = self.container.download_blob(self.key(actor, suffix))
            content = download.readall()
            if len(content) > 65536:
                raise Problem(503, "registry_document_size", "Lifecycle record exceeds its bound.")
            return Stored(value=model.model_validate_json(content), etag=download.properties.etag)
        except ResourceNotFoundError:
            return None
        except (AzureError, ValidationError) as exc:
            log.exception("Learning lifecycle record read failed")
            raise unavailable("Private learning lifecycle record") from exc

    def _write_record(self, actor, suffix, value: Model, etag: str | None) -> Stored:
        kwargs = (
            {} if etag is None else {"etag": etag, "match_condition": MatchConditions.IfNotModified}
        )
        try:
            result = self.container.get_blob_client(self.key(actor, suffix)).upload_blob(
                data=value.model_dump_json().encode(),
                overwrite=etag is not None,
                content_settings=ContentSettings(content_type="application/json"),
                **kwargs,
            )
            return Stored(value=value, etag=result["etag"])
        except (ResourceExistsError, ResourceModifiedError) as exc:
            raise Problem(409, "registry_revision_conflict", "Lifecycle record changed.") from exc
        except AzureError as exc:
            log.exception("Learning lifecycle record write failed")
            raise unavailable("Private learning lifecycle record") from exc

    def heartbeat(self, actor, target):
        return self._read_record(
            actor, f"jobs/{target.job_name}/reconciliation.json", ReconciliationHeartbeat
        )

    def record_heartbeat(self, actor, target, client_id):
        if target.actor_id != actor.object_id:
            raise Problem(403, "worker_scope_mismatch", "Reconciliation actor differs.")
        value = ReconciliationHeartbeat(
            target=target, worker_client_id=client_id, observed_at=utcnow()
        )
        for attempt in range(2):
            current = self.heartbeat(actor, target)
            if current is not None:
                if current.value.target != target or current.value.worker_client_id != client_id:
                    raise Problem(
                        409, "reconciliation_binding_changed", "Monitor enrollment is immutable."
                    )
                if current.value.observed_at >= value.observed_at:
                    return current
            try:
                return self._write_record(
                    actor,
                    f"jobs/{target.job_name}/reconciliation.json",
                    value,
                    current.etag if current else None,
                )
            except Problem as exc:
                if exc.code != "registry_revision_conflict" or attempt:
                    raise

    @staticmethod
    def _cancellation_binding(actor, specification, configuration):
        return {
            "owner_key": actor.owner_key,
            "job_name": specification.run.backend_job_name,
            "specification_sha256": specification.run.specification_sha256,
            "configuration_sha256": fingerprint(configuration),
        }

    def cancellation(self, actor, specification, configuration):
        current = self._read_record(
            actor,
            f"jobs/{specification.run.backend_job_name}/cancellation.json",
            CancellationClaim,
        )
        if current is not None and any(
            getattr(current.value, field) != value
            for field, value in self._cancellation_binding(
                actor, specification, configuration
            ).items()
        ):
            raise Problem(409, "cancellation_scope_changed", "Cancellation belongs to another job.")
        return current

    def claim_cancellation(self, actor, specification, configuration):
        value = CancellationClaim(
            **self._cancellation_binding(actor, specification, configuration), requested_at=utcnow()
        )
        try:
            return self._write_record(
                actor, f"jobs/{value.job_name}/cancellation.json", value, None
            ), True
        except Problem as exc:
            if exc.code != "registry_revision_conflict":
                raise
            current = self.cancellation(actor, specification, configuration)
            if current is None:
                raise unavailable("Existing cancellation claim") from exc
            return current, False

    def record_cancellation(self, actor, stored, state, error_code=None):
        if stored.value.owner_key != actor.owner_key or stored.value.state != "claimed":
            raise Problem(409, "cancellation_scope_changed", "Cancellation cannot be renewed.")
        updated = stored.value.model_copy(update={"state": state, "error_code": error_code})
        return self._write_record(
            actor, f"jobs/{updated.job_name}/cancellation.json", updated, stored.etag
        )

    def release(self, actor, release_id: UUID):
        value = self.get(actor, f"releases/{release_id}.json")
        if value is None:
            raise Problem(404, "policy_release_missing", "No reviewed policy is registered.")
        return PolicyRelease.model_validate(value)

    def training_parent(self, actor, artifact_id: UUID):
        value = self.get(actor, f"training-parents/{artifact_id}.json")
        if value is None:
            raise Problem(404, "training_parent_missing", "Train-only artifact is not registered.")
        return TrainingParent.model_validate(value)

    def artifact_index(self, actor, artifact_id):
        value = self.get(actor, f"artifacts/{artifact_id}/index.json")
        if not isinstance(value, dict) or value.get("owner_key") != actor.owner_key:
            raise unavailable("Scoped immutable artifact index")
        return value

    def download(self, actor, artifact_id, destination: Path, max_bytes=20 * 1024**3):
        index = self.artifact_index(actor, artifact_id)
        files = index.get("files")
        if not isinstance(files, dict) or not files or len(files) > 100000:
            raise unavailable("Bounded artifact inventory")
        total = 0
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in files.items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or not isinstance(expected, str)
            ):
                raise Problem(
                    503, "invalid_artifact_path", "Artifact inventory contains an invalid path."
                )
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            try:
                downloader = self.container.download_blob(
                    self.key(actor, f"artifacts/{artifact_id}/files/{name}")
                )
                with output.open("xb") as stream:
                    for chunk in downloader.chunks():
                        total += len(chunk)
                        if total > max_bytes:
                            raise Problem(
                                503,
                                "artifact_budget_exceeded",
                                "Artifact exceeds the approved download bound.",
                            )
                        stream.write(chunk)
                        digest.update(chunk)
            except AzureError as exc:
                raise unavailable("Private artifact download") from exc
            if digest.hexdigest() != expected:
                raise Problem(
                    503,
                    "artifact_digest_mismatch",
                    "Downloaded artifact did not match the immutable inventory.",
                )
        return index

    def upload(self, actor, artifact_id, root: Path, metadata: dict):
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise Problem(
                    422, "artifact_symlink", "Symlinks are not allowed in registered artifacts."
                )
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
            try:
                with path.open("rb") as stream:
                    self.container.upload_blob(
                        name=self.key(actor, f"artifacts/{artifact_id}/files/{name}"),
                        data=stream,
                        overwrite=False,
                    )
            except ResourceExistsError as exc:
                # Never silently accept an unverified pre-existing partial artifact.
                index = self.get(actor, f"artifacts/{artifact_id}/index.json")
                if not isinstance(index, dict) or index.get("files", {}).get(name) != files[name]:
                    raise Problem(
                        409, "artifact_exists", "Artifact upload requires a new registration ID."
                    ) from exc
            except AzureError as exc:
                raise unavailable("Private artifact upload") from exc
        self.put(
            actor,
            f"artifacts/{artifact_id}/index.json",
            {
                **metadata,
                "owner_key": actor.owner_key,
                "files": files,
            },
        )

    def close(self):
        self.client.close()
