from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, NamedTuple
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
ARTIFACT_INDEX_MAX_BYTES = 16 * 1024**2
ARTIFACT_MAX_FILES = 100000
ARTIFACT_MAX_BYTES = 20 * 1024**3
_INDEX_FIELDS = frozenset(
    {
        "owner_key",
        "manifest_sha256",
        "files",
        "role",
        "training_execution",
        "azure_job_type",
        "azure_job_id",
        "training_origin",
        "external_import_id",
        "project_id",
    }
)


class ArtifactIndex(NamedTuple):
    value: dict
    etag: str


class ImportBlob(NamedTuple):
    content: bytes
    modified_at: datetime
    etag: str


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
    def __init__(self, account_url: str, container: str, credential, *, budget=None):
        self.client = BlobServiceClient(
            account_url,
            credential=credential,
            connection_timeout=5,
            read_timeout=10,
            retry_total=0,
        )
        self.container = self.client.get_container_client(container)
        self.budget = budget

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

    def import_document(self, actor, suffix, *, max_bytes=2 * 1024**2):
        """Read exact private bytes and the same download's server modification timestamp."""
        result = self.import_blob(actor, suffix, max_bytes=max_bytes)
        return result.content, result.modified_at

    def import_blob(self, actor, suffix, *, max_bytes=2 * 1024**2):
        try:
            download = self.container.download_blob(self.key(actor, suffix))
            content = bytearray()
            budget = getattr(self, "budget", None)
            if budget:
                budget.consume(0, files=1)
            for chunk in download.chunks():
                if len(content) + len(chunk) > max_bytes:
                    raise Problem(503, "managed_import_size", "Import document exceeds its bound.")
                if budget:
                    budget.consume(len(chunk))
                content.extend(chunk)
            modified = getattr(download.properties, "last_modified", None)
            etag = getattr(download.properties, "etag", None)
            if (
                not isinstance(modified, datetime)
                or modified.tzinfo is None
                or modified.utcoffset() is None
                or not isinstance(etag, str)
                or not etag
            ):
                raise Problem(
                    503, "managed_import_timestamp", "Blob-observed UTC modification is required."
                )
            return ImportBlob(bytes(content), modified, etag)
        except ResourceNotFoundError as exc:
            raise Problem(
                404, "managed_import_missing", "The original private import record is missing."
            ) from exc
        except AzureError as exc:
            raise unavailable("Private managed evaluation import") from exc

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

    def reference_authorization(self, actor, project_id: UUID, case_id: str):
        from apps.api.reference_models import ReferenceAuthorization

        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", case_id):
            raise Problem(
                422,
                "invalid_reference_case",
                "A reference case must be a fixed approved identifier.",
            )
        value = self.get(actor, f"projects/{project_id}/reference-authorizations/{case_id}.json")
        if value is None:
            raise Problem(
                503,
                "reference_authority_missing",
                "No operator-installed reference authority exists.",
            )
        result = ReferenceAuthorization.model_validate(value)
        if result.project_id != project_id or result.case_id != case_id:
            raise Problem(409, "reference_grant_mismatch", "Reference catalog scope differs.")
        return result

    def training_parent(self, actor, artifact_id: UUID):
        value = self.get(actor, f"training-parents/{artifact_id}.json")
        if value is None:
            raise Problem(404, "training_parent_missing", "Train-only artifact is not registered.")
        return TrainingParent.model_validate(value)

    def _artifact_prefix(self, actor, artifact_id):
        try:
            identifier = UUID(str(artifact_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise Problem(422, "invalid_artifact_id", "Artifact identity must be a UUID.") from exc
        return self.key(actor, f"artifacts/{identifier}/")

    @staticmethod
    def _artifact_path(name):
        from learning.common import relative_path

        try:
            if not isinstance(name, str) or len(name) > 1024:
                raise ValueError("Invalid artifact path type or length")
            return relative_path(name)
        except ValueError as exc:
            raise Problem(
                503, "invalid_artifact_path", "Artifact paths must be bounded and relative."
            ) from exc

    def _validate_artifact_index(self, actor, value):
        if (
            not isinstance(value, dict)
            or value.keys() - _INDEX_FIELDS
            or value.get("owner_key") != actor.owner_key
            or not isinstance(value.get("manifest_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", value["manifest_sha256"]) is None
            or not isinstance(value.get("files"), dict)
            or not 1 <= len(value["files"]) <= ARTIFACT_MAX_FILES
        ):
            raise Problem(
                503, "artifact_index_invalid", "Artifact index scope or structure is invalid."
            )
        for name in value.keys() - {"files", "owner_key", "manifest_sha256"}:
            if not isinstance(value[name], str) or not value[name] or len(value[name]) > 2048:
                raise Problem(
                    503, "artifact_index_invalid", "Artifact metadata has an invalid type."
                )
        files = value["files"]
        for name, checksum in files.items():
            path = self._artifact_path(name)
            if (
                not isinstance(checksum, str)
                or re.fullmatch(r"[a-f0-9]{64}", checksum) is None
                or any(str(parent) in files for parent in path.parents if str(parent) != ".")
            ):
                raise Problem(
                    503, "artifact_index_invalid", "Artifact file hashes or paths conflict."
                )
        if not any(
            files.get(name) == value["manifest_sha256"]
            for name in (
                "manifest.json",
                "model.json",
                "report.json",
                "backbone.json",
                "checkpoint.json",
            )
        ):
            raise Problem(
                503, "artifact_index_invalid", "The declared artifact manifest is missing."
            )
        return value

    def _read_artifact_index(self, actor, artifact_id, *, etag=None):
        from learning.common import parse_json

        key = self._artifact_prefix(actor, artifact_id) + "index.json"
        conditional = (
            {}
            if etag is None
            else {
                "etag": etag,
                "match_condition": MatchConditions.IfNotModified,
            }
        )
        try:
            budget = getattr(self, "budget", None)
            if budget:
                budget.consume(0, files=1)
            download = self.container.download_blob(
                key,
                offset=0,
                length=ARTIFACT_INDEX_MAX_BYTES + 1,
                max_concurrency=1,
                retry_total=0,
                **conditional,
            )
            actual_etag = getattr(download.properties, "etag", None)
            if (
                not isinstance(actual_etag, str)
                or not actual_etag
                or (etag is not None and actual_etag != etag)
            ):
                raise Problem(
                    409, "artifact_index_changed", "Artifact index version is unconfirmed."
                )
            content = bytearray()
            for chunk in download.chunks():
                if budget:
                    budget.consume(len(chunk))
                if len(content) + len(chunk) > ARTIFACT_INDEX_MAX_BYTES:
                    raise Problem(
                        503, "artifact_index_size", "Artifact index exceeds its 16 MiB bound."
                    )
                content.extend(chunk)
            value = self._validate_artifact_index(actor, parse_json(bytes(content)))
            return ArtifactIndex(value, actual_etag)
        except ResourceNotFoundError as exc:
            if etag is not None:
                raise Problem(
                    409, "artifact_index_changed", "The original artifact index disappeared."
                ) from exc
            return None
        except ResourceModifiedError as exc:
            raise Problem(
                409, "artifact_index_changed", "The original artifact index changed."
            ) from exc
        except AzureError as exc:
            raise unavailable("Bounded private artifact index") from exc
        except ValueError as exc:
            raise Problem(503, "artifact_index_invalid", "Artifact index JSON is invalid.") from exc

    def artifact_index(self, actor, artifact_id):
        result = self._read_artifact_index(actor, artifact_id)
        if result is None:
            raise unavailable("Scoped immutable artifact index")
        return result.value

    @staticmethod
    def _payload_etag(value):
        if not isinstance(value, str) or not value or value.startswith("W/"):
            raise Problem(409, "artifact_payload_changed", "A strong payload ETag is required.")
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if not value or re.fullmatch(r"[!#-~]+", value) is None:
            raise Problem(409, "artifact_payload_changed", "Payload ETag syntax is invalid.")
        return f'"{value}"'

    def _payload_inventory(self, prefix):
        result, total = {}, 0
        budget = getattr(self, "budget", None)
        try:
            for blob in self.container.list_blobs(name_starts_with=prefix):
                if budget:
                    budget.check()
                if not isinstance(blob.name, str) or not blob.name.startswith(prefix):
                    raise Problem(
                        503, "artifact_inventory_invalid", "Payload inventory escaped its prefix."
                    )
                name = blob.name[len(prefix) :]
                self._artifact_path(name)
                if (
                    name in result
                    or len(result) >= ARTIFACT_MAX_FILES
                    or type(blob.size) is not int
                    or blob.size < 0
                    or not isinstance(blob.etag, str)
                    or not blob.etag
                ):
                    raise Problem(
                        503, "artifact_inventory_invalid", "Payload inventory is invalid."
                    )
                total += blob.size
                if total > ARTIFACT_MAX_BYTES:
                    raise Problem(
                        503, "artifact_budget_exceeded", "Payload inventory exceeds 20 GiB."
                    )
                result[name] = (blob.size, self._payload_etag(blob.etag))
        except AzureError as exc:
            raise unavailable("Private artifact payload inventory") from exc
        return result

    def _verify_payloads(self, prefix, files, sizes):
        observed = self._payload_inventory(prefix)
        if {name: size for name, (size, _) in observed.items()} != sizes:
            raise Problem(409, "artifact_payload_mismatch", "Existing payload inventory differs.")
        budget = getattr(self, "budget", None)
        for name, expected in files.items():
            size, etag = observed[name]
            digest, received = hashlib.sha256(), 0
            try:
                if budget:
                    budget.consume(0, files=1)
                download = self.container.download_blob(
                    prefix + name,
                    offset=0,
                    length=size + 1,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                    max_concurrency=1,
                    retry_total=0,
                )
                if self._payload_etag(getattr(download.properties, "etag", None)) != etag:
                    raise Problem(
                        409, "artifact_payload_changed", "Payload version changed during readback."
                    )
                for chunk in download.chunks():
                    if budget:
                        budget.consume(len(chunk))
                    received += len(chunk)
                    if received > size:
                        raise Problem(
                            409,
                            "artifact_payload_mismatch",
                            "Payload readback exceeds its original size.",
                        )
                    digest.update(chunk)
            except (ResourceModifiedError, ResourceNotFoundError) as exc:
                raise Problem(
                    409, "artifact_payload_changed", "Original payload disappeared or changed."
                ) from exc
            except AzureError as exc:
                raise unavailable("Private immutable artifact readback") from exc
            if received != size or digest.hexdigest() != expected:
                raise Problem(
                    409, "artifact_payload_mismatch", "Payload readback checksum or size differs."
                )
        if observed != self._payload_inventory(prefix):
            raise Problem(
                409, "artifact_payload_changed", "Payload inventory changed during verification."
            )

    def _source_inventory(self, root):
        if not root.is_dir() or root.is_symlink():
            raise Problem(422, "artifact_symlink", "The artifact source must be a real directory.")
        files, sizes, total = {}, {}, 0
        budget = getattr(self, "budget", None)
        for path in root.rglob("*"):
            if budget:
                budget.check()
            if path.is_symlink():
                raise Problem(
                    422, "artifact_symlink", "Symlinks are not allowed in registered artifacts."
                )
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            self._artifact_path(name)
            size = path.stat().st_size
            total += size
            if len(files) >= ARTIFACT_MAX_FILES or total > ARTIFACT_MAX_BYTES:
                raise Problem(
                    503, "artifact_budget_exceeded", "Artifact source exceeds its file/byte bounds."
                )
            digest, received = hashlib.sha256(), 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    if budget:
                        budget.check()
                    received += len(chunk)
                    if received > size:
                        raise Problem(
                            409, "artifact_source_changed", "Artifact source changed while hashing."
                        )
                    digest.update(chunk)
            if received != size:
                raise Problem(
                    409, "artifact_source_changed", "Artifact source size changed while hashing."
                )
            files[name], sizes[name] = digest.hexdigest(), size
        return files, sizes

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
            budget = getattr(self, "budget", None)
            if budget:
                budget.consume(0, files=1)
            try:
                downloader = self.container.download_blob(
                    self.key(actor, f"artifacts/{artifact_id}/files/{name}")
                )
                with output.open("xb") as stream:
                    for chunk in downloader.chunks():
                        if budget:
                            budget.consume(len(chunk))
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
        prefix = self._artifact_prefix(actor, artifact_id)
        if not isinstance(metadata, dict) or {"owner_key", "files"} & metadata.keys():
            raise Problem(
                422, "artifact_index_invalid", "Artifact metadata cannot replace owner or files."
            )
        files, sizes = self._source_inventory(root)
        value = self._validate_artifact_index(
            actor, {**metadata, "owner_key": actor.owner_key, "files": files}
        )
        payload = bytearray()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
        for fragment in encoder.iterencode(value):
            chunk = fragment.encode()
            if len(payload) + len(chunk) > ARTIFACT_INDEX_MAX_BYTES:
                raise Problem(
                    503, "artifact_index_size", "Artifact index exceeds its 16 MiB bound."
                )
            payload.extend(chunk)
        original = self._read_artifact_index(actor, artifact_id)
        if original is not None:
            if original.value != value:
                raise Problem(
                    409,
                    "immutable_registry_conflict",
                    "The complete existing artifact index differs.",
                )
            self._verify_payloads(prefix + "files/", files, sizes)
            if self._read_artifact_index(actor, artifact_id, etag=original.etag) != original:
                raise Problem(
                    409, "artifact_index_changed", "Artifact index changed during readback."
                )
            return
        if self._payload_inventory(prefix + "files/"):
            raise Problem(
                409,
                "artifact_exists",
                "Partial payloads without a completed index cannot be reused.",
            )
        budget = getattr(self, "budget", None)
        for name in sorted(files):
            path = root.joinpath(*self._artifact_path(name).parts)
            try:
                if budget:
                    budget.consume(sizes[name], files=1)
                with path.open("rb") as stream:
                    self.container.get_blob_client(prefix + "files/" + name).upload_blob(
                        data=stream,
                        overwrite=False,
                        max_concurrency=1,
                        retry_total=0,
                    )
            except ResourceExistsError as exc:
                raise Problem(
                    409, "artifact_exists", "A concurrent partial upload cannot be adopted."
                ) from exc
            except AzureError as exc:
                raise unavailable("Private artifact upload") from exc
        self._verify_payloads(prefix + "files/", files, sizes)
        try:
            if budget:
                budget.consume(len(payload), files=1)
            self.container.get_blob_client(prefix + "index.json").upload_blob(
                data=bytes(payload),
                overwrite=False,
                retry_total=0,
                content_settings=ContentSettings(content_type="application/json"),
            )
        except ResourceExistsError as exc:
            concurrent = self._read_artifact_index(actor, artifact_id)
            if concurrent is None or concurrent.value != value:
                raise Problem(
                    409,
                    "immutable_registry_conflict",
                    "Concurrent artifact index publication differs.",
                ) from exc
            self._verify_payloads(prefix + "files/", files, sizes)
            if self._read_artifact_index(actor, artifact_id, etag=concurrent.etag) != concurrent:
                raise Problem(
                    409,
                    "artifact_index_changed",
                    "Concurrent artifact index changed during verification.",
                ) from exc
            return
        except AzureError as exc:
            raise unavailable("Private artifact index publication") from exc
        published = self._read_artifact_index(actor, artifact_id)
        if published is None or published.value != value:
            raise Problem(
                409,
                "artifact_index_changed",
                "Published artifact index differs from verified payloads.",
            )

    def close(self):
        self.client.close()
