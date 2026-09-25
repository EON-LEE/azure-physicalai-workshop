from __future__ import annotations

import shutil
import time
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from azure.core.exceptions import AzureError, ResourceNotFoundError
from pydantic import AwareDatetime, TypeAdapter, ValidationError

from apps.api.artifact_models import ArtifactPolicy, ArtifactResult, ArtifactStatus, ArtifactWork
from apps.api.errors import Problem, unavailable
from apps.api.models import utcnow

TERMINAL = frozenset({"ready", "failed", "timed_out", "uncertain"})


class ArtifactBudget:
    def __init__(self, deadline, *, max_bytes, max_files):
        self.deadline = deadline
        self.monotonic_end = time.monotonic() + max(0, (deadline - utcnow()).total_seconds())
        self.max_bytes, self.max_files = max_bytes, max_files
        self.bytes, self.files = 0, 0
        self.directory = None

    def check(self):
        if utcnow() >= self.deadline or time.monotonic() >= self.monotonic_end:
            raise Problem(409, "artifact_deadline", "The original artifact wall budget expired.")
        if self.directory is not None and shutil.disk_usage(self.directory).free < 64 * 1024**2:
            raise Problem(503, "artifact_disk_budget", "Temporary disk reserve was exhausted.")

    def consume(self, size, *, files=0):
        self.check()
        self.bytes += size
        self.files += files
        if self.bytes > self.max_bytes:
            raise Problem(422, "artifact_byte_budget", "Aggregate transfer bytes exceed approval.")
        if self.files > self.max_files:
            raise Problem(422, "artifact_file_budget", "Aggregate file count exceeds approval.")

    def require_disk(self, input_bytes, directory):
        self.check()
        needed = input_bytes * 2 + 256 * 1024**2
        if shutil.disk_usage(Path(directory)).free < needed:
            raise Problem(
                503,
                "artifact_disk_budget",
                "Insufficient temporary disk for input and sealed copies; no work was started.",
            )
        self.directory = Path(directory)


class ArtifactOperations:
    def __init__(
        self,
        registry,
        *,
        enabled=False,
        actor_ids=frozenset(),
        maximum_seconds=1800,
        capture_bytes=4 * 1024**3,
        dataset_bytes=20 * 1024**3,
        maximum_files=100000,
    ):
        self.registry = registry
        self.enabled = enabled
        self.actor_ids = actor_ids
        self.maximum_seconds = maximum_seconds
        self.capture_bytes, self.dataset_bytes = capture_bytes, dataset_bytes
        self.maximum_files = maximum_files

    def _state(self, actor, operation_id):
        value = self.registry._read_record(
            actor, f"artifact-operations/{operation_id}/state.json", ArtifactStatus
        )
        if value is not None and value.value.owner_key != actor.owner_key:
            raise Problem(403, "artifact_owner_mismatch", "Artifact operation owner differs.")
        return value

    def policy(self, actor):
        if not self.enabled:
            raise Problem(
                503, "artifact_operations_disabled", "Resident artifact processing is off."
            )
        if actor.object_id not in self.actor_ids:
            raise Problem(403, "artifact_owner_unapproved", "Artifact owner is not allowlisted.")
        return ArtifactPolicy(
            maximum_seconds=self.maximum_seconds,
            capture_bytes=self.capture_bytes,
            dataset_bytes=self.dataset_bytes,
            maximum_files=self.maximum_files,
        )

    def work(self, actor, operation_id):
        value = self.registry.get(actor, f"artifact-operations/{operation_id}/request.json")
        if value is None:
            raise Problem(404, "artifact_operation_missing", "No owned artifact request exists.")
        work = ArtifactWork.model_validate(value)
        if work.actor != actor or work.id != operation_id:
            raise Problem(403, "artifact_owner_mismatch", "Artifact request scope differs.")
        return work

    def _save(self, actor, stored, **changes):
        value = ArtifactStatus.model_validate(
            {
                **stored.value.model_dump(),
                **changes,
                "updated_at": utcnow(),
            }
        )
        return self.registry._write_record(
            actor, f"artifact-operations/{value.id}/state.json", value, stored.etag
        ).value

    def begin(self, actor, work):
        if not self.enabled:
            raise Problem(
                503, "artifact_operations_disabled", "Resident artifact processing is off."
            )
        if actor.object_id not in self.actor_ids or actor != work.actor:
            raise Problem(403, "artifact_owner_unapproved", "Artifact owner is not allowlisted.")
        ceiling = self.capture_bytes if work.operation == "capture" else self.dataset_bytes
        if (
            work.max_bytes > ceiling
            or work.max_files > self.maximum_files
            or (work.deadline - work.created_at).total_seconds() > self.maximum_seconds
            or utcnow() >= work.deadline
        ):
            raise Problem(422, "artifact_budget_unapproved", "Original artifact budget is invalid.")
        self.registry.put(
            actor, f"artifact-operations/{work.id}/request.json", work.model_dump(mode="json")
        )
        current = self._state(actor, work.id)
        if current is None:
            value = ArtifactStatus(
                id=work.id,
                owner_key=actor.owner_key,
                project_id=work.project.id,
                target_id=work.target_id,
                operation=work.operation,
                work_sha256=work.sha256,
                created_at=work.created_at,
                updated_at=utcnow(),
                deadline=work.deadline,
                max_bytes=work.max_bytes,
                max_files=work.max_files,
                status="queued",
            )
            try:
                current = self.registry._write_record(
                    actor, f"artifact-operations/{work.id}/state.json", value, None
                )
            except Problem as exc:
                if exc.code != "registry_revision_conflict":
                    raise
                current = self._state(actor, work.id)
        if current is None or current.value.work_sha256 != work.sha256:
            raise Problem(
                409, "artifact_operation_conflict", "The immutable artifact inputs changed."
            )
        if current.value.status == "queued":
            self.registry.put(actor, f"artifact-pending/{work.id}.json", {"id": str(work.id)})
        return current.value

    def status(self, actor, operation_id):
        stored = self._state(actor, operation_id)
        return None if stored is None else stored.value

    def claim(self, actor, operation_id, claim_id):
        current = self._state(actor, operation_id)
        if current is None or current.value.status != "queued":
            return None
        if utcnow() >= current.value.deadline:
            self._save(
                actor,
                current,
                status="timed_out",
                phase="stopped",
                error_code="artifact_deadline",
                message="Queued time exhausted the original budget.",
            )
            return None
        try:
            return self._save(
                actor,
                current,
                status="running",
                phase="processing",
                claim_id=claim_id,
                heartbeat_at=utcnow(),
            )
        except Problem as exc:
            if exc.code != "registry_revision_conflict":
                raise
            return None

    def heartbeat(self, actor, operation_id, claim_id):
        current = self._state(actor, operation_id)
        if (
            current is None
            or current.value.status != "running"
            or current.value.claim_id != claim_id
        ):
            return False
        self._save(actor, current, heartbeat_at=utcnow())
        return True

    def finish(self, actor, operation_id, claim_id, *, status, error_code, message):
        current = self._state(actor, operation_id)
        if (
            current is None
            or current.value.status != "running"
            or current.value.claim_id != claim_id
        ):
            return self.status(actor, operation_id)
        return self._save(
            actor, current, status=status, phase="stopped", error_code=error_code, message=message
        )

    def recover(self, actor, operation_id):
        current = self._state(actor, operation_id)
        if current is None or current.value.status in ("ready", "failed", "timed_out"):
            return None if current is None else current.value
        completed = self.registry.get(actor, f"artifact-operations/{operation_id}/result.json")
        if completed is not None:
            try:
                result = ArtifactResult.model_validate(completed["result"])
                finished = TypeAdapter(AwareDatetime).validate_python(completed["completed_at"])
                if (
                    completed["work_sha256"] != current.value.work_sha256
                    or finished > current.value.deadline
                    or finished < current.value.created_at
                ):
                    raise ValueError("Original operation or deadline differs")
                if current.value.operation == "dataset" and (
                    result.artifact_id != current.value.target_id or result.capture is not None
                ):
                    raise ValueError("Sealed dataset identity differs from original target")
                if current.value.operation == "capture" and result.capture is None:
                    raise ValueError("Original capture receipt is missing")
            except (KeyError, ValueError, ValidationError) as exc:
                raise unavailable("Verified artifact completion metadata") from exc
            index = self.registry.artifact_index(actor, result.artifact_id)
            if index.get("manifest_sha256") != result.manifest_sha256 or (
                index.get("files", {}).get("manifest.json") != result.manifest_sha256
            ):
                raise Problem(
                    503, "artifact_manifest_unverified", "Final verified manifest is missing."
                )
            return self._save(
                actor,
                current,
                status="ready",
                phase="manifest_committed",
                result=result,
                error_code=None,
                message="The verified artifact manifest is committed.",
            )
        if current.value.status == "running" and (
            current.value.heartbeat_at is None
            or utcnow() - current.value.heartbeat_at >= timedelta(seconds=30)
        ):
            return self._save(
                actor,
                current,
                status="uncertain",
                phase="stopped",
                error_code="artifact_worker_lost",
                message="The previous processor was lost; incomplete heavy work was not replayed.",
            )
        return current.value

    def pending(self, actor):
        prefix = self.registry.key(actor, "artifact-pending/")
        result = []
        try:
            for item in self.registry.container.list_blobs(name_starts_with=prefix):
                if len(result) >= 50:
                    break
                name = item.name.removeprefix(prefix)
                if not item.name.startswith(prefix) or "/" in name or not name.endswith(".json"):
                    raise Problem(503, "artifact_queue_scope", "Unexpected pending operation path.")
                result.append(UUID(name[:-5]))
        except (AzureError, ValueError) as exc:
            raise unavailable("Scoped artifact operation queue") from exc
        return result

    def dequeue(self, actor, operation_id):
        try:
            self.registry.container.delete_blob(
                self.registry.key(actor, f"artifact-pending/{operation_id}.json")
            )
        except ResourceNotFoundError:
            pass
        except AzureError as exc:
            raise unavailable("Completed artifact queue pointer cleanup") from exc
