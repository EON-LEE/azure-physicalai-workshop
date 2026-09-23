from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from uuid import UUID

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import PolicyRelease, TrainingParent
from apps.api.learning_ports import JobSpecification
from apps.api.models import Principal


class BlobRegistry:
    def __init__(self, account_url: str, container: str, credential):
        self.client = BlobServiceClient(account_url, credential=credential)
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

    def claim_job(self, actor, specification):
        return self.put(
            actor,
            f"jobs/{specification.run.backend_job_name}/specification.json",
            specification.model_dump(mode="json"),
        )

    def job(self, actor, name):
        value = self.get(actor, f"jobs/{name}/specification.json")
        return None if value is None else JobSpecification.model_validate(value)

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
