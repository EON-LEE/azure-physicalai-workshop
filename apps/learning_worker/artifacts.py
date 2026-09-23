from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, UUID, uuid5

from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    CaptureReceipt,
    PolicyCandidate,
    TrainingParent,
    fingerprint,
)
from apps.api.models import DemonstrationResult, utcnow
from apps.learning_worker.policies import implementation


class VerifiedArtifacts:
    def __init__(
        self,
        registry,
        credential,
        capture_account_url: str,
        capture_container: str,
        *,
        allowed_policy_types=(),
    ):
        self.registry, self.credential = registry, credential
        self.capture_account_url = capture_account_url.rstrip("/")
        self.capture_container = capture_container
        self.allowed_policy_types = tuple(allowed_policy_types)

    @staticmethod
    def _scope(actor):
        from learning.contract import Scope

        return Scope(str(actor.tenant_id), actor.owner_key)

    def _model(self, actor, root, expected_sha, policy_type, *, inference=True):
        if policy_type not in self.allowed_policy_types:
            raise Problem(
                503, "learning_policy_unapproved", "Model license and hardware are unapproved."
            )
        module_name, _ = implementation(policy_type, model_use=True)
        try:
            module = importlib.import_module(f"{module_name}.artifacts")
        except ModuleNotFoundError as exc:
            raise unavailable("Pinned policy artifact validator") from exc
        try:
            result = module.validate_model(
                root,
                expected_scope=self._scope(actor),
                expected_model_sha256=expected_sha,
                for_inference=inference,
            )
        except ValueError as exc:
            raise Problem(
                422, "invalid_model_artifact", "Model inventory/provenance verification failed."
            ) from exc
        if result["policy_type"] != policy_type:
            raise Problem(409, "model_family_mismatch", "Model versions cannot be relabeled.")
        return result

    def register_parent(self, actor, root: Path, expected_sha: str, registered_by: UUID):
        model = self._model(
            actor,
            root,
            expected_sha,
            self._read_json(root / "model.json")["policy_type"],
            inference=False,
        )
        if model["role"] != "pretrained" or model["training"] is not None:
            raise Problem(
                422, "not_train_only_parent", "Register vendor weights only as a train-only parent."
            )
        now = utcnow()
        artifact_id = uuid5(NAMESPACE_URL, f"{actor.owner_key}:training-parent:{expected_sha}")
        existing = self.registry.get(actor, f"training-parents/{artifact_id}.json")
        if existing is not None:
            original = TrainingParent.model_validate(existing)
            if (
                original.owner_key != actor.owner_key
                or original.model_sha256 != expected_sha
                or original.policy_type != model["policy_type"]
                or original.processor_sha256 != model["processor_sha256"]
            ):
                raise Problem(
                    409,
                    "immutable_registry_conflict",
                    "The original training-parent registration differs.",
                )
            return original
        self.registry.upload(
            actor,
            artifact_id,
            root,
            {"manifest_sha256": expected_sha, "role": "pretrained_train_only"},
        )
        record = TrainingParent(
            id=artifact_id,
            owner_key=actor.owner_key,
            actor_id=actor.object_id,
            tenant_id=actor.tenant_id,
            created_at=now,
            updated_at=now,
            fingerprint=expected_sha,
            policy_type=model["policy_type"],
            artifact_id=artifact_id,
            model_sha256=expected_sha,
            processor_sha256=model["processor_sha256"],
            source_commit=model["upstream"]["source_commit"],
            model_revision=model["upstream"]["model_revision"],
            registered_by=registered_by,
        )
        self.registry.put(
            actor, f"training-parents/{record.id}.json", record.model_dump(mode="json")
        )
        return record

    @staticmethod
    def _read_json(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise unavailable("Verified artifact metadata") from exc
        if not isinstance(value, dict):
            raise unavailable("Verified artifact metadata")
        return value

    @staticmethod
    def _relative(name: str):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise Problem(
                422, "artifact_path_invalid", "Artifact paths cannot escape their immutable root."
            )
        return path.parts

    def _download_prefix(self, account_url, container, prefix, root, *, max_bytes=32 * 1024**3):
        if not prefix.endswith("/") or ".." in PurePosixPath(prefix).parts:
            raise Problem(
                422, "artifact_prefix_invalid", "Artifact prefix must be explicit and scoped."
            )
        root.mkdir(parents=True, exist_ok=False)
        total = 0
        try:
            with BlobServiceClient(account_url, credential=self.credential) as client:
                bucket = client.get_container_client(container)
                count = 0
                for item in bucket.list_blobs(name_starts_with=prefix):
                    count += 1
                    total += item.size
                    if count > 100000 or total > max_bytes:
                        raise Problem(
                            503,
                            "artifact_budget_exceeded",
                            "The artifact download exceeds its bound.",
                        )
                    relative = item.name[len(prefix) :]
                    destination = root.joinpath(*self._relative(relative))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as output:
                        bucket.download_blob(item.name).readinto(output)
        except AzureError as exc:
            raise unavailable("Private owner-scoped artifact download") from exc

    def verify_capture(self, actor, project, session, raw_receipt):
        receipt = DemonstrationResult.model_validate(raw_receipt)
        if receipt.status != "uploaded":
            raise Problem(409, "capture_not_ready", "A verified uploaded manifest is required.")
        expected_prefix = (
            f"{self.capture_account_url}/{self.capture_container}/"
            f"{actor.owner_key}/{receipt.episode_id}/"
        )
        if receipt.manifest_uri != expected_prefix + "manifest.json":
            raise Problem(
                403,
                "capture_scope_mismatch",
                "Capture must reside in the fixed owner/episode prefix.",
            )
        with TemporaryDirectory(prefix="physicalai-capture-") as folder:
            root = Path(folder) / "capture"
            self._download_prefix(
                self.capture_account_url,
                self.capture_container,
                f"{actor.owner_key}/{receipt.episode_id}/",
                root,
                max_bytes=4 * 1024**3,
            )
            from learning.contract import validate_dataset

            try:
                validated = validate_dataset(
                    root, expected_scope=self._scope(actor), require_live=True
                )
            except ValueError as exc:
                raise Problem(
                    422, "capture_invalid", "Raw teaching data failed strict validation."
                ) from exc
            manifest = validated.manifest
            if validated.manifest_sha256 != receipt.manifest_sha256 or len(validated.episodes) != 1:
                raise Problem(
                    409,
                    "capture_digest_mismatch",
                    "The exact single-episode capture digest changed.",
                )
            episode = validated.episodes[0].metadata
            source = episode.get("demonstration")
            profile = manifest.get("control_profile", {})
            if (
                not isinstance(source, dict)
                or source.get("kind") != session.source
                or source.get("task_id") != project.task_id
                or source.get("instruction") != project.instruction
                or source.get("goal_id") != project.goal_station_id
                or profile.get("profile_id") != project.control_profile_id
                or episode.get("environment_id") != project.environment_id
                or episode.get("revision") != project.revision
                or episode.get("episode_id") != str(receipt.episode_id)
                or episode.get("frame_count") != receipt.frame_count
            ):
                raise Problem(
                    409,
                    "capture_provenance_mismatch",
                    "Source/task/profile/scene differs from approved teaching.",
                )
            artifact_id = uuid5(
                NAMESPACE_URL, f"{actor.owner_key}:capture:{receipt.manifest_sha256}"
            )
            self.registry.upload(
                actor,
                artifact_id,
                root,
                {"manifest_sha256": receipt.manifest_sha256, "role": "teaching_capture"},
            )
            return CaptureReceipt(
                episode_id=receipt.episode_id,
                manifest_sha256=receipt.manifest_sha256,
                artifact_id=artifact_id,
                frame_count=receipt.frame_count,
                source=session.source,
                seed=episode["seed"],
                task_id=project.task_id,
                control_profile_id=project.control_profile_id,
            )

    def seal_dataset(self, actor, project, dataset_id, captures):
        from learning.capture import assemble_dataset
        from learning.common import file_digest

        with TemporaryDirectory(prefix="physicalai-seal-") as folder:
            roots = []
            for capture in captures:
                root = Path(folder) / str(capture.artifact_id)
                index = self.registry.download(actor, capture.artifact_id, root)
                if index.get("manifest_sha256") != capture.manifest_sha256:
                    raise Problem(
                        409,
                        "dataset_digest_mismatch",
                        "Capture was replaced before dataset sealing.",
                    )
                roots.append(root)
            output = Path(folder) / "sealed"
            try:
                assemble_dataset(
                    roots,
                    output,
                    dataset_id=str(dataset_id),
                    expected_scope=self._scope(actor),
                    require_live=True,
                )
            except ValueError as exc:
                raise Problem(
                    422, "dataset_invalid", "Source/split/profile validation failed."
                ) from exc
            digest = file_digest(output / "manifest.json")
            self.registry.upload(
                actor, dataset_id, output, {"manifest_sha256": digest, "role": "sealed_dataset"}
            )
            return dataset_id, digest

    def _output(self, actor, specification, output_name, destination):
        approval = self.registry.approved_plan(actor, specification)
        config = approval["config"]
        scoped = f"tenants/{actor.tenant_id}/owners/{actor.owner_key}/"
        prefix = config.get("output_prefix", "")
        if not prefix.startswith(scoped) or config.get("owner_id") != actor.owner_key:
            raise Problem(
                403, "output_scope_mismatch", "Azure outputs must stay in the approved owner scope."
            )
        account = config["storage_account_name"]
        if not re.fullmatch(r"[a-z0-9]{3,24}", account):
            raise Problem(503, "invalid_output_account", "Azure output account is not approved.")
        self._download_prefix(
            f"https://{account}.blob.core.windows.net",
            config["blob_container"],
            f"{prefix.rstrip('/')}/{specification.run.backend_job_name}/{output_name}/",
            destination,
        )
        return config

    def completed_candidate(self, actor, specification, azure_job_id):
        from learning.common import canonical, digest, file_digest

        with TemporaryDirectory(prefix="physicalai-checkpoint-") as folder:
            output = Path(folder) / "model"
            config = self._output(actor, specification, "model", output)
            result = self._read_json(output / "result.json")
            if result.get("azure_job_id") != azure_job_id or not re.fullmatch(
                r"candidates/step-[0-9]+", str(result.get("candidate", ""))
            ):
                raise Problem(
                    503, "candidate_job_mismatch", "No final result for the exact owned Azure job."
                )
            root = output.joinpath(*self._relative(result["candidate"]))
            expected = result.get("model_manifest_sha256")
            if not isinstance(expected, str) or file_digest(root / "model.json") != expected:
                raise Problem(
                    503,
                    "candidate_digest_mismatch",
                    "Final model manifest digest is missing or changed.",
                )
            model = self._model(actor, root, expected, specification.project.policy_type)
            training = model["training"]
            parent = specification.training_parent or specification.baseline
            if (
                parent is None
                or specification.dataset is None
                or training["azure_pipeline_job_id"] != azure_job_id
                or training["azure_job_id"] != result.get("azure_component_job_id")
                or training["parent_model_sha256"] != parent.model_sha256
                or training["raw_manifest_sha256"] != specification.dataset.manifest_sha256
                or training["optimizer_steps"] != result["optimizer_steps"]
                or model["task"]
                != {
                    "task_id": specification.project.task_id,
                    "instruction": specification.project.instruction,
                    "goal_id": specification.project.goal_station_id,
                }
                or digest(canonical(model["control_profile"])) != config["control_profile_sha256"]
                or digest(canonical(model["task"])) != config["task_sha256"]
            ):
                raise Problem(
                    503,
                    "candidate_provenance_mismatch",
                    "Actual optimizer output does not match this approved training.",
                )
            candidate_id = uuid5(
                NAMESPACE_URL, f"{actor.owner_key}:candidate:{azure_job_id}:{expected}"
            )
            self.registry.upload(
                actor,
                candidate_id,
                root,
                {"manifest_sha256": expected, "role": "trained_candidate"},
            )
            now = utcnow()
            candidate = PolicyCandidate(
                id=candidate_id,
                owner_key=actor.owner_key,
                actor_id=actor.object_id,
                tenant_id=actor.tenant_id,
                created_at=now,
                updated_at=now,
                fingerprint=fingerprint({"job": azure_job_id, "manifest": expected}),
                project_id=specification.project.id,
                dataset_id=specification.dataset.id,
                training_run_id=specification.run.id,
                parent_release_id=specification.run.parent_release_id,
                pretrained_artifact_id=specification.run.pretrained_artifact_id,
                policy_type=model["policy_type"],
                model_sha256=expected,
                parent_model_sha256=parent.model_sha256,
                processor_sha256=model["processor_sha256"],
                manifest_sha256=expected,
                artifact_id=candidate_id,
                optimizer_steps=training["optimizer_steps"],
                azure_job_id=azure_job_id,
                source_commit=model["upstream"]["source_commit"],
                model_revision=model["upstream"]["model_revision"],
                control_profile_id=specification.project.control_profile_id,
            )
            key = f"jobs/{specification.run.backend_job_name}/candidate.json"
            existing = self.registry.get(actor, key)
            if existing is not None:
                return PolicyCandidate.model_validate(existing)
            self.registry.put(actor, key, candidate.model_dump(mode="json"))
            return candidate

    def completed_report(self, actor, specification, azure_job_id, *, required=True):
        from apps.learning_worker.reports import project_report

        with TemporaryDirectory(prefix="physicalai-evaluation-") as folder:
            output = Path(folder) / "report"
            config = self._output(actor, specification, "report", output)
            if not (output / "report.json").is_file() and not required:
                return None
            report = self._read_json(output / "report.json")
            if report.get("azure_job_id") != azure_job_id:
                raise Problem(
                    503,
                    "evaluation_job_mismatch",
                    "Paired report does not bind the exact Azure evaluation job.",
                )
            run = specification.run.model_copy(update={"azure_job_id": azure_job_id})
            bound = specification.model_copy(update={"run": run})
            value = project_report(
                actor,
                bound,
                report,
                report_sha256=hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
                expected_plan_sha256=config["inputs"]["plan"]["sha256"],
            )
            self.registry.upload(
                actor,
                value.artifact_id,
                output,
                {"manifest_sha256": value.report_sha256, "role": "evaluation_report"},
            )
            return value

    def verify_candidate(self, actor, project, run, candidate):
        with TemporaryDirectory(prefix="physicalai-verify-model-") as folder:
            root = Path(folder) / "candidate"
            self.registry.download(actor, candidate.artifact_id, root)
            model = self._model(actor, root, candidate.manifest_sha256, candidate.policy_type)
            if model["training"]["azure_pipeline_job_id"] != run.azure_job_id:
                raise Problem(
                    503, "candidate_job_mismatch", "Candidate references a different training run."
                )

    def verify_report(self, actor, project, run, report):
        index = self.registry.artifact_index(actor, report.artifact_id)
        if index.get("manifest_sha256") != report.report_sha256:
            raise Problem(503, "report_digest_mismatch", "The immutable report digest changed.")
