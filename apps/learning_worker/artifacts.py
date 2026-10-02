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
from apps.api.reference_models import ReferenceCollection
from apps.learning_worker.candidate_provenance import (
    VerifiedCommandJob,
    command_config,
    command_model,
    read_command_result,
    training_root_id,
    verify_candidate_record,
    verify_command_job,
    verify_command_result,
    verify_pipeline_result,
)
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
        budget=None,
    ):
        self.registry, self.credential = registry, credential
        self.capture_account_url = capture_account_url.rstrip("/")
        self.capture_container = capture_container
        self.allowed_policy_types = tuple(allowed_policy_types)
        self.budget = budget

    @staticmethod
    def _scope(actor):
        from learning.contract import Scope

        return Scope(str(actor.tenant_id), actor.owner_key)

    def _model(
        self,
        actor,
        root,
        expected_sha,
        policy_type,
        *,
        inference=True,
        execution_timing=None,
        verified_command: VerifiedCommandJob | None = None,
    ):
        if policy_type not in self.allowed_policy_types:
            raise Problem(
                503, "learning_policy_unapproved", "Model license and hardware are unapproved."
            )
        if verified_command is not None and (
            not isinstance(verified_command, VerifiedCommandJob)
            or verified_command.owner_key != actor.owner_key
            or execution_timing != "paused_simulation"
            or policy_type != "smolvla"
            or not inference
        ):
            raise Problem(503, "candidate_job_mismatch", "Verified command context is invalid.")
        module_name, _ = implementation(policy_type, model_use=True)
        if execution_timing is not None:
            if execution_timing != "paused_simulation" or policy_type != "smolvla":
                raise Problem(
                    409, "model_timing_mismatch", "Model mode and family are not supported."
                )
            module_name = "learning.paused"
        module_path = f"{module_name}.artifacts"
        if verified_command is not None:
            module_path = "learning.paused.command_artifacts"
        try:
            module = importlib.import_module(module_path)
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
        if verified_command is not None and (
            not command_model(result)
            or result["training"]["azure_job_id"] != verified_command.azure_job_id
            or result["training"]["specification_sha256"] != verified_command.specification_sha256
        ):
            raise Problem(
                503, "candidate_job_mismatch", "Model does not match the verified command."
            )
        return result

    @staticmethod
    def _model_timing(model):
        if model.get("execution_timing") is None:
            return {}
        profile = model["control_profile"]
        return {
            "execution_timing": model["execution_timing"],
            "real_time_admission": model["real_time_admission"],
            "control_profile_id": profile["profile_id"],
            "control_profile_sha256": fingerprint(profile),
            "criteria_sha256": model["criteria_sha256"],
            "frozen_plan_sha256": model["frozen_plan_sha256"],
        }

    def register_parent(
        self, actor, root: Path, expected_sha: str, registered_by: UUID, *, execution_timing=None
    ):
        model = self._model(
            actor,
            root,
            expected_sha,
            self._read_json(root / "model.json")["policy_type"],
            inference=False,
            execution_timing=execution_timing,
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
                or original.timing_fields() != self._model_timing(model)
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
            **self._model_timing(model),
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
            with BlobServiceClient(
                account_url,
                credential=self.credential,
                connection_timeout=5,
                read_timeout=10,
                retry_total=0,
            ) as client:
                bucket = client.get_container_client(container)
                count = 0
                for item in bucket.list_blobs(name_starts_with=prefix):
                    if self.budget:
                        self.budget.consume(0, files=1)
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
                        for chunk in bucket.download_blob(item.name).chunks():
                            if self.budget:
                                self.budget.consume(len(chunk))
                            output.write(chunk)
        except AzureError as exc:
            raise unavailable("Private owner-scoped artifact download") from exc

    def verify_capture(self, actor, project, session, raw_receipt):
        if (
            session.owner_key != actor.owner_key
            or project.owner_key != actor.owner_key
            or session.project_id != project.id
        ):
            raise Problem(
                403,
                "capture_scope_mismatch",
                "Capture verification requires this owner's project and session.",
            )
        case = session.teaching_case
        if case is None or project.selected_case(case.case_id) != case:
            raise Problem(
                409,
                "capture_case_unverified",
                "The teaching session has no approved immutable case.",
            )
        receipt = DemonstrationResult.model_validate(raw_receipt)
        if receipt.status != "uploaded":
            raise Problem(409, "capture_not_ready", "A verified uploaded manifest is required.")
        if project.execution_timing is not None and receipt.episode_id != session.command_id:
            raise Problem(
                409,
                "capture_provenance_mismatch",
                "Paused capture is not from this original command.",
            )
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
            try:
                validated = self._raw_dataset(actor, project, root)
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
            self._check_case_metadata(project, case, session.source, episode, manifest)
            if isinstance(session, ReferenceCollection) and (
                episode.get("provenance", {}).get("code_revision") != session.source_revision
                or episode.get("provenance", {}).get("simulator_image_digest")
                != session.simulator_image_digest
            ):
                raise Problem(
                    409,
                    "capture_provenance_mismatch",
                    "Reference runtime source/image differs from its grant.",
                )
            if (
                episode.get("episode_id") != str(receipt.episode_id)
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
                case_id=case.case_id,
                environment_id=episode["environment_id"],
                revision=episode["revision"],
                split=episode["split"],
                task_id=project.task_id,
                **({"control_profile_id": project.control_profile_id} | project.timing_fields()),
            )

    def _raw_dataset(self, actor, project, root, *, expected_sha=None):
        if project.execution_timing == "paused_simulation":
            from learning.paused.capture import validate_dataset

            return validate_dataset(
                root,
                expected_scope=self._scope(actor),
                expected_manifest_sha256=expected_sha,
                require_live=True,
                require_demonstrations=True,
            )
        from learning.contract import validate_dataset

        return validate_dataset(
            root,
            expected_scope=self._scope(actor),
            expected_manifest_sha256=expected_sha,
            require_live=True,
        )

    @staticmethod
    def _check_case_metadata(project, case, source_kind, episode, manifest):
        source = episode.get("demonstration")
        profile = manifest.get("control_profile", {})
        if (
            not isinstance(source, dict)
            or source.get("kind") != source_kind
            or source.get("task_id") != project.task_id
            or source.get("instruction") != project.instruction
            or source.get("goal_id") != project.goal_station_id
            or profile.get("profile_id") != project.control_profile_id
            or (
                episode.get("environment_id"),
                episode.get("revision"),
                episode.get("seed"),
                episode.get("split"),
            )
            != (case.environment_id, case.revision, case.seed, case.split)
        ):
            raise Problem(
                409,
                "capture_provenance_mismatch",
                "Actual capture does not match its approved case, split, source, task and profile.",
            )
        if project.execution_timing is not None and (
            manifest.get("schema") != "physicalai.demonstrations/v3"
            or manifest.get("execution_timing") != project.execution_timing
            or manifest.get("real_time_admission") is not False
            or manifest.get("purpose") != "demonstration"
            or manifest.get("timestamp_basis") != "simulation_time"
            or manifest.get("criteria_sha256") != project.criteria_sha256
            or manifest.get("frozen_plan_sha256") != project.frozen_plan_sha256
            or fingerprint(profile) != project.control_profile_sha256
        ):
            raise Problem(
                409,
                "capture_provenance_mismatch",
                "Paused capture mode, profile or frozen criteria differs from its project.",
            )

    def seal_dataset(self, actor, project, dataset_id, captures):
        from learning.common import file_digest

        if project.execution_timing == "paused_simulation":
            from learning.paused.capture import assemble_dataset
        else:
            from learning.capture import assemble_dataset

        if project.owner_key != actor.owner_key:
            raise Problem(
                403, "dataset_scope_mismatch", "Dataset sealing requires this owner's project."
            )
        with TemporaryDirectory(prefix="physicalai-seal-") as folder:
            roots = []
            for capture in captures:
                case = capture.authorized_case(project)
                if capture.episode_id in project.evaluation_plan.held_out_episode_ids:
                    raise Problem(
                        422, "held_out_overlap", "Test evidence cannot enter a teaching dataset."
                    )
                root = Path(folder) / str(capture.artifact_id)
                index = self.registry.download(actor, capture.artifact_id, root)
                if index.get("manifest_sha256") != capture.manifest_sha256:
                    raise Problem(
                        409,
                        "dataset_digest_mismatch",
                        "Capture was replaced before dataset sealing.",
                    )
                try:
                    actual = self._raw_dataset(
                        actor, project, root, expected_sha=capture.manifest_sha256
                    )
                except ValueError as exc:
                    raise Problem(
                        422,
                        "dataset_invalid",
                        "Actual capture inventory/splits failed verification.",
                    ) from exc
                if len(actual.episodes) != 1:
                    raise Problem(
                        409,
                        "capture_provenance_mismatch",
                        "A receipt must bind one original episode.",
                    )
                episode = actual.episodes[0].metadata
                self._check_case_metadata(project, case, capture.source, episode, actual.manifest)
                if (
                    episode["episode_id"] != str(capture.episode_id)
                    or episode["frame_count"] != capture.frame_count
                ):
                    raise Problem(
                        409,
                        "capture_provenance_mismatch",
                        "Downloaded episode differs from its verified receipt.",
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

    def _output_location(self, actor, specification, output_name):
        config = self.registry.job_configuration(actor, specification)
        if config is None:
            config = self.registry.approved_plan(actor, specification)["config"]
            if not str(config.get("schema", "")).endswith("/v1"):
                raise unavailable("Original immutable job output configuration")
        scoped = f"tenants/{actor.tenant_id}/owners/{actor.owner_key}/"
        prefix = config.get("output_prefix", "")
        if not prefix.startswith(scoped) or config.get("owner_id") != actor.owner_key:
            raise Problem(
                403, "output_scope_mismatch", "Azure outputs must stay in the approved owner scope."
            )
        account = config["storage_account_name"]
        if not re.fullmatch(r"[a-z0-9]{3,24}", account):
            raise Problem(503, "invalid_output_account", "Azure output account is not approved.")
        if command_config(config) and (
            config["run_id"] != specification.run.backend_job_name
            or config["tenant_id"] != str(actor.tenant_id)
            or config["specification_sha256"] != specification.run.specification_sha256
        ):
            raise Problem(
                503, "candidate_configuration_mismatch", "Command output binding differs."
            )
        return (
            config,
            f"https://{account}.blob.core.windows.net",
            config["blob_container"],
            f"{prefix.rstrip('/')}/{specification.run.backend_job_name}/{output_name}/",
        )

    def _output(self, actor, specification, output_name, destination):
        config, account, container, prefix = self._output_location(
            actor, specification, output_name
        )
        self._download_prefix(
            account,
            container,
            prefix,
            destination,
        )
        return config

    def completed_candidate(self, actor, specification, azure_job_id):
        from learning.common import canonical, digest, file_digest

        with TemporaryDirectory(prefix="physicalai-checkpoint-") as folder:
            output = Path(folder) / "model"
            config = self._output(actor, specification, "model", output)
            result = (
                read_command_result(output / "result.json")
                if command_config(config)
                else self._read_json(output / "result.json")
            )
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
            verified_command = (
                verify_command_job(self, actor, specification, config, azure_job_id)
                if command_config(config)
                else None
            )
            model = self._model(
                actor,
                root,
                expected,
                specification.project.policy_type,
                execution_timing=specification.project.execution_timing,
                verified_command=verified_command,
            )
            training = model["training"]
            if command_config(config):
                verify_command_result(
                    self,
                    actor,
                    specification,
                    config,
                    model,
                    result,
                    azure_job_id,
                    verified_job=verified_command,
                )
            else:
                verify_pipeline_result(model, result, azure_job_id)
            parent = specification.training_parent or specification.baseline
            if (
                parent is None
                or specification.dataset is None
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
                or self._model_timing(model) != specification.project.timing_fields()
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
                {
                    "manifest_sha256": expected,
                    "role": "trained_candidate",
                    **(
                        {
                            "training_execution": "azureml_command",
                            "azure_job_type": "command",
                            "azure_job_id": azure_job_id,
                        }
                        if verified_command is not None
                        else {}
                    ),
                },
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
                **(
                    {"control_profile_id": specification.project.control_profile_id}
                    | specification.project.timing_fields()
                ),
            )
            key = f"jobs/{specification.run.backend_job_name}/candidate.json"
            existing = self.registry.get(actor, key)
            if existing is not None:
                original = PolicyCandidate.model_validate(existing)
                if verified_command is not None and original.model_dump(
                    exclude={"created_at", "updated_at"}
                ) != candidate.model_dump(exclude={"created_at", "updated_at"}):
                    raise Problem(
                        503, "candidate_provenance_mismatch", "Existing command candidate differs."
                    )
                return original
            self.registry.put(actor, key, candidate.model_dump(mode="json"))
            return candidate

    def completed_report(self, actor, specification, azure_job_id, *, required=True):
        if getattr(specification.run, "provider", "azure_ml") == "managed_batch":
            raise Problem(
                409, "managed_import_required", "Managed reports use the bounded artifact import."
            )
        if specification.project.execution_timing == "paused_simulation":
            from apps.learning_worker.paused_reports import complete_verified_report

            return complete_verified_report(
                self, actor, specification, azure_job_id, required=required
            )
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

    def _metadata_client(self, config):
        from azure.ai.ml import MLClient

        return MLClient(
            self.credential,
            config["subscription_id"],
            config["resource_group"],
            config["workspace"],
        )

    def _blob_inventory(self, account, container, prefix, *, exact=False):
        entries, size = {}, 0
        try:
            with BlobServiceClient(
                account,
                credential=self.credential,
                connection_timeout=5,
                read_timeout=10,
                retry_total=0,
            ) as client:
                bucket = client.get_container_client(container)
                for item in bucket.list_blobs(name_starts_with=prefix):
                    if not item.name.startswith(prefix):
                        raise unavailable("Exact scoped artifact inventory")
                    if exact and item.name != prefix:
                        continue
                    if not isinstance(item.etag, str) or not item.etag:
                        raise unavailable("Artifact ETag inventory")
                    size += item.size
                    if len(entries) >= 100000 or size > 32 * 1024**3:
                        raise Problem(
                            503, "artifact_budget_exceeded", "Inventory exceeds its bound."
                        )
                    entries[item.name] = {"etag": item.etag, "size": item.size}
        except AzureError as exc:
            raise unavailable("Private artifact inventory") from exc
        return fingerprint(entries)

    def _download_file(self, account, container, key, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with BlobServiceClient(
                account,
                credential=self.credential,
                connection_timeout=5,
                read_timeout=10,
                retry_total=0,
            ) as client:
                download = client.get_container_client(container).download_blob(key)
                with destination.open("xb") as output:
                    for chunk in download.chunks():
                        size += len(chunk)
                        if size > 32 * 1024 * 1024:
                            raise unavailable("Bounded plan/report document")
                        output.write(chunk)
        except AzureError as exc:
            raise unavailable("Private immutable document") from exc

    def verify_candidate(self, actor, project, run, candidate):
        if candidate.training_origin != "api_training_run":
            raise Problem(
                409, "external_import_required", "External candidates use their import receipt."
            )
        with TemporaryDirectory(prefix="physicalai-verify-model-") as folder:
            root = Path(folder) / "candidate"
            index = self.registry.download(actor, candidate.artifact_id, root)
            model = self.registered_model(
                actor,
                candidate,
                root,
                index,
                execution_timing=project.execution_timing,
                project=project,
                run=run,
            )
            verify_candidate_record(candidate, model)
            if (
                training_root_id(model) != run.azure_job_id
                or self._model_timing(model) != project.timing_fields()
                or not candidate.matches_timing(project)
            ):
                raise Problem(
                    503, "candidate_job_mismatch", "Candidate references a different training run."
                )

    def registered_model(
        self, actor, record, root, index, *, execution_timing, project=None, run=None
    ):
        if index.get("training_origin") == "external_native_import":
            if execution_timing != "paused_simulation" or run is not None:
                raise Problem(
                    409, "external_import_required", "External import is not an API training run."
                )
            from apps.learning_worker.external_training import registered_model

            return registered_model(self, actor, record, root, index, project=project)
        if getattr(record, "training_origin", "api_training_run") == "external_native_import":
            raise Problem(
                409, "external_import_certificate", "Original external artifact index is required."
            )
        if "training_origin" in index or "external_import_id" in index:
            raise Problem(
                409,
                "external_import_certificate",
                "Unknown or partial external artifact provenance.",
            )
        if not any(
            name in index for name in ("training_execution", "azure_job_type", "azure_job_id")
        ):
            return self._model(
                actor,
                root,
                record.manifest_sha256,
                record.policy_type,
                execution_timing=execution_timing,
            )
        if (
            index.get("training_execution") != "azureml_command"
            or index.get("azure_job_type") != "command"
            or index.get("role") != "trained_candidate"
            or index.get("manifest_sha256") != record.manifest_sha256
            or not isinstance(index.get("azure_job_id"), str)
            or "azure_pipeline_job_id" in index
            or "azure_component_job_id" in index
        ):
            raise Problem(
                503, "candidate_job_mismatch", "Registered command provenance is incomplete."
            )
        job_name = index["azure_job_id"].rsplit("/", 1)[-1]
        if not re.fullmatch(r"learning-[a-f0-9-]{1,91}", job_name):
            raise Problem(503, "candidate_job_mismatch", "Registered command name is invalid.")
        candidate = record
        if not isinstance(record, PolicyCandidate):
            original = self.registry.get(actor, f"jobs/{job_name}/candidate.json")
            if original is None:
                raise Problem(
                    503, "candidate_job_mismatch", "Original command candidate is missing."
                )
            candidate = PolicyCandidate.model_validate(original)
            if (
                candidate.id != record.candidate_id
                or candidate.model_sha256 != record.model_sha256
                or candidate.artifact_id != record.artifact_id
            ):
                raise Problem(
                    503, "candidate_job_mismatch", "Release and command candidate differ."
                )
        if candidate.azure_job_id != index["azure_job_id"]:
            raise Problem(503, "candidate_job_mismatch", "Original command identity differs.")
        specification = self.registry.job(actor, job_name)
        if specification is None or (
            specification.run.kind != "training"
            or specification.run.id != candidate.training_run_id
            or specification.run.backend_job_name != job_name
            or specification.project.id != candidate.project_id
            or candidate.dataset_id != specification.run.dataset_id
            or candidate.owner_key != actor.owner_key
            or candidate.tenant_id != actor.tenant_id
            or candidate.parent_release_id != specification.run.parent_release_id
            or candidate.pretrained_artifact_id != specification.run.pretrained_artifact_id
            or (project is not None and specification.project != project)
            or (
                run is not None
                and any(
                    getattr(specification.run, field) != getattr(run, field)
                    for field in (
                        "id",
                        "project_id",
                        "backend_job_name",
                        "specification_sha256",
                        "deadline",
                        "dataset_id",
                        "optimizer_steps",
                    )
                )
            )
        ):
            raise Problem(
                503,
                "candidate_job_mismatch",
                "Original command candidate job registration differs.",
            )
        config, account, container, prefix = self._output_location(actor, specification, "model")
        if not command_config(config):
            raise Problem(
                503, "candidate_job_mismatch", "Command model requires its original config."
            )
        verified_command = verify_command_job(
            self, actor, specification, config, candidate.azure_job_id
        )
        model = self._model(
            actor,
            root,
            candidate.manifest_sha256,
            candidate.policy_type,
            execution_timing=execution_timing,
            verified_command=verified_command,
        )
        with TemporaryDirectory(prefix="physicalai-command-result-") as temporary:
            path = Path(temporary) / "result.json"
            self._download_file(account, container, prefix + "result.json", path)
            result = read_command_result(path)
        if result.get("model_manifest_sha256") != candidate.manifest_sha256:
            raise Problem(
                503, "candidate_digest_mismatch", "Original command result model differs."
            )
        verify_command_result(
            self,
            actor,
            specification,
            config,
            model,
            result,
            candidate.azure_job_id,
            verified_job=verified_command,
        )
        verify_candidate_record(candidate, model)
        return model

    def verify_report(self, actor, project, run, report):
        from apps.api.simulation_reports import MANAGED_REPORT_SCHEMA, SimulationReport

        if isinstance(report, SimulationReport) and report.native_schema == MANAGED_REPORT_SCHEMA:
            from apps.learning_worker.managed_reports import context, verified_receipt

            value = context(self.registry, actor, project.id, run.id)
            if (
                run.provider != "managed_batch"
                or value.binding.specification.project != project
                or value.reference.specification_sha256 != run.specification_sha256
                or verified_receipt(self, actor, value.reference).report != report
            ):
                raise Problem(
                    409, "managed_import_certificate", "Original verified import differs."
                )
            return

        if isinstance(report, SimulationReport):
            specification = self.registry.job(actor, run.backend_job_name)
            if specification is None:
                raise unavailable("Original evaluation job claim")
            actual = self.completed_report(actor, specification, run.azure_job_id)
            if actual != report:
                raise Problem(503, "report_digest_mismatch", "Verified report projection differs.")
            return
        index = self.registry.artifact_index(actor, report.artifact_id)
        if index.get("manifest_sha256") != report.report_sha256:
            raise Problem(503, "report_digest_mismatch", "The immutable report digest changed.")

    def report_document(self, actor, specification, report_sha):
        from apps.api.simulation_reports import SimulationReport

        saved = self.registry.get(
            actor, f"jobs/{specification.run.backend_job_name}/reports/{report_sha}.json"
        )
        if (
            not isinstance(saved, dict)
            or saved.get("specification_sha256") != specification.run.specification_sha256
        ):
            raise Problem(
                404, "report_not_verified", "No verified report exists for this owned job."
            )
        report = SimulationReport.model_validate(saved["report"])
        if report.report_sha256 != report_sha:
            raise Problem(503, "report_digest_mismatch", "Verified report identity changed.")
        if getattr(specification.run, "provider", "azure_ml") == "managed_batch":
            self.verify_report(actor, specification.project, specification.run, report)
        with TemporaryDirectory(prefix="physicalai-report-download-") as folder:
            root = Path(folder) / "verified"
            index = self.registry.download(
                actor, report.artifact_id, root, max_bytes=32 * 1024 * 1024
            )
            if index.get("files") != {"report.json": report_sha}:
                raise Problem(503, "report_digest_mismatch", "Native report inventory changed.")
            content = (root / "report.json").read_bytes()
            if hashlib.sha256(content).hexdigest() != report_sha:
                raise Problem(503, "report_digest_mismatch", "Native report content changed.")
            return content
