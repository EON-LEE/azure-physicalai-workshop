"""Closed trained-model job provenance, separate from physical quality or release authority."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from azure.core.exceptions import AzureError
from pydantic import Field, ValidationError, field_validator

from apps.api.errors import Problem, unavailable
from apps.api.models import Model, Revision

COMMAND_MODEL_SCHEMA = "physicalai.smolvla-checkpoint/v3"
COMMAND_RESULT_SCHEMA = "physicalai.smolvla-command-training-result/v1"
COMMAND_EXECUTION = {
    "schema": "physicalai.smolvla-command-execution/v1",
    "kind": "command",
    "data_transport": "private_blob_mi",
}


@dataclass(frozen=True)
class VerifiedCommandJob:
    owner_key: str
    azure_job_id: str
    specification_sha256: str
    configuration_sha256: str
    snapshot_sha256: str


class CommandTrainingResult(Model):
    schema_version: Literal["physicalai.smolvla-command-training-result/v1"] = Field(alias="schema")
    azure_job_id: str = Field(min_length=1, max_length=2048)
    azure_job_type: Literal["command"]
    specification_sha256: Revision
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    timestamp_basis: Literal["simulation_time"]
    criteria_sha256: Revision
    frozen_plan_sha256: Revision
    optimizer_steps: int = Field(strict=True, ge=1, le=100000)
    candidate: str = Field(pattern=r"^candidates/step-[0-9]{6}\Z", max_length=64)
    model_manifest_sha256: Revision
    learning_quality_verified: Literal[False]

    @field_validator("candidate")
    @classmethod
    def positive_checkpoint_step(cls, value):
        if not 1 <= int(value.rsplit("-", 1)[1]) <= 100000:
            raise ValueError("Candidate path must use a positive six-digit native checkpoint step.")
        return value

    @field_validator("real_time_admission", "learning_quality_verified", mode="before")
    @classmethod
    def explicit_false(cls, value):
        if value is not False:
            raise ValueError(
                "Training outputs do not confer physical quality or real-time admission."
            )
        return value


def read_command_result(path):
    from learning.common import read_json

    try:
        value = read_json(path, max_bytes=1024**2)
        CommandTrainingResult.model_validate(value)
    except (OSError, ValueError) as exc:
        raise Problem(503, "candidate_job_mismatch", "Original command result is invalid.") from exc
    return value


def command_config(config):
    if "job_execution" not in config:
        return False
    from learning.smolvla.azure import validate_config

    if config["job_execution"] != COMMAND_EXECUTION:
        raise Problem(
            503, "candidate_job_mismatch", "Unknown or partial command execution selector."
        )
    try:
        validate_config(config)
    except ValueError as exc:
        raise Problem(
            503, "candidate_configuration_mismatch", "Original command configuration is invalid."
        ) from exc
    return True


def command_model(model):
    training = model["training"]
    if not isinstance(training, dict):
        raise Problem(503, "candidate_job_mismatch", "An actual trained candidate is required.")
    if model["schema"] == COMMAND_MODEL_SCHEMA:
        if (
            model.get("training_execution") != "azureml_command"
            or training.get("azure_job_type") != "command"
            or not isinstance(training.get("azure_job_id"), str)
            or "azure_pipeline_job_id" in training
            or "azure_component_job_id" in training
            or "resume_from_pipeline_job_id" in training
        ):
            raise Problem(503, "candidate_job_mismatch", "Command model has mixed job provenance.")
        return True
    if (
        "training_execution" in model
        or "azure_job_type" in training
        or "azure_component_job_id" in training
        or "resume_from_job_type" in training
        or not isinstance(training.get("azure_job_id"), str)
        or not isinstance(training.get("azure_pipeline_job_id"), str)
        or training["azure_job_id"].lower() == training["azure_pipeline_job_id"].lower()
    ):
        raise Problem(
            503, "candidate_job_mismatch", "Pipeline model requires distinct root/component jobs."
        )
    return False


def training_root_id(model):
    return (
        model["training"]["azure_job_id"]
        if command_model(model)
        else model["training"]["azure_pipeline_job_id"]
    )


def verify_pipeline_result(model, result, azure_job_id):
    if (
        command_model(model)
        or "azure_job_type" in result
        or "training_execution" in result
        or "azure_pipeline_job_id" in result
        or result.get("schema") == COMMAND_RESULT_SCHEMA
        or result["azure_job_id"] != azure_job_id
        or model["training"]["azure_pipeline_job_id"] != azure_job_id
        or model["training"]["azure_job_id"] != result.get("azure_component_job_id")
    ):
        raise Problem(503, "candidate_job_mismatch", "Pipeline result has mixed job provenance.")


def command_snapshot(config):
    from learning.common import canonical, digest
    from learning.smolvla.embedded_source import static_inventory

    try:
        files = static_inventory(Path(__file__).resolve().parents[2], direct=True)
    except (OSError, ValueError) as exc:
        raise Problem(
            503, "candidate_source_mismatch", "Pinned command source inventory is unavailable."
        ) from exc
    if digest(canonical(files)) != config["source_delivery"]["static_sha256"]:
        raise Problem(
            503,
            "candidate_source_mismatch",
            "Worker source differs from the approved image source.",
        )
    return digest(canonical({**files, "run-config.json": digest(canonical(config) + b"\n")}))


def verify_command_job(verifier, actor, specification, config, azure_job_id):
    from learning.common import canonical, digest, utc

    if not command_config(config):
        raise Problem(503, "candidate_job_mismatch", "Original command selector is required.")
    run, project = specification.run, specification.project
    deadline = utc(config["job_deadline_utc"])
    if (
        specification.owner_key != actor.owner_key
        or project.owner_key != actor.owner_key
        or project.tenant_id != actor.tenant_id
        or project.actor_id != actor.object_id
        or run.owner_key != actor.owner_key
        or run.tenant_id != actor.tenant_id
        or run.actor_id != actor.object_id
        or run.project_id != project.id
        or run.kind != "training"
        or config["tenant_id"] != str(actor.tenant_id)
        or config["owner_id"] != actor.owner_key
        or config["run_id"] != run.backend_job_name
        or config["specification_sha256"] != run.specification_sha256
        or (run.azure_job_id is not None and run.azure_job_id != azure_job_id)
        or deadline > run.deadline
        or (run.job_deadline_utc is not None and run.job_deadline_utc != deadline)
    ):
        raise Problem(503, "candidate_job_mismatch", "Original command owner or deadline differs.")
    from learning.paused.command import validate_command_job

    snapshot = command_snapshot(config)
    try:
        client = verifier._metadata_client(config)
        job = client.jobs.get(run.backend_job_name)
        observed = validate_command_job(
            client, config, job, snapshot_sha256=snapshot, expected_status="Completed"
        )
    except AzureError as exc:
        raise unavailable("Actual completed Azure ML command") from exc
    except ValueError as exc:
        raise Problem(
            503, "candidate_job_mismatch", "Actual Azure ML command metadata differs from approval."
        ) from exc
    if observed != {
        "azure_job_id": azure_job_id,
        "azure_job_type": "command",
        "specification_sha256": run.specification_sha256,
    }:
        raise Problem(503, "candidate_job_mismatch", "Actual root command receipt differs.")
    return VerifiedCommandJob(
        actor.owner_key, azure_job_id, run.specification_sha256, digest(canonical(config)), snapshot
    )


def verify_command_result(
    verifier, actor, specification, config, model, result, azure_job_id, *, verified_job
):
    from learning.common import canonical, digest, utc
    from learning.smolvla.train import TrainOptions

    if not command_config(config) or not command_model(model):
        raise Problem(
            503, "candidate_job_mismatch", "Command config and model must both be explicit."
        )
    try:
        receipt = CommandTrainingResult.model_validate(result)
    except ValidationError as exc:
        raise Problem(
            503, "candidate_job_mismatch", "Command result provenance is invalid."
        ) from exc
    run, project, training = specification.run, specification.project, model["training"]
    parent = specification.training_parent or specification.baseline
    deadline = utc(config["job_deadline_utc"])
    if (
        not isinstance(verified_job, VerifiedCommandJob)
        or verified_job.owner_key != actor.owner_key
        or verified_job.azure_job_id != azure_job_id
        or verified_job.specification_sha256 != run.specification_sha256
        or verified_job.configuration_sha256 != digest(canonical(config))
    ):
        raise Problem(503, "candidate_job_mismatch", "Actual command verification is required.")
    snapshot = verified_job.snapshot_sha256
    options = TrainOptions(**config["parameters"])
    resumed = config["checkpointing"]["resume"]
    approved_updates = options.max_steps - (
        resumed["step"] if options.resume_mode == "full_state" and resumed is not None else 0
    )
    if (
        specification.owner_key != actor.owner_key
        or project.owner_key != actor.owner_key
        or project.tenant_id != actor.tenant_id
        or project.actor_id != actor.object_id
        or run.owner_key != actor.owner_key
        or run.tenant_id != actor.tenant_id
        or run.actor_id != actor.object_id
        or run.kind != "training"
        or run.project_id != project.id
        or run.policy_type != "smolvla"
        or project.policy_type != "smolvla"
        or project.execution_timing != "paused_simulation"
        or not run.matches_timing(project)
        or config["tenant_id"] != str(actor.tenant_id)
        or config["owner_id"] != actor.owner_key
        or config["run_id"] != run.backend_job_name
        or config["specification_sha256"] != run.specification_sha256
        or receipt.specification_sha256 != run.specification_sha256
        or training["specification_sha256"] != run.specification_sha256
        or receipt.azure_job_id != azure_job_id
        or training["azure_job_id"] != azure_job_id
        or (run.azure_job_id is not None and run.azure_job_id != azure_job_id)
        or deadline > run.deadline
        or (run.job_deadline_utc is not None and run.job_deadline_utc != deadline)
        or parent is None
        or specification.dataset is None
        or specification.dataset.id != run.dataset_id
        or specification.dataset.project_id != project.id
        or approved_updates != run.optimizer_steps
        or options.timeout_seconds > project.budget.training_seconds
        or (
            specification.training_parent is not None
            and (run.pretrained_artifact_id != parent.id or run.parent_release_id is not None)
        )
        or (
            specification.baseline is not None
            and (run.parent_release_id != parent.id or run.pretrained_artifact_id is not None)
        )
        or any(
            record.owner_key != actor.owner_key
            or record.tenant_id != actor.tenant_id
            or not record.matches_timing(project)
            for record in (parent, specification.dataset)
        )
        or config["inputs"]["parent_model"]["sha256"] != parent.model_sha256
        or training["parent_model_sha256"] != parent.model_sha256
        or config["inputs"]["demonstrations"]["sha256"] != specification.dataset.manifest_sha256
        or training["raw_manifest_sha256"] != specification.dataset.manifest_sha256
        or config["inputs"]["backbone"]["sha256"] != model["backbone_manifest_sha256"]
        or training["config_sha256"] != digest(canonical(asdict(options)))
        or training["code_snapshot_sha256"] != snapshot
        or training["optimizer_steps"] != receipt.optimizer_steps
        or receipt.optimizer_steps > run.optimizer_steps
        or training["checkpoint_step"] > options.max_steps
        or training["resume_mode"] != options.resume_mode
        or receipt.candidate != f"candidates/step-{training['checkpoint_step']:06d}"
        or receipt.real_time_admission is not False
        or result["learning_quality_verified"] is not False
        or verifier._model_timing(model) != project.timing_fields()
        or model["task"]
        != {
            "task_id": project.task_id,
            "instruction": project.instruction,
            "goal_id": project.goal_station_id,
        }
        or any(
            getattr(receipt, name) != getattr(project, name)
            or config[name] != getattr(project, name)
            for name in ("criteria_sha256", "frozen_plan_sha256")
        )
        or config["control_profile_sha256"] != project.control_profile_sha256
        or config["task_sha256"] != digest(canonical(model["task"]))
    ):
        raise Problem(
            503,
            "candidate_provenance_mismatch",
            "Command output differs from original job authority.",
        )
    if resumed is not None and (
        resumed.get("source_azure_job_type") != "command"
        or "source_azure_pipeline_job_id" in resumed
        or training.get("resume_from_job_id") != resumed["source_azure_job_id"]
        or training.get("resume_from_job_type") != "command"
        or training.get("resume_from_checkpoint_sha256") != resumed["checkpoint_sha256"]
        or training.get("resume_from_checkpoint_step") != resumed["step"]
        or training["conversion_sha256"] != config["inputs"]["converted_dataset"]["sha256"]
    ):
        raise Problem(
            503, "candidate_provenance_mismatch", "Original command resume source differs."
        )


def verify_candidate_record(candidate, model):
    training = model["training"]
    if (
        candidate.azure_job_id != training_root_id(model)
        or candidate.optimizer_steps != training["optimizer_steps"]
        or candidate.parent_model_sha256 != training["parent_model_sha256"]
        or candidate.source_commit != model["upstream"]["source_commit"]
        or candidate.model_revision != model["upstream"]["model_revision"]
        or candidate.processor_sha256 != model["processor_sha256"]
        or candidate.manifest_sha256 != candidate.model_sha256
    ):
        raise Problem(
            503, "candidate_provenance_mismatch", "Registered candidate metadata differs."
        )
