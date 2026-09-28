"""A genuine standalone AML command, with scoped private MI transfer and native CUDA training."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from learning.common import canonical, digest, integer, keys, require, sha256, token
from learning.deadlines import JobDeadline

EXECUTION = {
    "schema": "physicalai.smolvla-command-execution/v1",
    "kind": "command",
    "data_transport": "private_blob_mi",
}
MODEL_SCHEMA = "physicalai.smolvla-checkpoint/v3"
CHECKPOINT_SCHEMA = "physicalai.smolvla-training-checkpoint/v2"
RESULT_SCHEMA = "physicalai.smolvla-command-training-result/v1"
P1_TRAINING_COHORT = {
    "schema": "physicalai.smolvla-training-cohort/v1",
    "kind": "p1_additional20",
}


class CommandDeadline(JobDeadline):
    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config["job_deadline_utc"], **kwargs)
        self._execution_end = self._monotonic() + integer(
            config["parameters"]["timeout_seconds"],
            "command execution timeout",
            1,
            86400,
        )

    def remaining_seconds(self) -> float:
        return min(super().remaining_seconds(), self._execution_end - self._monotonic())


def is_direct(config: dict) -> bool:
    return config.get("job_execution") == EXECUTION


def validate_execution(config: dict) -> None:
    require(
        config.get("job_execution") == EXECUTION, "Unknown standalone command execution variant"
    )
    require(
        config.get("schema") == "physicalai.smolvla-azure/v2"
        and config.get("kind") == "train"
        and config.get("execution_timing") == "paused_simulation"
        and config.get("real_time_admission") is False
        and config.get("source_delivery", {}).get("mode") == "image_embedded"
        and isinstance(config.get("checkpointing"), dict),
        "Direct command requires explicit paused-v2 image delivery and checkpointing",
    )
    require(
        re.fullmatch(r"[a-z0-9]{3,24}", config["storage_account_name"]) is not None,
        "Unapproved storage account name",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", config["blob_container"]) is not None,
        "Unapproved Blob container name",
    )
    require(
        config["compute_size"] == "Standard_NC24ads_A100_v4"
        and config["compute_tier"] == "LowPriority",
        "Standalone training is limited to the approved one-A100 LowPriority compute",
    )
    retained = f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/learning/outputs"
    require(
        config["output_prefix"] == retained or config["output_prefix"].startswith(retained + "/"),
        "Standalone training output must use the approved retained owner prefix",
    )


def validate_training_cohort(config: dict) -> None:
    require(
        config.get("training_cohort") == P1_TRAINING_COHORT,
        "Unknown explicit additional training cohort",
    )
    validate_execution(config)
    require(
        config["parameters"].get("resume_mode") == "weights_only"
        and "resume" in config["checkpointing"]
        and config["checkpointing"]["resume"] is None,
        "P1 requires P0 weights with a fresh optimizer, not checkpoint continuation",
    )


def validate_training_episodes(config: dict, episodes: list[dict]) -> None:
    additional = "training_cohort" in config
    if additional:
        validate_training_cohort(config)
    expected = set(range(11001, 11021)) if additional else set(range(10001, 10021))
    require(
        isinstance(episodes, list)
        and len(episodes) == 20
        and {episode["seed"] for episode in episodes} == expected
        and all(episode["split"] == "train" for episode in episodes),
        "Standalone training requires exactly the explicitly approved twenty TRAIN seeds",
    )
    if additional:
        require(
            len({episode["episode_id"] for episode in episodes}) == 20
            and all(
                episode["demonstration"]["kind"] == "reference_controller" for episode in episodes
            ),
            "P1 requires twenty unique reference demonstrations, not learned/relabeled data",
        )


def validate_p1_training(config: dict, data: dict, parent: dict) -> None:
    """Check native-validated data/model bindings; never infer a selector from data."""
    from learning.gr00t.azure import workspace_id
    from learning.paused.contract import PausedControlProfile

    validate_training_cohort(config)
    validate_training_episodes(config, data["episodes"])
    training = parent.get("training")
    require(
        parent.get("schema") == MODEL_SCHEMA
        and parent.get("role") == "candidate"
        and parent.get("training_execution") == "azureml_command"
        and isinstance(training, dict)
        and training.get("azure_job_type") == "command",
        "P1 must warm-start the genuine command-trained P0 candidate",
    )
    expected_scope = {"tenant_id": config["tenant_id"], "owner_id": config["owner_id"]}
    require(
        data["scope"] == parent["scope"] == expected_scope
        and parent["control_profile"] == data["control_profile"]
        and PausedControlProfile(**data["control_profile"]).sha256
        == config["control_profile_sha256"]
        and all(
            data[name] == parent[name] == config[name]
            for name in ("criteria_sha256", "frozen_plan_sha256")
        ),
        "P1 data/parent scope, profile or frozen conditions changed",
    )
    require(
        digest(canonical(parent["task"])) == config["task_sha256"]
        and all(
            {name: episode["demonstration"][name] for name in ("task_id", "instruction", "goal_id")}
            == parent["task"]
            for episode in data["episodes"]
        ),
        "P1 original reference task differs from the P0 parent",
    )
    prior = training["episodes"]
    require(
        len(prior) == 20
        and {episode["seed"] for episode in prior} == set(range(10001, 10021))
        and len({episode["episode_id"] for episode in prior}) == 20,
        "P1 parent must contain exactly the original P0 TRAIN20 lineage",
    )
    require(
        {episode["episode_id"] for episode in prior}.isdisjoint(
            episode["episode_id"] for episode in data["episodes"]
        )
        and {episode["seed"] for episode in prior}.isdisjoint(
            episode["seed"] for episode in data["episodes"]
        )
        and training["raw_manifest_sha256"] != config["inputs"]["demonstrations"]["sha256"],
        "Additional P1 data overlaps or relabels the original P0 data",
    )
    job = training["azure_job_id"]
    workspace = workspace_id(config) + "/jobs/"
    require(
        isinstance(job, str)
        and job.startswith(workspace)
        and re.fullmatch(r"[A-Za-z0-9_.-]+", job[len(workspace) :])
        and job != workspace + config["run_id"],
        "P0 parent must be a distinct actual command in the approved workspace",
    )
    if "raw_manifest_sha256" in data:
        require(
            data["raw_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"],
            "P1 converted data is not the approved additional raw manifest",
        )


def validate_p1_parent_job(client, config: dict, parent: dict) -> None:
    """Confirm the actual completed P0 root without substituting its historical config/source."""
    training = parent["training"]
    job_id = training["azure_job_id"]
    name = job_id.rsplit("/", 1)[-1]
    actual = client.jobs.get(name)
    require(
        actual.id == job_id
        and actual.name == name
        and actual.type == "command"
        and actual.status == "Completed"
        and not getattr(actual, "parent_job_name", None),
        "P1 parent is not the actual completed standalone P0 command",
    )
    expected = {
        "scope_tenant": config["tenant_id"],
        "scope_owner": config["owner_id"],
        "job_execution": "command",
        "policy_type": "smolvla",
        "execution_timing": "paused_simulation",
        "real_time_admission": "false",
        "specification_sha256": training["specification_sha256"],
        "code_snapshot_sha256": training["code_snapshot_sha256"],
        "control_profile_sha256": config["control_profile_sha256"],
        "task_sha256": config["task_sha256"],
        "criteria_sha256": config["criteria_sha256"],
        "frozen_plan_sha256": config["frozen_plan_sha256"],
    }
    require(
        all((actual.tags or {}).get(key) == value for key, value in expected.items())
        and "training_cohort" not in (actual.tags or {}),
        "P0 parent job scope/source/provenance does not match the approved trained model",
    )


def validate_job(job, config: dict, *, snapshot_sha256: str | None = None) -> None:
    from learning.gr00t.azure import job_tags, workspace_id

    require(
        getattr(job, "id", "").lower()
        == (workspace_id(config) + "/jobs/" + config["run_id"]).lower()
        and job.name == config["run_id"]
        and getattr(job, "type", None) == "command"
        and not getattr(job, "parent_job_name", None),
        "Actual Azure job is not the approved standalone command",
    )
    require(
        not getattr(job, "code", None) and not getattr(job, "inputs", None),
        "Standalone command unexpectedly uses service-prepared code or inputs",
    )
    # Azure can expose its system workspace artifacts output, but never the custom mounts.
    outputs = getattr(job, "outputs", {}) or {}
    require(
        set(outputs).issubset({"default"}), "Standalone command unexpectedly mounts custom outputs"
    )
    if outputs:
        system = outputs["default"]
        require(
            getattr(system, "path", None)
            == "azureml://datastores/workspaceartifactstore/ExperimentRun/dcid." + config["run_id"]
            and getattr(system, "type", None) == "uri_folder"
            and getattr(system, "mode", None) == "rw_mount",
            "Only the observed job-bound system artifact output is permitted",
        )
    compute = getattr(job, "compute", None)
    require(
        compute
        in (
            config["compute"],
            f"azureml:{config['compute']}",
            workspace_id(config) + "/computes/" + config["compute"],
        ),
        "Actual standalone command uses unapproved compute",
    )
    identity = getattr(job, "identity", None)
    kind = getattr(identity, "type", None)
    require(
        getattr(kind, "value", kind) in ("managed", "Managed", "managed_identity")
        and getattr(identity, "client_id", None) == config["managed_identity_client_id"],
        "Actual standalone command is not using the approved explicit managed identity",
    )
    checksum = sha256(snapshot_sha256 or (job.tags or {}).get("code_snapshot_sha256"))
    expected = job_tags(config, checksum, policy_type="smolvla")
    require(
        all((job.tags or {}).get(key) == value for key, value in expected.items()),
        "Actual command source/config/deadline/owner tags differ",
    )


def validate_command_job(
    client, config: dict, job, *, snapshot_sha256: str, expected_status: str
) -> dict:
    """Validate an actual root GET without re-admitting expired terminal work."""
    from learning.gr00t.azure import workspace_id
    from learning.offline import OFFLINE_ENV
    from learning.smolvla.azure import validate_config
    from learning.smolvla.embedded_source import embedded_command

    validate_config(config)
    validate_execution(config)
    validate_job(job, config, snapshot_sha256=snapshot_sha256)
    require(
        expected_status in ("Running", "Completed", "Failed", "Canceled")
        and job.status == expected_status,
        "Azure did not confirm the required actual command status",
    )
    require(
        getattr(job, "command", None)
        == embedded_command(config, snapshot_sha256, stage="command-train"),
        "Actual command entry/config/source binding differs from the approved native command",
    )
    resources = getattr(job, "resources", None)
    require(
        getattr(resources, "instance_count", None) == 1
        and getattr(job, "distribution", None) is None
        and getattr(getattr(job, "limits", None), "timeout", None)
        == config["parameters"]["timeout_seconds"],
        "Actual command changed its single-node execution/time budget",
    )
    require(
        all(
            (getattr(job, "environment_variables", None) or {}).get(name) == value
            for name, value in {**OFFLINE_ENV, "PYTHONDONTWRITEBYTECODE": "1"}.items()
        ),
        "Actual command changed required offline/Hugging Face publication guards",
    )
    environment = getattr(job, "environment", None)
    if isinstance(environment, str):
        prefix = workspace_id(config) + "/environments/"
        if environment.startswith(prefix):
            parts = environment[len(prefix) :].split("/")
            require(len(parts) == 3 and parts[1] == "versions", "Unversioned command environment")
            name, version = parts[0], parts[2]
        else:
            # jobs.get() shortens only workspace-local IDs to name:version.
            parts = environment.split(":")
            require(len(parts) == 2, "Command environment is not an exact versioned reference")
            name, version = parts
        token(name, "environment name")
        token(version, "environment version")
        environment = client.environments.get(name, version=version)
    require(
        getattr(environment, "image", None) == config["environment_image"],
        "Actual command image digest differs from its approved config",
    )
    return {
        "azure_job_id": job.id,
        "azure_job_type": "command",
        "specification_sha256": config["specification_sha256"],
    }


def running_command_binding(client, config: dict, *, snapshot_sha256: str) -> dict:
    from learning.smolvla.azure import job_deadline, validate_config

    validate_config(config)
    deadline = job_deadline(config)
    deadline.check()
    require(
        os.environ.get("AZUREML_RUN_ID") == config["run_id"],
        "Missing/different actual AML command context",
    )
    job = client.jobs.get(config["run_id"])
    binding = validate_command_job(
        client,
        config,
        job,
        snapshot_sha256=snapshot_sha256,
        expected_status="Running",
    )
    deadline.check()
    return binding


def checkpoint_origin(binding: dict, *, config: dict, snapshot_sha256: str) -> dict:
    value = {
        "azure_job_id": binding["azure_job_id"],
        "azure_job_type": "command",
        "specification_sha256": binding["specification_sha256"],
        "code_snapshot_sha256": snapshot_sha256,
        "job_deadline_utc": config["job_deadline_utc"],
        "test_only": False,
    }
    return value


def build_command(config: dict, snapshot_sha256: str, job_name: str) -> dict:
    from learning.gr00t.azure import job_tags
    from learning.offline import OFFLINE_ENV
    from learning.smolvla.embedded_source import embedded_command

    validate_execution(config)
    sha256(snapshot_sha256)
    require(
        job_name == config["run_id"], "Standalone job name must equal the newly approved run_id"
    )
    token(job_name, "new standalone AML job")
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
        "type": "command",
        "name": job_name,
        "display_name": job_name,
        "experiment_name": "physicalai-smolvla-command",
        "command": embedded_command(config, snapshot_sha256, stage="command-train"),
        "environment": {"image": config["environment_image"]},
        "environment_variables": {**OFFLINE_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        "identity": {"type": "managed", "client_id": config["managed_identity_client_id"]},
        "compute": f"azureml:{config['compute']}",
        "resources": {"instance_count": 1, "shm_size": "8g"},
        "limits": {"timeout": config["parameters"]["timeout_seconds"]},
        "tags": job_tags(config, snapshot_sha256, policy_type="smolvla"),
    }


def _manifest_scope(manifest: dict, config: dict) -> None:
    require(
        manifest.get("scope") == {"tenant_id": config["tenant_id"], "owner_id": config["owner_id"]},
        "Private manifest tenant/owner differs",
    )


def _registered_inputs(client, config: dict) -> None:
    from learning.azure import registered_input_matches

    datastore = client.datastores.get(config["datastore"])
    credentials = getattr(datastore, "credentials", None)
    credential_type = getattr(credentials, "type", None)
    require(
        getattr(datastore.type, "value", datastore.type) == "AzureBlob"
        and datastore.account_name == config["storage_account_name"]
        and datastore.container_name == config["blob_container"]
        and getattr(credential_type, "value", credential_type) == "None",
        "Command transport requires the exact registered private keyless Blob datastore",
    )
    for asset in config["inputs"].values():
        actual = client.data.get(asset["name"], version=asset["version"])
        require(
            registered_input_matches(actual.type, actual.path, asset),
            "Actual registered input version/location changed before direct transfer",
        )


def _download_inputs(transfer, config: dict, root: Path) -> tuple[Path, Path, Path, Path | None]:
    from learning.common import parse_json
    from learning.contract import CAMERAS, Scope
    from learning.paused.blob_transfer import input_prefix
    from learning.paused.capture import MAX_FRAME_BYTES, validate_dataset
    from learning.paused.command_artifacts import validate_parent
    from learning.paused.contract import PausedControlProfile
    from learning.smolvla import UPSTREAM
    from learning.smolvla.artifacts import validate_backbone

    scope = Scope(config["tenant_id"], config["owner_id"])
    assets = config["inputs"]
    prefix = input_prefix(config, "demonstrations")
    raw, payload = transfer.manifest(prefix, "manifest.json", assets["demonstrations"]["sha256"])
    _manifest_scope(raw, config)
    require(
        raw.get("schema") == "physicalai.demonstrations/v3"
        and raw.get("purpose") == "demonstration"
        and raw.get("execution_timing") == "paused_simulation"
        and raw.get("real_time_admission") is False
        and raw.get("criteria_sha256") == config["criteria_sha256"]
        and raw.get("frozen_plan_sha256") == config["frozen_plan_sha256"]
        and PausedControlProfile(**raw["control_profile"]).sha256
        == config["control_profile_sha256"],
        "Private raw dataset is not the exact approved paused TRAIN data",
    )
    episodes = raw["episodes"]
    validate_training_episodes(config, episodes)
    require(
        all(
            digest(
                canonical(
                    {
                        name: episode["demonstration"][name]
                        for name in ("task_id", "instruction", "goal_id")
                    }
                )
            )
            == config["task_sha256"]
            for episode in episodes
        ),
        "Approved TRAIN task/instruction differs from runtime config",
    )
    metadata, files = {"manifest.json": payload}, {}
    for episode in episodes:
        episode_id = token(episode["episode_id"], "actual episode")
        count = integer(episode["frame_count"], "actual frame count", 2, 600)
        require(episode["path"] == f"episodes/{episode_id}/frames.jsonl", "Unsafe raw episode path")
        # Frame streams are signed by the raw manifest and bound before any image reads.
        _, frame_bytes = transfer.manifest_bytes(
            prefix, episode["path"], episode["sha256"], maximum=count * MAX_FRAME_BYTES
        )
        lines = frame_bytes.splitlines()
        require(len(lines) == count, "Raw frame count differs from the approved manifest")
        metadata[episode["path"]] = frame_bytes
        for index, line in enumerate(lines):
            require(len(line) <= MAX_FRAME_BYTES, "Oversized raw frame")
            frame = parse_json(line)
            for camera, image in keys(frame["images"], set(CAMERAS), "actual cameras").items():
                name = f"episodes/{episode_id}/{camera}/{index:08d}.png"
                require(
                    image["path"] == name and name not in files, "Unsafe or repeated raw image path"
                )
                files[name] = sha256(image["sha256"])
    raw_root = root / "raw"
    transfer.download_files(prefix, files, raw_root, metadata=metadata)
    validate_dataset(
        raw_root,
        expected_scope=scope,
        expected_manifest_sha256=assets["demonstrations"]["sha256"],
        require_live=True,
        require_demonstrations=True,
    )
    parent_prefix = input_prefix(config, "parent_model")
    parent, parent_bytes = transfer.manifest(
        parent_prefix, "model.json", assets["parent_model"]["sha256"]
    )
    _manifest_scope(parent, config)
    require(
        parent.get("control_profile") == raw["control_profile"]
        and parent.get("criteria_sha256") == config["criteria_sha256"]
        and parent.get("frozen_plan_sha256") == config["frozen_plan_sha256"],
        "Parent model metadata differs from actual data/profile/criteria",
    )
    parent_files = {
        "checkpoint/" + name: checksum for name, checksum in parent["checkpoint_files"].items()
    }
    if parent["role"] == "pretrained":
        parent_files["LEROBOT-LICENSE"] = UPSTREAM["source_license_sha256"]
    parent_root = root / "parent"
    transfer.download_files(
        parent_prefix, parent_files, parent_root, metadata={"model.json": parent_bytes}
    )
    validated_parent = validate_parent(
        parent_root,
        expected_scope=scope,
        expected_model_sha256=assets["parent_model"]["sha256"],
        for_inference=False,
    )
    if "training_cohort" in config:
        validate_p1_training(config, raw, validated_parent)
    backbone_prefix = input_prefix(config, "backbone")
    backbone, backbone_bytes = transfer.manifest(
        backbone_prefix, "backbone.json", assets["backbone"]["sha256"]
    )
    _manifest_scope(backbone, config)
    require(
        parent["backbone_manifest_sha256"] == assets["backbone"]["sha256"],
        "Approved backbone linkage changed",
    )
    backbone_root = root / "backbone"
    transfer.download_files(
        backbone_prefix,
        backbone["files"],
        backbone_root,
        metadata={"backbone.json": backbone_bytes},
    )
    validate_backbone(backbone_root, scope=scope, expected_sha256=assets["backbone"]["sha256"])
    checkpoint_root = None
    if config["checkpointing"]["resume"] is not None:
        from learning.smolvla.checkpoints import CheckpointLimits, restore_checkpoint

        checkpoint_root = root / "resume"
        resume_prefix = input_prefix(config, "resume_checkpoint")
        resume, _ = transfer.manifest(
            resume_prefix, "checkpoint.json", assets["resume_checkpoint"]["sha256"]
        )
        require(
            resume["schema"] == CHECKPOINT_SCHEMA
            and resume["origin"].get("azure_job_type") == "command",
            "Resume requires the exact complete standalone command checkpoint",
        )
        transfer._listed(resume_prefix, set(resume["files"]) | {"checkpoint.json"})
        restore_checkpoint(
            transfer.container,
            checkpoint_prefix=resume_prefix,
            expected_sha256=assets["resume_checkpoint"]["sha256"],
            expected_scope=scope,
            destination=checkpoint_root,
            check_deadline=transfer._check,
            limits=CheckpointLimits(**config["checkpointing"]["limits"]),
        )
    return raw_root, parent_root, backbone_root, checkpoint_root


def run_workload(config: dict, client, transfer, directory: Path, snapshot_sha256: str) -> dict:
    from learning.common import file_digest, read_json
    from learning.contract import Scope
    from learning.paused.blob_transfer import input_prefix
    from learning.paused.dataset import convert_dataset, validate_conversion
    from learning.paused.train import run_training
    from learning.smolvla.train import TrainOptions

    binding = running_command_binding(client, config, snapshot_sha256=snapshot_sha256)
    _registered_inputs(client, config)
    transfer._check()
    print(
        "Genuine Azure ML command: downloading exact approved TRAIN/model/backbone via MI.",
        flush=True,
    )
    raw, parent, backbone, checkpoint = _download_inputs(transfer, config, directory)
    scope = Scope(config["tenant_id"], config["owner_id"])
    dataset_root = directory / "converted"
    dataset = dataset_root / "dataset"
    output_prefix = config["output_prefix"] + "/" + config["run_id"]
    if checkpoint is not None:
        prefix = input_prefix(config, "converted_dataset")
        conversion, body = transfer.manifest(
            prefix, "dataset/conversion.json", config["inputs"]["converted_dataset"]["sha256"]
        )
        _manifest_scope(conversion, config)
        transfer.download_files(
            prefix,
            {"dataset/" + name: sha for name, sha in conversion["files"].items()},
            dataset_root,
            metadata={"dataset/conversion.json": body},
        )
        validate_conversion(dataset, scope)
    else:
        print("Genuine Azure ML command: native paused LeRobot conversion.", flush=True)
        convert_dataset(
            raw,
            dataset,
            expected_scope=scope,
            expected_manifest_sha256=config["inputs"]["demonstrations"]["sha256"],
            expected_criteria_sha256=config["criteria_sha256"],
            expected_frozen_plan_sha256=config["frozen_plan_sha256"],
        )
        transfer._check()
        transfer.publish_files(
            dataset_root, output_prefix + "/dataset", marker="dataset/conversion.json"
        )
    conversion = validate_conversion(dataset, scope)
    from learning.paused.contract import PausedControlProfile

    require(
        conversion["raw_manifest_sha256"] == config["inputs"]["demonstrations"]["sha256"]
        and PausedControlProfile(**conversion["control_profile"]).sha256
        == config["control_profile_sha256"],
        "Converted data differs from the exact approved raw dataset/profile",
    )
    fields = ("episode_id", "environment_id", "revision", "seed", "frame_count")
    original_episodes = read_json(raw / "manifest.json")["episodes"]
    require(
        sorted(tuple(episode[name] for name in fields) for episode in conversion["episodes"])
        == sorted(tuple(episode[name] for name in fields) for episode in original_episodes),
        "Conversion lost/rebound any of the twenty original TRAIN episodes or frames",
    )
    model_root = directory / "model"
    running_command_binding(client, config, snapshot_sha256=snapshot_sha256)
    print(
        "Genuine Azure ML command: native CUDA optimizer and full-state checkpoint publisher.",
        flush=True,
    )
    run_training(
        dataset,
        parent,
        backbone,
        model_root,
        scope=scope,
        parent_model_sha256=config["inputs"]["parent_model"]["sha256"],
        conversion_sha256=file_digest(dataset / "conversion.json"),
        code_snapshot_sha256=snapshot_sha256,
        config=config,
        client=client,
        options=TrainOptions(**config["parameters"]),
        resume_checkpoint=checkpoint,
    )
    transfer._check()
    result = read_json(model_root / "result.json")
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("azure_job_type") == "command"
        and result["azure_job_id"] == binding["azure_job_id"],
        "Actual native command result provenance is incomplete",
    )
    # Native checkpoints already use their own readback-verified publication. Publish only the
    # sealed candidate and bounded logs/context, not the mutable upstream last-checkpoint link.
    receipt = transfer.publish_candidate(model_root, output_prefix + "/model")
    running_command_binding(client, config, snapshot_sha256=snapshot_sha256)
    transfer.publish_document(
        output_prefix + "/transfer",
        "completion.json",
        {
            "schema": "physicalai.smolvla-command-transfer/v1",
            **binding,
            "model_result": receipt,
            "optimizer_steps": result["optimizer_steps"],
            "learning_quality_verified": False,
        },
    )
    return result


def _preserve_failure(
    transfer, config: dict, directory: Path, binding: dict, error: Exception
) -> None:
    import shutil
    import sys

    from learning.common import file_digest, safe_path, write_json

    failure_root = directory / "failure"
    failure_root.mkdir()
    files = {}
    for name in ("training.log", "training-context.json"):
        path = directory / "model" / name
        if path.exists():
            path = safe_path(directory / "model", name)
            require(path.stat().st_size <= 16 * 1024**2, "Failure log exceeds fixed byte budget")
            if name == "training.log":
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    print(stream.read().decode("utf-8", errors="replace"), file=sys.stderr)
            shutil.copyfile(path, failure_root / name)
            files[name] = file_digest(failure_root / name)
    write_json(
        failure_root / "failure.json",
        {
            "schema": "physicalai.smolvla-command-failure/v1",
            **binding,
            "failure_type": type(error).__name__,
            "candidate_complete": False,
            "learning_quality_verified": False,
        },
    )
    files["failure.json"] = file_digest(failure_root / "failure.json")
    transfer.publish_files(
        failure_root,
        config["output_prefix"] + "/" + config["run_id"] + "/transfer/failure",
        marker="failure.json",
        files=files,
    )


def main() -> None:
    import sys
    import tempfile

    from learning.common import read_json
    from learning.deadlines import run_bounded
    from learning.smolvla.azure import (
        clients_for_managed_identity,
        validate_config,
        verify_code,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--run-command", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = read_json(args.runtime_config)
    validate_config(config)
    require(is_direct(config), "Standalone command needs explicit reviewed execution mode")
    deadline = CommandDeadline(config)
    deadline.check()
    verify_code(Path.cwd(), args.snapshot_sha256, config=config)
    if not args.run_command:
        run_bounded(
            [sys.executable, "-B", "-m", "learning.paused.command", *sys.argv[1:], "--run-command"],
            deadline,
        )
        return
    client, _ = clients_for_managed_identity(
        config, caller_client_id=config["managed_identity_client_id"]
    )
    binding = running_command_binding(client, config, snapshot_sha256=args.snapshot_sha256)
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient

    from learning.paused.blob_transfer import PrivateBlobTransfer

    with (
        ManagedIdentityCredential(client_id=config["managed_identity_client_id"]) as credential,
        BlobServiceClient(
            f"https://{config['storage_account_name']}.blob.core.windows.net",
            credential=credential,
            retry_total=0,
            connection_timeout=10,
            read_timeout=30,
        ) as service,
        tempfile.TemporaryDirectory(prefix="physicalai-command-data-") as temporary,
    ):
        transfer = PrivateBlobTransfer(
            service.get_container_client(config["blob_container"]), config, remaining=deadline.check
        )
        try:
            run_workload(config, client, transfer, Path(temporary), args.snapshot_sha256)
        except Exception as error:
            import traceback

            traceback.print_exc()
            if deadline.remaining_seconds() > 0:
                try:
                    _preserve_failure(transfer, config, Path(temporary), binding, error)
                except Exception as publication_error:
                    print(
                        "Failure-proof publication also failed: "
                        + type(publication_error).__name__,
                        file=sys.stderr,
                    )
            raise


if __name__ == "__main__":
    main()
