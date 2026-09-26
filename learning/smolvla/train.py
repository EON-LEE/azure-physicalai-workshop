from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    inventory,
    read_json,
    require,
    sha256,
    write_json,
)
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.gr00t.train import Gr00tTrainOptions
from learning.offline import OFFLINE_ENV, require_lerobot
from learning.smolvla.adaptation import adapted_config
from learning.smolvla.artifacts import model_contract, validate_backbone, validate_model
from learning.train import validate_conversion


@dataclass(frozen=True)
class TrainOptions(Gr00tTrainOptions):
    def validate(self) -> None:
        super().validate(allowed_resume_modes=("new", "weights_only", "full_state"))


def training_command(dataset: Path, seed: Path, output: Path, options: TrainOptions) -> list[str]:
    options.validate()
    require(
        options.resume_mode != "full_state", "Full-state uses the native saved config resume entry"
    )
    warmup = min(100, options.max_steps // 10)
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={seed.resolve()}",
        "--dataset.repo_id=azure-local/physicalai-reference-arm",
        f"--dataset.root={dataset.resolve()}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=false",
        "--dataset.image_transforms.enable=false",
        f"--output_dir={output.resolve()}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--wandb.mode=disabled",
        "--eval_freq=0",
        "--num_workers=0",
        "--save_checkpoint=true",
        "--log_freq=1",
        f"--steps={options.max_steps}",
        f"--save_freq={options.checkpoint_steps}",
        f"--batch_size={options.batch_size}",
        f"--seed={options.seed}",
        f"--policy.optimizer_lr={options.learning_rate}",
        f"--policy.scheduler_warmup_steps={warmup}",
        f"--policy.scheduler_decay_steps={max(options.max_steps, warmup + 1)}",
    ]


def full_state_training_command(
    checkpoint: Path, dataset: Path, backbone: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "learning.smolvla.checkpoint_runner",
        f"--config_path={(checkpoint / 'pretrained_model' / 'train_config.json').resolve()}",
        "--resume=true",
        f"--output_dir={output.resolve()}",
        f"--dataset.root={dataset.resolve()}",
        f"--policy.vlm_model_name={backbone.resolve()}",
    ]


def prepare_seed(parent: Path, backbone: Path, dataset: Path, output: Path) -> None:
    """Reuse actual pretrained weights but create NEW nine-dimensional dataset normalization."""
    require(not output.exists(), "Never overwrite an initialization checkpoint")
    output.mkdir(parents=True)
    original = read_json(parent / "checkpoint" / "config.json")
    config_json = adapted_config(original, backbone_path=backbone)
    write_json(output / "config.json", config_json)
    shutil.copyfile(parent / "checkpoint" / "model.safetensors", output / "model.safetensors")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(output, local_files_only=True)
    data = LeRobotDataset(
        "azure-local/physicalai-reference-arm",
        root=dataset,
        video_backend="pyav",
        download_videos=False,
    )
    pre, post = make_pre_post_processors(config, dataset_stats=data.meta.stats)
    pre.save_pretrained(output, push_to_hub=False)
    post.save_pretrained(output, push_to_hub=False)


def parameter_fingerprint(path: Path) -> str:
    import hashlib

    from safetensors import safe_open

    checksum = hashlib.sha256()
    names = (
        "model.state_proj.weight",
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
    )
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        for name in names:
            require(name in tensors.keys(), "Missing actual SmolVLA trainable projection")
            tensor = tensors.get_tensor(name).reshape(-1)[:4096].float()
            require(bool(tensor.isfinite().all()), "Nonfinite actual trained projection")
            checksum.update(name.encode())
            checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def validate_resume(parent: dict, *, mode: str) -> None:
    require(mode in ("new", "weights_only"), "Unsupported optimizer/pickle resume")
    if mode == "weights_only":
        require(
            parent["role"] == "candidate" and parent["training"] is not None,
            "Weights-only continuation requires a real previously trained checkpoint",
        )


def run_training(
    dataset: Path,
    parent_root: Path,
    backbone_root: Path,
    output: Path,
    *,
    scope: Scope,
    parent_model_sha256: str,
    conversion_sha256: str,
    code_snapshot_sha256: str,
    config: dict,
    client,
    options: TrainOptions,
    resume_checkpoint: Path | None = None,
) -> dict:
    require("execution_timing" not in config, "Use the explicitly selected paused training entry")
    return _run_training(
        dataset,
        parent_root,
        backbone_root,
        output,
        scope=scope,
        parent_model_sha256=parent_model_sha256,
        conversion_sha256=conversion_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
        config=config,
        client=client,
        options=options,
        resume_checkpoint=resume_checkpoint,
    )


def _run_training(
    dataset: Path,
    parent_root: Path,
    backbone_root: Path,
    output: Path,
    *,
    scope: Scope,
    parent_model_sha256: str,
    conversion_sha256: str,
    code_snapshot_sha256: str,
    config: dict,
    client,
    options: TrainOptions,
    model_validator=validate_model,
    conversion_validator=validate_conversion,
    profile_type=ControlProfile,
    model_builder=model_contract,
    mode_metadata: dict | None = None,
    resume_checkpoint: Path | None = None,
) -> dict:
    from learning.smolvla.azure import job_deadline

    deadline = job_deadline(config)
    deadline.check()
    options.validate()
    require(
        options.gradient_accumulation_steps == 1,
        "Pinned LeRobot CLI has no gradient accumulation flag; use an explicit batch size",
    )
    require(
        file_digest(dataset / "conversion.json") == conversion_sha256,
        "Conversion checksum mismatch",
    )
    converted = conversion_validator(dataset, scope)
    deadline.check()
    require(
        converted["test_only"] is False and converted.get("control_profile") is not None,
        "Production SmolVLA requires actual scoped v2 demonstrations",
    )
    parent = model_validator(
        parent_root,
        expected_scope=scope,
        expected_model_sha256=parent_model_sha256,
        for_inference=False,
    )
    deadline.check()
    if resume_checkpoint is None:
        require(
            config.get("checkpointing", {}).get("resume") is None,
            "Approved resume checkpoint was not mounted",
        )
        validate_resume(parent, mode=options.resume_mode)
    else:
        require(
            config.get("checkpointing", {}).get("resume") is not None,
            "Unapproved checkpoint resume input",
        )
    profile = profile_type(**converted["control_profile"])
    require(
        profile.sha256 == profile_type(**parent["control_profile"]).sha256,
        "Parent and real training data use different servo profiles",
    )
    if mode_metadata is not None:
        require(
            profile.sha256 == config["control_profile_sha256"]
            and digest(canonical(parent["task"])) == config["task_sha256"]
            and all(
                parent.get(key) == converted.get(key) == value
                for key, value in mode_metadata.items()
            ),
            "Paused parent and actual data differ from the approved frozen mode/plan",
        )
    require(
        all(
            {name: episode["demonstration"][name] for name in ("task_id", "instruction", "goal_id")}
            == parent["task"]
            for episode in converted["episodes"]
        ),
        "Actual demonstrations differ from the approved task",
    )
    backbone = validate_backbone(
        backbone_root, scope=scope, expected_sha256=parent["backbone_manifest_sha256"]
    )
    deadline.check()
    require_lerobot()
    import torch

    require(torch.cuda.is_available(), "Real SmolVLA optimization requires an Azure CUDA GPU")
    from learning.gr00t.azure import running_job_binding

    binding = running_job_binding(client, config)
    deadline.check()
    checkpoint_context, resumed = None, None
    if "checkpointing" in config:
        from learning.smolvla.checkpoint_runner import (
            CONTEXT_SCHEMA,
            make_binding,
            native_runtime,
            validate_policy,
            validate_resume_checkpoint,
        )

        limits = validate_policy(config["checkpointing"], parameters=asdict(options))
        checkpoint_binding = make_binding(
            config,
            parent,
            converted,
            conversion_sha256,
            native_runtime(environment_image=config["environment_image"]),
        )
        origin = {
            "azure_job_id": binding["azure_component_job_id"],
            "azure_pipeline_job_id": binding["azure_job_id"],
            "specification_sha256": binding["specification_sha256"],
            "code_snapshot_sha256": code_snapshot_sha256,
            "job_deadline_utc": config["job_deadline_utc"],
            "test_only": False,
        }
        if resume_checkpoint is not None:
            resumed = validate_resume_checkpoint(
                resume_checkpoint,
                config=config,
                expected_binding=checkpoint_binding,
                current_origin=origin,
            )
        checkpoint_context = {
            "schema": CONTEXT_SCHEMA,
            "binding": checkpoint_binding,
            "origin": origin,
            "limits": asdict(limits),
            "storage_account_name": config["storage_account_name"],
            "environment_image": config["environment_image"],
            "blob_container": config["blob_container"],
            "managed_identity_client_id": config["managed_identity_client_id"],
            "blob_prefix": config["output_prefix"]
            + "/"
            + binding["azure_job_id"].rsplit("/", 1)[-1]
            + "/checkpoints",
            "resume_from_checkpoint_sha256": (
                config["checkpointing"]["resume"]["checkpoint_sha256"] if resumed else None
            ),
            "prior_optimizer_steps": resumed["cumulative_optimizer_steps"]
            - (resumed["step"] if options.resume_mode == "full_state" else 0)
            if resumed
            else (parent["training"]["cumulative_optimizer_steps"] if parent["training"] else 0),
            "training_parameters": asdict(options),
            "resume_checkpoint_root": str(resume_checkpoint.resolve()) if resumed else None,
        }
    require(not output.exists() or not any(output.iterdir()), "Output folder is not empty")
    output.mkdir(parents=True, exist_ok=True)
    seed = output / "initialization"
    full_resume = resumed is not None and options.resume_mode == "full_state"
    if full_resume:
        seed = resume_checkpoint / "pretrained_model"
    else:
        prepare_seed(parent_root, backbone, dataset, seed)
    if resumed is not None and not full_resume:
        shutil.copyfile(
            resume_checkpoint / "pretrained_model" / "model.safetensors", seed / "model.safetensors"
        )
    deadline.check()
    before = parameter_fingerprint(seed / "model.safetensors")
    deadline.check()
    gpu = {"cuda": True, "name": torch.cuda.get_device_name()}
    write_json(
        output / "training-context.json",
        {
            "schema": "physicalai.smolvla-training-context/v2"
            if mode_metadata is not None
            else "physicalai.smolvla-training-context/v1",
            **(mode_metadata or {}),
            **binding,
            "scope": asdict(scope),
            "parent_model_sha256": parent_model_sha256,
            "conversion_sha256": conversion_sha256,
            "code_snapshot_sha256": code_snapshot_sha256,
            "parameters": asdict(options),
            "gpu": gpu,
            "updated_parameter_sample_before": before,
            "resume": None
            if resumed is None
            else {
                "mode": options.resume_mode,
                "checkpoint_sha256": config["checkpointing"]["resume"]["checkpoint_sha256"],
                "step": resumed["step"],
                "source_origin": resumed["origin"],
                "optimizer_state_restore_requested": full_resume,
                "bitwise_continuation_claimed": False,
            },
            "note": (
                "Only a readback-verified checkpoint manifest proves Blob publication; "
                "rw_mount alone does not."
            ),
        },
    )
    command = (
        full_state_training_command(resume_checkpoint, dataset, backbone, output / "training")
        if full_resume
        else training_command(dataset, seed, output / "training", options)
    )
    if checkpoint_context is not None:
        context_path = output / "checkpoint-run.json"
        write_json(context_path, checkpoint_context)
        command[2] = "learning.smolvla.checkpoint_runner"
        command.extend(
            (
                f"--checkpoint-context={context_path.resolve()}",
                f"--checkpoint-context-sha256={file_digest(context_path)}",
            )
        )
    with (output / "training.log").open("xb") as log:
        subprocess.run(
            command,
            check=True,
            timeout=min(options.timeout_seconds, deadline.check()),
            env={**os.environ, **OFFLINE_ENV},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline.check()
    result = _seal_checkpoint(
        output / "training" / "checkpoints" / f"{options.max_steps:06d}",
        output,
        step=options.max_steps,
        scope=scope,
        parent=parent,
        converted=converted,
        profile=profile,
        options=options,
        binding=binding,
        gpu=gpu,
        before=before,
        parent_model_sha256=parent_model_sha256,
        conversion_sha256=conversion_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
        model_builder=model_builder,
        model_validator=model_validator,
        mode_metadata=mode_metadata,
        resumed=resumed,
        resume_manifest_sha256=None
        if resumed is None
        else config["checkpointing"]["resume"]["checkpoint_sha256"],
    )
    deadline.check()
    return result


def _seal_checkpoint(
    step_dir: Path,
    output: Path,
    *,
    step: int,
    scope: Scope,
    parent: dict,
    converted: dict,
    profile: ControlProfile,
    options: TrainOptions,
    binding: dict,
    gpu: dict,
    before: str,
    parent_model_sha256: str,
    conversion_sha256: str,
    code_snapshot_sha256: str,
    model_builder=model_contract,
    model_validator=validate_model,
    mode_metadata: dict | None = None,
    resumed: dict | None = None,
    resume_manifest_sha256: str | None = None,
) -> dict:
    marker = read_json(step_dir / "training_state" / "training_step.json")
    require(
        marker.get("step") == step,
        "Missing actual upstream optimizer checkpoint marker",
    )
    saved = step_dir / "pretrained_model"
    after = parameter_fingerprint(saved / "model.safetensors")
    require(before != after, "No measured update to actual SmolVLA trainable parameters")
    destination = output / "candidates" / f"step-{step:06d}"
    checkpoint = destination / "checkpoint"
    checkpoint.mkdir(parents=True)
    for name in inventory(saved):
        require("/" not in name, "Unexpected nested upstream checkpoint file")
        if name in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ) or (
            name.startswith(("policy_preprocessor_step_", "policy_postprocessor_step_"))
            and name.endswith(".safetensors")
        ):
            shutil.copyfile(saved / name, checkpoint / name)
    ancestor_episodes = parent["training"]["episodes"] if parent["training"] else []
    episodes = {entry["episode_id"]: entry for entry in ancestor_episodes}
    for entry in converted["episodes"]:
        item = {key: entry[key] for key in ("episode_id", "environment_id", "revision", "seed")}
        require(
            item["episode_id"] not in episodes or episodes[item["episode_id"]] == item,
            "Conflicting model/dataset episode ancestry",
        )
        episodes[item["episode_id"]] = item
    previous_steps = parent["training"]["cumulative_optimizer_steps"] if parent["training"] else 0
    full_resume = resumed is not None and options.resume_mode == "full_state"
    actual_updates = step - resumed["step"] if full_resume else step
    require(actual_updates > 0, "Resumed job did not complete new optimizer updates")
    if resumed is not None:
        previous_steps = resumed["cumulative_optimizer_steps"]
    training_metadata = {}
    if mode_metadata is not None:
        training_metadata = {
            **mode_metadata,
            "raw_schema": "physicalai.demonstrations/v3",
            "conversion_schema": converted["schema"],
        }
    model = model_builder(
        checkpoint=checkpoint,
        scope=scope,
        profile=profile,
        task=DemonstrationSource(kind="reference_controller", **parent["task"]),
        backbone_manifest_sha256=parent["backbone_manifest_sha256"],
        role="candidate",
        training={
            **training_metadata,
            "parent_model_sha256": parent_model_sha256,
            "parent_weights_sha256": (
                resumed["files"]["pretrained_model/model.safetensors"]["sha256"]
                if resumed
                else parent["weights_sha256"]
            ),
            "raw_manifest_sha256": converted["raw_manifest_sha256"],
            "conversion_sha256": conversion_sha256,
            "config_sha256": digest(canonical(asdict(options))),
            "code_snapshot_sha256": code_snapshot_sha256,
            "specification_sha256": binding["specification_sha256"],
            "azure_job_id": binding["azure_component_job_id"],
            "azure_pipeline_job_id": binding["azure_job_id"],
            "optimizer_steps": actual_updates,
            "checkpoint_step": step,
            "cumulative_optimizer_steps": previous_steps + actual_updates,
            "test_only": False,
            "resume_mode": options.resume_mode,
            "resume_from_checkpoint_sha256": resume_manifest_sha256,
            "resume_from_checkpoint_step": resumed["step"] if resumed else None,
            "resume_from_cumulative_optimizer_steps": (
                resumed["cumulative_optimizer_steps"] if resumed else None
            ),
            "optimizer_state_restored": full_resume,
            "bitwise_continuation_claimed": False,
            "resume_from_job_id": resumed["origin"]["azure_job_id"]
            if resumed
            else parent["training"]["azure_job_id"]
            if options.resume_mode == "weights_only"
            else None,
            "resume_from_pipeline_job_id": (
                resumed["origin"]["azure_pipeline_job_id"] if resumed else None
            ),
            "ancestor_model_sha256s": [
                *(parent["training"]["ancestor_model_sha256s"] if parent["training"] else []),
                parent_model_sha256,
            ],
            "episodes": list(episodes.values()),
            "gpu": gpu,
            "updated_parameter_sample_before": before,
            "updated_parameter_sample_after": after,
            "loss": None,
        },
    )
    write_json(destination / "model.json", model)
    model_sha = file_digest(destination / "model.json")
    model_validator(destination, expected_scope=scope, expected_model_sha256=model_sha)
    write_json(
        output / "result.json",
        {
            **binding,
            **(mode_metadata or {}),
            "optimizer_steps": actual_updates,
            "candidate": destination.relative_to(output).as_posix(),
            "model_manifest_sha256": model_sha,
            "learning_quality_verified": False,
        },
    )
    return model


def recover_checkpoint(
    interrupted_output: Path,
    dataset: Path,
    parent_root: Path,
    output: Path,
    *,
    step: int,
    scope: Scope,
    expected_context_sha256: str,
    client,
) -> dict:
    """Seal a completed model checkpoint; never submit/restart a paid job automatically."""
    path = interrupted_output / "training-context.json"
    require(
        file_digest(path) == sha256(expected_context_sha256), "Training context checksum mismatch"
    )
    context = read_json(path)
    require(
        context.get("schema") == "physicalai.smolvla-training-context/v1"
        and context.get("scope") == asdict(scope),
        "Wrong recovery scope/context",
    )
    options = TrainOptions(**context["parameters"])
    options.validate()
    integer(step, "recoverable optimizer step", 1, options.max_steps)
    require(
        step % options.checkpoint_steps == 0 or step == options.max_steps,
        "Requested step was not an approved checkpoint boundary",
    )
    job = client.jobs.get(context["azure_component_job_id"].rsplit("/", 1)[1])
    require(
        job.id == context["azure_component_job_id"]
        and job.status in ("Failed", "Canceled", "Completed"),
        "Recovery requires a terminal actual job",
    )
    require(
        job.parent_job_name == context["azure_job_id"].rsplit("/", 1)[1],
        "Checkpoint component/pipeline lineage mismatch",
    )
    parent_job = client.jobs.get(job.parent_job_name)
    require(
        parent_job.id == context["azure_job_id"]
        and (parent_job.tags or {}).get("specification_sha256") == context["specification_sha256"]
        and (parent_job.tags or {}).get("scope_owner") == scope.owner_id
        and (parent_job.tags or {}).get("scope_tenant") == scope.tenant_id,
        "Recovery job owner/specification mismatch",
    )
    require(
        file_digest(dataset / "conversion.json") == context["conversion_sha256"],
        "Original conversion changed",
    )
    converted = validate_conversion(dataset, scope)
    require(
        converted["test_only"] is False,
        "Fixture checkpoints are not resumable production artifacts",
    )
    parent = validate_model(
        parent_root,
        expected_scope=scope,
        expected_model_sha256=context["parent_model_sha256"],
        for_inference=False,
    )
    require(
        parameter_fingerprint(parent_root / "checkpoint" / "model.safetensors")
        == context["updated_parameter_sample_before"],
        "Recovery initialization fingerprint differs from the approved parent",
    )
    require(
        converted["control_profile"] == parent["control_profile"],
        "Recovery dataset/parent servo profiles differ",
    )
    require(not output.exists(), "Never overwrite a recovered candidate")
    return _seal_checkpoint(
        interrupted_output / "training" / "checkpoints" / f"{step:06d}",
        output,
        step=step,
        scope=scope,
        parent=parent,
        converted=converted,
        profile=ControlProfile(**converted["control_profile"]),
        options=options,
        binding={
            key: context[key]
            for key in ("azure_job_id", "azure_component_job_id", "specification_sha256")
        },
        gpu=context["gpu"],
        before=context["updated_parameter_sample_before"],
        parent_model_sha256=context["parent_model_sha256"],
        conversion_sha256=context["conversion_sha256"],
        code_snapshot_sha256=context["code_snapshot_sha256"],
    )
