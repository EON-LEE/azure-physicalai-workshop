from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    finite,
    integer,
    inventory,
    require,
    sha256,
    write_json,
)
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.gr00t.artifacts import model_contract, validate_model
from learning.gr00t.dataset import validate_export
from learning.gr00t.franka_modality import FrankaDataConfig
from learning.gr00t.source import activate_source


@dataclass(frozen=True)
class Gr00tTrainOptions:
    max_steps: int = 1000
    checkpoint_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0001
    seed: int = 42
    timeout_seconds: int = 3600
    compute_tier: str = "Dedicated"
    resume_mode: str = "new"

    def validate(self) -> None:
        integer(self.max_steps, "optimizer step budget", 1, 100000)
        integer(self.checkpoint_steps, "checkpoint interval", 1, self.max_steps)
        integer(self.batch_size, "GPU batch size", 1, 64)
        integer(self.gradient_accumulation_steps, "gradient accumulation", 1, 64)
        integer(self.seed, "training seed", 0, 2**32 - 1)
        integer(self.timeout_seconds, "job time budget", 1, 86400)
        require(0 < finite(self.learning_rate, "learning rate") <= 0.001, "Invalid learning rate")
        require(self.compute_tier in ("Dedicated", "LowPriority"), "Explicit compute tier required")
        require(self.resume_mode in ("new", "weights_only"), "No pickle/optimizer-state resume")
        if self.compute_tier == "LowPriority":
            require(
                self.checkpoint_steps < self.max_steps and self.checkpoint_steps <= 100,
                "Spot training requires incremental persisted checkpoints, not only final steps",
            )


def training_arguments(output: Path, options: Gr00tTrainOptions) -> dict:
    options.validate()
    return {
        "output_dir": str(output),
        "remove_unused_columns": False,
        "bf16": True,
        "tf32": True,
        "per_device_train_batch_size": options.batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "optim": "adamw_torch",
        "learning_rate": options.learning_rate,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "max_steps": options.max_steps,
        "save_strategy": "steps",
        "save_steps": options.checkpoint_steps,
        "save_total_limit": 2,
        "save_safetensors": True,
        "save_only_model": True,
        "report_to": [],
        "push_to_hub": False,
        "seed": options.seed,
        "do_eval": False,
        "ddp_find_unused_parameters": False,
    }


def azure_job_identity(config: dict) -> str:
    run_id = os.environ.get("AZUREML_RUN_ID")
    require(
        bool(run_id), "Real Azure ML execution context is required; never synthesize AZUREML_RUN_ID"
    )
    from learning.azure import _uuid
    from learning.common import token

    _uuid(config["subscription_id"], "subscription")
    for name in ("resource_group", "workspace"):
        token(config[name], name)
    token(run_id, "actual Azure ML run ID")
    return (
        f"/subscriptions/{config['subscription_id']}/resourceGroups/{config['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{config['workspace']}/jobs/{run_id}"
    )


def _parameter_digest(model) -> tuple[str, int]:
    result, count = hashlib.sha256(), 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().reshape(-1)[:4096].float().cpu().numpy().tobytes())
            count += 1
    require(count > 0, "Upstream GR00T model exposes no trainable parameters")
    return result.hexdigest(), count


def run_training(
    dataset_root: Path,
    parent_root: Path,
    output: Path,
    *,
    source_root: Path,
    scope: Scope,
    export_sha256: str,
    parent_model_sha256: str,
    code_snapshot_sha256: str,
    azure_config: dict,
    client,
    options: Gr00tTrainOptions,
) -> dict:
    options.validate()
    sha256(code_snapshot_sha256)
    exported = validate_export(dataset_root, scope=scope, expected_sha256=export_sha256)
    require(exported["test_only"] is False, "GR00T production training rejects test fixtures")
    parent = validate_model(
        parent_root,
        expected_scope=scope,
        expected_model_sha256=parent_model_sha256,
        for_inference=False,
    )
    require(parent["control_profile"] == exported["control_profile"], "Parent/data servo mismatch")
    require(
        all(
            ep["demonstration"]["task_id"] == parent["task"]["task_id"]
            and ep["demonstration"]["instruction"] == parent["task"]["instruction"]
            and ep["demonstration"]["goal_id"] == parent["task"]["goal_id"]
            for ep in exported["episodes"]
        ),
        "Dataset task differs from the approved parent task",
    )
    if options.resume_mode == "weights_only":
        require(
            parent["role"] == "candidate", "Resume requires a verified prior training checkpoint"
        )
    job_id = azure_job_identity(azure_config)
    job = client.jobs.get(os.environ["AZUREML_RUN_ID"])
    require(
        job.id == job_id and str(job.status).lower() == "running",
        "Azure ML did not confirm this actual running job",
    )
    require(not output.exists() or not any(output.iterdir()), "Never overwrite candidate artifacts")
    activate_source(source_root)
    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.runner import TrainRunner
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from transformers import TrainerCallback, TrainingArguments

    require(torch.cuda.is_available(), "Actual GR00T optimizer execution requires CUDA")
    require(torch.cuda.device_count() == 1, "Only the explicitly approved single GPU is supported")
    config = FrankaDataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_root,
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag="new_embodiment",
        video_backend="decord",
    )
    model = GR00T_N1_5.from_pretrained(
        str((parent_root / "checkpoint").resolve()),
        local_files_only=True,
        use_safetensors=True,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    model.compute_dtype = model.config.compute_dtype = "bfloat16"
    before_digest, tensor_count = _parameter_digest(model)
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)
    arguments = training_arguments(output / "upstream", options)
    runner = TrainRunner(
        model=model,
        training_args=TrainingArguments(**arguments),
        train_dataset=dataset,
        resume_from_checkpoint=False,
    )
    started = time.monotonic()
    published: dict[int, dict] = {}
    ancestor_episodes = parent["training"]["episodes"] if parent["training"] else []
    episodes = {ep["episode_id"]: ep for ep in ancestor_episodes}
    for ep in exported["episodes"]:
        selected = {name: ep[name] for name in ("episode_id", "seed", "environment_id", "revision")}
        require(
            ep["episode_id"] not in episodes or episodes[ep["episode_id"]] == selected,
            "Parent/additional episode lineage collision",
        )
        episodes[ep["episode_id"]] = selected
    prior_steps = parent["training"]["cumulative_optimizer_steps"] if parent["training"] else 0

    def publish(step: int, checkpoint: Path) -> dict:
        if step in published:
            return published[step]
        changed, _ = _parameter_digest(model)
        require(
            changed != before_digest, "Optimizer produced no observed trainable-parameter update"
        )
        destination = output / "candidates" / f"step-{step:06d}"
        staged = destination / "checkpoint"
        staged.mkdir(parents=True)
        for name in inventory(checkpoint):
            if name in (
                "config.json",
                "model.safetensors.index.json",
                "experiment_cfg/metadata.json",
            ) or name.endswith(".safetensors"):
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint / name, target)
        value = model_contract(
            scope=scope,
            profile=ControlProfile(**exported["control_profile"]),
            task=DemonstrationSource(kind="human_teleop", **parent["task"]),
            checkpoint=staged,
            role="candidate",
            training={
                "parent_model_sha256": parent_model_sha256,
                "parent_weights_sha256": parent["weights_sha256"],
                "raw_manifest_sha256": exported["raw_manifest_sha256"],
                "export_sha256": export_sha256,
                "config_sha256": digest(canonical(asdict(options))),
                "code_snapshot_sha256": code_snapshot_sha256,
                "azure_job_id": job_id,
                "optimizer_steps": step,
                "checkpoint_step": step,
                "cumulative_optimizer_steps": prior_steps + step,
                "resume_from_job_id": parent["training"]["azure_job_id"]
                if options.resume_mode == "weights_only"
                else None,
                "resume_mode": options.resume_mode,
                "test_only": False,
                "episodes": list(episodes.values()),
                "gpu": {
                    "cuda": True,
                    "name": torch.cuda.get_device_name(),
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                    "updated_parameter_sample_before": before_digest,
                    "updated_parameter_sample_after": changed,
                    "trainable_tensors": tensor_count,
                },
            },
        )
        write_json(destination / "model.json", value)
        from learning.common import file_digest

        validate_model(
            destination,
            expected_scope=scope,
            expected_model_sha256=file_digest(destination / "model.json"),
        )
        published[step] = value
        return value

    class PersistedCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            require(time.monotonic() - started < options.timeout_seconds, "GR00T training timeout")

        def on_save(self, args, state, control, **kwargs):
            publish(state.global_step, Path(args.output_dir) / f"checkpoint-{state.global_step}")

    runner.trainer.add_callback(PersistedCheckpoint())
    runner.train()
    require(
        runner.trainer.state.global_step == options.max_steps, "Incomplete upstream optimizer run"
    )
    final = publish(options.max_steps, output / "upstream")
    write_json(
        output / "result.json",
        {
            "azure_job_id": job_id,
            "optimizer_steps": options.max_steps,
            "candidate": f"candidates/step-{options.max_steps:06d}",
            "model_manifest_sha256": digest(canonical(final) + b"\n"),
            "learning_quality_verified": False,
        },
    )
    return final
