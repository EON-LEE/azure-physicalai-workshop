"""External, zero-optimizer qualification of real TRAIN batches on the fixed native runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from uuid import UUID

from learning.common import canonical, file_digest, keys, read_json, require, sha256
from learning.contract import Scope

PLAN_SCHEMA = "physicalai.smolvla-training-capacity-plan/v1"
REPORT_SCHEMA = "physicalai.smolvla-training-capacity/v1"
BATCHES = (1, 8, 16, 32, 64)
STATIC_SHA = "4e0ca2728d962b6f551ef74b55f23bb8eb847ea0edcdba023ef2adf9218a16d8"
IMAGE = (
    "factory20n3ig3ttsxayp2.azurecr.io/physicalai-smolvla@sha256:"
    "aec374cd45f7ec4eea939cf22cd3eef55db4a857ec5e531c226ef98fd6d06aa4"
)


def validate_plan(plan: dict) -> None:
    keys(
        plan,
        {
            "schema",
            "audit_id",
            "audit_script_sha256",
            "scope",
            "environment_image",
            "static_source_sha256",
            "model_sha256",
            "raw_manifest_sha256",
            "conversion_sha256",
            "backbone_sha256",
            "control_profile_sha256",
            "criteria_sha256",
            "frozen_plan_sha256",
            "random_seed",
            "max_wall_seconds",
            "max_report_bytes",
            "batch_sizes",
            "iterations_per_batch",
            "optimizer_steps",
        },
        "training capacity plan",
    )
    require(plan["schema"] == PLAN_SCHEMA, "Unknown capacity plan")
    require(str(UUID(plan["audit_id"])) == plan["audit_id"], "Invalid diagnostic identity")
    Scope(**plan["scope"]).validate()
    require(
        plan["environment_image"] == IMAGE and plan["static_source_sha256"] == STATIC_SHA,
        "Capacity qualification requires the actual corrected training image",
    )
    require(
        plan["batch_sizes"] == list(BATCHES)
        and type(plan["iterations_per_batch"]) is int
        and plan["iterations_per_batch"] == 3
        and type(plan["optimizer_steps"]) is int
        and plan["optimizer_steps"] == 0,
        "Only the fixed zero-optimizer batch schedule is authorized",
    )
    require(
        type(plan["max_wall_seconds"]) is int
        and 1 <= plan["max_wall_seconds"] <= 1800
        and type(plan["max_report_bytes"]) is int
        and 1 <= plan["max_report_bytes"] <= 1024**2
        and type(plan["random_seed"]) is int
        and 0 <= plan["random_seed"] < 2**32,
        "Invalid finite qualification limits",
    )
    for name in (
        "audit_script_sha256",
        "model_sha256",
        "raw_manifest_sha256",
        "conversion_sha256",
        "backbone_sha256",
        "control_profile_sha256",
        "criteria_sha256",
        "frozen_plan_sha256",
    ):
        sha256(plan[name], name)


def batch_indices(episodes: list[dict], size: int) -> list[int]:
    require(size in BATCHES and len(episodes) == 20, "Unexpected TRAIN batch selection")
    offsets, total = [], 0
    for index, episode in enumerate(episodes):
        require(
            episode["episode_index"] == index
            and type(episode["frame_count"]) is int
            and episode["frame_count"] >= 50
            and episode["split"] == "train"
            and 10001 <= episode["seed"] <= 10020,
            "Capacity probe cannot use non-TRAIN or malformed episodes",
        )
        offsets.append(total)
        total += episode["frame_count"]
    return [
        offsets[index % 20]
        + (0, episodes[index % 20]["frame_count"] // 2, episodes[index % 20]["frame_count"] - 1)[
            (index // 20) % 3
        ]
        for index in range(size)
    ]


def training_schedule(frames: int, batch: int, passes: int = 20) -> dict:
    require(type(frames) is int and frames > 0 and batch in BATCHES, "Invalid native data size")
    require(type(passes) is int and passes > 0, "Invalid complete pass count")
    steps = math.ceil(frames / batch) * passes
    checkpoints = math.ceil(steps / 100)
    return {
        "complete_passes": passes,
        "native_drop_last": False,
        "batch_size": batch,
        "optimizer_steps": steps,
        "actual_frame_anchors": frames * passes,
        "checkpoint_interval": 100,
        "checkpoint_count": checkpoints,
        "fits_existing_32_checkpoint_limit": checkpoints <= 32,
        "training_authorized": False,
    }


class StopBeforeOptimizer(Exception):
    """An intentional diagnostic boundary after real native backward."""


class ForbiddenOptimizer:
    def step(self, *args, **kwargs):
        raise RuntimeError("Capacity qualification must never execute an optimizer step")

    def zero_grad(self, *args, **kwargs):
        raise RuntimeError("Native update crossed the diagnostic backward boundary")


class BackwardBoundary:
    def __init__(self, accelerator, observe):
        self.accelerator, self.observe = accelerator, observe
        self.reached = False

    def autocast(self):
        return self.accelerator.autocast()

    def backward(self, loss):
        self.accelerator.backward(loss)
        self.observe(loss)
        self.reached = True
        raise StopBeforeOptimizer


def state_digest(policy) -> str:
    import torch

    checksum = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        checksum.update(
            canonical({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
        )
        checksum.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def parameter_inventory(policy) -> dict:
    values = {}
    for _, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        require(
            parameter.is_floating_point(), "Native trainable parameters must be floating tensors"
        )
        key = str(parameter.dtype)
        entry = values.setdefault(key, {"parameters": 0, "elements": 0, "bytes": 0})
        entry["parameters"] += 1
        entry["elements"] += parameter.numel()
        entry["bytes"] += parameter.numel() * parameter.element_size()
    require(values, "Native policy has no trainable parameters")
    return {
        "by_dtype": values,
        "elements": sum(value["elements"] for value in values.values()),
        "bytes": sum(value["bytes"] for value in values.values()),
        "dtype_changed": False,
    }


def write_report(path: Path, value: dict, maximum: int) -> None:
    payload = canonical(value) + b"\n"
    require(len(payload) <= maximum, "Capacity report exceeds its bound")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run(
    plan: dict,
    *,
    raw_root,
    conversion_root,
    model_root,
    backbone_root,
    source_root,
    static_manifest,
    output,
) -> dict:
    from learning.offline import enforce_offline, require_lerobot
    from learning.paused.capture import validate_dataset
    from learning.paused.command_model import LocalCommandPausedSmolVLAPolicy
    from learning.paused.dataset import validate_conversion
    from learning.smolvla.checkpoint_runner import (
        PADDING_RECIPE,
        native_runtime,
        training_padding_batch,
    )
    from learning.smolvla.image_bootstrap import verify_static

    validate_plan(plan)
    require(file_digest(Path(__file__)) == plan["audit_script_sha256"], "Capacity script changed")
    require(not output.exists(), "Never replace a previous qualification")
    require(
        all(
            not output.resolve().is_relative_to(p.resolve())
            for p in (raw_root, conversion_root, model_root, backbone_root, source_root)
        ),
        "Output must be separate from immutable inputs",
    )
    started = time.monotonic()

    def check():
        require(time.monotonic() - started < plan["max_wall_seconds"], "Capacity deadline expired")

    verify_static(source_root, static_manifest, STATIC_SHA, check=check)
    import learning

    require(
        Path(learning.__file__).resolve().parent == (source_root / "learning").resolve(),
        "Unqualified learning import",
    )
    enforce_offline()
    require_lerobot()
    scope = Scope(**plan["scope"])
    raw = validate_dataset(
        raw_root,
        expected_scope=scope,
        expected_manifest_sha256=plan["raw_manifest_sha256"],
        require_live=True,
        require_demonstrations=True,
    )
    require(
        file_digest(conversion_root / "conversion.json") == plan["conversion_sha256"],
        "Original conversion changed",
    )
    converted = validate_conversion(conversion_root, scope)
    require(
        converted["raw_manifest_sha256"] == plan["raw_manifest_sha256"]
        and {ep["seed"] for ep in converted["episodes"]} == set(range(10001, 10021))
        and len(converted["episodes"]) == len(raw.episodes) == 20,
        "Use only the original complete P0 TRAIN cohort",
    )
    require(
        file_digest(backbone_root / "backbone.json") == plan["backbone_sha256"],
        "Original backbone changed",
    )
    check()
    import torch
    from accelerate import Accelerator
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.scripts.lerobot_train import update_policy
    from torch.utils.data import default_collate

    runtime = native_runtime(environment_image=IMAGE, training_recipe=PADDING_RECIPE)
    require(
        runtime["device"] == "cuda"
        and runtime["device_count"] == 1
        and "A100" in torch.cuda.get_device_name()
        and torch.cuda.get_device_properties(0).total_memory >= 80_000_000_000,
        "A real single A100 80GB is required",
    )
    loaded = LocalCommandPausedSmolVLAPolicy(
        model_root,
        backbone_root=backbone_root,
        scope=scope,
        model_sha256=plan["model_sha256"],
        expected_control_profile_sha256=plan["control_profile_sha256"],
        expected_criteria_sha256=plan["criteria_sha256"],
        expected_frozen_plan_sha256=plan["frozen_plan_sha256"],
    )
    require(loaded.policy.config.use_amp is False, "Do not change precision for qualification")
    accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=1)
    require(
        accelerator.num_processes == 1 and accelerator.device.type == "cuda",
        "Unexpected distributed or CPU execution",
    )
    meta = LeRobotDatasetMetadata("azure-local/physicalai-reference-arm", root=conversion_root)
    dataset = LeRobotDataset(
        "azure-local/physicalai-reference-arm",
        root=conversion_root,
        download_videos=False,
        video_backend="pyav",
        image_transforms=None,
        delta_timestamps=resolve_delta_timestamps(loaded.policy.config, meta),
    )
    require(
        len(dataset) == sum(e["frame_count"] for e in converted["episodes"]),
        "Native dataset length differs",
    )
    before = state_digest(loaded.policy)
    parameters = [p for p in loaded.policy.parameters() if p.requires_grad]
    parameter_info = parameter_inventory(loaded.policy)
    trainable_count = parameter_info["elements"]
    output.mkdir(parents=True, exist_ok=False)
    probes, failure = [], None
    for batch_size in BATCHES:
        check()
        indices = batch_indices(converted["episodes"], batch_size)
        try:
            decode_started = time.monotonic()
            batch = default_collate([dataset[index] for index in indices])
            decode_seconds = time.monotonic() - decode_started
            iterations = []
            for iteration in range(3):
                check()
                loaded.reset()
                loaded.policy.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(plan["random_seed"] + iteration)
                torch.cuda.manual_seed_all(plan["random_seed"] + iteration)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                step_started = time.monotonic()
                prepared = training_padding_batch(loaded.preprocessor(batch))
                measured = {}

                def observe(loss, measured=measured, step_started=step_started):
                    torch.cuda.synchronize()
                    require(bool(torch.isfinite(loss)), "Nonfinite native training loss")
                    require(
                        all(
                            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters
                        ),
                        "Nonfinite native gradients",
                    )
                    measured.update(
                        native_loss=loss.detach().cpu().item(),
                        forward_backward_seconds=time.monotonic() - step_started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        free_device_bytes=torch.cuda.mem_get_info()[0],
                    )

                boundary = BackwardBoundary(accelerator, observe)
                try:
                    update_policy(None, loaded.policy, prepared, ForbiddenOptimizer(), 1, boundary)
                except StopBeforeOptimizer:
                    require(
                        boundary.reached, "Native backward did not reach the diagnostic boundary"
                    )
                else:
                    raise RuntimeError("Native update returned past the optimizer boundary")
                iterations.append({"iteration": iteration, "warmup": iteration == 0, **measured})
                del prepared, boundary
            probes.append(
                {
                    "batch_size": batch_size,
                    "dataset_indices": indices,
                    "decode_collate_seconds": decode_seconds,
                    "iterations": iterations,
                    "optimizer_steps": 0,
                    "native_update_policy_used": True,
                    "native_padding_recipe": PADDING_RECIPE,
                }
            )
            del batch
            loaded.policy.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            write_report(output / f"batch-{batch_size:02d}.json", probes[-1], 64 * 1024)
        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as exc:
            failure = {
                "batch_size": batch_size,
                "type": type(exc).__name__,
                "message": str(exc)[:2048],
            }
            break
    loaded.policy.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    after = state_digest(loaded.policy)
    require(
        before == after, "Model parameters or buffers changed during zero-optimizer qualification"
    )
    verify_static(source_root, static_manifest, STATIC_SHA, check=check)
    report = {
        "schema": REPORT_SCHEMA,
        "audit_id": plan["audit_id"],
        "status": "completed" if failure is None else "failed",
        "diagnostic_only": True,
        "physical_quality_verified": False,
        "optimizer_steps": 0,
        "plan": plan,
        "runtime": runtime,
        "weights_before_sha256": before,
        "weights_after_sha256": after,
        "weights_unchanged": True,
        "trainable_parameter_count": trainable_count,
        "trainable_parameters": parameter_info,
        "adamw_two_fp32_moments_estimated_bytes": trainable_count * 8,
        "adamw_two_parameter_dtype_moments_estimated_bytes": parameter_info["bytes"] * 2,
        "optimizer_state_actually_allocated": False,
        "disk_free_bytes": shutil.disk_usage(output).free,
        "probes": probes,
        "failure": failure,
        "schedule_proposals": [training_schedule(len(dataset), p["batch_size"]) for p in probes],
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "No optimizer state/step executed; moment memory and update time are not measured.",
            "Decode timing uses a fixed TRAIN batch, not a sustained full-dataset I/O benchmark.",
            "Checkpoint publication throughput and full-run duration require separate budgeting.",
            "No clipping, precision changes, held-out inputs or physical actions.",
        ],
    }
    write_report(output / "report.json", report, plan["max_report_bytes"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "raw-root",
        "conversion-root",
        "model-root",
        "backbone-root",
        "source-root",
        "static-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(file_digest(args.plan) == sha256(args.plan_sha256), "Capacity plan checksum differs")
    plan = read_json(args.plan, max_bytes=16384)
    report = run(
        plan,
        **{
            name: getattr(args, name)
            for name in (
                "raw_root",
                "conversion_root",
                "model_root",
                "backbone_root",
                "source_root",
                "static_manifest",
                "output",
            )
        },
    )
    print(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "status": report["status"],
                "optimizer_steps": 0,
                "batch_sizes_completed": [p["batch_size"] for p in report["probes"]],
                "failure": report["failure"],
                "physical_quality_verified": False,
            }
        ),
        flush=True,
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
