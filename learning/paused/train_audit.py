"""Bounded TRAIN-only SmolVLA diagnostics, never a simulator, controller or quality release."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import struct
import sys
import time
from importlib.metadata import distribution, version
from io import BytesIO
from pathlib import Path

from learning.common import (
    ContractError,
    canonical,
    digest,
    file_digest,
    integer,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
    token,
    vector,
)
from learning.contract import (
    CAMERAS,
    JOINT_LOWER,
    JOINT_NAMES,
    JOINT_UNITS,
    JOINT_UPPER,
    Scope,
    bounded_joints,
)
from learning.paused.contract import PausedControlProfile
from learning.paused.dataset import TRAIN_SEEDS

PLAN_SCHEMA = "physicalai.smolvla-train-audit-plan/v1"
REPORT_SCHEMA = "physicalai.smolvla-train-audit/v1"
REQUEST_SCHEMA = "physicalai.smolvla-train-audit-request/v1"
SAMPLE_SCHEMA = "physicalai.smolvla-train-audit-sample/v1"
MAX_SAMPLE_BYTES = 128 * 1024
NATIVE_FILES = {
    "lerobot/policies/smolvla/modeling_smolvla.py": (
        "3bdbaeecbd0dd3908d08507c13ed3517e63d2a653555322e2428066efb77b5f4"
    ),
    "lerobot/policies/smolvla/configuration_smolvla.py": (
        "6c55f3dea30a3c9571ecaa3dccf599dab9240a6eaa658b3fbc507273778b49aa"
    ),
    "lerobot/policies/smolvla/processor_smolvla.py": (
        "eeb3714ca4b926d7f4d72c63ce5877497705108736eaf53cdc7362d8dfef2644"
    ),
    "lerobot/processor/normalize_processor.py": (
        "5b610f5be7d3bcf371d52e3ccca433846c9ef49adca84ba5ef3c54ad8809338e"
    ),
    "lerobot/processor/converters.py": (
        "2b97131f34c4881e93e7b864091637fac8ce1642a6620ed9330e84372e113640"
    ),
    "lerobot/datasets/lerobot_dataset.py": (
        "e0930de3c1dac7aedec3458cdabc6f5838e61c020927de2f7fe41308bd639809"
    ),
    "lerobot/datasets/factory.py": (
        "adc786de92b9d79f84f1b950e141fce7ad28d48c885be3b66176a3f6d36c63ef"
    ),
    "lerobot/scripts/lerobot_train.py": (
        "827a2b7e74341f1d89d5c508d62ce52592bb9cab0c74fe8a3de38e9f03886410"
    ),
}


def validate_plan(value: dict) -> None:
    hashes = {
        "model_sha256",
        "backbone_sha256",
        "raw_manifest_sha256",
        "conversion_sha256",
        "control_profile_sha256",
        "criteria_sha256",
        "frozen_plan_sha256",
        "audit_script_sha256",
        "static_source_sha256",
    }
    keys(
        value,
        hashes
        | {
            "schema",
            "audit_id",
            "scope",
            "environment_image",
            "random_seed",
            "max_wall_seconds",
            "max_report_bytes",
        },
        "TRAIN audit plan",
    )
    require(value["schema"] == PLAN_SCHEMA, "Wrong TRAIN-only audit schema")
    token(value["audit_id"], "audit ID")
    Scope(**keys(value["scope"], {"tenant_id", "owner_id"}, "audit scope")).validate()
    for name in hashes:
        sha256(value[name], name)
    image = value["environment_image"]
    require(
        isinstance(image, str) and ".azurecr.io/" in image and "@sha256:" in image,
        "An immutable qualified dependency image is required",
    )
    sha256(image.rsplit("@sha256:", 1)[1], "dependency image")
    integer(value["random_seed"], "fixed audit RNG seed", 0, 2**32 - 1)
    integer(value["max_wall_seconds"], "audit process wall bound", 1, 1800)
    integer(value["max_report_bytes"], "total diagnostic bytes", 1024, 16 * 1024**2)


def validate_input_bindings(plan: dict, raw: dict, converted: dict, model: dict) -> None:
    validate_plan(plan)
    require(
        raw["schema"] == "physicalai.demonstrations/v3"
        and raw["purpose"] == "demonstration"
        and converted["schema"] == "physicalai.lerobot-conversion/v3"
        and converted["test_only"] is False
        and converted["split"] == "train",
        "Only original live TRAIN demonstrations and conversion may be audited",
    )
    for document in (raw, converted):
        require(
            document["joint_names"] == list(JOINT_NAMES)
            and document["joint_units"] == list(JOINT_UNITS)
            and document["fps"] == 10
            and document["physics_hz"] == 60,
            "Changed nine-joint order/units or held-control cadence",
        )
    require(
        converted["image_size"] == 256 and converted["image_transform"] == "rgb-bilinear-square/v1",
        "Changed original camera conversion",
    )
    for document in (raw, converted, model):
        require(
            document["scope"] == plan["scope"]
            and document["criteria_sha256"] == plan["criteria_sha256"]
            and document["frozen_plan_sha256"] == plan["frozen_plan_sha256"]
            and PausedControlProfile(**document["control_profile"]).sha256
            == plan["control_profile_sha256"],
            "Audit owner/profile/conditions were rebound",
        )
    require(
        converted["raw_manifest_sha256"]
        == model["training"]["raw_manifest_sha256"]
        == plan["raw_manifest_sha256"]
        and model["training"]["conversion_sha256"] == plan["conversion_sha256"]
        and model["backbone_manifest_sha256"] == plan["backbone_sha256"],
        "Audit does not use the original checkpoint's data/backbone",
    )
    episodes = raw["episodes"]
    require(
        isinstance(episodes, list)
        and 1 <= len(episodes) <= 40
        and len(converted["episodes"]) == len(episodes),
        "Audit cohort exceeds40 TRAIN episodes",
    )
    lineage = ("episode_id", "environment_id", "revision", "seed")
    expected = []
    for index, (episode, converted_episode) in enumerate(
        zip(episodes, converted["episodes"], strict=True)
    ):
        require(
            episode["split"] == converted_episode["split"] == "train"
            and episode["seed"] in TRAIN_SEEDS,
            "Validation/heldout/integration data cannot enter a TRAIN audit",
        )
        integer(episode["frame_count"], "TRAIN episode frames", 2, 600)
        require(
            converted_episode["episode_index"] == index
            and converted_episode["terminated"] is True
            and converted_episode["truncated"] is False
            and all(episode[name] == converted_episode[name] for name in (*lineage, "frame_count")),
            "Changed/truncated original TRAIN episode",
        )
        expected.append({name: episode[name] for name in lineage})
    require(
        len({ep["episode_id"] for ep in expected}) == len(expected)
        and model["training"]["episodes"] == expected,
        "Original training episode lineage differs",
    )


def sample_anchors(converted: dict) -> list[dict]:
    anchors, offset = [], 0
    for episode in converted["episodes"]:
        count = integer(episode["frame_count"], "episode frames", 2, 600)
        for index in sorted({0, (count - 1) // 2, count - 1}):
            anchors.append(
                {
                    "episode_id": episode["episode_id"],
                    "seed": episode["seed"],
                    "episode_index": episode["episode_index"],
                    "frame_index": index,
                    "dataset_index": offset + index,
                }
            )
        offset += count
    require(1 <= len(anchors) <= 120, "Fixed TRAIN sample bound exceeded")
    return anchors


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def target_horizon(frames, index: int) -> tuple[list, list[bool]]:
    integer(index, "original frame index", 0, len(frames) - 1)
    values = [
        [
            float32(value)
            for value in bounded_joints(
                frames[min(index + row, len(frames) - 1)]["commanded_joint_targets"],
                "TRAIN targets",
            )
        ]
        for row in range(50)
    ]
    return values, [index + row >= len(frames) for row in range(50)]


def json_tensor(value):
    if isinstance(value, (list, tuple)):
        return [json_tensor(item) for item in value]
    require(type(value) in (int, float), "Expected primitive numeric diagnostic tensor")
    return value if math.isfinite(value) else {"nonfinite": str(value)}


def tensor_record(tensor) -> dict:
    metadata = {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
    }
    return {**metadata, "values": json_tensor(tensor.detach().cpu().tolist())}


def check_native_state(loaded, training_batch, inference_batch, torch) -> None:
    require(
        torch.equal(
            loaded.policy.prepare_state(training_batch),
            loaded.policy.prepare_state(inference_batch),
        ),
        "Training and inference native state preprocessing differ",
    )


def summarize_predictions(predictions, targets, padding, observed) -> dict:
    require(
        len(predictions) == len(targets) == len(padding) == 50
        and all(type(row) in (list, tuple) and len(row) == 9 for row in predictions)
        and all(type(flag) is bool for flag in padding)
        and padding[0] is False,
        "Audit must retain all50x9 action rows and original padding",
    )
    observed = vector(observed, 9, "original observed state")
    violations, guard_errors = [], []
    for index, row in enumerate(predictions):
        require(all(type(value) in (int, float) for value in row), "Nonnumeric native output")
        try:
            bounded_joints(row, "paused Smol action")
        except ContractError as exc:
            guard_errors.append({"row": index, "error": str(exc)})
        for joint, (value, low, high) in enumerate(zip(row, JOINT_LOWER, JOINT_UPPER, strict=True)):
            if not math.isfinite(value) or not low - 1e-6 <= value <= high + 1e-6:
                violations.append(
                    {
                        "row": index,
                        "joint_index": joint,
                        "joint": JOINT_NAMES[joint],
                        "value": json_tensor(value),
                        "low": low,
                        "high": high,
                    }
                )
    finite = all(math.isfinite(value) for row in predictions for value in row)
    valid = [index for index, pad in enumerate(padding) if not pad]
    padded = [index for index, pad in enumerate(padding) if pad]

    def mae(rows):
        return (
            [
                sum(abs(predictions[row][joint] - targets[row][joint]) for row in rows) / len(rows)
                for joint in range(9)
            ]
            if finite and rows
            else None
        )

    return {
        "all_finite": finite,
        "all_rows_within_joint_limits": not guard_errors,
        "row_zero_within_joint_limits": not any(error["row"] == 0 for error in guard_errors),
        "valid_target_rows": len(valid),
        "padded_target_rows": 50 - len(valid),
        "joint_violations": violations,
        "original_guard_errors": guard_errors,
        "mae_per_joint": mae(valid),
        "all_rows_mae_per_joint": mae(list(range(50))),
        "padded_rows_mae_per_joint": mae(padded),
        "row_zero_abs_error": [abs(predictions[0][joint] - targets[0][joint]) for joint in range(9)]
        if finite
        else None,
        "row_zero_tracking_delta": [
            abs(predictions[0][joint] - observed[joint]) for joint in range(9)
        ]
        if finite
        else None,
    }


def training_exposure(counts: list[int], *, steps: int, batch_size: int) -> dict:
    for count in counts:
        integer(count, "episode frames", 2, 600)
    frames = sum(counts)
    integer(frames, "total TRAIN frames", 2, 24000)
    integer(steps, "optimizer updates", 1, 100000)
    integer(batch_size, "batch size", 1, 64)
    batches = math.ceil(frames / batch_size)
    passes, remainder = divmod(steps, batches)
    samples = passes * frames + min(remainder * batch_size, frames)
    return {
        "frames": frames,
        "episodes": len(counts),
        "steps_per_pass": batches,
        "completed_passes": passes,
        "anchor_samples": samples,
        "anchor_passes": samples / frames,
        "sampler": "shuffled-frame-anchors-without-replacement;drop_last=false;one-GPU",
        "historical_exact_sample_order_known": False,
        "anchors_with_tail_padding": sum(min(count - 1, 49) for count in counts),
        "repeated_tail_target_rows_per_pass": sum(
            max(0, index + 50 - count) for count in counts for index in range(count)
        ),
    }


def prospective_training(counts: list[int]) -> dict:
    steps_per_pass = math.ceil(sum(counts) / 8)
    steps = 20 * steps_per_pass
    require(steps <= 100000, "Prospective training exceeds existing step contract")
    return {
        "schema": "physicalai.smolvla-prospective-training/v1",
        "automatic_submission": False,
        "quality_release": False,
        "parameters": {
            "max_steps": steps,
            "checkpoint_steps": 2 * steps_per_pass,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0001,
            "seed": 42,
            "timeout_seconds": 43200,
            "compute_tier": "Dedicated",
            "resume_mode": "weights_only",
        },
        "job_timeout_seconds": 57600,
        "exposure": training_exposure(counts, steps=steps, batch_size=8),
        "checkpoints": {
            "count": 10,
            "max_bytes_each": 8 * 1024**3,
            "max_bytes_total": 80 * 1024**3,
            "publication_timeout_seconds_each": 180,
        },
        "prerequisites": [
            "Native TRAIN audit completed; no heldout reused.",
            "Review separately versioned training-only padding-mask correction before submission.",
            "Measure batch8 memory/throughput; fail, do not extend, if fixed deadline cannot fit.",
            "Mint a new approved context, original finite deadline and output prefix.",
        ],
        "train_checks": [
            "Original data/processors,9-joint order/units and RGB transforms agree.",
            "All fixed TRAIN predictions finite; all50 rows satisfy unchanged joint guards.",
            "Report physical9-channel valid/padded horizon and row-zero error separately.",
            "Compare fixed TRAIN errors against original P0; this is not a quality release.",
        ],
    }


def write_bounded(path: Path, value: dict, *, maximum: int) -> int:
    payload = canonical(value) + b"\n"
    require(len(payload) <= maximum, "Diagnostic artifact exceeds its byte bound")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return len(payload)


def native_sources(package_root: Path) -> dict:
    for name, checksum in NATIVE_FILES.items():
        require(
            file_digest(safe_path(package_root, name)) == checksum,
            f"Actual pinned native source differs: {name}",
        )
    source = ast.parse((package_root / "lerobot/policies/smolvla/modeling_smolvla.py").read_bytes())
    policy = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "SmolVLAPolicy"
    )
    forward = next(
        node for node in policy.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    pad_keys = [
        node.args[0].value
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "batch"
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    require(pad_keys == ["actions_id_pad"], "Unexpected native padding implementation")
    return {
        "files": NATIVE_FILES,
        "dataset_padding_key": "action_is_pad",
        "policy_loss_padding_key": pad_keys[0],
        "padding_keys_match": False,
        "loss_action_channels": 32,
        "physical_action_channels": 9,
        "scalar_loss_is_physical_joint_error": False,
        "physical_failure_cause_established": False,
    }


def _tensor_inputs(frame: dict, root: Path, instruction: str):
    import numpy as np
    import torch
    from PIL import Image

    inputs = {
        "observation.state": torch.tensor(frame["joint_positions"], dtype=torch.float32),
        "task": instruction,
    }
    for name in CAMERAS:
        image = frame["images"][name]
        payload = safe_path(root, image["path"]).read_bytes()
        require(digest(payload) == image["sha256"], "Original TRAIN image changed")
        with Image.open(BytesIO(payload)) as decoded:
            pixels = np.array(
                decoded.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        inputs[f"observation.images.{name}"] = torch.from_numpy(pixels).permute(2, 0, 1) / 255
    return inputs


def _saved_statistics(loaded, dataset, checkpoint: Path) -> dict:
    import torch
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from safetensors import safe_open

    normalizers = [
        step for step in loaded.preprocessor.steps if isinstance(step, NormalizerProcessorStep)
    ]
    unnormalizers = [
        step for step in loaded.postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
    ]
    require(
        len(normalizers) == len(unnormalizers) == 1, "Expected one saved native pre/post normalizer"
    )
    pre, post = normalizers[0], unnormalizers[0]
    pre_state, post_state = pre.state_dict(), post.state_dict()
    values, tensor_metadata = {}, {"loaded_preprocessor": {}, "loaded_postprocessor": {}}
    for feature in ("action", "observation.state"):
        values[feature] = {}
        for stat in ("mean", "std"):
            key = feature + "." + stat
            actual = pre_state[key].detach().cpu()
            expected = torch.as_tensor(dataset.meta.stats[feature][stat], dtype=torch.float32)
            require(
                tuple(actual.shape) == (9,)
                and bool(torch.isfinite(actual).all())
                and torch.equal(actual, expected),
                "Saved normalizer differs from TRAIN statistics",
            )
            if feature == "action":
                require(
                    torch.equal(actual, post_state[key].detach().cpu()),
                    "Saved action normalizer/unnormalizer differ",
                )
            if stat == "std":
                require(bool((actual > 0).all()), "Degenerate TRAIN normalization")
            values[feature][stat] = actual.tolist()
            tensor_metadata["loaded_preprocessor"][key] = tensor_record(pre_state[key])
            if feature == "action":
                tensor_metadata["loaded_postprocessor"][key] = tensor_record(post_state[key])
    configs = {
        name: read_json(checkpoint / name)
        for name in ("policy_preprocessor.json", "policy_postprocessor.json")
    }
    saved = {}
    for name, config in configs.items():
        for step in config["steps"]:
            if step["registry_name"] not in {"normalizer_processor", "unnormalizer_processor"}:
                continue
            source = safe_path(checkpoint, step["state_file"])
            with safe_open(str(source), framework="pt", device="cpu") as tensors:
                saved[name] = {
                    key: tensor_record(tensors.get_tensor(key))
                    for key in (
                        "action.mean",
                        "action.std",
                        "observation.state.mean",
                        "observation.state.std",
                    )
                    if key in tensors.keys()
                }
    require(
        pre.get_config()["norm_map"]
        == {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "Audit expects the original Smol mean/std mapping",
    )
    return {
        "statistics": values,
        "preprocessor": configs["policy_preprocessor.json"],
        "postprocessor": configs["policy_postprocessor.json"],
        "normalizer_eps": pre.eps,
        "unnormalizer_eps": post.eps,
        "actual_saved_stats_match_train": True,
        "tensor_metadata": tensor_metadata,
        "saved_safetensors": saved,
    }


def run_audit(
    plan: dict,
    *,
    raw_root: Path,
    conversion_root: Path,
    model_root: Path,
    backbone_root: Path,
    output: Path,
    source_root: Path,
    static_manifest: Path,
) -> dict:
    from learning.offline import enforce_offline, require_lerobot
    from learning.paused.capture import validate_dataset
    from learning.paused.command_model import LocalCommandPausedSmolVLAPolicy
    from learning.paused.dataset import validate_conversion
    from learning.smolvla.image_bootstrap import verify_static

    started = time.monotonic()

    def check():
        require(
            time.monotonic() - started < plan["max_wall_seconds"], "Original audit deadline expired"
        )

    validate_plan(plan)
    require(
        file_digest(Path(__file__)) == plan["audit_script_sha256"], "Audit script checksum mismatch"
    )
    require(not output.exists(), "Never overwrite an audit or its partial evidence")
    require(
        all(
            not output.resolve().is_relative_to(path.resolve())
            for path in (raw_root, conversion_root, model_root, backbone_root, source_root)
        ),
        "Audit output must be outside immutable inputs",
    )
    static = verify_static(source_root, static_manifest, plan["static_source_sha256"], check=check)
    import learning

    require(
        Path(learning.__file__).resolve().parent == (source_root / "learning").resolve(),
        "Audit imported learning code outside the qualified static source",
    )
    enforce_offline()
    require_lerobot()
    package = distribution("lerobot")
    sources = native_sources(Path(package.locate_file("")))
    for root, name, expected in (
        (raw_root, "manifest.json", plan["raw_manifest_sha256"]),
        (conversion_root, "conversion.json", plan["conversion_sha256"]),
        (model_root, "model.json", plan["model_sha256"]),
        (backbone_root, "backbone.json", plan["backbone_sha256"]),
    ):
        require(file_digest(safe_path(root, name)) == expected, f"Original {name} checksum changed")
    scope = Scope(**plan["scope"])
    raw = validate_dataset(
        raw_root,
        expected_scope=scope,
        expected_manifest_sha256=plan["raw_manifest_sha256"],
        require_live=True,
        require_demonstrations=True,
    )
    check()
    converted = validate_conversion(conversion_root, scope)
    model = read_json(model_root / "model.json")
    validate_input_bindings(plan, raw.manifest, converted, model)
    check()
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torch.utils.data import default_collate

    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "A genuine single CUDA GPU is required; no CPU/model bypass",
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
    check()
    checkpoint = model_root / "checkpoint"
    config = read_json(checkpoint / "config.json")
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
        len(dataset) == sum(ep["frame_count"] for ep in converted["episodes"])
        and not hasattr(loaded.policy.config, "drop_n_last_frames"),
        "Unexpected native sampling",
    )
    stats = _saved_statistics(loaded, dataset, checkpoint)
    anchors = sample_anchors(converted)
    output.mkdir(parents=True, exist_ok=False)
    bytes_written, samples = 0, []
    torch.cuda.reset_peak_memory_stats()
    for ordinal, anchor in enumerate(anchors):
        check()
        frames = raw.episodes[anchor["episode_index"]].frames
        frame = frames[anchor["frame_index"]]
        targets, padding = target_horizon(frames, anchor["frame_index"])
        item = dataset[anchor["dataset_index"]]
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        require(
            torch.equal(item["action"].cpu(), target_tensor)
            and item["action_is_pad"].tolist() == padding,
            "Native action horizon/padding differs from RAW",
        )
        inputs = _tensor_inputs(frame, raw_root, model["task"]["instruction"])
        require(
            int(item["episode_index"]) == anchor["episode_index"]
            and int(item["frame_index"]) == anchor["frame_index"]
            and item["task"] == inputs["task"],
            "Native dataset anchor was rebound",
        )
        for key in ("observation.state", *(f"observation.images.{name}" for name in CAMERAS)):
            require(
                torch.equal(item[key][0].cpu(), inputs[key]),
                f"RAW/conversion input mismatch: {key}",
            )
        request = {
            "schema": REQUEST_SCHEMA,
            "audit_id": plan["audit_id"],
            "ordinal": ordinal,
            "live_control_authority": False,
            "model_sha256": plan["model_sha256"],
            "scope": plan["scope"],
            "raw_manifest_sha256": plan["raw_manifest_sha256"],
            "conversion_sha256": plan["conversion_sha256"],
            **anchor,
            "source_frame_sha256": digest(canonical(frame)),
            "original_observation_sha256": frame["observation_sha256"],
            "original_environment_id": frame["environment_id"],
            "original_revision": frame["revision"],
            "original_physics_step": frame["physics_step"],
            "original_captured_at_utc": frame["captured_at_utc"],
            "image_sha256": {name: frame["images"][name]["sha256"] for name in CAMERAS},
            "task_sha256": digest(canonical(model["task"])),
            "random_seed": plan["random_seed"] + ordinal,
        }
        loaded.reset()
        with torch.inference_mode():
            train_batch = loaded.preprocessor(default_collate([item]))
            inference_batch = loaded.preprocessor(inputs)
            check_native_state(loaded, train_batch, inference_batch, torch)
            normalized_targets = train_batch["action"]
            round_trip = loaded.postprocessor(normalized_targets.clone()).detach().cpu()
            require(
                torch.allclose(round_trip[0], target_tensor, atol=1e-6, rtol=0),
                "Saved target normalization round trip differs",
            )
            torch.manual_seed(request["random_seed"])
            torch.cuda.manual_seed_all(request["random_seed"])
            torch.cuda.synchronize()
            predict_started = time.monotonic_ns()
            normalized = loaded.policy.predict_action_chunk(inference_batch)
            normalized_metadata = {"dtype": str(normalized.dtype), "device": str(normalized.device)}
            denormalized = loaded.postprocessor(normalized.clone())
            denormalized_metadata = {
                "dtype": str(denormalized.dtype),
                "device": str(denormalized.device),
            }
            torch.cuda.synchronize()
            predict_ended = time.monotonic_ns()
        require(
            tuple(normalized.shape) == tuple(denormalized.shape) == (1, 50, 9),
            "Unexpected actual model50x9 shape",
        )
        values = denormalized.detach().cpu()[0].tolist()
        summary = summarize_predictions(values, targets, padding, frame["joint_positions"])
        sample = {
            "schema": SAMPLE_SCHEMA,
            "request": request,
            "request_sha256": digest(canonical(request)),
            "normalized_actions": json_tensor(normalized.detach().cpu()[0].tolist()),
            "denormalized_actions": json_tensor(values),
            "native_dtype_path": {
                "normalized_prediction": normalized_metadata,
                "denormalized_prediction": denormalized_metadata,
                "normalized_state": str(inference_batch["observation.state"].dtype),
                "normalized_targets": str(normalized_targets.dtype),
                "postprocessor_stats_after_prediction": {
                    key: str(tensor.dtype)
                    for step in loaded.postprocessor.steps
                    for key, tensor in step.state_dict().items()
                    if key in {"action.mean", "action.std"}
                },
            },
            "targets": targets,
            "normalized_targets": json_tensor(normalized_targets.detach().cpu()[0].tolist()),
            "action_is_pad": padding,
            "physical_metrics": summary,
            "training_batch_padding_keys": sorted(key for key in train_batch if "_pad" in key),
            "inference_latency_ms": (predict_ended - predict_started) / 1e6,
            "inference_started_monotonic_ns": predict_started,
            "inference_finished_monotonic_ns": predict_ended,
            "original_observation_state": list(frame["joint_positions"]),
        }
        name = f"sample-{ordinal:03d}.json"
        bytes_written += write_bounded(
            output / name,
            sample,
            maximum=min(MAX_SAMPLE_BYTES, plan["max_report_bytes"] - bytes_written),
        )
        samples.append(
            {
                "path": name,
                "sha256": file_digest(output / name),
                "all_finite": summary["all_finite"],
                "all_rows_within_joint_limits": summary["all_rows_within_joint_limits"],
            }
        )
    check()
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "diagnostic_only": True,
        "physical_quality_verified": False,
        "live_control_authority": False,
        "plan": plan,
        "plan_canonical_sha256": digest(canonical(plan)),
        "samples": samples,
        "all_sampled_predictions_finite": all(sample["all_finite"] for sample in samples),
        "all_sampled_horizons_within_joint_limits": all(
            sample["all_rows_within_joint_limits"] for sample in samples
        ),
        "native_sources": sources,
        "static_source_sha256": static["sha256"],
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("torch", "lerobot", "numpy", "safetensors", "transformers")
            },
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
        },
        "checkpoint_config": config,
        "processors": stats,
        "checkpoint_files": model["checkpoint_files"],
        "actual_model_manifest_sha256": file_digest(model_root / "model.json"),
        "training_exposure": training_exposure(
            [ep["frame_count"] for ep in converted["episodes"]],
            steps=model["training"]["optimizer_steps"],
            batch_size=1,
        )
        if model["training"]["config_sha256"]
        == digest(
            canonical(
                {
                    "max_steps": 1000,
                    "checkpoint_steps": 100,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 0.0001,
                    "seed": 42,
                    "timeout_seconds": 3600,
                    "compute_tier": "Dedicated",
                    "resume_mode": "new",
                }
            )
        )
        else None,
        "prospective_training": prospective_training(
            [ep["frame_count"] for ep in converted["episodes"]]
        ),
        "elapsed_wall_seconds": time.monotonic() - started,
        "notes": [
            "Offline TRAIN observations retain original IDs/hashes, not reconstructed2a4 inputs.",
            "Native prediction is diagnostic only; unchanged joint guard checked on all50 rows.",
            "No model, image, dataset, control or training-loss changes were made.",
        ],
    }
    write_bounded(
        output / "report.json",
        report,
        maximum=min(256 * 1024, plan["max_report_bytes"] - bytes_written),
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "raw-root",
        "conversion-root",
        "model-root",
        "backbone-root",
        "output",
        "source-root",
        "static-manifest",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    require(file_digest(args.plan) == sha256(args.plan_sha256), "Audit plan checksum mismatch")
    plan = read_json(args.plan, max_bytes=16384)
    try:
        report = run_audit(
            plan,
            **{
                name: getattr(args, name)
                for name in (
                    "raw_root",
                    "conversion_root",
                    "model_root",
                    "backbone_root",
                    "output",
                    "source_root",
                    "static_manifest",
                )
            },
        )
    except (ContractError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "schema": REPORT_SCHEMA,
                    "status": "failed",
                    "diagnostic_only": True,
                    "failure_type": type(exc).__name__,
                    "failure": str(exc)[:2048],
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema",
                    "status",
                    "all_sampled_predictions_finite",
                    "all_sampled_horizons_within_joint_limits",
                    "physical_quality_verified",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
