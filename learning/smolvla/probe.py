"""Real CUDA/vendor-weight diagnostic; no actuation or training claims."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from learning.common import file_digest, read_json, require, write_json
from learning.contract import Scope
from learning.offline import require_lerobot
from learning.smolvla.artifacts import validate_backbone, validate_model
from learning.smolvla.train import prepare_seed
from learning.train import validate_conversion


def run_probe(
    dataset: Path,
    parent: Path,
    backbone_root: Path,
    output: Path,
    *,
    scope: Scope,
    model_sha256: str,
    conversion_sha256: str,
) -> dict:
    require(
        file_digest(dataset / "conversion.json") == conversion_sha256,
        "Probe conversion checksum mismatch",
    )
    converted = validate_conversion(dataset, scope)
    require(
        converted["test_only"] is False and converted.get("control_profile") is not None,
        "Actual GPU probe requires real v2 ten-Hz demonstrations, never fixtures",
    )
    model = validate_model(
        parent, expected_scope=scope, expected_model_sha256=model_sha256, for_inference=False
    )
    backbone = validate_backbone(
        backbone_root, scope=scope, expected_sha256=model["backbone_manifest_sha256"]
    )
    require(not output.exists() or not any(output.iterdir()), "Probe output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    require_lerobot()
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    require(torch.cuda.is_available(), "Actual CUDA GPU is required")
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    initialization = output / "initialization"
    prepare_seed(parent, backbone, dataset, initialization)
    config = PreTrainedConfig.from_pretrained(initialization, local_files_only=True)
    config.device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    policy = SmolVLAPolicy.from_pretrained(
        initialization, config=config, local_files_only=True, strict=True
    )
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(initialization),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(backbone)},
        },
    )
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    data = LeRobotDataset(
        "azure-local/physicalai-reference-arm",
        root=dataset,
        video_backend="pyav",
        download_videos=False,
    )
    row = data[0]
    allowed = {
        "observation.state",
        "observation.images.inspection",
        "observation.images.overview",
        "task",
    }
    inputs = {name: row[name] for name in allowed}
    require(
        inputs["observation.state"].shape == (9,), "Probe has no nine-dimensional measured state"
    )
    latencies = []
    for _ in range(6):
        torch.cuda.synchronize()
        start = time.monotonic_ns()
        with torch.inference_mode():
            actions = post(policy.predict_action_chunk(pre(inputs)))
        torch.cuda.synchronize()
        latencies.append((time.monotonic_ns() - start) / 1e6)
        require(
            tuple(actions.shape) == (1, 50, 9) and bool(actions.isfinite().all()),
            "Real SmolVLA produced invalid physical actions",
        )
    report = {
        "schema": "physicalai.smolvla-gpu-probe/v1",
        "policy_type": "smolvla",
        "nvidia_smi": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "load_seconds": load_seconds,
        "actual_inference_calls": 6,
        "physical_action_shape": [1, 50, 9],
        "real_input_cameras": 2,
        "physical_state_dimensions": 9,
        "cold_latency_ms": latencies[0],
        "steady_latencies_without_ipc_ms": latencies[1:],
        "latency_budget_passed": max(latencies[1:]) <= 80,
        "model_manifest_sha256": model_sha256,
        "backbone_manifest_sha256": model["backbone_manifest_sha256"],
        "conversion_sha256": conversion_sha256,
        "optimizer_steps": 0,
        "ready_for_live_execution": False,
        "learning_quality_verified": False,
        "normalization_source": (
            "Actual nine-DOF training data; vendor initialization, not trained Franka P0"
        ),
    }
    write_json(output / "probe.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("dataset", "parent", "backbone", "output", "binding"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--conversion-sha256", required=True)
    args = parser.parse_args()
    result = run_probe(
        args.dataset,
        args.parent,
        args.backbone,
        args.output,
        scope=Scope(**read_json(args.binding)["scope"]),
        model_sha256=args.model_sha256,
        conversion_sha256=args.conversion_sha256,
    )
    print(json.dumps(result, indent=2))
    if not result["latency_budget_passed"]:
        raise SystemExit(1)
