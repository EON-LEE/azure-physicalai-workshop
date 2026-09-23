"""G0 measurement with real vendor weights and real captured data; never an executable P0."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from learning.common import read_json, require, write_json
from learning.contract import Scope
from learning.gr00t.artifacts import validate_model
from learning.gr00t.dataset import validate_export
from learning.gr00t.franka_modality import FrankaDataConfig
from learning.gr00t.inference import physical_actions
from learning.gr00t.source import activate_source


def validate_probe_inputs(
    dataset: Path,
    model: Path,
    *,
    tenant_id: str,
    owner_id: str,
    export_sha256: str,
    model_sha256: str,
) -> tuple[dict, dict]:
    scope = Scope(tenant_id, owner_id)
    exported = validate_export(dataset, scope=scope, expected_sha256=export_sha256)
    require(exported["test_only"] is False, "G0 GPU probe needs real captured v2 data")
    checkpoint = validate_model(
        model,
        expected_scope=scope,
        expected_model_sha256=model_sha256,
        for_inference=False,
    )
    require(
        checkpoint["role"] == "pretrained", "G0 vendor probe is not a trained-policy comparison"
    )
    require(
        checkpoint["control_profile"] == exported["control_profile"], "Probe servo profile mismatch"
    )
    return exported, checkpoint


def run_probe(
    dataset: Path,
    model_root: Path,
    output: Path,
    *,
    source_root: Path,
    tenant_id: str,
    owner_id: str,
    export_sha256: str,
    model_sha256: str,
) -> dict:
    validate_probe_inputs(
        dataset,
        model_root,
        tenant_id=tenant_id,
        owner_id=owner_id,
        export_sha256=export_sha256,
        model_sha256=model_sha256,
    )
    activate_source(source_root)
    import numpy as np
    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    require(torch.cuda.is_available(), "Actual GPU/CUDA is unavailable")
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
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = (
        GR00T_N1_5.from_pretrained(
            str((model_root / "checkpoint").resolve()),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        .to("cuda")
        .eval()
    )
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - start
    config = FrankaDataConfig()
    data = LeRobotSingleDataset(
        dataset_path=dataset,
        modality_configs=config.modality_config(),
        embodiment_tag="new_embodiment",
        video_backend="decord",
    )
    transforms = config.transform()
    transforms.set_metadata(data.metadata)
    transforms.eval()
    observations = data.get_step_data(int(data.trajectory_ids[0]), 0)
    observations = {
        name: value for name, value in observations.items() if not name.startswith("action.")
    }
    require(
        set(observations)
        == {
            "video.inspection",
            "video.overview",
            "state.arm",
            "state.fingers",
            "annotation.human.task_description",
        },
        "Unexpected information in actual probe observations",
    )
    observations = {name: np.expand_dims(value, 0) for name, value in observations.items()}
    latencies, shapes = [], []
    for _ in range(6):
        torch.cuda.synchronize()
        started = time.monotonic_ns()
        batch = transforms(observations)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions = model.get_action(batch)["action_pred"].float().cpu()
        outputs = transforms.unapply({"action": actions})
        result = {name: value[0].tolist() for name, value in outputs.items()}
        require(
            set(result) == {"action.arm", "action.fingers"}
            and len(result["action.arm"]) == len(result["action.fingers"]) == 16
            and np.isfinite(np.asarray(result["action.arm"])).all()
            and np.isfinite(np.asarray(result["action.fingers"])).all(),
            "Actual GR00T prediction has invalid physical dimensions/nonfinite outputs",
        )
        torch.cuda.synchronize()
        latencies.append((time.monotonic_ns() - started) / 1e6)
        shapes.append(
            [
                len(result["action.arm"]),
                len(result["action.arm"][0]) + len(result["action.fingers"][0]),
            ]
        )
    from learning.common import ContractError

    try:
        physical_actions(result)
        within_bounds = True
    except ContractError:
        within_bounds = False
    report = {
        "schema": "physicalai.gr00t-gpu-probe/v1",
        "nvidia_smi": driver,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_load_seconds": load_seconds,
        "vendor_model_sha256": model_sha256,
        "export_sha256": export_sha256,
        "actual_inference_calls": 6,
        "physical_action_shapes": shapes,
        "cold_latency_ms": latencies[0],
        "steady_latencies_without_ipc_ms": latencies[1:],
        "latency_budget_passed": max(latencies[1:]) <= 80,
        "last_prediction_in_joint_bounds": within_bounds,
        "normalization_source": "actual scoped teaching dataset; not a trained Franka policy",
        "optimizer_steps": 0,
        "ready_for_live_execution": False,
        "learning_quality_verified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probe.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model-root", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()
    scope = read_json(args.binding)["scope"]
    result = run_probe(
        args.dataset,
        args.model_root,
        args.output,
        source_root=args.source_root,
        **scope,
        export_sha256=args.export_sha256,
        model_sha256=args.model_sha256,
    )
    print(json.dumps(result, indent=2))
    if not result["latency_budget_passed"]:
        raise SystemExit(1)
