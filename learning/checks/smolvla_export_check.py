"""Actual LeRobot v3 conversion, with explicitly synthetic v2 input and no policy weights."""

import argparse
import json
from pathlib import Path

from learning.checks.fixtures import SCOPE
from learning.checks.groot_fixtures import TASK, teaching
from learning.common import file_digest, require, write_json
from learning.smolvla.dataset import convert_dataset
from learning.train import validate_conversion


def run(output: Path) -> dict:
    require(not output.exists(), "Choose a new explicit test-only output directory")
    raw = teaching(output / "raw")
    converted = output / "v3"
    manifest = convert_dataset(
        raw,
        converted,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(raw / "manifest.json"),
        allow_test_fixture=True,
    )
    validate_conversion(converted, SCOPE)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "azure-local/physicalai-reference-arm",
        root=converted,
        video_backend="pyav",
        download_videos=False,
    )
    frame = dataset[0]
    require(
        tuple(frame["observation.state"].shape) == tuple(frame["action"].shape) == (9,),
        "Converted physical data must be nine actual dimensions",
    )
    require(
        frame["task"] == TASK.instruction, "Actual approved task was replaced by a generic caption"
    )
    require(dataset.num_episodes == 1 and len(dataset) == 18, "Dropped/resampled physical frames")
    report = {
        "policy_type": "smolvla",
        "dataset_format": dataset.meta.info["codebase_version"],
        "conversion_schema": manifest["schema"],
        "physical_dimensions": 9,
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "task_preserved": True,
        "actual_image_shapes": {
            name: list(frame[f"observation.images.{name}"].shape)
            for name in ("inspection", "overview")
        },
        "conversion_sha256": file_digest(converted / "conversion.json"),
        "test_only": True,
        "vendor_weights_loaded": False,
        "optimizer_steps": 0,
        "learning_quality_verified": False,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
