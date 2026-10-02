"""Verify pinned upstream APIs without importing model code or downloading weights."""

import argparse
import ast
import json
from pathlib import Path

from learning.common import file_digest, read_json, require
from learning.gr00t import SOURCE_COMMIT
from learning.gr00t.source import activate_source


def run(root: Path) -> dict:
    activate_source(root)
    signatures = {}
    for relative, class_name, method, required in (
        (
            "gr00t/data/dataset.py",
            "LeRobotSingleDataset",
            "__init__",
            {"dataset_path", "modality_configs", "embodiment_tag", "video_backend", "transforms"},
        ),
        (
            "gr00t/model/policy.py",
            "Gr00tPolicy",
            "__init__",
            {"model_path", "embodiment_tag", "modality_config", "modality_transform", "device"},
        ),
        ("gr00t/model/policy.py", "Gr00tPolicy", "get_action", {"observations"}),
        (
            "gr00t/experiment/runner.py",
            "TrainRunner",
            "__init__",
            {"model", "training_args", "train_dataset", "resume_from_checkpoint"},
        ),
        ("gr00t/experiment/runner.py", "TrainRunner", "train", set()),
    ):
        tree = ast.parse((root / relative).read_text())
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        function = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method
        )
        arguments = [argument.arg for argument in function.args.args]
        require(required.issubset(arguments), f"Upstream signature mismatch: {class_name}.{method}")
        signatures[f"{class_name}.{method}"] = arguments
    eagle = root / "gr00t" / "model" / "backbone" / "eagle2_hg_model"
    config = read_json(eagle / "config.json")
    require(
        all("--" not in value for value in config["auto_map"].values()),
        "Remote Eagle code reference",
    )
    return {
        "source_commit": SOURCE_COMMIT,
        "api_signatures_from_actual_source": signatures,
        "bundled_eagle_config_sha256": file_digest(eagle / "config.json"),
        "model_imported": False,
        "weights_downloaded": False,
        "gpu_training_verified": False,
        "learning_quality_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().source_root), indent=2))
