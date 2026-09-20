from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from learning import LEROBOT_VERSION
from learning.common import (
    digest,
    integer,
    inventory,
    require,
    safe_path,
    token,
    write_json,
)
from learning.contract import CAMERAS, JOINT_NAMES, JOINT_UNITS, Scope, validate_dataset
from learning.offline import require_lerobot

CONVERSION_SCHEMA = "physicalai.lerobot-conversion/v1"
LOCAL_REPO_ID = "azure-local/physicalai-reference-arm"
TASK = "Inspect the visible part and place it in its approved tray."


def feature_spec(image_size: int) -> dict:
    integer(image_size, "image size", 32, 512)
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(JOINT_NAMES),
        },
        "action": {"dtype": "float32", "shape": (9,), "names": list(JOINT_NAMES)},
        **{
            f"observation.images.{camera}": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERAS
        },
    }


def convert_dataset(
    source: Path,
    output: Path,
    *,
    expected_scope: Scope,
    expected_manifest_sha256: str,
    split: str = "train",
    image_size: int = 224,
    allow_test_fixture: bool = False,
) -> dict:
    validated = validate_dataset(
        source,
        expected_scope=expected_scope,
        expected_manifest_sha256=expected_manifest_sha256,
        require_live=not allow_test_fixture,
    )
    episodes = validated.split(split)
    features = feature_spec(image_size)
    require(not output.exists(), "Conversion output already exists")
    require(
        not output.resolve().is_relative_to(source.resolve()), "Output must be outside raw data"
    )
    require_lerobot()
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    dataset = LeRobotDataset.create(
        repo_id=LOCAL_REPO_ID,
        root=output,
        fps=validated.manifest["fps"],
        robot_type="isaac_franka_9dof",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
        video_backend="pyav",
    )
    try:
        for episode in episodes:
            for raw in episode.frames:
                frame = {
                    "observation.state": np.asarray(raw["joint_positions"], dtype=np.float32),
                    "action": np.asarray(raw["commanded_joint_targets"], dtype=np.float32),
                    "task": TASK,
                }
                for camera in CAMERAS:
                    image = raw["images"][camera]
                    payload = safe_path(source, image["path"]).read_bytes()
                    require(digest(payload) == image["sha256"], "Image changed after validation")
                    with Image.open(BytesIO(payload)) as decoded:
                        frame[f"observation.images.{camera}"] = np.asarray(
                            decoded.convert("RGB").resize(
                                (image_size, image_size), Image.Resampling.BILINEAR
                            ),
                            dtype=np.uint8,
                        )
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.finalize()
    manifest = {
        "schema": CONVERSION_SCHEMA,
        "lerobot_version": LEROBOT_VERSION,
        "repo_id": LOCAL_REPO_ID,
        "scope": asdict(expected_scope),
        "raw_manifest_sha256": validated.manifest_sha256,
        "dataset_id": token(validated.manifest["dataset_id"], "dataset ID"),
        "split": split,
        "fps": validated.manifest["fps"],
        "physics_hz": validated.manifest["physics_hz"],
        "joint_names": list(JOINT_NAMES),
        "joint_units": list(JOINT_UNITS),
        "image_size": image_size,
        "image_transform": "rgb-bilinear-square/v1",
        "test_only": any(
            ep.metadata["provenance"]["source_kind"] != "isaac_sim" for ep in episodes
        ),
        "episodes": [
            {
                "episode_index": index,
                **episode.metadata,
                "terminated": episode.frames[-1]["terminated"],
                "truncated": episode.frames[-1]["truncated"],
            }
            for index, episode in enumerate(episodes)
        ],
        "files": inventory(output),
    }
    write_json(output / "conversion.json", manifest)
    return manifest
