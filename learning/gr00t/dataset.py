from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    integer,
    inventory,
    read_json,
    require,
    safe_path,
    sha256,
    verify_inventory,
    write_json,
)
from learning.contract import (
    CAMERAS,
    TEACHING_CONTRACT_VERSION,
    Scope,
    ValidatedDataset,
    ValidatedEpisode,
    validate_dataset,
)
from learning.gr00t import SOURCE_COMMIT

EXPORT_SCHEMA = "physicalai.gr00t-dataset/v1"


def modality_spec() -> dict:
    return {
        **{
            group: {
                name: {
                    "start": start,
                    "end": end,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": original,
                }
                for name, start, end in (("arm", 0, 7), ("fingers", 7, 9))
            }
            for group, original in (("state", "observation.state"), ("action", "action"))
        },
        "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def prepare_export(
    source: Path,
    *,
    expected_scope: Scope,
    allow_test_fixture: bool = False,
    expected_manifest_sha256: str | None = None,
) -> ValidatedDataset:
    dataset = validate_dataset(
        source,
        expected_scope=expected_scope,
        require_live=not allow_test_fixture,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    require(
        dataset.manifest["schema"] == TEACHING_CONTRACT_VERSION,
        "GR00T requires actual v2 10Hz hold evidence; never relabel/decimate v1 data",
    )
    return dataset


def export_rows(
    episode: ValidatedEpisode, *, episode_index: int, start_index: int, task_index: int
) -> list[dict]:
    return [
        {
            "observation.state": frame["joint_positions"],
            "action": frame["commanded_joint_targets"],
            "timestamp": index / 10,
            "frame_index": index,
            "episode_index": episode_index,
            "index": start_index + index,
            "task_index": task_index,
        }
        for index, frame in enumerate(episode.frames)
    ]


def v21_info(*, episodes: int, frames: int, tasks: int, image_size: int) -> dict:
    for value in (episodes, frames, tasks):
        integer(value, "dataset count", 1)
    integer(image_size, "GR00T image size", 32, 512)
    return {
        "codebase_version": "v2.1",
        "robot_type": "isaac_franka_9dof",
        "fps": 10,
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "total_videos": episodes * 2,
        "total_chunks": (episodes + 999) // 1000,
        "chunks_size": 1000,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            **{
                key: {
                    "dtype": "float32",
                    "shape": [9],
                    "names": [
                        *(f"panda_joint{i}" for i in range(1, 8)),
                        "panda_finger_joint1",
                        "panda_finger_joint2",
                    ],
                }
                for key in ("observation.state", "action")
            },
            **{
                f"observation.images.{camera}": {
                    "dtype": "video",
                    "shape": [image_size, image_size, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 10,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for camera in CAMERAS
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            **{
                key: {"dtype": "int64", "shape": [1], "names": None}
                for key in ("frame_index", "episode_index", "index", "task_index")
            },
        },
    }


def _write_video(
    path: Path, episode: ValidatedEpisode, source: Path, camera: str, size: int
) -> None:
    import av
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for index, frame in enumerate(episode.frames):
            image = frame["images"][camera]
            payload = safe_path(source, image["path"]).read_bytes()
            require(digest(payload) == image["sha256"], "Camera changed during export")
            with Image.open(BytesIO(payload)) as decoded:
                pixels = np.array(
                    decoded.convert("RGB").resize(
                        (size, size),
                        Image.Resampling.BILINEAR,
                    )
                )
            encoded = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            encoded.pts, encoded.time_base = index, Fraction(1, 10)
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path), "r") as result:
        timestamps = [float(frame.pts * frame.time_base) for frame in result.decode(video=0)]
    require(
        len(timestamps) == len(episode.frames)
        and all(abs(value - index / 10) < 1e-5 for index, value in enumerate(timestamps)),
        "Encoded video lost frames or changed the actual control cadence",
    )


def export_dataset(
    source: Path,
    output: Path,
    *,
    expected_scope: Scope,
    expected_manifest_sha256: str,
    image_size: int = 224,
    allow_test_fixture: bool = False,
) -> dict:
    dataset = prepare_export(
        source,
        expected_scope=expected_scope,
        allow_test_fixture=allow_test_fixture,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    episodes = dataset.split("train")
    require(not output.exists(), "GR00T output already exists")
    require(
        not output.resolve().is_relative_to(source.resolve()), "Export must be outside raw data"
    )
    integer(image_size, "image size", 32, 512)
    require(image_size % 2 == 0, "H264 image size must be even")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks, records, all_rows = {}, [], []
    for episode in episodes:
        instruction = episode.metadata["demonstration"]["instruction"]
        tasks.setdefault(instruction, len(tasks))
    output.mkdir(parents=True)
    (output / "meta").mkdir()
    info = v21_info(
        episodes=len(episodes),
        frames=sum(len(ep.frames) for ep in episodes),
        tasks=len(tasks),
        image_size=image_size,
    )
    schema = pa.schema(
        [
            ("observation.state", pa.list_(pa.float32(), 9)),
            ("action", pa.list_(pa.float32(), 9)),
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )
    for index, episode in enumerate(episodes):
        instruction = episode.metadata["demonstration"]["instruction"]
        rows = export_rows(
            episode,
            episode_index=index,
            start_index=len(all_rows),
            task_index=tasks[instruction],
        )
        path = output / info["data_path"].format(episode_chunk=index // 1000, episode_index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="snappy")
        for camera in CAMERAS:
            video = info["video_path"].format(
                episode_chunk=index // 1000,
                episode_index=index,
                video_key=f"observation.images.{camera}",
            )
            _write_video(output / video, episode, source, camera, image_size)
        records.append({"episode_index": index, "tasks": [instruction], "length": len(rows)})
        all_rows.extend(rows)
    statistics = {}
    for key in all_rows[0]:
        values = np.asarray([row[key] for row in all_rows], dtype=np.float64).reshape(
            len(all_rows), -1
        )
        statistics[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    write_json(output / "meta" / "info.json", info)
    write_json(output / "meta" / "modality.json", modality_spec())
    write_json(output / "meta" / "stats.json", statistics)
    for name, values in (
        ("episodes.jsonl", records),
        ("tasks.jsonl", [{"task_index": index, "task": task} for task, index in tasks.items()]),
    ):
        (output / "meta" / name).write_bytes(b"".join(canonical(value) + b"\n" for value in values))
    manifest = {
        "schema": EXPORT_SCHEMA,
        "upstream_source_commit": SOURCE_COMMIT,
        "scope": asdict(expected_scope),
        "raw_manifest_sha256": dataset.manifest_sha256,
        "control_profile": dataset.manifest["control_profile"],
        "image_size": image_size,
        "split": "train",
        "frame_count": len(all_rows),
        "episode_count": len(episodes),
        "test_only": any(
            ep.metadata["provenance"]["source_kind"] != "isaac_sim" for ep in episodes
        ),
        "source_counts": dict(Counter(ep.metadata["demonstration"]["kind"] for ep in episodes)),
        "episodes": [
            {
                **ep.metadata,
                "terminated": ep.frames[-1]["terminated"],
                "truncated": ep.frames[-1]["truncated"],
            }
            for ep in episodes
        ],
        "files": inventory(output),
    }
    write_json(output / "export.json", manifest)
    return manifest


def validate_export(root: Path, *, scope: Scope, expected_sha256: str) -> dict:
    from learning.common import file_digest

    require(
        file_digest(safe_path(root, "export.json")) == sha256(expected_sha256),
        "Export checksum mismatch",
    )
    value = read_json(root / "export.json")
    require(
        value.get("schema") == EXPORT_SCHEMA
        and value.get("scope") == asdict(scope)
        and value.get("upstream_source_commit") == SOURCE_COMMIT
        and value.get("split") == "train",
        "Unapproved GR00T export identity",
    )
    verify_inventory(root, value["files"], exclude={"export.json"})
    require(
        read_json(root / "meta" / "modality.json") == modality_spec()
        and read_json(root / "meta" / "info.json")["codebase_version"] == "v2.1",
        "Wrong Franka modality or LeRobot version",
    )
    require(
        len(value["episodes"]) == value["episode_count"] > 0
        and all(ep["split"] == "train" for ep in value["episodes"]),
        "Held-out/empty episode set in training export",
    )
    return value
