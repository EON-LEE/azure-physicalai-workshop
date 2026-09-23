import argparse
import json
from importlib.metadata import version
from pathlib import Path

from learning.checks.fixtures import SCOPE
from learning.checks.groot_fixtures import teaching
from learning.common import file_digest, read_json, require, write_json
from learning.gr00t.dataset import export_dataset, validate_export


def run(output: Path) -> dict:
    import av
    import pyarrow.parquet as pq

    require(not output.exists(), "Choose a new test-only artifact directory")
    raw = teaching(output / "raw")
    exported = output / "v21"
    export_dataset(
        raw,
        exported,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(raw / "manifest.json"),
        allow_test_fixture=True,
        image_size=224,
    )
    manifest = validate_export(
        exported,
        scope=SCOPE,
        expected_sha256=file_digest(exported / "export.json"),
    )
    table = pq.read_table(exported / "data" / "chunk-000" / "episode_000000.parquet")
    require(table.num_rows == 18, "Real Parquet export omitted frames")
    require(len(table["action"][0].as_py()) == 9, "Wrong actual Parquet actuator dimensions")
    for camera in ("inspection", "overview"):
        path = (
            exported
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / "episode_000000.mp4"
        )
        with av.open(str(path)) as video:
            frames = list(video.decode(video=0))
        require(
            len(frames) == 18
            and all((frame.width, frame.height) == (224, 224) for frame in frames),
            "Real MP4 frame count/shape mismatch",
        )
    report = {
        "check": "real-v21-export-on-explicit-test-fixtures",
        "versions": {name: version(name) for name in ("numpy", "pyarrow", "av")},
        "lerobot_format": read_json(exported / "meta" / "info.json")["codebase_version"],
        "frames": table.num_rows,
        "videos": 2,
        "action_dimensions": 9,
        "export_manifest_sha256": file_digest(exported / "export.json"),
        "test_only": manifest["test_only"],
        "upstream_gr00t_execution": False,
        "learning_quality_verified": False,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
