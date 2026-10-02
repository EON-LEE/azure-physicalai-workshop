import argparse
from pathlib import Path

from learning.common import file_digest, read_json, require
from learning.contract import TEACHING_CONTRACT_VERSION, Scope, validate_dataset
from learning.convert import convert_dataset as convert_lerobot
from learning.smolvla.adaptation import IMAGE_SIZE


def convert_dataset(
    source: Path,
    output: Path,
    *,
    expected_scope: Scope,
    expected_manifest_sha256: str,
    allow_test_fixture: bool = False,
) -> dict:
    dataset = validate_dataset(
        source,
        expected_scope=expected_scope,
        expected_manifest_sha256=expected_manifest_sha256,
        require_live=not allow_test_fixture,
    )
    require(
        dataset.manifest["schema"] == TEACHING_CONTRACT_VERSION,
        "SmolVLA requires real v2 ten-Hz hold/tracking evidence, not decimated baseline data",
    )
    return convert_lerobot(
        source,
        output,
        expected_scope=expected_scope,
        expected_manifest_sha256=expected_manifest_sha256,
        split="train",
        image_size=IMAGE_SIZE,
        allow_test_fixture=allow_test_fixture,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Private real v2 -> LeRobot v3 Smol conversion.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    convert_dataset(
        args.source,
        args.output,
        expected_scope=Scope(**read_json(args.binding)["scope"]),
        expected_manifest_sha256=args.manifest_sha256,
    )
    print(file_digest(args.output / "conversion.json"))
