"""Offline import of operator-acquired, license-reviewed exact NVIDIA weights."""

import argparse
import hashlib
import shutil
from pathlib import Path

from learning.common import file_digest, keys, read_json, require, safe_path, write_json
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.gr00t import MODEL_REVISION, POLICY_TYPE
from learning.gr00t.artifacts import PRETRAINED_WEIGHTS, model_contract, validate_model
from learning.gr00t.licensing import require_commercial_model

PRETRAINED_METADATA = {
    "config.json": "dc65cdaf211ac2368ec9f896b1b401bbf8d9c32d",
    "experiment_cfg/metadata.json": "da0f878bd43488eed700f3e6b3693e41b4b66d78",
    "model.safetensors.index.json": "7683f029898daa196d61df0a041107ac72921730",
}


def import_pretrained(
    source: Path,
    output: Path,
    *,
    scope: Scope,
    profile: ControlProfile,
    task: DemonstrationSource,
    acknowledge_license_review: bool,
) -> str:
    require(acknowledge_license_review is True, "Operator model-license review is required")
    require_commercial_model(POLICY_TYPE, MODEL_REVISION)
    require(not output.exists(), "Never overwrite an approved pretrained artifact")
    scope.validate()
    profile.validate()
    task.validate()
    for name, expected in PRETRAINED_WEIGHTS.items():
        require(
            file_digest(safe_path(source, name)) == expected, "Wrong pinned NVIDIA weight shard"
        )
    for name, expected in PRETRAINED_METADATA.items():
        payload = safe_path(source, name).read_bytes()
        blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
        require(hashlib.sha1(blob).hexdigest() == expected, "Wrong pinned NVIDIA metadata revision")
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(parents=True)
    for name in (*PRETRAINED_WEIGHTS, *PRETRAINED_METADATA):
        destination = checkpoint / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    write_json(
        output / "model.json",
        model_contract(
            scope=scope,
            profile=profile,
            task=task,
            checkpoint=checkpoint,
            role="pretrained",
            training=None,
        ),
    )
    checksum = file_digest(output / "model.json")
    validate_model(
        output, expected_scope=scope, expected_model_sha256=checksum, for_inference=False
    )
    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--acknowledge-license-review", action="store_true")
    args = parser.parse_args()
    binding = keys(
        read_json(args.binding), {"scope", "control_profile", "task"}, "operator binding"
    )
    print(
        import_pretrained(
            args.source,
            args.output,
            scope=Scope(**binding["scope"]),
            profile=ControlProfile(**binding["control_profile"]),
            task=DemonstrationSource(kind="reference_controller", **binding["task"]),
            acknowledge_license_review=args.acknowledge_license_review,
        )
    )
