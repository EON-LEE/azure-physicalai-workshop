"""Import exact operator-acquired Apache model assets; no Hub network operation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    inventory,
    read_json,
    require,
    safe_path,
    write_json,
)
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.smolvla import UPSTREAM

VENDOR_SHA256 = {
    "model": {
        "model.safetensors": "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
            "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8"
        ),
        "policy_preprocessor_step_5_normalizer_processor.safetensors": (
            "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8"
        ),
        "README.md": UPSTREAM["model_license_sha256"],
    },
    "backbone": {
        "model.safetensors": "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
        "README.md": UPSTREAM["backbone_license_sha256"],
    },
}
VENDOR_GIT_BLOBS = {
    "model": {
        "config.json": "36d4d51c1b01be11b2c7a074ab34bf5ee6db9021",
        "policy_preprocessor.json": "4df8f7cbce54095074f9cf31b663e5af5c006f85",
        "policy_postprocessor.json": "6f8997e5be18c67bad9377dd2cd9622ba38b5ae3",
    },
    "backbone": {
        "added_tokens.json": "71dc811700b6aa9274d3697ee7d4fb2d5375e766",
        "chat_template.json": "1d342e4f070c962a14ba4ece4e06212bc70d8e36",
        "config.json": "2463e842695989a207176fdb249327a494b05cfb",
        "generation_config.json": "eace9aa6d392fd94dbe0c90825074c53fd7ecd4a",
        "merges.txt": "69503b13f727ba3812b6803e97442a6de05ef5eb",
        "preprocessor_config.json": "bf7669a38692ad141d333db3be18bd55cb6e2c59",
        "processor_config.json": "83df8c48da1f41e1a9129a4bc2aba000eb2b529f",
        "special_tokens_map.json": "2b5aee28d56c0e9d89ff6bc70818fba986a90ca9",
        "tokenizer.json": "a4005d1cf3170a31600a5c96f95768166cbc2b28",
        "tokenizer_config.json": "e4042a1126290fdb96ece2a4ad8dd7108c5de484",
        "vocab.json": "0ad5ecc2035b7031b88afb544ee95e2d49baa484",
    },
}


def verify_vendor_files(root: Path, *, kind: str) -> dict[str, str]:
    require(kind in VENDOR_SHA256, "Unknown vendor asset family")
    result = {}
    for name, expected in VENDOR_SHA256[kind].items():
        path = safe_path(root, name)
        require(file_digest(path) == expected, f"Wrong pinned {kind} file checksum: {name}")
        result[name] = expected
    for name, expected in VENDOR_GIT_BLOBS[kind].items():
        path = safe_path(root, name)
        data = path.read_bytes()
        require(
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected,
            f"Wrong exact {kind} metadata revision: {name}",
        )
        result[name] = digest(data)
    return result


def _source_license() -> Path:
    package = distribution("lerobot")
    paths = [
        package.locate_file(path)
        for path in package.files
        if str(path).endswith("/licenses/LICENSE")
    ]
    require(
        len(paths) == 1 and file_digest(paths[0]) == UPSTREAM["source_license_sha256"],
        "Installed LeRobot source license differs from the approved pin",
    )
    return paths[0]


def import_assets(
    model_source: Path,
    backbone_source: Path,
    output: Path,
    *,
    scope: Scope,
    profile: ControlProfile,
    task: DemonstrationSource,
) -> dict:
    scope.validate()
    profile.validate()
    task.validate()
    require(not output.exists(), "Never overwrite an immutable Smol asset bundle")
    model_files = verify_vendor_files(model_source, kind="model")
    backbone_files = verify_vendor_files(backbone_source, kind="backbone")
    license_path = _source_license()
    for kind, source, files in (
        ("model", model_source, model_files),
        ("backbone", backbone_source, backbone_files),
    ):
        destination = output / kind
        destination.mkdir(parents=True)
        for name in files:
            path = destination / ("checkpoint" if kind == "model" else "assets") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        shutil.copyfile(license_path, destination / "LEROBOT-LICENSE")
    backbone = {
        "schema": "physicalai.smolvla-backbone/v1",
        "scope": asdict(scope),
        "upstream": UPSTREAM,
        "files": inventory(output / "backbone"),
    }
    write_json(output / "backbone" / "backbone.json", backbone)
    backbone_sha = file_digest(output / "backbone" / "backbone.json")
    from learning.smolvla.artifacts import model_contract, validate_model

    model = model_contract(
        checkpoint=output / "model" / "checkpoint",
        scope=scope,
        profile=profile,
        task=task,
        backbone_manifest_sha256=backbone_sha,
        role="pretrained",
        training=None,
    )
    write_json(output / "model" / "model.json", model)
    checksum = file_digest(output / "model" / "model.json")
    validate_model(
        output / "model", expected_scope=scope, expected_model_sha256=checksum, for_inference=False
    )
    return {
        "model_manifest_sha256": checksum,
        "backbone_manifest_sha256": backbone_sha,
        "policy_type": "smolvla",
        "role": "pretrained",
        "ready_for_execution": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--backbone-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    args = parser.parse_args()
    binding = read_json(args.binding)
    result = import_assets(
        args.model_source,
        args.backbone_source,
        args.output,
        scope=Scope(**binding["scope"]),
        profile=ControlProfile(**binding["control_profile"]),
        task=DemonstrationSource(kind="reference_controller", **binding["task"]),
    )
    print(canonical(result).decode())
