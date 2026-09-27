"""Create a static, config-free code layer and bind it to reviewed native AML job plans."""

from __future__ import annotations

import argparse
import base64
import shutil
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    keys,
    require,
    safe_path,
    sha256,
    write_json,
)
from learning.smolvla.image_bootstrap import (
    BOOTSTRAP_PATH,
    BOOTSTRAP_RELATIVE,
    DELIVERY_SCHEMA,
    MAX_CONFIG_BYTES,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_SOURCE_BYTES,
    NATIVE_PYTHON,
    STATIC_SCHEMA,
)

QUALIFIED_BASE_IMAGE = (
    "factory20n3ig3ttsxayp2.azurecr.io/physicalai-smolvla@sha256:"
    "441f2a33a8bb0c534a10ad7d56c4f7be611dccde39e8ee0b99610b4a7a75e8f9"
)
SOURCE_FILES = ("learning/smolvla/embedded_source.py", BOOTSTRAP_RELATIVE)


def static_code_files() -> tuple[str, ...]:
    from learning.smolvla.azure import CHECKPOINT_CODE_FILES, PAUSED_CODE_FILES

    return PAUSED_CODE_FILES + CHECKPOINT_CODE_FILES + SOURCE_FILES


def validate_delivery(config: dict) -> None:
    value = keys(config["source_delivery"], {"schema", "mode", "static_sha256"}, "source delivery")
    require(
        config.get("schema") == "physicalai.smolvla-azure/v2"
        and config.get("execution_timing") == "paused_simulation"
        and config.get("real_time_admission") is False
        and config.get("kind") == "train"
        and value["schema"] == DELIVERY_SCHEMA
        and value["mode"] == "image_embedded",
        "Embedded source delivery is only the explicit paused-v2 export/train variant",
    )
    sha256(value["static_sha256"], "approved static image source")
    require(
        len(canonical(config)) + 1 <= MAX_CONFIG_BYTES, "Embedded runtime config exceeds byte limit"
    )


def static_inventory(root: Path) -> dict[str, str]:
    names = static_code_files()
    require(len(names) == len(set(names)) <= MAX_FILES, "Invalid static source allowlist")
    files, total = {}, 0
    for name in names:
        path = safe_path(root, name)
        size = path.stat().st_size
        total += size
        require(
            size <= MAX_FILE_BYTES and total <= MAX_SOURCE_BYTES,
            "Static source exceeds frozen byte limits",
        )
        files[name] = file_digest(path)
    return files


def verify_static_binding(config: dict, root: Path) -> None:
    validate_delivery(config)
    require(
        digest(canonical(static_inventory(root))) == config["source_delivery"]["static_sha256"],
        "Static image source differs from the reviewed plan source bytes",
    )


def prepare_context(output: Path, *, source_root: Path | None = None) -> dict:
    root = source_root or Path(__file__).resolve().parents[2]
    files = static_inventory(root)
    static_sha = digest(canonical(files))
    require(
        not output.exists() and not output.is_symlink(),
        "Never overwrite a frozen image build context",
    )
    payload = output / "source"
    payload.mkdir(parents=True)
    for name, checksum in files.items():
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(root, name), target)
        require(file_digest(target) == checksum, "Static source changed while packaging")
    write_json(
        output / "static-manifest.json",
        {
            "schema": STATIC_SCHEMA,
            "files": files,
            "sha256": static_sha,
        },
    )
    delivery = {"schema": DELIVERY_SCHEMA, "mode": "image_embedded", "static_sha256": static_sha}
    write_json(output / "source-delivery.json", delivery)
    shutil.copyfile(safe_path(root, "learning/smolvla/Dockerfile.embedded"), output / "Dockerfile")
    receipt = {
        "schema": "physicalai.smolvla-static-source-package/v1",
        "base_image": QUALIFIED_BASE_IMAGE,
        "static_sha256": static_sha,
        "file_count": len(files),
        "runtime_config_embedded": False,
        "image_built": False,
        "cloud_calls": 0,
        "build_arguments": [
            "docker",
            "build",
            "--file",
            str(output / "Dockerfile"),
            "--build-arg",
            f"STATIC_SOURCE_SHA256={static_sha}",
            str(output),
        ],
        "next_step": (
            "After an explicitly authorized image build/readback, put its immutable digest "
            "in environment_image and source-delivery.json in source_delivery. "
            "Then create a new reviewed plan."
        ),
    }
    write_json(output / "package.json", receipt)
    return receipt


def embedded_command(config: dict, snapshot_sha256: str, *, stage: str) -> str:
    validate_delivery(config)
    sha256(snapshot_sha256)
    require(stage in ("convert", "train"), "Unsupported embedded component")
    payload = canonical(config) + b"\n"
    encoded = base64.b64encode(payload).decode("ascii")
    command = (
        f"{NATIVE_PYTHON} -I -B {BOOTSTRAP_PATH} run "
        + ("export" if stage == "convert" else "train")
        + f" --static-sha256 {config['source_delivery']['static_sha256']}"
        + f" --config-base64 {encoded} --config-sha256 {digest(payload)}"
        + f" --snapshot-sha256 {snapshot_sha256}"
    )
    if stage == "convert":
        command += " --input '${{inputs.raw}}' --output '${{outputs.dataset}}'"
    else:
        command += (
            " --input '${{inputs.dataset}}' --output '${{outputs.model}}'"
            " --parent '${{inputs.parent}}' --backbone '${{inputs.backbone}}'"
        )
        if config.get("checkpointing", {}).get("resume") is not None:
            command += " --resume-checkpoint '${{inputs.resume}}'"
    require(len(command.encode()) <= 32768, "Embedded command exceeds the fixed byte budget")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(canonical(prepare_context(args.output)).decode())


if __name__ == "__main__":
    main()
