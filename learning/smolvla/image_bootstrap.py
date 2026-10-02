"""Standalone stdlib-only entry for a frozen image payload, never a generic command launcher."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

DELIVERY_SCHEMA = "physicalai.smolvla-source-delivery/v1"
STATIC_SCHEMA = "physicalai.smolvla-static-source/v1"
COMMAND_EXECUTION = {
    "schema": "physicalai.smolvla-command-execution/v1",
    "kind": "command",
    "data_transport": "private_blob_mi",
}
NATIVE_PYTHON = "/opt/smolvla-venv/bin/python"
IMAGE_ROOT = "/opt/physicalai"
BOOTSTRAP_RELATIVE = "learning/smolvla/image_bootstrap.py"
BOOTSTRAP_PATH = IMAGE_ROOT + "/source/" + BOOTSTRAP_RELATIVE
MAX_CONFIG_BYTES = 16 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_FILES = 128
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024


class ImageSourceError(ValueError):
    """A source/config/authority mismatch stops execution before native imports or work."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImageSourceError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "An explicit SHA-256 binding is required",
    )
    return value


def _parse(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key in frozen source/config")
            result[key] = value
        return result

    def nonfinite(value):
        raise ImageSourceError(f"Nonfinite JSON value: {value}")

    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ImageSourceError("Invalid bounded source/config JSON") from exc
    _require(isinstance(result, dict), "Frozen source/config must be a JSON object")
    return result


def _no_symlinks(path: Path) -> None:
    _require(path.is_absolute(), "Static/materialized source paths must be absolute")
    for candidate in (path, *path.parents):
        _require(not candidate.is_symlink(), "Symlinks are forbidden in frozen source paths")


def _read(path: Path, maximum: int) -> bytes:
    _no_symlinks(path)
    _require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Missing regular source file")
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    _require(len(value) <= maximum, "Frozen source/config exceeds its byte limit")
    return value


def _relative(name: object, *, static_only: bool) -> str:
    _require(
        isinstance(name, str)
        and bool(name)
        and re.fullmatch(r"[A-Za-z0-9_./-]+", name) is not None
        and not PurePosixPath(name).is_absolute()
        and all(part not in ("", ".", "..") for part in name.split("/")),
        "Unsafe frozen source path",
    )
    if static_only:
        _require(
            name.startswith("learning/"), "Static payload may contain only approved learning source"
        )
    return name


def _inventory(root: Path, files: dict, *, extra_files=(), check=None) -> dict[str, bytes]:
    _no_symlinks(root)
    _require(root.is_dir(), "Frozen source directory is missing")
    _require(
        isinstance(files, dict) and 1 <= len(files) <= MAX_FILES, "Source file count exceeds bounds"
    )
    allowed = set(files) | set(extra_files)
    directories = set()
    for name in allowed:
        _relative(name, static_only=False)
        directories.update(
            str(parent) for parent in PurePosixPath(name).parents if str(parent) != "."
        )
    seen = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            _require(not path.is_symlink(), "Symlink in approved source inventory")
            name = path.relative_to(root).as_posix()
            if path.is_dir():
                _require(name in directories, "Unexpected directory in frozen source inventory")
                pending.append(path)
            else:
                _require(
                    name in allowed and path.is_file(), "Unexpected file in frozen source inventory"
                )
                seen.add(name)
    _require(seen == allowed, "Missing file in frozen source inventory")
    payloads, total = {}, 0
    for name, checksum in files.items():
        if check is not None:
            check()
        data = _read(root.joinpath(*PurePosixPath(name).parts), MAX_FILE_BYTES)
        total += len(data)
        _require(total <= MAX_SOURCE_BYTES, "Static source exceeds its total byte limit")
        _require(_digest(data) == _sha(checksum), "Frozen source checksum mismatch")
        payloads[name] = data
    return payloads


def verify_static(
    source_root: Path, manifest_path: Path, static_sha256: str, *, check=None
) -> dict:
    body = _read(manifest_path, MAX_MANIFEST_BYTES)
    manifest = _parse(body)
    _require(
        set(manifest) == {"schema", "files", "sha256"}
        and manifest["schema"] == STATIC_SCHEMA
        and body == _canonical(manifest) + b"\n",
        "Unknown or noncanonical static-source manifest",
    )
    files = manifest["files"]
    _require(isinstance(files, dict) and 1 <= len(files) <= MAX_FILES, "Invalid static inventory")
    for name in files:
        _relative(name, static_only=True)
    _require(BOOTSTRAP_RELATIVE in files, "Frozen source does not bind its bootstrap")
    _require(
        manifest["sha256"] == _sha(static_sha256) == _digest(_canonical(files)),
        "Static-source manifest differs from the approved image payload",
    )
    _inventory(source_root, files, check=check)
    return manifest


class _Deadline:
    def __init__(self, config: dict) -> None:
        value = config.get("job_deadline_utc")
        _require(
            isinstance(value, str) and value.endswith("Z"), "Explicit UTC job deadline required"
        )
        try:
            self.expiry = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ImageSourceError("Invalid absolute job deadline") from exc
        _require(
            self.expiry.tzinfo == UTC and self.expiry.isoformat().replace("+00:00", "Z") == value,
            "Job deadline must be canonical UTC",
        )
        self.monotonic_end = time.monotonic() + (self.expiry - datetime.now(UTC)).total_seconds()

    def check(self) -> None:
        _require(
            datetime.now(UTC) < self.expiry and time.monotonic() < self.monotonic_end,
            "Original approved job deadline expired before source admission",
        )


def _configuration(encoded: str, checksum: str, static_sha256: str) -> tuple[dict, bytes]:
    _require(
        isinstance(encoded, str) and 1 <= len(encoded) <= 4 * ((MAX_CONFIG_BYTES + 2) // 3),
        "Encoded runtime config exceeds its bounded command payload",
    )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageSourceError("Invalid runtime config base64") from exc
    _require(
        len(payload) <= MAX_CONFIG_BYTES and _digest(payload) == _sha(checksum),
        "Runtime config checksum/size mismatch",
    )
    config = _parse(payload)
    _require(
        payload == _canonical(config) + b"\n",
        "Runtime config is not the exact canonical snapshot bytes",
    )
    _require(
        config.get("schema") == "physicalai.smolvla-azure/v2"
        and config.get("kind") == "train"
        and config.get("execution_timing") == "paused_simulation"
        and config.get("real_time_admission") is False
        and config.get("source_delivery")
        == {
            "schema": DELIVERY_SCHEMA,
            "mode": "image_embedded",
            "static_sha256": _sha(static_sha256),
        },
        "Runtime config is not the closed approved paused image-delivery variant",
    )
    _require(
        isinstance(config.get("environment_image"), str)
        and re.fullmatch(
            r"[a-z0-9]{5,50}\.azurecr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}",
            config["environment_image"],
        )
        is not None,
        "Runtime image must remain digest-pinned",
    )
    if "job_execution" in config:
        _require(
            config["job_execution"] == COMMAND_EXECUTION, "Unapproved standalone job execution"
        )
    if "training_cohort" in config:
        _require(
            config.get("job_execution") == COMMAND_EXECUTION
            and config["training_cohort"]
            == {
                "schema": "physicalai.smolvla-training-cohort/v1",
                "kind": "p1_additional20",
            }
            and config.get("parameters", {}).get("resume_mode") == "weights_only"
            and isinstance(config.get("checkpointing"), dict)
            and "resume" in config["checkpointing"]
            and config["checkpointing"]["resume"] is None,
            "Additional P1 source admission requires the closed weights-only cohort authority",
        )
    return config, payload


def materialize_source(
    source_root: Path,
    manifest_path: Path,
    *,
    config_base64: str,
    config_sha256: str,
    snapshot_sha256: str,
    static_sha256: str,
    destination: Path,
) -> dict:
    config, payload = _configuration(config_base64, config_sha256, static_sha256)
    deadline = _Deadline(config)
    deadline.check()
    manifest = verify_static(source_root, manifest_path, static_sha256, check=deadline.check)
    files = {**manifest["files"], "run-config.json": _digest(payload)}
    _require(
        _digest(_canonical(files)) == _sha(snapshot_sha256),
        "Full static-plus-config snapshot differs from the approved plan",
    )
    _no_symlinks(destination)
    _require(not destination.exists(), "Materialization requires a fresh task-local directory")
    destination.mkdir(parents=True, mode=0o700)
    complete = False
    try:
        for name, checksum in manifest["files"].items():
            deadline.check()
            source = source_root.joinpath(*PurePosixPath(name).parts)
            data = _read(source, MAX_FILE_BYTES)
            _require(_digest(data) == checksum, "Source changed during materialization")
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        with (destination / "run-config.json").open("xb") as stream:
            stream.write(payload)
        _inventory(destination, files, check=deadline.check)
        snapshot = {"files": files, "sha256": snapshot_sha256}
        with (destination / "snapshot.json").open("xb") as stream:
            stream.write(_canonical(snapshot) + b"\n")
        deadline.check()
        complete = True
        return config
    finally:
        if not complete:
            shutil.rmtree(destination)


def execute_component(
    root: Path,
    config: dict,
    *,
    command: str,
    snapshot_sha256: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
    parent_path: Path | None = None,
    backbone_path: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> None:
    _require(
        command in ("export", "train", "command-train"),
        "Only the approved export/train native components may execute",
    )
    direct = command == "command-train"
    _require(direct == ("job_execution" in config), "Wrong native command execution variant")
    _require(
        all(
            value is None
            for value in (input_path, output_path, parent_path, backbone_path, resume_checkpoint)
        )
        if direct
        else (
            input_path is not None
            and output_path is not None
            and (
                parent_path is not None and backbone_path is not None
                if command == "train"
                else parent_path is None and backbone_path is None and resume_checkpoint is None
            )
        ),
        "Wrong arguments for the selected native component",
    )
    _require(
        resume_checkpoint is None or config.get("checkpointing", {}).get("resume") is not None,
        "An unapproved resume checkpoint cannot be passed to the native component",
    )
    deadline = _Deadline(config)
    deadline.check()
    snapshot = _parse(_read(root / "snapshot.json", MAX_MANIFEST_BYTES))
    _require(
        set(snapshot) == {"files", "sha256"}
        and snapshot["sha256"] == _sha(snapshot_sha256)
        and _digest(_canonical(snapshot["files"])) == snapshot_sha256,
        "Materialized snapshot binding changed before execution",
    )
    _require(
        _read(root / "run-config.json", MAX_CONFIG_BYTES) == _canonical(config) + b"\n",
        "Materialized runtime config changed before execution",
    )
    _inventory(root, snapshot["files"], extra_files=("snapshot.json",), check=deadline.check)
    args = [
        sys.executable,
        "-B",
        "-m",
        "learning.paused.command" if direct else "learning.paused.components",
    ]
    if not direct:
        args.extend(
            (command, "--input", str(input_path.resolve()), "--output", str(output_path.resolve()))
        )
    args.extend(
        [
            "--runtime-config",
            "run-config.json",
            "--snapshot-sha256",
            snapshot_sha256,
        ]
    )
    for name, value in (
        ("--parent", parent_path),
        ("--backbone", backbone_path),
        ("--resume-checkpoint", resume_checkpoint),
    ):
        if value is not None:
            args.extend((name, str(value.resolve())))
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONINSPECT"):
        environment.pop(name, None)
    environment.update(
        PYTHONPATH=str(root),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    deadline.check()
    os.chdir(root)
    os.execve(sys.executable, args, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    verify = actions.add_parser("verify-static")
    verify.add_argument("--static-sha256", required=True)
    run = actions.add_parser("run")
    run.add_argument("command", choices=("export", "train", "command-train"))
    for name in ("static-sha256", "config-base64", "config-sha256", "snapshot-sha256"):
        run.add_argument(f"--{name}", required=True)
    for name in ("input", "output"):
        run.add_argument(f"--{name}", type=Path)
    for name in ("parent", "backbone", "resume-checkpoint"):
        run.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    _require(
        os.name == "posix" and sys.version_info[:2] == (3, 11),
        "Embedded native source requires the qualified Linux Python 3.11 interpreter",
    )
    source = Path(__file__).absolute().parents[2]
    manifest_path = source.parent / "static-manifest.json"
    if args.action == "verify-static":
        manifest = verify_static(source, manifest_path, args.static_sha256)
        print(
            json.dumps(
                {
                    "static_sha256": manifest["sha256"],
                    "files": len(manifest["files"]),
                    "runtime_config_embedded": False,
                    "model_loaded": False,
                    "cloud_calls": 0,
                }
            )
        )
        return
    with tempfile.TemporaryDirectory(prefix="physicalai-embedded-") as temporary:
        root = Path(temporary) / "code"
        config = materialize_source(
            source,
            manifest_path,
            config_base64=args.config_base64,
            config_sha256=args.config_sha256,
            snapshot_sha256=args.snapshot_sha256,
            static_sha256=args.static_sha256,
            destination=root,
        )
        execute_component(
            root,
            config,
            command=args.command,
            snapshot_sha256=args.snapshot_sha256,
            input_path=args.input,
            output_path=args.output,
            parent_path=args.parent,
            backbone_path=args.backbone,
            resume_checkpoint=args.resume_checkpoint,
        )


if __name__ == "__main__":
    try:
        main()
    except ImageSourceError as error:
        raise SystemExit(f"Embedded source admission failed: {error}") from error
