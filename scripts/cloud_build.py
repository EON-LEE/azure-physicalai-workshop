"""Build an allowlisted production source snapshot in the approved Azure Container Registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
SINGLE_FILES = ("Dockerfile", "pyproject.toml", "uv.lock", ".python-version", ".dockerignore")
DIRECTORIES = ("apps", "agents", "contracts", "examples", "scripts", "simulation", "learning")
EXCLUDED = {
    "node_modules",
    ".venv",
    "dist",
    "tests",
    "test-results",
    ".fixture-dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    "playwright-report",
    "coverage",
}


def source_files(root: Path) -> list[Path]:
    paths = [root / name for name in SINGLE_FILES]
    for directory in DIRECTORIES:
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Unexpected build-input symlink: {relative}")
            if path.is_file():
                if path.suffix in {".pem", ".key", ".pfx"} or path.name.startswith(".env"):
                    raise ValueError(
                        f"Credential-shaped file cannot enter the cloud build: {relative}"
                    )
                paths.append(path)
    return sorted(paths)


def snapshot(root: Path, output: Path) -> str:
    digest = hashlib.sha256()
    with tarfile.open(output, "w:gz") as archive:
        for path in source_files(root):
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            digest.update(name.encode() + b"\0" + content)
            archive.add(path, arcname=name, recursive=False)
    return digest.hexdigest()


@contextmanager
def reviewed_source(root: Path):
    with tempfile.TemporaryDirectory(prefix="physicalai-build-") as directory:
        staging = Path(directory)
        bundle = staging / "source.tar.gz"
        checksum = snapshot(root, bundle)
        source = staging / "source"
        source.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(source, filter="data")
        yield source, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", required=True, type=UUID)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with reviewed_source(ROOT) as (source, checksum):
        if not args.apply:
            print(json.dumps({"source_sha256": checksum, "azure_changed": False}))
            return
        command = [
            "az",
            "acr",
            "build",
            "--subscription",
            str(args.subscription),
            "--registry",
            args.registry,
            "--image",
            args.tag,
            "--file",
            "Dockerfile",
            str(source),
            "--only-show-errors",
        ]
        completed = subprocess.run(command, text=True, check=False, timeout=1800)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        result = subprocess.run(
            [
                "az",
                "acr",
                "repository",
                "show",
                "--subscription",
                str(args.subscription),
                "--name",
                args.registry,
                "--image",
                args.tag,
                "--output",
                "json",
                "--only-show-errors",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        manifest = json.loads(result.stdout)
        report = {
            "source_sha256": checksum,
            "tag": args.tag,
            "digest": manifest["digest"],
            "registry": args.registry,
            "subscription_id": str(args.subscription),
            "built_on": "Azure Container Registry",
            "runtime_verified": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
