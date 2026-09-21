"""Mirror the approved reference robot and its bounded dependencies into a private bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import tarfile
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

ORIGIN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
PREFIX = "/Assets/Isaac/5.1/"
ROOT_ASSET = "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
BUILTIN = frozenset({"OmniPBR.mdl", "OmniGlass.mdl", "UsdPreviewSurface.mdl"})


def resolve_asset(base_url: str, reference: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(base_url, reference))
    path = urllib.parse.unquote(parsed.path)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.netloc != urllib.parse.urlsplit(ORIGIN).netloc
        or parsed.query
        or parsed.fragment
        or not path.startswith(PREFIX)
        or "\\" in path
        or ".." in Path(path).parts
        or "[" in path
        or "<" in path
    ):
        raise ValueError(f"Unapproved or unsupported asset dependency: {reference}")
    relative = path[len(PREFIX) :]
    if not relative or relative.startswith("/"):
        raise ValueError("Asset must be a file inside the pinned reference collection.")
    return ORIGIN + urllib.parse.quote(path, safe="/"), relative


def prepare(output: Path, *, approved: bool) -> dict:
    if not approved:
        raise ValueError("Explicit reference-asset use approval is required.")
    if output.exists():
        raise ValueError("Choose a new output directory; existing assets are not overwritten.")
    from pxr import Sdf, UsdUtils

    output.mkdir(parents=True)
    assets = output / "assets"
    assets.mkdir()
    initial, _ = resolve_asset(ORIGIN + PREFIX, ROOT_ASSET)
    pending = deque([initial])
    visited: set[str] = set()
    inventory = []
    total = 0
    while pending:
        url = pending.popleft()
        if url in visited:
            continue
        if len(visited) >= 512:
            raise ValueError("Reference asset exceeds the 512-file limit.")
        visited.add(url)
        url, relative = resolve_asset(url, "")
        request = urllib.request.Request(
            url, headers={"User-Agent": "PhysicalAIReferenceAssetBuilder/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.geturl().split("?", 1)[0] != url:
                raise ValueError("Asset download unexpectedly redirected.")
            data = response.read(128 * 1024 * 1024 + 1)
        total += len(data)
        if len(data) > 128 * 1024 * 1024 or total > 1024**3:
            raise ValueError("Reference bundle exceeds its download budget.")
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        source_hash = hashlib.sha256(data).hexdigest()
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise ValueError(f"Could not open reference USD layer: {relative}")

            def rewrite(reference: str, current_url=url, current_relative=relative) -> str:
                if not reference or reference in BUILTIN:
                    return reference
                dependency, local_name = resolve_asset(current_url, reference)
                pending.append(dependency)
                return posixpath.relpath(local_name, posixpath.dirname(current_relative))

            UsdUtils.ModifyAssetPaths(layer, rewrite)
            converted = path.with_suffix(path.suffix + ".converted.usda")
            if not layer.Export(str(converted)):
                raise ValueError(f"Could not export self-contained USD: {relative}")
            converted.replace(path)
        inventory.append(
            {
                "path": relative,
                "source_url": url,
                "source_sha256": source_hash,
                "packaged_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"mirrored {len(inventory)}: {relative}", flush=True)
    manifest = {
        "schema": "physicalai.reference-assets/v1",
        "source": ORIGIN + PREFIX + ROOT_ASSET,
        "isaac_version": "5.1.0",
        "root_asset": ROOT_ASSET,
        "files": inventory,
        "downloaded_bytes": total,
        "usage": "Private Azure demonstration under the approved NVIDIA asset terms.",
    }
    (assets / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = output / "franka.tar"
    with tarfile.open(archive_path, "w") as archive:
        for path in sorted(assets.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(assets).as_posix(), recursive=False)
    with archive_path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "archive_sha256": checksum,
        "root_asset": ROOT_ASSET,
        "file_count": len(inventory),
        "source_manifest": manifest,
    }
    (output / "bundle.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--approved-reference-assets", action="store_true")
    args = parser.parse_args()
    result = prepare(args.output, approved=args.approved_reference_assets)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "source_manifest"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
