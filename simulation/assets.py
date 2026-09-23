"""Download a licensed, checksummed asset bundle from this deployment's Azure Blob."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from azure.storage.blob import BlobServiceClient


def extract_assets(archive: Path, destination: Path) -> None:
    total = 0
    seen = set()
    with tarfile.open(archive, "r:*") as bundle:
        for item in bundle:
            name = PurePosixPath(item.name)
            if name.is_absolute() or ".." in name.parts or "\\" in item.name:
                raise ValueError(
                    "Asset archives must contain relative POSIX paths without traversal."
                )
            if not (item.isfile() or item.isdir()):
                raise ValueError("Asset archives cannot contain links, devices, or special files.")
            target = destination.joinpath(*name.parts)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not name.parts or target in seen:
                raise ValueError("Asset archive contains an empty or duplicate file path.")
            seen.add(target)
            total += item.size
            if total > 10 * 1024**3:
                raise ValueError("Expanded asset bundle exceeds 10 GiB.")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(item)
            if source is None:
                raise ValueError("Asset archive file cannot be read.")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def prepare_assets(credential) -> Path:
    baked = os.environ.get("USE_BAKED_REFERENCE_ASSETS") == "true"
    bundle = None
    if baked:
        import json

        bundle = json.loads(
            Path("/opt/physicalai/reference/bundle.json").read_text(encoding="utf-8")
        )
        os.environ["FRANKA_ASSET_SHA256"] = bundle["archive_sha256"]
        os.environ["FRANKA_USD_RELATIVE_PATH"] = bundle["root_asset"]
    checksum = os.environ["FRANKA_ASSET_SHA256"]
    if re.fullmatch(r"[a-f0-9]{64}", checksum) is None:
        raise ValueError("FRANKA_ASSET_SHA256 must pin the licensed archive.")
    relative = PurePosixPath(os.environ["FRANKA_USD_RELATIVE_PATH"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("The robot USD must be a relative path inside the asset bundle.")
    cache = Path("/data/assets")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = cache / checksum
    marker = destination / ".complete"
    if not marker.is_file():
        if destination.exists():
            raise RuntimeError("An incomplete asset cache exists; inspect it before retrying.")
        with TemporaryDirectory(prefix="download-", dir=cache) as temporary:
            staging = Path(temporary)
            archive = staging / "assets.tar"
            if baked:
                shutil.copyfile("/opt/physicalai/reference/franka.tar", archive)
            else:
                with BlobServiceClient(
                    account_url=os.environ["STORAGE_ACCOUNT_URL"], credential=credential
                ) as storage:
                    blob = storage.get_blob_client(
                        container=os.environ.get("STORAGE_CONTAINER", "artifacts"),
                        blob=os.environ["FRANKA_ASSET_BLOB"],
                    )
                    if blob.get_blob_properties().size > 4 * 1024**3:
                        raise ValueError("The licensed asset archive exceeds 4 GiB.")
                    with archive.open("wb") as stream:
                        blob.download_blob().readinto(stream)
            with archive.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != checksum:
                raise ValueError("The Azure asset archive does not match the approved checksum.")
            extracted = staging / "extracted"
            extracted.mkdir()
            extract_assets(archive, extracted)
            if not extracted.joinpath(*relative.parts).is_file():
                raise ValueError("The declared robot USD is missing from the archive.")
            (extracted / ".complete").write_text(checksum, encoding="ascii")
            extracted.rename(destination)
    if marker.read_text(encoding="ascii") != checksum:
        raise ValueError("The asset cache marker does not match the approved bundle.")
    result = destination.joinpath(*relative.parts)
    if not result.is_file():
        raise ValueError("The robot USD is missing from the verified asset cache.")
    return result


def configure_asset_environment(credential) -> Path:
    asset = prepare_assets(credential)
    root = Path("/data/assets") / os.environ["FRANKA_ASSET_SHA256"]
    if not asset.is_relative_to(root):
        raise ValueError("The verified robot asset is outside the pinned cache root.")
    os.environ["FRANKA_USD_PATH"] = str(asset)
    os.environ["FRANKA_ASSET_ROOT"] = str(root)
    return asset
