"""Bounded direct Blob MI transfer of the existing approved manifest formats."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from learning.azure import datastore_prefix
from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    inventory,
    parse_json,
    read_json,
    relative_path,
    require,
    safe_path,
    sha256,
)
from learning.contract import Scope
from learning.smolvla.checkpoints import _readback

MAX_FILES = 25000
MAX_FILE_BYTES = 4 * 1024**3
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_MANIFEST_BYTES = 4 * 1024**2


def input_prefix(config: dict, name: str) -> str:
    from learning.smolvla.azure import validate_config

    validate_config(config)
    asset = config["inputs"][name]
    prefix = datastore_prefix(config)
    require(asset["uri"].startswith(prefix), "Input escapes the approved datastore")
    result = asset["uri"][len(prefix) :]
    relative_path(result)
    require(
        result.startswith(f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/"),
        "Input escapes its approved owner namespace",
    )
    return result


class PrivateBlobTransfer:
    def __init__(self, container, config: dict, *, remaining: Callable[[], float]) -> None:
        from learning.smolvla.azure import validate_config

        validate_config(config)
        self.container, self.config, self.remaining = container, config, remaining
        self.scope = Scope(config["tenant_id"], config["owner_id"])
        self.scope.validate()
        self.input_prefixes = {input_prefix(config, name) for name in config["inputs"]}
        self.downloaded_bytes, self.published_bytes = 0, 0

    def _check(self) -> float:
        value = self.remaining()
        require(value > 0, "Original direct-transfer deadline expired")
        return value

    def _prefix(self, prefix: str) -> None:
        relative_path(prefix)
        require(
            prefix.startswith(f"tenants/{self.scope.tenant_id}/owners/{self.scope.owner_id}/"),
            "Blob prefix escapes the approved tenant/owner namespace",
        )

    def manifest(self, prefix: str, name: str, expected_sha: str) -> tuple[dict, bytes]:
        _, raw = self.manifest_bytes(prefix, name, expected_sha, maximum=MAX_MANIFEST_BYTES)
        return parse_json(raw), raw

    def manifest_bytes(
        self, prefix: str, name: str, expected_sha: str, *, maximum: int
    ) -> tuple[dict, bytes]:
        from azure.core import MatchConditions

        self._prefix(prefix)
        require(prefix in self.input_prefixes, "Blob input is not an exact approved asset prefix")
        relative_path(name)
        seconds = self._check()
        blob = self.container.get_blob_client(prefix + "/" + name)
        props = blob.get_blob_properties(
            retry_total=0, connection_timeout=min(10, seconds), read_timeout=min(30, seconds)
        )
        require(isinstance(props.etag, str) and bool(props.etag), "Missing manifest ETag")
        integer(maximum, "bounded document size", 1, 32 * 1024**2)
        integer(props.size, "manifest byte count", 1, maximum)
        body = bytearray()
        for part in blob.download_blob(
            etag=props.etag,
            match_condition=MatchConditions.IfNotModified,
            max_concurrency=1,
            retry_total=0,
            connection_timeout=min(10, seconds),
            read_timeout=min(30, seconds),
            timeout=max(1, math.ceil(seconds)),
        ).chunks():
            self._check()
            body.extend(part)
            require(len(body) <= props.size, "Manifest grew beyond the approved byte count")
        raw = bytes(body)
        require(
            len(raw) == props.size and digest(raw) == sha256(expected_sha),
            "Manifest checksum mismatch",
        )
        return {"etag": props.etag, "bytes": len(raw)}, raw

    def _listed(self, prefix: str, allowed: set[str]) -> dict:
        self._prefix(prefix)
        require(prefix in self.input_prefixes, "Blob input is not an exact approved asset prefix")
        require(0 < len(allowed) <= MAX_FILES, "Artifact file inventory exceeds budget")
        for name in allowed:
            relative_path(name)
        listing, total = {}, 0
        for item in self.container.list_blobs(
            name_starts_with=prefix + "/",
            results_per_page=100,
            retry_total=0,
            timeout=max(1, math.ceil(self._check())),
            connection_timeout=min(10, self._check()),
            read_timeout=min(30, self._check()),
        ):
            self._check()
            require(item.name.startswith(prefix + "/"), "Blob listing escaped the fixed prefix")
            name = item.name[len(prefix) + 1 :]
            require(
                name in allowed and name not in listing,
                "Unsigned or duplicate artifact inventory path",
            )
            total += integer(item.size, "artifact file size", 1, MAX_FILE_BYTES)
            require(
                total <= MAX_TOTAL_BYTES and len(listing) < MAX_FILES,
                "Artifact inventory exceeds byte/count bounds",
            )
            listing[name] = item
        require(set(listing) == allowed, "Missing files in the signed artifact inventory")
        self.downloaded_bytes += total
        require(
            self.downloaded_bytes <= MAX_TOTAL_BYTES,
            "Complete command input transfer exceeds byte budget",
        )
        return listing

    def download_files(
        self,
        prefix: str,
        files: dict[str, str],
        destination: Path,
        *,
        metadata: dict[str, bytes] | None = None,
    ) -> dict:
        self._check()
        metadata = metadata or {}
        require(not (set(files) & set(metadata)), "Manifest cannot alias a payload file")
        expected = {**files, **{name: digest(data) for name, data in metadata.items()}}
        for checksum in expected.values():
            sha256(checksum)
        listed = self._listed(prefix, set(expected))
        require(
            not destination.exists() and not destination.is_symlink(),
            "Download requires a fresh task-local directory",
        )
        destination.mkdir(parents=True)
        require(
            shutil.disk_usage(destination).free > sum(item.size for item in listed.values()),
            "Insufficient bounded task-local disk for approved inputs",
        )

        # Completion manifests are always written after payloads.
        def download(name):
            path = safe_path(destination, name, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = {"sha256": files[name], "bytes": listed[name].size}
            with path.open("xb") as stream:
                etag = _readback(
                    self.container, prefix + "/" + name, descriptor, self._check, destination=stream
                )
            return name, {"sha256": files[name], "bytes": listed[name].size, "etag": etag}

        with ThreadPoolExecutor(max_workers=4) as executor:
            receipts = dict(executor.map(download, files))
        for name, data in reversed(tuple(metadata.items())):
            descriptor = {"sha256": expected[name], "bytes": len(data)}
            _readback(self.container, prefix + "/" + name, descriptor, self._check)
            path = safe_path(destination, name, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        return {
            "files": receipts,
            "manifest_sha256s": {name: expected[name] for name in metadata},
            "readback_verified": True,
        }

    def publish_files(
        self, root: Path, prefix: str, *, marker: str, files: dict | None = None
    ) -> dict:
        from azure.core.exceptions import ResourceExistsError

        self._prefix(prefix)
        require(
            prefix.startswith(self.config["output_prefix"] + "/" + self.config["run_id"] + "/"),
            "Output must remain under this exact approved command job",
        )
        self._check()
        checksums = inventory(root) if files is None else files
        require(
            marker in checksums and 1 <= len(checksums) <= MAX_FILES,
            "Missing output completion marker",
        )
        sizes = {
            name: integer(safe_path(root, name).stat().st_size, "output size", 1, MAX_FILE_BYTES)
            for name in checksums
        }
        require(sum(sizes.values()) <= MAX_TOTAL_BYTES, "Output exceeds fixed total byte budget")
        self.published_bytes += sum(sizes.values())
        require(
            self.published_bytes <= MAX_TOTAL_BYTES,
            "Complete command output transfer exceeds byte budget",
        )
        order = sorted(set(checksums) - {marker}) + [marker]
        etags = {}
        for name in order:
            seconds = self._check()
            path = safe_path(root, name)
            with path.open("rb") as stream:
                try:
                    self.container.upload_blob(
                        prefix + "/" + name,
                        stream,
                        overwrite=False,
                        max_concurrency=1,
                        retry_total=0,
                        metadata={"sha256": checksums[name]},
                        connection_timeout=min(10, seconds),
                        read_timeout=min(30, seconds),
                        timeout=max(1, math.ceil(seconds)),
                    )
                except ResourceExistsError:
                    pass  # Existing bytes must still pass the exact readback below.
            etags[name] = _readback(
                self.container,
                prefix + "/" + name,
                {"bytes": sizes[name], "sha256": checksums[name]},
                self._check,
            )
        return {
            "prefix": prefix,
            "marker": marker,
            "marker_sha256": checksums[marker],
            "file_count": len(checksums),
            "bytes": sum(sizes.values()),
            "readback_verified": True,
            "etag": etags[marker],
        }

    def publish_document(self, prefix: str, name: str, value: dict) -> None:
        import io

        self._prefix(prefix)
        require(
            prefix.startswith(self.config["output_prefix"] + "/" + self.config["run_id"] + "/"),
            "Diagnostic output escaped command job",
        )
        relative_path(name)
        body = canonical(value) + b"\n"
        require(len(body) <= MAX_MANIFEST_BYTES, "Diagnostic JSON exceeds fixed budget")
        seconds = self._check()
        self.container.upload_blob(
            prefix + "/" + name,
            io.BytesIO(body),
            overwrite=False,
            retry_total=0,
            connection_timeout=min(10, seconds),
            read_timeout=min(30, seconds),
            timeout=max(1, math.ceil(seconds)),
        )
        _readback(
            self.container,
            prefix + "/" + name,
            {"bytes": len(body), "sha256": digest(body)},
            self._check,
        )

    def publish_candidate(self, root: Path, prefix: str) -> dict:
        import re

        result = read_json(safe_path(root, "result.json"))
        name = result.get("candidate")
        require(
            isinstance(name, str) and re.fullmatch(r"candidates/step-\d{6}", name),
            "Unapproved sealed candidate path",
        )
        candidate = root.joinpath(*relative_path(name).parts)
        require(
            file_digest(safe_path(candidate, "model.json")) == result["model_manifest_sha256"],
            "Final candidate manifest checksum mismatch",
        )
        files = {
            name + "/" + relative: checksum for relative, checksum in inventory(candidate).items()
        }
        for relative in ("training-context.json", "training.log", "result.json"):
            path = safe_path(root, relative)
            require(
                path.stat().st_size <= 16 * 1024**2, "Training diagnostic file exceeds bounded size"
            )
            files[relative] = file_digest(path)
        return self.publish_files(root, prefix, marker="result.json", files=files)
