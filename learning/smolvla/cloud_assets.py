"""Bounded private-Blob asset verification/preparation using an explicit managed identity."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    inventory,
    keys,
    parse_json,
    read_json,
    relative_path,
    require,
    safe_path,
    sha256,
)
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.smolvla import BACKBONE_ID, BACKBONE_REVISION, MODEL_ID, MODEL_REVISION
from learning.smolvla.artifacts import validate_backbone
from learning.smolvla.prepare import VENDOR_GIT_BLOBS, VENDOR_SHA256, import_assets

SPEC_SCHEMA = "physicalai.smolvla-private-preparation/v1"


def validate_prefixes(source: str, destination: str, scope: Scope) -> None:
    scope.validate()
    approved = f"tenants/{scope.tenant_id}/owners/{scope.owner_id}/learning/"
    for prefix in (source, destination):
        relative_path(prefix)
        require(prefix.startswith(approved), "Asset path escapes the approved tenant/owner scope")
    require(
        source != destination
        and not destination.startswith(source + "/")
        and not source.startswith(destination + "/"),
        "Prepared assets must use a distinct immutable output prefix",
    )


def validate_vendor_inventory(value: dict, *, prefix: str, scope: Scope) -> dict:
    scope.validate()
    relative_path(prefix)
    require(
        prefix.startswith(f"tenants/{scope.tenant_id}/owners/{scope.owner_id}/learning/"),
        "Vendor asset prefix escapes owner scope",
    )
    keys(
        value,
        {
            "source",
            "scope",
            "customer_data_sent_to_vendor",
            "weights_loaded",
            "optimizer_steps",
            "files",
            "passed",
            "manifest_prefix",
        },
        "actual vendor inventory",
    )
    require(
        value["source"] == "actual_azure_vendor_asset_staging"
        and value["scope"] == "immutable_public_apache_assets_only"
        and value["customer_data_sent_to_vendor"] is False
        and value["weights_loaded"] is False
        and type(value["optimizer_steps"]) is int
        and value["optimizer_steps"] == 0
        and value["passed"] is True
        and value["manifest_prefix"] == prefix,
        "Vendor staging is incomplete or differently scoped",
    )
    expected = {
        (kind, name)
        for kind in ("model", "backbone")
        for name in set(VENDOR_SHA256[kind]) | set(VENDOR_GIT_BLOBS[kind])
    }
    require(
        isinstance(value["files"], list) and len(value["files"]) == len(expected),
        "Vendor inventory is incomplete or has extra files",
    )
    versions = {"model": (MODEL_ID, MODEL_REVISION), "backbone": (BACKBONE_ID, BACKBONE_REVISION)}
    result = {}
    for item in value["files"]:
        keys(item, {"kind", "name", "sha256", "bytes", "blob", "repo", "revision"}, "vendor file")
        pair = (item["kind"], item["name"])
        require(pair in expected and pair not in result, "Unexpected/duplicate vendor file")
        require((item["repo"], item["revision"]) == versions[item["kind"]], "Wrong vendor revision")
        require(
            item["blob"] == f"{prefix}/{item['kind']}/{item['name']}", "Unsafe vendor Blob path"
        )
        sha256(item["sha256"])
        integer(item["bytes"], "bounded vendor file bytes", 1, 3 * 1024**3)
        if item["name"] in VENDOR_SHA256[item["kind"]]:
            require(
                item["sha256"] == VENDOR_SHA256[item["kind"]][item["name"]],
                "Vendor file checksum differs from the pinned model/license",
            )
        result[pair] = item
    require(set(result) == expected, "Incomplete verified vendor inventory")
    return result


def prepare_private_assets(spec: dict) -> dict:
    keys(
        spec,
        {
            "schema",
            "scope",
            "control_profile",
            "task",
            "storage_account_url",
            "container",
            "vendor_prefix",
            "output_prefix",
            "managed_identity_client_id",
            "image",
            "code_snapshot_sha256",
            "timeout_seconds",
        },
        "private preparation specification",
    )
    require(spec["schema"] == SPEC_SCHEMA, "Unknown private asset preparation version")
    scope = Scope(**keys(spec["scope"], {"tenant_id", "owner_id"}, "preparation scope"))
    profile = ControlProfile(
        **keys(
            spec["control_profile"],
            set(ControlProfile.__dataclass_fields__),
            "reviewed servo profile",
        )
    )
    task = DemonstrationSource(
        kind="reference_controller",
        **keys(
            spec["task"],
            {"task_id", "instruction", "goal_id"},
            "approved task",
        ),
    )
    validate_prefixes(spec["vendor_prefix"], spec["output_prefix"], scope)
    profile.validate()
    task.validate()
    from learning.azure import _uuid

    _uuid(spec["managed_identity_client_id"], "explicit asset-job identity")
    require(
        re.fullmatch(
            r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", spec["storage_account_url"]
        ),
        "Only the explicitly approved Azure Blob endpoint is supported",
    )
    require(spec["container"] == "artifacts", "Only the approved artifact container is supported")
    require(
        re.fullmatch(r"[a-z0-9]+\.azurecr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}", spec["image"]),
        "Preparation image must be digest-pinned",
    )
    sha256(spec["code_snapshot_sha256"])
    code = read_json(Path("/work/code-manifest.json"))
    require(
        code.get("sha256") == spec["code_snapshot_sha256"]
        and digest(canonical(code.get("files"))) == spec["code_snapshot_sha256"],
        "Runtime image code differs from the reviewed preparation snapshot",
    )
    for name, checksum in code["files"].items():
        require(
            file_digest(safe_path(Path("/work"), name)) == checksum,
            "Runtime preparation source changed after image build",
        )
    timeout = integer(spec["timeout_seconds"], "preparation deadline", 1, 1800)
    deadline = time.monotonic() + timeout

    def within_budget() -> None:
        require(time.monotonic() < deadline, "Private asset preparation deadline exceeded")

    from azure.core import MatchConditions
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    with (
        ManagedIdentityCredential(client_id=spec["managed_identity_client_id"]) as credential,
        BlobServiceClient(
            spec["storage_account_url"],
            credential=credential,
            connection_timeout=10,
            read_timeout=60,
            retry_total=1,
        ) as client,
        tempfile.TemporaryDirectory(prefix="physicalai-approved-assets-") as temporary,
    ):
        container = client.get_container_client(spec["container"])
        require(
            next(iter(container.list_blobs(name_starts_with=spec["output_prefix"] + "/")), None)
            is None,
            "Preparation output already exists, including incomplete prior attempts",
        )
        source = container.get_blob_client(spec["vendor_prefix"] + "/vendor-inventory.json")
        metadata = source.get_blob_properties()
        require(metadata.size <= 256 * 1024, "Oversized vendor inventory")
        encoded = source.download_blob(
            etag=metadata.etag,
            match_condition=MatchConditions.IfNotModified,
        ).readall()
        vendor = validate_vendor_inventory(
            parse_json(encoded), prefix=spec["vendor_prefix"], scope=scope
        )
        root = Path(temporary)
        download_root = root / "vendor"
        download_root.mkdir()
        for (kind, name), record in vendor.items():
            within_budget()
            blob = container.get_blob_client(record["blob"])
            properties = blob.get_blob_properties()
            require(
                properties.size == record["bytes"]
                and properties.metadata.get("sha256") == record["sha256"]
                and properties.metadata.get("vendor_revision") == record["revision"],
                "Staged Blob properties differ from the actual vendor inventory",
            )
            path = safe_path(download_root, f"{kind}/{name}", must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            checksum, count = hashlib.sha256(), 0
            with path.open("xb") as output:
                for chunk in blob.download_blob(
                    etag=properties.etag,
                    match_condition=MatchConditions.IfNotModified,
                    max_concurrency=2,
                ).chunks():
                    within_budget()
                    count += len(chunk)
                    require(count <= record["bytes"], "Blob exceeded the reviewed file size")
                    checksum.update(chunk)
                    output.write(chunk)
            require(
                count == record["bytes"] and checksum.hexdigest() == record["sha256"],
                "Downloaded private asset checksum mismatch",
            )
            print(
                canonical({"verified_private_asset": f"{kind}/{name}", "bytes": count}).decode(),
                flush=True,
            )
        prepared = root / "prepared"
        result = import_assets(
            download_root / "model",
            download_root / "backbone",
            prepared,
            scope=scope,
            profile=profile,
            task=task,
        )
        validate_backbone(
            prepared / "backbone",
            scope=scope,
            expected_sha256=result["backbone_manifest_sha256"],
        )
        files = inventory(prepared)
        ordered = sorted(
            files, key=lambda name: (name.endswith(("model.json", "backbone.json")), name)
        )
        for name in ordered:
            within_budget()
            with (prepared / name).open("rb") as stream:
                container.upload_blob(
                    spec["output_prefix"] + "/" + name,
                    stream,
                    overwrite=False,
                    metadata={"sha256": files[name]},
                    content_settings=ContentSettings(content_type="application/octet-stream"),
                )
        proof = {
            "schema": "physicalai.smolvla-private-preparation-proof/v1",
            "source": "actual_azure_managed_identity_private_preparation",
            "scope": asdict(scope),
            "control_profile": asdict(profile),
            "task": spec["task"],
            "vendor_prefix": spec["vendor_prefix"],
            "output_prefix": spec["output_prefix"],
            "vendor_inventory_sha256": digest(encoded),
            "image": spec["image"],
            "code_snapshot_sha256": spec["code_snapshot_sha256"],
            "files": files,
            **result,
            "weights_loaded": False,
            "optimizer_steps": 0,
            "learning_quality_verified": False,
        }
        body = canonical(proof) + b"\n"
        container.upload_blob(
            spec["output_prefix"] + "/prepared-inventory.json",
            body,
            overwrite=False,
            content_settings=ContentSettings(content_type="application/json"),
        )
        receipt = {
            "prepared_inventory_sha256": digest(body),
            "output_prefix": spec["output_prefix"],
            "file_count": len(files),
            **result,
        }
        print("PHYSICALAI_SMOL_PRIVATE_PREPARATION " + canonical(receipt).decode(), flush=True)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.spec is None:
        encoded = os.environ["PHYSICALAI_PREPARATION_SPEC_BASE64"]
        require(len(encoded) <= 32768, "Oversized preparation specification")
        spec = parse_json(base64.b64decode(encoded, validate=True))
    else:
        spec = read_json(args.spec)
    prepare_private_assets(spec)


if __name__ == "__main__":
    main()
