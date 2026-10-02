"""Bounded, manifest-last native training checkpoints. No Torch or Azure calls on import."""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    keys,
    parse_json,
    read_json,
    relative_path,
    require,
    safe_path,
    sha256,
    utc,
)
from learning.contract import Scope

SCHEMA = "physicalai.smolvla-training-checkpoint/v1"
COMMAND_SCHEMA = "physicalai.smolvla-training-checkpoint/v2"
MANIFEST = "checkpoint.json"
CONTINUATION_SCHEMA = "physicalai.smolvla-continuation/v1"
BINDING_KEYS = frozenset(
    {
        "scope",
        "raw_manifest_sha256",
        "conversion_sha256",
        "parent_model_sha256",
        "parent_weights_sha256",
        "control_profile_sha256",
        "task_sha256",
        "criteria_sha256",
        "frozen_plan_sha256",
        "training_config_sha256",
        "training_code_sha256",
        "runtime_sha256",
    }
)
ORIGIN_KEYS = frozenset(
    {
        "azure_job_id",
        "azure_pipeline_job_id",
        "specification_sha256",
        "code_snapshot_sha256",
        "job_deadline_utc",
        "test_only",
    }
)
MODEL_FILES = frozenset(
    {
        "pretrained_model/model.safetensors",
        "pretrained_model/config.json",
        "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_postprocessor.json",
        "training_state/training_step.json",
    }
)
FULL_STATE_FILES = frozenset(
    {
        "training_state/optimizer_state.safetensors",
        "training_state/optimizer_param_groups.json",
        "training_state/scheduler_state.json",
        "training_state/rng_state.safetensors",
        "training_state/continuation.safetensors",
        "training_state/continuation.json",
    }
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


@dataclass(frozen=True)
class CheckpointLimits:
    max_files: int = 64
    max_file_bytes: int = 4 * 1024**3
    max_total_bytes: int = 8 * 1024**3
    max_json_bytes: int = 1024 * 1024
    max_header_bytes: int = 4 * 1024 * 1024
    max_checkpoints: int = 32
    publication_timeout_seconds: int = 180

    def validate(self) -> None:
        integer(self.max_files, "checkpoint file budget", 6, 64)
        integer(self.max_file_bytes, "checkpoint file bytes", 1, 4 * 1024**3)
        integer(self.max_total_bytes, "checkpoint total bytes", 1, 8 * 1024**3)
        integer(self.max_json_bytes, "checkpoint JSON bytes", 256, 1024 * 1024)
        integer(self.max_header_bytes, "safetensors header bytes", 256, 4 * 1024 * 1024)
        integer(self.max_checkpoints, "checkpoint count budget", 1, 32)
        integer(self.publication_timeout_seconds, "checkpoint publication seconds", 1, 600)


DEFAULT_LIMITS = CheckpointLimits()


def validate_binding(value: dict) -> None:
    keys(value, set(BINDING_KEYS), "checkpoint binding")
    Scope(**keys(value["scope"], {"tenant_id", "owner_id"}, "checkpoint scope")).validate()
    for name in BINDING_KEYS - {"scope"}:
        if name in ("criteria_sha256", "frozen_plan_sha256") and value[name] is None:
            continue
        sha256(value[name], name)
    require(
        (value["criteria_sha256"] is None) == (value["frozen_plan_sha256"] is None),
        "Partial paused checkpoint binding",
    )


def _origin(value: dict) -> None:
    command = value.get("azure_job_type") == "command"
    expected = (
        (set(ORIGIN_KEYS) - {"azure_pipeline_job_id"}) | {"azure_job_type"}
        if command
        else set(ORIGIN_KEYS)
    )
    keys(value, expected, "checkpoint origin")
    for name in ("azure_job_id",) if command else ("azure_job_id", "azure_pipeline_job_id"):
        require(
            isinstance(value[name], str)
            and re.fullmatch(
                r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
                r"Microsoft.MachineLearningServices/workspaces/[A-Za-z0-9_.-]+/jobs/[A-Za-z0-9_.-]+",
                value[name],
            ),
            "Missing actual scoped checkpoint job identity",
        )
    if not command:
        require(
            value["azure_job_id"].rsplit("/jobs/", 1)[0]
            == value["azure_pipeline_job_id"].rsplit("/jobs/", 1)[0],
            "Checkpoint job identities cross workspaces",
        )
    sha256(value["specification_sha256"])
    sha256(value["code_snapshot_sha256"])
    utc(value["job_deadline_utc"])
    require(type(value["test_only"]) is bool, "Explicit checkpoint test provenance is required")


def _allowed_file(name: str) -> None:
    relative_path(name)
    require(
        name in MODEL_FILES | FULL_STATE_FILES
        or re.fullmatch(
            r"pretrained_model/policy_(pre|post)processor_step_\d+_[a-z_]+\.safetensors", name
        ),
        f"Unapproved checkpoint file (pickle/executable/link forbidden): {name}",
    )


def _safe_tensor(path: Path, limits: CheckpointLimits) -> None:
    size = path.stat().st_size
    require(size >= 8, "Incomplete safetensors header")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        require(
            2 <= length <= limits.max_header_bytes and 8 + length <= size,
            "Invalid or oversized safetensors header",
        )
        header = parse_json(stream.read(length))
    spans = []
    for name, item in header.items():
        if name == "__metadata__":
            require(
                isinstance(item, dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in item.items()),
                "Invalid tensor metadata",
            )
            continue
        keys(item, {"dtype", "shape", "data_offsets"}, "tensor descriptor")
        require(
            isinstance(item["dtype"], str) and item["dtype"] in _DTYPE_BYTES,
            "Unsupported native tensor dtype",
        )
        require(
            isinstance(item["shape"], list) and len(item["shape"]) <= 16, "Invalid tensor shape"
        )
        shape = [integer(v, "tensor dimension", 0, 2**32) for v in item["shape"]]
        offsets = item["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2, "Invalid tensor offsets")
        low, high = [integer(v, "tensor offset", 0, limits.max_file_bytes) for v in offsets]
        require(
            high - low == math.prod(shape) * _DTYPE_BYTES[item["dtype"]],
            "Tensor shape/byte range mismatch",
        )
        spans.append((low, high))
    end = 0
    for low, high in sorted(spans):
        require(low == end and high >= low, "Overlapping or missing tensor bytes")
        end = high
    require(bool(spans) and 8 + length + end == size, "Incomplete or extra safetensors bytes")


def _files(root: Path, limits: CheckpointLimits) -> dict:
    require(root.is_dir() and not root.is_symlink(), "Checkpoint root must be a real directory")
    files, total = {}, 0
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Checkpoint symlinks are not accepted")
        if path.is_dir():
            require(
                path.parent == root and path.name in ("pretrained_model", "training_state"),
                "Unexpected checkpoint subdirectory",
            )
            continue
        require(path.is_file(), "Checkpoint entries must be regular files or approved directories")
        name = path.relative_to(root).as_posix()
        if name == MANIFEST:
            continue
        _allowed_file(name)
        safe_path(root, name)
        size = integer(path.stat().st_size, "checkpoint file bytes", 1, limits.max_file_bytes)
        total += size
        require(
            len(files) < limits.max_files and total <= limits.max_total_bytes,
            "Checkpoint exceeds the reviewed file/byte budget",
        )
        if name.endswith(".safetensors"):
            _safe_tensor(path, limits)
        else:
            require(size <= limits.max_json_bytes, "Checkpoint JSON exceeds its byte budget")
            # Optimizer parameter groups are a native JSON list; parse with the same duplicate/
            # nonfinite rejection as object manifests, never pickle or dynamic class import.
            parse_json(b'{"value":' + path.read_bytes() + b"}")
        files[name] = {"sha256": file_digest(path), "bytes": size}
    return files


def _manifest(value: dict, limits: CheckpointLimits) -> None:
    keys(
        value,
        {
            "schema",
            "step",
            "cumulative_optimizer_steps",
            "state_kind",
            "binding",
            "origin",
            "resume_from_checkpoint_sha256",
            "files",
            "bitwise_continuation_claimed",
            "learning_quality_verified",
        },
        "complete checkpoint manifest",
    )
    require(
        value["schema"]
        == (COMMAND_SCHEMA if value["origin"].get("azure_job_type") == "command" else SCHEMA)
        and value["state_kind"] in ("full_state", "weights_only"),
        "Unknown checkpoint/state schema",
    )
    integer(value["step"], "completed optimizer step", 1, 100000)
    integer(value["cumulative_optimizer_steps"], "cumulative actual optimizer steps", value["step"])
    validate_binding(value["binding"])
    _origin(value["origin"])
    if value["resume_from_checkpoint_sha256"] is not None:
        sha256(value["resume_from_checkpoint_sha256"])
    require(
        value["bitwise_continuation_claimed"] is False
        and value["learning_quality_verified"] is False,
        "A checkpoint is not a cross-device determinism or learned-quality attestation",
    )
    require(
        isinstance(value["files"], dict) and MODEL_FILES.issubset(value["files"]),
        "Missing complete model/processors/step files",
    )
    if value["state_kind"] == "full_state":
        require(
            FULL_STATE_FILES.issubset(value["files"]), "Missing full-state optimizer/RNG/data state"
        )
    total = 0
    require(len(value["files"]) <= limits.max_files, "Checkpoint file count exceeds budget")
    for name, file in value["files"].items():
        _allowed_file(name)
        keys(file, {"sha256", "bytes"}, "checkpoint file")
        sha256(file["sha256"])
        total += integer(file["bytes"], "checkpoint file bytes", 1, limits.max_file_bytes)
    require(total <= limits.max_total_bytes, "Checkpoint exceeds total byte budget")


def seal_checkpoint(
    root: Path,
    *,
    step: int,
    binding: dict,
    origin: dict,
    state_kind: str,
    prior_optimizer_steps: int = 0,
    resume_from_checkpoint_sha256: str | None = None,
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> str:
    limits.validate()
    require(not (root / MANIFEST).exists(), "Never overwrite a completed checkpoint")
    integer(prior_optimizer_steps, "prior actual optimizer steps")
    value = {
        "schema": COMMAND_SCHEMA if origin.get("azure_job_type") == "command" else SCHEMA,
        "step": step,
        "cumulative_optimizer_steps": prior_optimizer_steps + step,
        "state_kind": state_kind,
        "binding": binding,
        "origin": origin,
        "resume_from_checkpoint_sha256": resume_from_checkpoint_sha256,
        "files": _files(root, limits),
        "bitwise_continuation_claimed": False,
        "learning_quality_verified": False,
    }
    _manifest(value, limits)
    require(
        read_json(root / "training_state" / "training_step.json") == {"step": step},
        "Actual native optimizer step differs from checkpoint",
    )
    if state_kind == "full_state":
        continuation = read_json(
            root / "training_state" / "continuation.json", limits.max_json_bytes
        )
        require(
            continuation.get("schema") == CONTINUATION_SCHEMA and continuation.get("step") == step,
            "Missing completed data/RNG continuation state",
        )
    body = canonical(value) + b"\n"
    require(len(body) <= limits.max_json_bytes, "Checkpoint manifest exceeds its byte budget")
    pending = root / ".checkpoint.pending"
    with pending.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(root / MANIFEST)
    return digest(body)


def validate_checkpoint(
    root: Path,
    *,
    expected_sha256: str,
    expected_binding: dict,
    require_full_state: bool = False,
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> dict:
    limits.validate()
    validate_binding(expected_binding)
    path = safe_path(root, MANIFEST)
    require(path.stat().st_size <= limits.max_json_bytes, "Checkpoint manifest exceeds budget")
    require(file_digest(path) == sha256(expected_sha256), "Complete checkpoint checksum mismatch")
    value = read_json(path, limits.max_json_bytes)
    _manifest(value, limits)
    require(
        value["binding"] == expected_binding,
        "Checkpoint binding differs from the approved continuation",
    )
    if require_full_state:
        require(
            value["state_kind"] == "full_state",
            "Weights-only checkpoint is not full-state continuation",
        )
    require(_files(root, limits) == value["files"], "Checkpoint inventory/checksum differs")
    require(
        read_json(root / "training_state" / "training_step.json") == {"step": value["step"]},
        "Completed native step changed",
    )
    if value["state_kind"] == "full_state":
        state = read_json(root / "training_state" / "continuation.json", limits.max_json_bytes)
        require(
            state.get("schema") == CONTINUATION_SCHEMA and state.get("step") == value["step"],
            "Incomplete full-state continuation marker",
        )
    return value


def latest_complete_checkpoint(
    root: Path,
    *,
    expected_binding: dict,
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> dict:
    limits.validate()
    validate_binding(expected_binding)
    candidates = []
    require(root.is_dir() and not root.is_symlink(), "Missing real checkpoint directory")
    entries = list(root.iterdir())
    require(len(entries) <= limits.max_checkpoints + 1, "Checkpoint scan exceeds count budget")
    for path in entries:
        require(not path.is_symlink(), "Do not recover from a mutable latest symlink")
        require(
            path.is_dir() and re.fullmatch(r"step-\d{6}", path.name),
            "Unexpected checkpoint scan path",
        )
        marker = path / MANIFEST
        if not marker.exists():
            continue
        checksum = file_digest(marker)
        value = validate_checkpoint(
            path, expected_sha256=checksum, expected_binding=expected_binding, limits=limits
        )
        require(path.name == f"step-{value['step']:06d}", "Checkpoint directory/step mismatch")
        candidates.append(
            {
                "path": path,
                "sha256": checksum,
                "step": value["step"],
                "state_kind": value["state_kind"],
            }
        )
    require(bool(candidates), "No verified complete checkpoint; partial native files cannot resume")
    return max(candidates, key=lambda item: item["step"])


def _prefix(prefix: str, binding: dict) -> None:
    validate_binding(binding)
    relative_path(prefix)
    scope = Scope(**binding["scope"])
    require(
        prefix.startswith(f"tenants/{scope.tenant_id}/owners/{scope.owner_id}/learning/outputs/"),
        "Checkpoint Blob prefix escapes approved tenant/owner outputs",
    )


def _budget(check_deadline: Callable[[], float], limits: CheckpointLimits):
    end = time.monotonic() + limits.publication_timeout_seconds

    def remaining():
        seconds = min(check_deadline(), end - time.monotonic())
        require(seconds > 0, "Checkpoint publication deadline expired")
        return seconds

    return remaining


def _readback(container, name: str, expected: dict, remaining, *, destination=None) -> str:
    from azure.core import MatchConditions

    seconds = remaining()
    blob = container.get_blob_client(name)
    props = blob.get_blob_properties(
        timeout=max(1, math.ceil(seconds)),
        retry_total=0,
        connection_timeout=min(10, seconds),
        read_timeout=min(30, seconds),
    )
    require(
        isinstance(props.etag, str) and bool(props.etag), "Missing Blob checkpoint readback ETag"
    )
    require(props.size == expected["bytes"], "Blob checkpoint readback byte count differs")
    result = hashlib.sha256()
    count = 0
    seconds = remaining()
    stream = blob.download_blob(
        etag=props.etag,
        match_condition=MatchConditions.IfNotModified,
        retry_total=0,
        timeout=max(1, math.ceil(seconds)),
        connection_timeout=min(10, seconds),
        read_timeout=min(30, seconds),
        max_concurrency=1,
    )
    for chunk in stream.chunks():
        remaining()
        count += len(chunk)
        require(count <= expected["bytes"], "Blob readback exceeded the checkpoint byte budget")
        result.update(chunk)
        if destination is not None:
            destination.write(chunk)
    require(
        count == expected["bytes"] and result.hexdigest() == expected["sha256"],
        "Blob checkpoint readback checksum differs",
    )
    remaining()
    return props.etag


def publish_checkpoint(
    root: Path,
    container,
    *,
    prefix: str,
    expected_sha256: str,
    expected_binding: dict,
    check_deadline: Callable[[], float],
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> dict:
    limits.validate()
    _prefix(prefix, expected_binding)
    remaining = _budget(check_deadline, limits)
    remaining()
    manifest = validate_checkpoint(
        root, expected_sha256=expected_sha256, expected_binding=expected_binding, limits=limits
    )
    target = f"{prefix}/step-{manifest['step']:06d}"
    from azure.core.exceptions import ResourceExistsError

    files = {
        **manifest["files"],
        MANIFEST: {"sha256": expected_sha256, "bytes": (root / MANIFEST).stat().st_size},
    }
    etags = {}
    for name, file in files.items():
        seconds = remaining()
        blob_name = f"{target}/{name}"
        with safe_path(root, name).open("rb") as stream:
            try:
                container.upload_blob(
                    blob_name,
                    stream,
                    overwrite=False,
                    retry_total=0,
                    max_concurrency=1,
                    timeout=max(1, math.ceil(seconds)),
                    connection_timeout=min(10, seconds),
                    read_timeout=min(30, seconds),
                    metadata={"sha256": file["sha256"]},
                )
            except ResourceExistsError:
                # Idempotent publication only when the immutable existing bytes match exactly.
                pass
        etags[name] = _readback(container, blob_name, file, remaining)
    return {
        "checkpoint_sha256": expected_sha256,
        "manifest_blob": f"{target}/{MANIFEST}",
        "step": manifest["step"],
        "state_kind": manifest["state_kind"],
        "readback_verified": True,
        "blob_etags": etags,
        "resume_capability": manifest["state_kind"],
        "learning_quality_verified": False,
    }


def restore_checkpoint(
    container,
    *,
    checkpoint_prefix: str,
    expected_sha256: str,
    expected_scope: Scope,
    destination: Path,
    check_deadline: Callable[[], float],
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> dict:
    """Restore a named complete bundle into fresh local storage, never by following latest links."""
    limits.validate()
    expected_scope.validate()
    relative_path(checkpoint_prefix)
    scope_prefix = (
        f"tenants/{expected_scope.tenant_id}/owners/{expected_scope.owner_id}/learning/outputs/"
    )
    require(checkpoint_prefix.startswith(scope_prefix), "Resume checkpoint escapes owner outputs")
    require(
        re.fullmatch(r"step-\d{6}", checkpoint_prefix.rsplit("/", 1)[-1]),
        "Resume requires an immutable step prefix",
    )
    require(not destination.exists(), "Never overwrite a checkpoint restore directory")
    remaining = _budget(check_deadline, limits)
    seconds = remaining()
    from azure.core import MatchConditions

    blob = container.get_blob_client(f"{checkpoint_prefix}/{MANIFEST}")
    props = blob.get_blob_properties(timeout=max(1, math.ceil(seconds)), retry_total=0)
    require(0 < props.size <= limits.max_json_bytes, "Resume checkpoint manifest exceeds budget")
    stream = blob.download_blob(
        etag=props.etag,
        match_condition=MatchConditions.IfNotModified,
        retry_total=0,
        timeout=max(1, math.ceil(remaining())),
        max_concurrency=1,
        connection_timeout=min(10, seconds),
        read_timeout=min(30, seconds),
    )
    body = bytearray()
    for chunk in stream.chunks():
        remaining()
        require(len(body) + len(chunk) <= props.size, "Oversized checkpoint manifest readback")
        body.extend(chunk)
    require(
        len(body) == props.size and digest(bytes(body)) == sha256(expected_sha256),
        "Complete remote checkpoint manifest checksum mismatch",
    )
    value = parse_json(bytes(body))
    _manifest(value, limits)
    require(
        value["binding"]["scope"]
        == {"tenant_id": expected_scope.tenant_id, "owner_id": expected_scope.owner_id},
        "Remote checkpoint manifest owner binding mismatch",
    )
    require(
        checkpoint_prefix.endswith(f"/step-{value['step']:06d}"), "Remote checkpoint step differs"
    )
    destination.mkdir(parents=True)
    for name, descriptor in value["files"].items():
        path = safe_path(destination, name, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            _readback(
                container, f"{checkpoint_prefix}/{name}", descriptor, remaining, destination=target
            )
            target.flush()
            os.fsync(target.fileno())
    require(_files(destination, limits) == value["files"], "Restored checkpoint inventory differs")
    # A failed restore has no complete manifest, even when some individual files arrived.
    with (destination / MANIFEST).open("xb") as marker:
        marker.write(body)
        marker.flush()
        os.fsync(marker.fileno())
    validate_checkpoint(
        destination,
        expected_sha256=expected_sha256,
        expected_binding=value["binding"],
        limits=limits,
    )
    return value


def latest_remote_checkpoint(
    container,
    *,
    prefix: str,
    expected_binding: dict,
    check_deadline: Callable[[], float],
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> dict:
    """Scan one approved owner/job prefix, never a mutable latest pointer or auto-submit."""
    limits.validate()
    _prefix(prefix, expected_binding)
    remaining = _budget(check_deadline, limits)
    names, inspected = [], 0
    for item in container.list_blobs(
        name_starts_with=prefix + "/",
        results_per_page=64,
        retry_total=0,
        timeout=max(1, math.ceil(remaining())),
        connection_timeout=min(10, remaining()),
        read_timeout=min(30, remaining()),
    ):
        remaining()
        inspected += 1
        require(
            inspected <= (limits.max_files + 1) * (limits.max_checkpoints + 1),
            "Remote checkpoint scan exceeded the file/count budget",
        )
        require(
            item.name.startswith(prefix + "/"),
            "Checkpoint listing escaped its exact owner/job prefix",
        )
        relative = item.name[len(prefix) + 1 :]
        relative_path(relative)
        if relative.endswith("/checkpoint.json"):
            require(
                re.fullmatch(r"step-\d{6}/checkpoint\.json", relative),
                "Malformed checkpoint completion marker path",
            )
            names.append(item.name)
    require(len(names) <= limits.max_checkpoints, "Remote checkpoint marker count exceeded budget")
    require(bool(names), "No complete remote checkpoint; newer partial files are not resumable")
    selected = sorted(names)[-1]
    from azure.core import MatchConditions

    blob = container.get_blob_client(selected)
    props = blob.get_blob_properties(retry_total=0, timeout=max(1, math.ceil(remaining())))
    require(0 < props.size <= limits.max_json_bytes, "Remote checkpoint manifest exceeds budget")
    body = bytearray()
    stream = blob.download_blob(
        etag=props.etag,
        match_condition=MatchConditions.IfNotModified,
        max_concurrency=1,
        retry_total=0,
        timeout=max(1, math.ceil(remaining())),
    )
    for chunk in stream.chunks():
        remaining()
        require(len(body) + len(chunk) <= props.size, "Remote checkpoint manifest grew")
        body.extend(chunk)
    require(len(body) == props.size, "Incomplete remote checkpoint manifest")
    value = parse_json(bytes(body))
    _manifest(value, limits)
    require(value["binding"] == expected_binding, "Remote checkpoint binding differs")
    checkpoint_prefix = selected.rsplit("/", 1)[0]
    require(
        checkpoint_prefix.endswith(f"/step-{value['step']:06d}"), "Remote manifest step mismatch"
    )
    etags = {}
    for name, descriptor in value["files"].items():
        etags[name] = _readback(container, f"{checkpoint_prefix}/{name}", descriptor, remaining)
    return {
        "checkpoint_prefix": checkpoint_prefix,
        "checkpoint_sha256": digest(bytes(body)),
        "step": value["step"],
        "resume_capability": value["state_kind"],
        "manifest_etag": props.etag,
        "blob_etags": etags,
        "readback_verified": True,
        "jobs_submitted": 0,
    }
