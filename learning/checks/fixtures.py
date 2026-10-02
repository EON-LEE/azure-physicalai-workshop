from __future__ import annotations

import json
import struct
import zlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from learning.common import canonical, file_digest, read_json
from learning.contract import CameraSample, FrameSample, Provenance, Scope

SCOPE = Scope("11111111-1111-4111-8111-111111111111", "a" * 64)
JOINTS = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.02, 0.02)
PROVENANCE = Provenance(
    source_kind="test_fixture",
    simulator_version="5.1.0",
    simulator_image_digest="sha256:" + "b" * 64,
    robot_asset_sha256="c" * 64,
    scene_builder_id="inspection-cell-custom-v1",
    scene_builder_sha256="d" * 64,
    code_revision="e" * 40,
    capture_host="test_cpu",
    gpu_model="none",
)
LIVE_PROVENANCE = replace(
    PROVENANCE,
    source_kind="isaac_sim",
    capture_host="azure_gpu",
    gpu_model="test-attestation-not-live",
)


def png(width: int = 32, height: int = 32, color: int = 80) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + bytes([color] * width * 3)) * height))
        + chunk(b"IEND", b"")
    )


def frame(index: int, *, terminal: bool = False) -> FrameSample:
    timestamp = 1_000_000_000 + index * 100_000_000
    images = {
        name: CameraSample(png(color=80 + index), index, index * 6, timestamp)
        for name in ("inspection", "overview")
    }
    captured = datetime(2026, 9, 20, tzinfo=UTC) + timedelta(milliseconds=100 * index)
    return FrameSample(
        captured_at_utc=captured.isoformat().replace("+00:00", "Z"),
        monotonic_ns=timestamp,
        physics_step=index * 6,
        joint_positions=JOINTS,
        commanded_joint_targets=JOINTS,
        images=images,
        terminated=terminal,
    )


def overwrite(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def edit_frames(root: Path, mutate) -> None:
    manifest = read_json(root / "manifest.json")
    path = root / manifest["episodes"][0]["path"]
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(frames)
    path.write_bytes(b"".join(canonical(value) + b"\n" for value in frames))
    manifest["episodes"][0]["sha256"] = file_digest(path)
    overwrite(root / "manifest.json", manifest)
