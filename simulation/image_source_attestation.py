"""Operator-only source attestation; it never changes or executes simulator image code."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4


def verify_recorded_source(expected: dict, observed: dict, *, receipt_path: Path) -> None:
    accepted = bool(expected.get("files")) and expected == observed
    record = {
        "schema": "physicalai.recorded-image-source/v1",
        "expected_basis": "canonical_git_archive_bytes",
        "expected": expected,
        "observed": observed,
        "accepted": accepted,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(f".{uuid4()}.new")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    if not accepted:
        raise ValueError(
            "Actual image source readback differs from its canonical archive authority."
        )
