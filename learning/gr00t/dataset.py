from __future__ import annotations

from pathlib import Path

from learning.common import require
from learning.contract import (
    CAMERAS,
    TEACHING_CONTRACT_VERSION,
    Scope,
    ValidatedDataset,
    validate_dataset,
)


def modality_spec() -> dict:
    return {
        **{
            group: {
                name: {
                    "start": start,
                    "end": end,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": original,
                }
                for name, start, end in (("arm", 0, 7), ("fingers", 7, 9))
            }
            for group, original in (("state", "observation.state"), ("action", "action"))
        },
        "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def prepare_export(
    source: Path,
    *,
    expected_scope: Scope,
    allow_test_fixture: bool = False,
    expected_manifest_sha256: str | None = None,
) -> ValidatedDataset:
    dataset = validate_dataset(
        source,
        expected_scope=expected_scope,
        require_live=not allow_test_fixture,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    require(
        dataset.manifest["schema"] == TEACHING_CONTRACT_VERSION,
        "GR00T requires actual v2 10Hz hold evidence; never relabel/decimate v1 data",
    )
    return dataset
