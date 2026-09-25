from __future__ import annotations

from pathlib import Path

from learning.common import file_digest, read_json, require, safe_path, sha256
from learning.contract import DemonstrationSource, Scope
from learning.paused.contract import (
    CONVERSION_SCHEMA,
    EXECUTION_TIMING,
    MODEL_SCHEMA,
    RAW_SCHEMA,
    PausedControlProfile,
)
from learning.smolvla import artifacts as shared

MODE_FIELDS = frozenset(
    {
        "execution_timing",
        "real_time_admission",
        "timestamp_basis",
        "criteria_sha256",
        "frozen_plan_sha256",
    }
)


def model_contract(
    *,
    checkpoint: Path,
    scope: Scope,
    profile: PausedControlProfile,
    task: DemonstrationSource,
    backbone_manifest_sha256: str,
    role: str,
    training: dict | None,
    criteria_sha256: str,
    frozen_plan_sha256: str,
) -> dict:
    scope.validate()
    profile.validate()
    task.validate()
    value = shared.model_contract(
        checkpoint=checkpoint,
        scope=scope,
        profile=profile,
        task=task,
        backbone_manifest_sha256=backbone_manifest_sha256,
        role=role,
        training=training,
    )
    value.update(
        schema=MODEL_SCHEMA,
        execution_timing=EXECUTION_TIMING,
        real_time_admission=False,
        timestamp_basis="simulation_time",
        criteria_sha256=sha256(criteria_sha256),
        frozen_plan_sha256=sha256(frozen_plan_sha256),
    )
    return value


def validate_model(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool = True,
) -> dict:
    expected_scope.validate()
    path = safe_path(root, "model.json")
    require(file_digest(path) == sha256(expected_model_sha256), "Model manifest checksum mismatch")
    value = read_json(path)
    require(
        value.get("schema") == MODEL_SCHEMA
        and value.get("execution_timing") == EXECUTION_TIMING
        and value.get("real_time_admission") is False
        and value.get("timestamp_basis") == "simulation_time",
        "Paused checkpoints require explicit v2 simulation-time provenance",
    )
    sha256(value.get("criteria_sha256"), "frozen criteria")
    sha256(value.get("frozen_plan_sha256"), "new frozen conditions plan")
    if value.get("role") == "candidate":
        training = value.get("training")
        require(
            isinstance(training, dict)
            and training.get("raw_schema") == RAW_SCHEMA
            and training.get("conversion_schema") == CONVERSION_SCHEMA
            and all(training.get(name) == value[name] for name in MODE_FIELDS),
            "Trained paused model is not bound to its actual v3 data and frozen plan",
        )
    return shared._validate_model(
        root,
        expected_scope=expected_scope,
        expected_model_sha256=expected_model_sha256,
        for_inference=for_inference,
        model_schema=MODEL_SCHEMA,
        profile_type=PausedControlProfile,
        extra_fields=MODE_FIELDS,
    )
