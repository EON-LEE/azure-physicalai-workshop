"""Bind verified private vendor bytes to a new paused training-only initialization."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from learning.common import canonical, read_json, require
from learning.contract import DemonstrationSource, Scope
from learning.paused.artifacts import model_contract, validate_model
from learning.paused.contract import EXECUTION_TIMING, PausedControlProfile
from learning.smolvla.prepare import _import_assets


def import_assets(
    model_source: Path,
    backbone_source: Path,
    output: Path,
    *,
    scope: Scope,
    profile: PausedControlProfile,
    task: DemonstrationSource,
    criteria_sha256: str,
    frozen_plan_sha256: str,
) -> dict:
    return _import_assets(
        model_source,
        backbone_source,
        output,
        scope=scope,
        profile=profile,
        task=task,
        model_builder=partial(
            model_contract,
            criteria_sha256=criteria_sha256,
            frozen_plan_sha256=frozen_plan_sha256,
        ),
        model_validator=validate_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--backbone-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    args = parser.parse_args()
    binding = read_json(args.binding)
    require(
        binding.get("execution_timing") == EXECUTION_TIMING
        and binding.get("real_time_admission") is False,
        "Explicit new paused vendor preparation binding is required",
    )
    result = import_assets(
        args.model_source,
        args.backbone_source,
        args.output,
        scope=Scope(**binding["scope"]),
        profile=PausedControlProfile(**binding["control_profile"]),
        task=DemonstrationSource(kind="reference_controller", **binding["task"]),
        criteria_sha256=binding["criteria_sha256"],
        frozen_plan_sha256=binding["frozen_plan_sha256"],
    )
    print(canonical(result).decode())
