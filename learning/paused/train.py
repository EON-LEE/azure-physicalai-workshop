from __future__ import annotations

from functools import partial
from pathlib import Path

from learning.common import require
from learning.contract import Scope
from learning.paused.artifacts import model_contract, validate_model
from learning.paused.contract import EXECUTION_TIMING, PausedControlProfile
from learning.paused.dataset import validate_conversion
from learning.smolvla.azure import job_deadline, validate_config
from learning.smolvla.train import TrainOptions, _run_training


def run_training(
    dataset: Path,
    parent_root: Path,
    backbone_root: Path,
    output: Path,
    *,
    scope: Scope,
    parent_model_sha256: str,
    conversion_sha256: str,
    code_snapshot_sha256: str,
    config: dict,
    client,
    options: TrainOptions,
) -> dict:
    job_deadline(config).check()
    validate_config(config)
    require(
        config.get("execution_timing") == EXECUTION_TIMING
        and config.get("real_time_admission") is False,
        "Explicit paused AML training authority is required",
    )
    mode = {
        "execution_timing": EXECUTION_TIMING,
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "criteria_sha256": config["criteria_sha256"],
        "frozen_plan_sha256": config["frozen_plan_sha256"],
    }
    return _run_training(
        dataset,
        parent_root,
        backbone_root,
        output,
        scope=scope,
        parent_model_sha256=parent_model_sha256,
        conversion_sha256=conversion_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
        config=config,
        client=client,
        options=options,
        model_validator=validate_model,
        conversion_validator=validate_conversion,
        profile_type=PausedControlProfile,
        model_builder=partial(
            model_contract,
            criteria_sha256=config["criteria_sha256"],
            frozen_plan_sha256=config["frozen_plan_sha256"],
        ),
        mode_metadata=mode,
    )
