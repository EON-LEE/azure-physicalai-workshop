from pathlib import Path

import pytest

from learning.checks.fixtures import SCOPE
from learning.common import ContractError, write_json


def test_spot_recovery_requires_reviewed_context_before_reading_job_or_weights(tmp_path):
    from learning.smolvla.train import recover_checkpoint

    write_json(tmp_path / "training-context.json", {"untrusted": True})
    with pytest.raises(ContractError, match="checksum"):
        recover_checkpoint(
            tmp_path,
            Path("/approved/data"),
            Path("/approved/parent"),
            tmp_path / "recovered",
            step=10,
            scope=SCOPE,
            expected_context_sha256="0" * 64,
            client=None,
        )


def test_smol_resume_is_not_a_vendor_initialization_alias():
    from learning.smolvla.train import validate_resume

    with pytest.raises(ContractError, match="trained"):
        validate_resume({"role": "pretrained", "training": None}, mode="weights_only")
    validate_resume({"role": "pretrained", "training": None}, mode="new")
