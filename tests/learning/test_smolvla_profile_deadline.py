from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.checks import smolvla_vendor_profile as profile
from learning.common import ContractError, canonical, digest, file_digest
from tests.learning.test_smolvla_profile_entry import template


@pytest.fixture
def specification():
    return {
        **template(),
        "schema": "physicalai.smolvla-vendor-profile/v2",
        "start_deadline_utc": "2026-09-23T12:01:00Z",
        "entry_code_sha256": file_digest(Path("learning/checks/smolvla_profile_entry.py")),
        "deadline_code_sha256": file_digest(Path("learning/deadlines.py")),
    }


@pytest.fixture
def clock(monkeypatch):
    from learning import deadlines

    value = SimpleNamespace(utc=datetime(2026, 9, 23, 12, tzinfo=UTC))
    monkeypatch.setattr(deadlines, "utcnow", lambda: value.utc)
    return value


def test_start_authority_is_new_versioned_and_hash_bound(specification, clock):
    profile.validate_specification(specification)
    receipt = profile.admit_start(specification)
    assert receipt["start_deadline_utc"] == specification["start_deadline_utc"]
    assert receipt["start_deadline_sha256"] == digest(
        canonical({"start_deadline_utc": specification["start_deadline_utc"]})
    )
    assert receipt["checked_at_utc"] == "2026-09-23T12:00:00Z"
    assert receipt["admitted"] is True


def test_old_profile_configuration_has_no_new_start_authority():
    profile.validate_specification(template())
    with pytest.raises(ContractError, match="v2|deadline"):
        profile.run_profile(template())


@pytest.mark.parametrize("seconds", [60, 61])
def test_expired_start_rejects_before_identity_assets_or_cuda(
    specification, clock, monkeypatch, seconds
):
    clock.utc += timedelta(seconds=seconds)
    monkeypatch.setattr(
        profile, "_helper", lambda spec: pytest.fail("Expired admission cannot start asset work")
    )
    with pytest.raises(ContractError, match="deadline"):
        profile.run_profile(specification)


def test_new_phase_cannot_start_loading_weights_after_admission_deadline(
    specification, clock, tmp_path
):
    assert profile.admit_start(specification)["admitted"] is True
    clock.utc += timedelta(seconds=61)
    with pytest.raises(ContractError, match="deadline"):
        profile._build_policy(tmp_path, specification, {})


@pytest.mark.parametrize(
    "field", ["entry_code_sha256", "deadline_code_sha256", "profile_code_sha256"]
)
def test_actual_start_guard_and_entry_bytes_must_match_reviewed_spec(specification, clock, field):
    specification[field] = "0" * 64
    with pytest.raises(ContractError, match="checksum|code"):
        profile.admit_start(specification)


@pytest.mark.parametrize(
    "value", [None, "2026-09-23T12:01:00", "2026-09-23T12:01:00+00:00", "invalid"]
)
def test_start_deadline_cannot_be_missing_or_relative(specification, value):
    if value is None:
        specification.pop("start_deadline_utc")
    else:
        specification["start_deadline_utc"] = value
    with pytest.raises(ContractError):
        profile.validate_specification(specification)
