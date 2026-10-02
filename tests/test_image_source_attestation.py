"""CPU operator attestation tests; no registry, VM or image process is used."""

import hashlib
import json

import pytest

from simulation.image_source_attestation import verify_recorded_source


def proof(content: bytes):
    return {
        "source_revision": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "servo_sha256": "c" * 64,
        "profile_sha256": "d" * 64,
        "builder_sha256": "e" * 64,
        "files": {"simulation/paused_control.py": hashlib.sha256(content).hexdigest()},
    }


def test_canonical_archive_proof_accepts_exact_image_bytes_and_is_preserved_before_return(tmp_path):
    expected = proof(b"line one\nline two\n")
    path = tmp_path / "readback.json"
    verify_recorded_source(expected, dict(expected), receipt_path=path)
    actual = json.loads(path.read_text())
    assert actual["observed"] == expected
    assert actual["accepted"] is True


def test_crlf_checkout_hash_cannot_replace_the_expected_lf_archive_inventory(tmp_path):
    archive = proof(b"line one\nline two\n")
    checkout = proof(b"line one\r\nline two\r\n")
    path = tmp_path / "rejected-readback.json"
    with pytest.raises(ValueError, match="source"):
        verify_recorded_source(archive, checkout, receipt_path=path)
    actual = json.loads(path.read_text())
    assert actual["observed"] == checkout
    assert actual["expected"] == archive
    assert actual["accepted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "f" * 40),
        ("profile_sha256", "f" * 64),
        ("image_digest", "sha256:" + "f" * 64),
        ("files", {}),
    ],
)
def test_mismatched_image_or_missing_inventory_is_recorded_without_grant_authority(
    tmp_path, field, value
):
    expected = proof(b"actual\n")
    observed = expected | {field: value}
    path = tmp_path / "failure.json"
    grant_minted = False
    with pytest.raises(ValueError):
        verify_recorded_source(expected, observed, receipt_path=path)
        grant_minted = True
    assert not grant_minted
    assert json.loads(path.read_text())["observed"] == observed


def test_previous_failure_readback_cannot_be_overwritten_by_a_new_attempt(tmp_path):
    expected = proof(b"actual\n")
    path = tmp_path / "readback.json"
    verify_recorded_source(expected, expected, receipt_path=path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        verify_recorded_source(expected, expected, receipt_path=path)
    assert path.read_bytes() == before
