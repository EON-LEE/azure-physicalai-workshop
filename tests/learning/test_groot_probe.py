from pathlib import Path

import pytest

from learning.common import ContractError


def test_probe_requires_real_export_and_never_generates_observations(tmp_path):
    from learning.gr00t.probe import validate_probe_inputs

    with pytest.raises((ContractError, FileNotFoundError)):
        validate_probe_inputs(
            tmp_path / "absent",
            tmp_path / "model",
            tenant_id="11111111-1111-4111-8111-111111111111",
            owner_id="a" * 64,
            export_sha256="b" * 64,
            model_sha256="c" * 64,
        )


def test_gpu_image_is_exactly_pinned_and_does_not_bundle_vendor_weights():
    path = Path(__file__).resolve().parents[2] / "learning" / "gr00t" / "Dockerfile"
    text = path.read_text()
    assert "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:" in text
    assert "uv sync --frozen --extra runtime --extra azure" in text
    assert "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96" in text
    assert "huggingface-cli download" not in text and "ACCEPT_EULA" not in text


def test_vendor_import_never_downloads_or_accepts_a_license(tmp_path):
    from learning.gr00t.prepare import import_pretrained

    with pytest.raises(ContractError, match="license"):
        import_pretrained(
            tmp_path / "input",
            tmp_path / "output",
            scope=None,
            profile=None,
            task=None,
            acknowledge_license_review=False,
        )
    assert not (tmp_path / "output").exists()
