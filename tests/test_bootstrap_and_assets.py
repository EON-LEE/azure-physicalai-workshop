import io
import tarfile

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

from scripts.bootstrap import certificates, main
from simulation.asset_references import local_reference
from simulation.assets import extract_assets


def archive(tmp_path, name="Franka/franka.usd", kind=tarfile.REGTYPE, payload=b"#usda 1.0"):
    path = tmp_path / "test.tar"
    with tarfile.open(path, "w") as bundle:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.size = len(payload) if kind == tarfile.REGTYPE else 0
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            item.linkname = "/outside/secret"
        bundle.addfile(item, io.BytesIO(payload) if item.isfile() else None)
    return path


def test_tls_leaf_and_root_are_separate_and_only_the_leaf_key_is_exported():
    ca_pem, certificate_pem, key_pem = certificates("sim.physicalai.internal", "10.42.4.4")
    ca = x509.load_pem_x509_certificate(ca_pem)
    certificate = x509.load_pem_x509_certificate(certificate_pem)
    key = serialization.load_pem_private_key(key_pem, password=None)
    ca.public_key().verify(
        certificate.signature,
        certificate.tbs_certificate_bytes,
        padding.PKCS1v15(),
        certificate.signature_hash_algorithm,
    )
    assert ca.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value.ca
    assert not certificate.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value.ca
    assert (
        ExtendedKeyUsageOID.SERVER_AUTH
        in certificate.extensions.get_extension_for_oid(ExtensionOID.EXTENDED_KEY_USAGE).value
    )
    assert key.public_key().public_numbers() == certificate.public_key().public_numbers()
    assert key.public_key().public_numbers() != ca.public_key().public_numbers()
    assert "PRIVATE" not in ca_pem.decode()


def test_bootstrap_refuses_writes_without_an_explicit_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_BOOTSTRAP_WRITE", raising=False)
    with pytest.raises(RuntimeError, match="Explicit"):
        main()


def test_normal_asset_bundle_extracts_into_the_selected_directory(tmp_path):
    destination = tmp_path / "assets"
    extract_assets(archive(tmp_path), destination)
    assert (destination / "Franka" / "franka.usd").read_bytes() == b"#usda 1.0"


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("windows\\path", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        ("device", tarfile.CHRTYPE),
    ],
)
def test_invalid_asset_members_are_rejected(tmp_path, name, kind):
    with pytest.raises(ValueError):
        extract_assets(archive(tmp_path, name, kind), tmp_path / "assets")


def test_duplicate_asset_files_are_rejected(tmp_path):
    path = tmp_path / "duplicate.tar"
    with tarfile.open(path, "w") as bundle:
        for _ in range(2):
            member = tarfile.TarInfo("same.usd")
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="duplicate"):
        extract_assets(path, tmp_path / "assets")


@pytest.mark.parametrize(
    "reference",
    [
        "https://external.example/asset.usd",
        "omniverse://server/asset.usd",
        "/etc/passwd",
        "../../outside.usd",
        "C:\\outside.usd",
    ],
)
def test_external_usd_dependencies_are_rejected(tmp_path, reference):
    with pytest.raises(ValueError):
        local_reference(tmp_path, tmp_path / "robot.usd", reference)


def test_local_usd_dependencies_and_bundled_materials_are_supported(tmp_path):
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"fixture")
    assert local_reference(tmp_path, tmp_path / "robot.usd", "texture.png") == texture
    assert local_reference(tmp_path, tmp_path / "robot.usd", "OmniPBR.mdl") is None
