"""Run as an approved Azure Container Apps Job, not as part of API startup."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from datetime import UTC, datetime, timedelta

from azure.ai.projects import AIProjectClient
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient, ContentSettings
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from pydantic import AwareDatetime, BaseModel, ConfigDict

from agents.inspection import agent_definition


class BootstrapManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_endpoint: str
    agent_name: str
    agent_version: str
    sim_ca_sha256: str
    created_at: AwareDatetime
    probe_response_id: str | None = None
    probe_scope: str | None = None


def certificates(hostname: str, private_ip: str) -> tuple[bytes, bytes, bytes]:
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Physical AI deployment CA")])
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(issuer)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(hostname),
                    x509.IPAddress(ipaddress.ip_address(private_ip)),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM)
    private_key = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return ca_certificate.public_bytes(serialization.Encoding.PEM), pem, private_key


def main() -> None:
    if os.environ.get("ALLOW_BOOTSTRAP_WRITE") != "true":
        raise RuntimeError("Explicit ALLOW_BOOTSTRAP_WRITE=true is required for this Azure job.")
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    name = os.environ["FOUNDRY_AGENT_NAME"]
    credential = ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
    with (
        credential,
        AIProjectClient(endpoint=endpoint, credential=credential) as project,
        SecretClient(vault_url=os.environ["KEY_VAULT_URL"], credential=credential) as vault,
        BlobServiceClient(
            account_url=os.environ["STORAGE_ACCOUNT_URL"], credential=credential
        ) as storage,
    ):
        agent = project.agents.create_version(
            agent_name=name, definition=agent_definition(os.environ["FOUNDRY_MODEL_DEPLOYMENT"])
        )
        with project.get_openai_client().with_options(timeout=45, max_retries=0) as inference:
            probe = inference.responses.create(
                input="Deployment connectivity probe only. Do not inspect a part or move a robot.",
                tool_choice="none",
                extra_body={
                    "agent_reference": {
                        "type": "agent_reference",
                        "name": agent.name,
                        "version": agent.version,
                    }
                },
            )
        if not probe.id:
            raise RuntimeError("Foundry returned no response ID for the deployment probe.")
        ca, certificate, private_key = certificates(
            os.environ["SIM_HOSTNAME"], os.environ["SIM_PRIVATE_IP"]
        )
        vault.set_secret("simulator-tls-certificate", certificate.decode("ascii"))
        vault.set_secret("simulator-tls-key", private_key.decode("ascii"))
        container = storage.get_container_client(os.environ.get("STORAGE_CONTAINER", "artifacts"))
        container.upload_blob(
            "bootstrap/simulator-ca.pem",
            ca,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/x-pem-file"),
        )
        manifest = BootstrapManifest(
            project_endpoint=endpoint,
            agent_name=agent.name,
            agent_version=agent.version,
            sim_ca_sha256=hashlib.sha256(ca).hexdigest(),
            created_at=datetime.now(UTC),
            probe_response_id=probe.id,
            probe_scope="agent_connectivity_only_not_physical_inspection",
        )
        container.upload_blob(
            "bootstrap/runtime.json",
            manifest.model_dump_json().encode(),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        print(
            json.dumps(
                {
                    "agent_name": agent.name,
                    "agent_version": agent.version,
                    "probe_response_id": probe.id,
                    "probe_scope": "agent_connectivity_only_not_physical_inspection",
                    "bootstrap": "written",
                }
            )
        )


if __name__ == "__main__":
    main()
