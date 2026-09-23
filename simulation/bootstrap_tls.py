"""Read the simulator TLS material from Azure Key Vault using the VM identity."""

import os
from pathlib import Path

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

from simulation.assets import configure_asset_environment


def main() -> None:
    directory = Path("/run/physicalai/tls")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (
        ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"]) as credential,
        SecretClient(vault_url=os.environ["KEY_VAULT_URL"], credential=credential) as vault,
    ):
        for name, filename in (
            ("simulator-tls-certificate", "server.crt"),
            ("simulator-tls-key", "server.key"),
        ):
            value = vault.get_secret(name).value
            if not value:
                raise RuntimeError(f"Required Key Vault secret is empty: {name}")
            path = directory / filename
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(value)
        configure_asset_environment(credential)
    os.environ["SIM_TLS_CERT_FILE"] = str(directory / "server.crt")
    os.environ["SIM_TLS_KEY_FILE"] = str(directory / "server.key")
    from simulation.run_isaac import main as run

    run()


if __name__ == "__main__":
    main()
