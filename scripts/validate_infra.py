"""Compile and inspect IaC offline; this is not Azure deployment validation."""

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resources(template: dict, resource_type: str) -> list[dict]:
    values = template["resources"]
    if isinstance(values, dict):
        values = values.values()
    return [item for item in values if item["type"].lower() == resource_type.lower()]


def validate() -> None:
    templates = {}
    with tempfile.TemporaryDirectory(prefix="physicalai-bicep-") as temporary:
        for name in ("foundation", "bootstrap", "bootstrap-aci", "runtime", "web", "private-data"):
            output = Path(temporary) / f"{name}.json"
            subprocess.run(
                [
                    "az",
                    "bicep",
                    "build",
                    "--file",
                    str(ROOT / "infra" / f"{name}.bicep"),
                    "--outfile",
                    str(output),
                ],
                check=True,
            )
            templates[name] = json.loads(output.read_text(encoding="utf-8"))
    foundation, runtime = templates["foundation"], templates["runtime"]
    storage = resources(foundation, "Microsoft.Storage/storageAccounts")[0]
    assert storage["properties"]["allowSharedKeyAccess"] is False
    assert storage["properties"]["allowBlobPublicAccess"] is False
    assert storage["properties"]["publicNetworkAccess"] == "Disabled"
    cosmos = resources(foundation, "Microsoft.DocumentDB/databaseAccounts")[0]
    assert cosmos["properties"]["disableLocalAuth"] is True
    assert cosmos["properties"]["publicNetworkAccess"] == "Disabled"
    vault = resources(foundation, "Microsoft.KeyVault/vaults")[0]
    assert vault["properties"]["publicNetworkAccess"] == "Disabled"
    foundry = resources(foundation, "Microsoft.CognitiveServices/accounts")[0]
    assert foundry["properties"]["disableLocalAuth"] is True
    assert resources(foundation, "Microsoft.CognitiveServices/accounts/projects")
    assert resources(templates["bootstrap"], "Microsoft.App/jobs")
    nic = resources(runtime, "Microsoft.Network/networkInterfaces")[0]
    for configuration in nic["properties"]["ipConfigurations"]:
        assert "publicIPAddress" not in configuration["properties"]
    app = resources(templates["web"], "Microsoft.App/containerApps")[0]
    assert app["properties"]["configuration"]["ingress"]["allowInsecure"] is False
    assert app["identity"]["type"] == "UserAssigned"
    assert app["properties"]["template"]["scale"]["minReplicas"] == 0
    assert "defaultValue" not in runtime["parameters"]["gpuVmSize"]
    assert "defaultValue" not in runtime["parameters"]["acceptNvidiaEula"]
    subprocess.run(["bash", "-n", str(ROOT / "infra" / "start-simulator.sh")], check=True)
    print("Six Bicep templates and baseline invariants passed. Azure/GPU execution NOT verified.")


if __name__ == "__main__":
    validate()
