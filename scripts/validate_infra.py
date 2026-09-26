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
        for name in (
            "foundation",
            "bootstrap",
            "bootstrap-aci",
            "runtime",
            "web",
            "private-data",
            "spot-live-network",
            "reference-presentation",
            "learning",
            "learning-worker",
            "learning-reconciler",
            "gpu-capacity-probe",
            "simulation-batch-foundation",
            "simulation-batch-access",
            "simulation-batch",
            "simulation-batch-node",
            "simulation-batch-warmup",
        ):
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
    registry = resources(foundation, "Microsoft.ContainerRegistry/registries")[0]
    assert registry["sku"]["name"] == "[parameters('registrySku')]"
    assert foundation["parameters"]["registrySku"]["defaultValue"] == "Basic"
    assert set(foundation["parameters"]["registrySku"]["allowedValues"]) == {
        "Basic",
        "Standard",
        "Premium",
    }
    storage = resources(foundation, "Microsoft.Storage/storageAccounts")[0]
    assert storage["properties"]["allowSharedKeyAccess"] is False
    assert storage["properties"]["allowBlobPublicAccess"] is False
    assert storage["properties"]["publicNetworkAccess"] == "Disabled"
    cosmos = resources(foundation, "Microsoft.DocumentDB/databaseAccounts")[0]
    assert cosmos["properties"]["disableLocalAuth"] is True
    assert cosmos["properties"]["publicNetworkAccess"] == "Disabled"
    state = resources(foundation, "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers")
    assert len(state) == 1 and state[0]["properties"]["resource"]["defaultTtl"] == -1
    assert state[0]["properties"]["resource"]["partitionKey"]["paths"] == ["/owner_key"]
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
    assert (
        app["properties"]["template"]["scale"]["minReplicas"].replace(" ", "")
        == "[if(parameters('publicDemoPublishLive'),1,0)]"
    )
    assert templates["web"]["parameters"]["publicDemoPublishLive"]["defaultValue"] is False
    assert "defaultValue" not in runtime["parameters"]["gpuVmSize"]
    assert "defaultValue" not in runtime["parameters"]["acceptNvidiaEula"]
    presentation = resources(templates["reference-presentation"], "Microsoft.App/jobs")[0]
    assert presentation["properties"]["configuration"]["replicaRetryLimit"] == 0
    assert presentation["properties"]["configuration"]["manualTriggerConfig"]["parallelism"] == 1
    assert templates["reference-presentation"]["parameters"]["cycles"]["maxValue"] == 1000
    workspace = resources(templates["learning"], "Microsoft.MachineLearningServices/workspaces")[0]
    assert workspace["properties"]["publicNetworkAccess"] == "Disabled"
    assert workspace["properties"]["systemDatastoresAuthMode"] == "Identity"
    assert workspace["properties"]["managedNetwork"]["isolationMode"] == "AllowOnlyApprovedOutbound"
    compute = resources(
        templates["learning"], "Microsoft.MachineLearningServices/workspaces/computes"
    )[0]["properties"]
    assert compute["disableLocalAuth"] is True
    assert compute["properties"]["enableNodePublicIp"] is False
    assert compute["properties"]["vmPriority"] == "[parameters('computeTier')]"
    assert compute["properties"]["scaleSettings"]["minNodeCount"] == 0
    assert compute["properties"]["scaleSettings"]["maxNodeCount"] == 1
    worker = resources(templates["learning-worker"], "Microsoft.App/containerApps")[0]
    worker_ingress = worker["properties"]["configuration"]["ingress"]
    assert worker_ingress["external"] is False
    assert worker_ingress["allowInsecure"] is False
    assert worker["properties"]["template"]["scale"]["maxReplicas"] == 1
    assert templates["learning-worker"]["parameters"]["allowedPolicyTypes"]["defaultValue"] == []
    reconciler = resources(templates["learning-reconciler"], "Microsoft.App/jobs")[0]
    assert reconciler["condition"] == "[parameters('enabled')]"
    assert templates["learning-reconciler"]["parameters"]["enabled"]["defaultValue"] is False
    assert (
        templates["learning-reconciler"]["parameters"]["reconciliationTargets"]["defaultValue"]
        == []
    )
    scheduled = reconciler["properties"]["configuration"]
    assert scheduled["triggerType"] == "Schedule"
    assert scheduled["replicaRetryLimit"] == 0
    assert scheduled["replicaTimeout"] == 120
    assert scheduled["scheduleTriggerConfig"]["parallelism"] == 1
    managed_simulation = templates["simulation-batch-foundation"]
    batch = resources(managed_simulation, "Microsoft.Batch/batchAccounts")[0]
    assert managed_simulation["parameters"]["enabled"]["defaultValue"] is False
    assert batch["properties"]["publicNetworkAccess"] == "Disabled"
    assert batch["properties"]["allowedAuthenticationModes"] == ["AAD"]
    assert not resources(managed_simulation, "Microsoft.Compute/virtualMachines")
    assert not resources(managed_simulation, "Microsoft.Batch/batchAccounts/pools")
    managed_pool = templates["simulation-batch"]
    assert managed_pool["parameters"]["provisionPool"]["defaultValue"] is False
    assert "defaultValue" not in managed_pool["parameters"]["allocationStartUtc"]
    for script in (
        ROOT / "infra" / "start-simulator.sh",
        ROOT / "scripts" / "start-live-simulator.sh",
        ROOT / "scripts" / "build-simulator-on-gpu.sh",
        ROOT / "scripts" / "install-rtx-grid.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(
        f"{len(templates)} Bicep templates and baseline invariants passed. "
        "Azure/GPU execution NOT verified."
    )


if __name__ == "__main__":
    validate()
