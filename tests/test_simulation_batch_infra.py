"""Offline compilation of the pool-only template; never deploys an Azure resource."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_managed_simulator_pool_is_zero_default_single_low_priority_and_private(tmp_path):
    source = Path("infra/simulation-batch.bicep")
    assert source.is_file(), "A reviewed Batch pool-only infrastructure module is required."
    compiler = shutil.which("bicep") or str(Path.home() / ".azure/bin/bicep")
    if not Path(compiler).is_file():
        pytest.skip("Bicep compiler not installed; compile this module in infrastructure CI.")
    output = tmp_path / "pool.json"
    result = subprocess.run(
        [compiler, "build", str(source), "--outfile", str(output)],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    template = json.loads(output.read_text())
    assert template["parameters"]["provisionPool"]["defaultValue"] is False
    assert template["parameters"]["allocationMinutes"]["maxValue"] == 60
    assert len(template["resources"]) == 1
    pool = template["resources"][0]
    assert pool["type"].lower() == "microsoft.batch/batchaccounts/pools"
    assert pool["condition"] == "[parameters('provisionPool')]"
    properties = pool["properties"]
    assert properties["vmSize"] == "Standard_NV36ads_A10_v5"
    assert properties["taskSlotsPerNode"] == 1
    assert properties["interNodeCommunication"] == "Disabled"
    assert (
        properties["networkConfiguration"]["publicIPAddressConfiguration"]["provision"]
        == "NoPublicIPAddresses"
    )
    assert "userAccounts" not in properties
    vm = properties["deploymentConfiguration"]["virtualMachineConfiguration"]
    assert vm["nodeAgentSkuId"] == "batch.node.ubuntu 24.04"
    assert vm["imageReference"] == {
        "publisher": "microsoft-dsvm",
        "offer": "ubuntu-hpc",
        "sku": "2404",
        "version": "24.04.2026092501",
    }
    extension = vm["extensions"][0]
    assert extension["publisher"] == "Microsoft.HpcCompute"
    assert extension["type"] == "NvidiaGpuDriverLinux"
    assert extension["typeHandlerVersion"] == "1.14.0.6"
    assert extension["autoUpgradeMinorVersion"] is False
    assert extension["enableAutomaticUpgrade"] is False
    assert extension["settings"] == {
        "driverVersion": "570.237",
        "installCUDA": False,
        "updateOS": False,
    }
    start = properties["startTask"]
    assert start["waitForSuccess"] is True and start["maxTaskRetryCount"] == 0
    assert "simulation.batch_task preflight" in start["commandLine"]
    assert "--kill-after=5s 60s" in start["commandLine"]
    assert start["containerSettings"]["containerRunOptions"] == "[variables('containerOptions')]"
    from simulation.batch import PREFLIGHT_COMMAND, PREFLIGHT_CONTAINER_OPTIONS, allocation_formula

    assert template["variables"]["containerOptions"] == PREFLIGHT_CONTAINER_OPTIONS
    assert start["commandLine"] == PREFLIGHT_COMMAND
    formula = template["variables"]["autoscale"]
    assert "$TargetDedicatedNodes = 0" in formula
    assert "min(1," in formula and "$PendingTasks.GetSample" in formula
    assert "$NodeDeallocationOption = terminate" in formula
    assert allocation_formula("{0}") in formula.replace("\r\n", "\n")
    assert "dateTimeAdd" in template["variables"]["allocationDeadlineUtc"]
    assert "roleAssignments" not in json.dumps(template)
    assert "ssh" not in json.dumps(template).lower()
