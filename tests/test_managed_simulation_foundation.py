import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.validate_infra import resources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infra" / "simulation-batch-foundation.bicep"


def compile_template(source, output):
    assert source.is_file(), "Managed simulation needs explicit private infrastructure."
    compiler = shutil.which("bicep")
    if compiler is None:
        installed = Path.home() / ".azure" / "bin" / "bicep"
        if installed.is_file():
            compiler = str(installed)
    if compiler is None:
        pytest.skip("Offline Bicep compiler not installed.")
    subprocess.run([compiler, "build", str(source), "--outfile", str(output)], check=True)
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    return compile_template(
        SOURCE, tmp_path_factory.mktemp("managed-simulation") / "foundation.json"
    )


def test_batch_foundation_is_disabled_by_default_without_a_vm_or_pool(compiled):
    assert compiled["parameters"]["enabled"]["defaultValue"] is False
    assert not resources(compiled, "Microsoft.Compute/virtualMachines")
    assert not resources(compiled, "Microsoft.Batch/batchAccounts/pools")
    assert not resources(compiled, "Microsoft.Authorization/roleAssignments")
    items = compiled["resources"]
    if isinstance(items, dict):
        items = items.values()
    assert all(item["condition"] == "[parameters('enabled')]" for item in items)


def test_batch_foundation_requires_identity_and_private_connectivity(compiled):
    account = resources(compiled, "Microsoft.Batch/batchAccounts")[0]["properties"]
    assert account["allowedAuthenticationModes"] == ["AAD"]
    assert account["publicNetworkAccess"] == "Disabled"
    assert account["poolAllocationMode"] == "BatchService"
    assert "autoStorage" not in account
    endpoints = resources(compiled, "Microsoft.Network/privateEndpoints")
    assert len(endpoints) == 2
    assert {
        item["properties"]["privateLinkServiceConnections"][0]["properties"]["groupIds"][0]
        for item in endpoints
    } == {"batchAccount", "nodeManagement"}
    assert all(
        item["properties"]["subnet"]["id"] == "[parameters('privateEndpointSubnetId')]"
        for item in endpoints
    )


def test_batch_nodes_use_explicit_egress_and_no_public_inbound(compiled):
    subnet = resources(compiled, "Microsoft.Network/virtualNetworks/subnets")[0]["properties"]
    assert subnet["defaultOutboundAccess"] is False
    assert subnet["natGateway"]["id"] == "[parameters('natGatewayId')]"
    assert subnet["addressPrefix"] == "[parameters('nodeSubnetPrefix')]"
    assert "defaultValue" not in compiled["parameters"]["nodeSubnetPrefix"]
    security = resources(compiled, "Microsoft.Network/networkSecurityGroups")[0]["properties"]
    rules = security["securityRules"]
    assert len(rules) == 1
    rule = rules[0]["properties"]
    assert (rule["direction"], rule["access"], rule["destinationPortRange"]) == (
        "Inbound",
        "Deny",
        "*",
    )
    assert not resources(compiled, "Microsoft.Network/publicIPAddresses")


def test_job_submission_access_cannot_resize_pools_or_grant_broader_access(tmp_path):
    template = compile_template(
        ROOT / "infra" / "simulation-batch-access.bicep", tmp_path / "access.json"
    )
    assert template["parameters"]["enabled"]["defaultValue"] is False
    grants = resources(template, "Microsoft.Authorization/roleAssignments")
    assert len(grants) == 1
    grant = grants[0]
    assert grant["condition"] == "[parameters('enabled')]"
    assert "Microsoft.Batch/batchAccounts" in grant["scope"]
    assert "batchAccountName" in grant["scope"]
    assert "48e5e92e-a480-4e71-aa9c-2778f4c13781" in grant["properties"]["roleDefinitionId"]
    assert grant["properties"]["principalType"] == "ServicePrincipal"
    assert grant["properties"]["principalId"] == "[parameters('submitterPrincipalId')]"
