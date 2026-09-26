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
    assert len(grants) == 2
    expected_roles = {
        "48e5e92e-a480-4e71-aa9c-2778f4c13781",
        "11076f67-66f6-4be0-8f6b-f0609fd05cc9",
    }
    for grant in grants:
        assert grant["condition"] == "[parameters('enabled')]"
        assert "Microsoft.Batch/batchAccounts" in grant["scope"]
        assert "batchAccountName" in grant["scope"]
        assert grant["properties"]["principalType"] == "ServicePrincipal"
        assert grant["properties"]["principalId"] == "[parameters('submitterPrincipalId')]"
    assert all(
        any(role in grant["properties"]["roleDefinitionId"] for grant in grants)
        for role in expected_roles
    )


def test_batch_node_identity_has_no_legacy_tls_or_control_permissions(tmp_path):
    template = compile_template(
        ROOT / "infra" / "simulation-batch-node.bicep", tmp_path / "node.json"
    )
    assert template["parameters"]["enabled"]["defaultValue"] is False
    assert len(resources(template, "Microsoft.ManagedIdentity/userAssignedIdentities")) == 1
    grants = resources(template, "Microsoft.Authorization/roleAssignments")
    expected = {
        "7f951dda-4ed3-4680-a7ca-43fe172d538d": "Microsoft.ContainerRegistry/registries",
        "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1": "'artifacts'",
        "ba92f5b4-2d11-453d-a403-e96b0029c9fe": "'demonstrations'",
    }
    assert len(grants) == len(expected)
    for role, scope in expected.items():
        matching = [g for g in grants if role in g["properties"]["roleDefinitionId"]]
        assert len(matching) == 1 and scope in matching[0]["scope"]
        assert matching[0]["condition"] == "[parameters('enabled')]"
        assert matching[0]["properties"]["principalType"] == "ServicePrincipal"
    assert "Microsoft.KeyVault" not in json.dumps(template)
    assert not resources(template, "Microsoft.Compute/virtualMachines")


def test_private_controller_selects_the_existing_worker_identity(tmp_path):
    template = compile_template(ROOT / "infra" / "learning-worker.bicep", tmp_path / "worker.json")
    worker = resources(template, "Microsoft.App/containerApps")[0]
    env = {
        item["name"]: item["value"]
        for item in worker["properties"]["template"]["containers"][0]["env"]
    }
    assert env["AZURE_CLIENT_ID"] == env["LEARNING_WORKER_MANAGED_IDENTITY_CLIENT_ID"]


def test_managed_warmup_controller_is_bounded_and_cannot_dispatch_physics(tmp_path):
    template = compile_template(
        ROOT / "infra" / "simulation-batch-warmup.bicep", tmp_path / "warmup.json"
    )
    assert template["parameters"]["enabled"]["defaultValue"] is False
    job = resources(template, "Microsoft.App/jobs")[0]
    assert job["condition"] == "[parameters('enabled')]"
    config = job["properties"]["configuration"]
    assert config["triggerType"] == "Manual"
    assert config["replicaTimeout"] == 180 and config["replicaRetryLimit"] == 0
    assert config["manualTriggerConfig"] == {"parallelism": 1, "replicaCompletionCount": 1}
    assert "'warmup'" in template["variables"]["invocation"]
    assert "'submit'" not in template["variables"]["invocation"]
    container = job["properties"]["template"]["containers"][0]
    assert "@sha256:" in container["image"]
    assert container["command"][:2] == ["/srv/apps/learning_worker/.venv/bin/python", "-c"]
    assert not resources(template, "Microsoft.Authorization/roleAssignments")
