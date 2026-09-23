from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.azure import validate_compute
from learning.common import ContractError


@pytest.mark.parametrize("tier", ["Dedicated", "LowPriority"])
def test_approved_compute_priority_is_explicit_and_verified(tier):
    compute = SimpleNamespace(
        type="amlcompute",
        size="Standard_NC24ads_A100_v4",
        min_instances=0,
        max_instances=1,
        enable_node_public_ip=False,
        tier=tier.lower(),
    )
    config = {"compute_size": compute.size, "compute_tier": tier}
    validate_compute(compute, config)
    compute.tier = "dedicated" if tier == "LowPriority" else "low_priority"
    with pytest.raises(ContractError, match="tier"):
        validate_compute(compute, config)


def test_private_single_node_learning_infrastructure_exists():
    path = Path(__file__).resolve().parents[2] / "infra" / "learning.bicep"
    template = path.read_text()
    for required in (
        "publicNetworkAccess: 'Disabled'",
        "isolationMode: 'AllowOnlyApprovedOutbound'",
        "enableNodePublicIp: false",
        "remoteLoginPortPublicAccess: 'Disabled'",
        "minNodeCount: 0",
        "maxNodeCount: 1",
        "vmPriority: computeTier",
        "param storageAccountId string",
        "param keyVaultId string",
        "param containerRegistryId string",
        "param applicationInsightsId string",
        "applicationInsights: applicationInsightsId",
        "systemDatastoresAuthMode: 'Identity'",
        "Microsoft.MachineLearningServices/workspaces@2025-06-01",
        "param workspaceIdentityId string",
        "param computeIdentityId string",
    ):
        assert required in template
    assert "listKeys(" not in template
    assert "adminUserPassword" not in template
    assert "userAccountCredentials" not in template


def test_associated_dependencies_do_not_duplicate_auto_generated_outbound_rules():
    path = Path(__file__).resolve().parents[2] / "infra" / "learning.bicep"
    template = path.read_text()
    assert "outboundRules:" not in template
    assert "storageAccount: storageAccountId" in template
    assert "keyVault: keyVaultId" in template
    assert "containerRegistry: containerRegistryId" in template
    assert "isolationMode: 'AllowOnlyApprovedOutbound'" in template


def test_workspace_and_scoped_identity_approval_precede_optional_compute_creation():
    path = Path(__file__).resolve().parents[2] / "infra" / "learning.bicep"
    text = path.read_text()
    assert "param provisionCompute bool = false" in text
    assert "computes@2024-04-01' = if (provisionCompute)" in text


def test_postdeploy_network_verification_requires_actual_active_dependency_endpoints():
    from learning.azure import validate_managed_network_dependencies

    rules = []
    workspace = SimpleNamespace(
        storage_account="/storage",
        key_vault="/vault",
        container_registry="/registry",
        managed_network=SimpleNamespace(
            isolation_mode="AllowOnlyApprovedOutbound",
            outbound_rules=rules,
        ),
    )
    for resource, subresource in (
        ("/storage", "blob"),
        ("/storage", "file"),
        ("/vault", "vault"),
        ("/registry", "registry"),
    ):
        rules.append(
            SimpleNamespace(
                type="private_endpoint",
                status="Active",
                service_resource_id=resource,
                subresource_target=subresource,
            )
        )
    validate_managed_network_dependencies(workspace)
    rules[0].status = "Inactive"
    with pytest.raises(ContractError, match="private endpoint"):
        validate_managed_network_dependencies(workspace)
