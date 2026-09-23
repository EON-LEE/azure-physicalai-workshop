from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.azure import validate_compute
from learning.common import ContractError


@pytest.mark.parametrize("tier", ["Dedicated", "LowPriority"])
def test_approved_compute_priority_is_explicit_and_verified(tier):
    compute = SimpleNamespace(
        type="amlcompute", size="Standard_NC24ads_A100_v4",
        min_instances=0, max_instances=1, enable_node_public_ip=False, tier=tier.lower(),
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
        "publicNetworkAccess: 'Disabled'", "isolationMode: 'AllowOnlyApprovedOutbound'",
        "enableNodePublicIp: false", "remoteLoginPortPublicAccess: 'Disabled'",
        "minNodeCount: 0", "maxNodeCount: 1", "vmPriority: computeTier",
        "param storageAccountId string", "param keyVaultId string",
        "param containerRegistryId string", "param workspaceIdentityId string",
        "param computeIdentityId string",
    ):
        assert required in template
    assert "listKeys(" not in template
    assert "adminUserPassword" not in template
    assert "userAccountCredentials" not in template
