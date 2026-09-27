"""Regional SKU support and account quota are distinct from actual GPU availability."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from simulation import batch


@pytest.fixture
def platform():
    return batch.BatchPlatform.model_validate(
        json.loads(Path("docs/batch-platform.example.json").read_text())
    )


@pytest.fixture
def records(platform):
    identity = platform.node_identity_resource_id.split("/")
    account_id = "/".join(identity[:5]) + "/providers/Microsoft.Batch/batchAccounts/unit"
    return (
        {
            "id": account_id,
            "location": "eastus2",
            "properties": {
                "accountEndpoint": "unit.eastus2.batch.azure.com",
                "poolAllocationMode": "BatchService",
                "provisioningState": "Succeeded",
                "lowPriorityCoreQuota": 36,
            },
        },
        {
            "value": [
                {
                    "name": platform.vm_size,
                    "familyName": "StandardNVADSA10v5Family",
                    "capabilities": [
                        {"name": "LowPriorityCapable", "value": "True"},
                        {"name": "vCPUs", "value": "36"},
                        {"name": "GPUs", "value": "1"},
                    ],
                }
            ]
        },
    )


def test_supported_sku_and_quota_permit_request_not_claim_capacity(platform, records):
    account, catalog = records
    result = batch.validate_regional_capacity(platform, account, catalog)
    assert result["can_request_one_node"] is True
    assert result["low_priority_core_quota"] == 36
    assert result["capacity_guaranteed"] is False
    assert result["physical_episode_started"] is False


def test_account_quota_does_not_make_an_unsupported_regional_sku_eligible(platform, records):
    account, catalog = records
    account["properties"]["lowPriorityCoreQuota"] = 500
    catalog["value"][0]["capabilities"][0]["value"] = "False"
    with pytest.raises(ValueError, match="LowPriorityCapable"):
        batch.validate_regional_capacity(platform, account, catalog)


@pytest.mark.parametrize("quota", [0, 35, None, True, "36"])
def test_insufficient_or_invalid_quota_blocks_before_submission(platform, records, quota):
    account, catalog = records
    account["properties"]["lowPriorityCoreQuota"] = quota
    with pytest.raises(ValueError, match="quota"):
        batch.validate_regional_capacity(platform, account, catalog)


@pytest.mark.parametrize("change", ["account", "region", "mode", "sku", "truncated", "duplicate"])
def test_unbound_or_ambiguous_management_evidence_is_rejected(platform, records, change):
    account, catalog = deepcopy(records)
    if change == "account":
        account["id"] += "-other"
    elif change == "region":
        account["location"] = "westus2"
    elif change == "mode":
        account["properties"]["poolAllocationMode"] = "UserSubscription"
    elif change == "sku":
        catalog["value"][0]["name"] = "Standard_NV12ads_A10_v5"
    elif change == "truncated":
        catalog["nextLink"] = "https://management.azure.com/more"
    else:
        catalog["value"].append(deepcopy(catalog["value"][0]))
    with pytest.raises(ValueError):
        batch.validate_regional_capacity(platform, account, catalog)


def test_management_reads_are_bound_to_account_region_and_existing_subscription(
    platform, records, monkeypatch
):
    calls = []
    credential = object()

    def read(url, actual_credential):
        assert actual_credential is credential
        calls.append(url)
        return records[len(calls) - 1]

    monkeypatch.setattr(batch, "_management_document", read)
    result = batch.check_regional_capacity(platform, credential)
    assert result["can_request_one_node"]
    assert len(calls) == 2
    assert "/batchAccounts/unit?api-version=2025-06-01" in calls[0]
    assert "/locations/eastus2/virtualMachineSkus?" in calls[1]
    assert all(url.startswith("https://management.azure.com/subscriptions/") for url in calls)
