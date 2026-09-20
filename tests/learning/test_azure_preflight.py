from __future__ import annotations

from types import SimpleNamespace

import pytest

from learning import azure
from learning.common import ContractError

pytest_plugins = ("learning.checks.pytest_fixtures",)


@pytest.fixture
def preflight_context(azure_config, monkeypatch):
    account = {
        "id": azure_config["subscription_id"],
        "tenantId": azure_config["tenant_id"],
        "state": "Enabled",
    }
    identity = SimpleNamespace(
        resource_id=azure_config["managed_identity_resource_id"],
        client_id=azure_config["managed_identity_client_id"],
    )
    workspace = SimpleNamespace(
        managed_network=SimpleNamespace(isolation_mode="allow_only_approved_outbound")
    )
    compute = SimpleNamespace(
        type="amlcompute",
        size=azure_config["compute_size"],
        min_instances=0,
        max_instances=1,
        enable_node_public_ip=False,
        identity=SimpleNamespace(user_assigned_identities=[identity]),
    )
    datastore = SimpleNamespace(
        type=SimpleNamespace(value="AzureBlob"),
        account_name=azure_config["storage_account_name"],
        container_name=azure_config["blob_container"],
        credentials=SimpleNamespace(type=SimpleNamespace(value="None")),
    )
    asset = SimpleNamespace(type="uri_folder", path=azure_config["inputs"]["demonstrations"]["uri"])
    client = SimpleNamespace(
        workspaces=SimpleNamespace(get=lambda name: workspace),
        compute=SimpleNamespace(get=lambda name: compute),
        datastores=SimpleNamespace(get=lambda name: datastore),
        data=SimpleNamespace(get=lambda name, version: asset),
    )

    def read_azure(arguments):
        assert "--subscription" in arguments
        assert azure_config["subscription_id"] in arguments
        if arguments[0] == "account":
            return account
        return {
            "policy": {
                "rules": [
                    {
                        "enabled": True,
                        "definition": {
                            "filters": {
                                "prefixMatch": [
                                    azure_config["blob_container"]
                                    + "/"
                                    + azure_config["output_prefix"]
                                    + "/"
                                ]
                            },
                            "actions": {
                                "baseBlob": {"delete": {"daysAfterModificationGreaterThan": 7}}
                            },
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(azure, "_az_json", read_azure)
    return client, account, workspace, compute, datastore, asset


def test_preflight_handles_pinned_sdk_response_shapes(azure_config, preflight_context):
    azure.preflight(preflight_context[0], azure_config)


@pytest.mark.parametrize(
    "kind",
    [
        "subscription",
        "tenant",
        "disabled",
        "network",
        "compute",
        "scale",
        "public-ip",
        "identity-resource",
        "identity-client",
        "container",
        "credentials",
        "data-version",
    ],
)
def test_preflight_refuses_scope_identity_network_or_asset_changes(
    azure_config, preflight_context, kind
):
    client, account, workspace, compute, datastore, asset = preflight_context
    if kind == "subscription":
        account["id"] = "default-subscription"
    elif kind == "tenant":
        account["tenantId"] = "other-tenant"
    elif kind == "disabled":
        account["state"] = "Disabled"
    elif kind == "network":
        workspace.managed_network.isolation_mode = "allow_internet_outbound"
    elif kind == "compute":
        compute.size = "Standard_ND96amsr_A100_v4"
    elif kind == "scale":
        compute.max_instances = 2
    elif kind == "public-ip":
        compute.enable_node_public_ip = True
    elif kind == "identity-resource":
        compute.identity.user_assigned_identities[0].resource_id = "another-identity"
    elif kind == "identity-client":
        compute.identity.user_assigned_identities[0].client_id = "other-client"
    elif kind == "container":
        datastore.container_name = "other-tenant-data"
    elif kind == "credentials":
        datastore.credentials.type.value = "AccountKey"
    else:
        asset.path = "https://unapproved.example/observations"
    with pytest.raises(ContractError):
        azure.preflight(client, azure_config)
