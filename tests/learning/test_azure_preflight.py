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


@pytest.fixture
def compute_wire_identity(azure_config, preflight_context):
    client, _, _, compute, _, _ = preflight_context
    compute.identity.user_assigned_identities[0].client_id = None
    record = SimpleNamespace(
        id=(
            f"/subscriptions/{azure_config['subscription_id']}"
            f"/resourceGroups/{azure_config['resource_group']}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{azure_config['workspace']}"
            f"/computes/{azure_config['compute']}"
        ),
        identity=SimpleNamespace(
            tenant_id=azure_config["tenant_id"],
            user_assigned_identities={
                azure_config["managed_identity_resource_id"].lower(): SimpleNamespace(
                    client_id=azure_config["managed_identity_client_id"]
                )
            },
        ),
    )
    calls = []

    def get(group, workspace, name):
        calls.append((group, workspace, name))
        return record

    client.compute._operation = SimpleNamespace(get=get)
    return record, calls


def test_compute_identity_lost_by_sdk_is_verified_from_original_wire_metadata(
    azure_config, preflight_context, compute_wire_identity
):
    azure.preflight(preflight_context[0], azure_config)
    assert compute_wire_identity[1] == [
        (azure_config["resource_group"], azure_config["workspace"], azure_config["compute"])
    ]
    assert preflight_context[3].identity.user_assigned_identities[0].client_id is None


@pytest.mark.parametrize(
    "change", ["compute", "tenant", "resource", "client", "missing", "no-wire"]
)
def test_missing_sdk_identity_never_weakens_the_approved_identity_binding(
    azure_config, preflight_context, compute_wire_identity, change
):
    record, _ = compute_wire_identity
    if change == "compute":
        record.id += "-other"
    elif change == "tenant":
        record.identity.tenant_id = "another-tenant"
    elif change == "resource":
        record.identity.user_assigned_identities = {
            "another-identity": SimpleNamespace(client_id=None)
        }
    elif change == "client":
        next(iter(record.identity.user_assigned_identities.values())).client_id = "another-client"
    elif change == "missing":
        record.identity.user_assigned_identities = {}
    else:
        del preflight_context[0].compute._operation
    with pytest.raises(ContractError, match="identity"):
        azure.preflight(preflight_context[0], azure_config)


def test_explicit_sdk_identity_mismatch_is_not_replaced_by_a_matching_wire_record(
    azure_config, preflight_context, compute_wire_identity
):
    preflight_context[3].identity.user_assigned_identities[0].client_id = "another-client"
    with pytest.raises(ContractError, match="identity"):
        azure.preflight(preflight_context[0], azure_config)
    assert compute_wire_identity[1] == []


def test_folder_registration_accepts_single_provider_added_trailing_slash(
    azure_config, preflight_context
):
    preflight_context[-1].path += "/"
    azure.preflight(preflight_context[0], azure_config)


@pytest.mark.parametrize(
    "path",
    [
        "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/data/paths/x//",
        "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/data/paths/x/child",
        "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/data/paths/X/",
        "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/data/paths/x/?query=1",
    ],
)
def test_folder_location_matching_does_not_normalize_other_differences(path):
    asset = {
        "type": "uri_folder",
        "uri": "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/data/paths/x",
    }
    assert not azure.registered_input_matches("uri_folder", path, asset)


def test_file_location_never_accepts_a_folder_suffix():
    asset = {"type": "uri_file", "uri": "azureml://approved/file.json"}
    assert not azure.registered_input_matches("uri_file", asset["uri"] + "/", asset)
    assert not azure.registered_input_matches("uri_folder", asset["uri"], asset)


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
