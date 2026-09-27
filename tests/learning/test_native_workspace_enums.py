from enum import Enum
from types import SimpleNamespace

import pytest

from learning.checks.smolvla_aml_check import example_config
from learning.common import ContractError
from learning.gr00t.azure import workspace_id
from learning.smolvla.azure import PolicyJobs


class PublicNetworkAccessType(str, Enum):  # noqa: UP042 -- Match Azure ML's legacy enum string form.
    DISABLED = "Disabled"
    ENABLED = "Enabled"


class IsolationMode(str, Enum):  # noqa: UP042 -- Match Azure ML's legacy enum string form.
    ALLOW_ONLY_APPROVED_OUTBOUND = "AllowOnlyApprovedOutbound"
    ALLOW_INTERNET_OUTBOUND = "AllowInternetOutbound"


class ComputeReached(Exception):
    pass


def backend(public, isolation):
    config = {
        **example_config(),
        "schema": "physicalai.smolvla-azure/v2",
        "job_deadline_utc": "2030-01-01T00:00:00Z",
    }
    workspace = SimpleNamespace(
        id=workspace_id(config),
        public_network_access=public,
        system_datastores_auth_mode="identity",
        storage_account="/storage",
        key_vault="/vault",
        container_registry="/registry",
        managed_network=SimpleNamespace(isolation_mode=isolation, outbound_rules=[]),
    )
    rules = [
        SimpleNamespace(
            type="private_endpoint",
            status="Active",
            service_resource_id=resource,
            subresource_target=subresource,
        )
        for resource, subresource in (
            ("/storage", "blob"),
            ("/storage", "file"),
            ("/vault", "vault"),
            ("/registry", "registry"),
        )
    ]

    def compute(_):
        raise ComputeReached

    client = SimpleNamespace(
        workspaces=SimpleNamespace(get=lambda _: workspace),
        workspace_outbound_rules=SimpleNamespace(list=lambda **_: rules),
        compute=SimpleNamespace(get=compute),
    )
    return PolicyJobs(client, config, storage_client=None)


@pytest.mark.parametrize("public", ["Disabled", PublicNetworkAccessType.DISABLED])
@pytest.mark.parametrize(
    "isolation", ["AllowOnlyApprovedOutbound", IsolationMode.ALLOW_ONLY_APPROVED_OUTBOUND]
)
def test_exact_private_network_strings_and_sdk_enum_values_pass_admission(public, isolation):
    with pytest.raises(ComputeReached):
        backend(public, isolation).preflight()


@pytest.mark.parametrize(
    "public,isolation",
    [
        (PublicNetworkAccessType.ENABLED, IsolationMode.ALLOW_ONLY_APPROVED_OUTBOUND),
        (PublicNetworkAccessType.DISABLED, IsolationMode.ALLOW_INTERNET_OUTBOUND),
        ("PublicNetworkAccessType.DISABLED", IsolationMode.ALLOW_ONLY_APPROVED_OUTBOUND),
        (None, IsolationMode.ALLOW_ONLY_APPROVED_OUTBOUND),
    ],
)
def test_wrong_or_missing_private_network_values_still_fail_closed(public, isolation):
    with pytest.raises(ContractError, match="networking"):
        backend(public, isolation).preflight()
