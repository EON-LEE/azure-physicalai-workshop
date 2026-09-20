"""Explicit fixtures avoid shadowing existing tests' unqualified conftest imports."""

from __future__ import annotations

import pytest

from learning.capture import EpisodeWriter, assemble_dataset
from learning.checks.fixtures import LIVE_PROVENANCE, PROVENANCE, SCOPE, frame
from learning.contract import EpisodeSpec


@pytest.fixture
def make_capture(tmp_path):
    def make(name="demo", *, seed=1, split="train", live=False, count=3, **limits):
        root = tmp_path / name
        writer = EpisodeWriter(
            root,
            dataset_id="customer-demonstrations",
            scope=SCOPE,
            episode=EpisodeSpec(name, "customer-line", "f" * 64, seed, split),
            provenance=LIVE_PROVENANCE if live else PROVENANCE,
            **limits,
        )
        for index in range(count):
            writer.append(frame(index, terminal=index == count - 1))
        writer.finalize()
        return root

    return make


@pytest.fixture
def make_dataset(make_capture, tmp_path):
    def make(*, test_count=1, live=False):
        roots = [
            make_capture("train-episode", seed=1, live=live),
            make_capture("validation-episode", seed=2, split="validation", live=live),
            *[
                make_capture(f"test-{index}", seed=100 + index, split="test", live=live)
                for index in range(test_count)
            ],
        ]
        output = tmp_path / "dataset"
        assemble_dataset(
            roots, output, dataset_id="customer-data", expected_scope=SCOPE, require_live=live
        )
        return output

    return make


@pytest.fixture
def azure_config():
    prefix = f"tenants/{SCOPE.tenant_id}/owners/{SCOPE.owner_id}/"
    subscription = "22222222-2222-4222-8222-222222222222"
    uri = (
        f"azureml://subscriptions/{subscription}/resourcegroups/approved-rg"
        f"/workspaces/approved-ml/datastores/ownerdata/paths/{prefix}raw/v1"
    )
    return {
        "schema": "physicalai.azure-learning/v1",
        "kind": "train",
        "subscription_id": subscription,
        "tenant_id": SCOPE.tenant_id,
        "owner_id": SCOPE.owner_id,
        "resource_group": "approved-rg",
        "workspace": "approved-ml",
        "compute": "approved-gpu",
        "compute_size": "Standard_NC4as_T4_v3",
        "managed_identity_client_id": "33333333-3333-4333-8333-333333333333",
        "managed_identity_resource_id": (
            f"/subscriptions/{subscription}/resourceGroups/approved-rg/providers/"
            "Microsoft.ManagedIdentity/userAssignedIdentities/learning-identity"
        ),
        "datastore": "ownerdata",
        "storage_account_name": "approvedstorage",
        "storage_account_resource_group": "approved-rg",
        "blob_container": "learning",
        "output_prefix": prefix + "outputs",
        "retention_days": 7,
        "run_id": "training-001",
        "environment_image": "approvedacr.azurecr.io/learning@sha256:" + "c" * 64,
        "model_name": "franka-act",
        "model_version": "1",
        "inputs": {
            "demonstrations": {
                "name": "approved-demonstrations",
                "version": "1",
                "type": "uri_folder",
                "uri": uri,
                "sha256": "d" * 64,
            }
        },
        "parameters": {
            "steps": 100,
            "batch_size": 2,
            "seed": 12345,
            "chunk_size": 10,
            "n_action_steps": 1,
            "timeout_seconds": 1800,
            "conversion_timeout_seconds": 600,
            "image_size": 224,
        },
    }
