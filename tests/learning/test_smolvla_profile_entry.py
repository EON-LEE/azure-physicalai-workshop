from copy import deepcopy
from pathlib import Path

import pytest

from learning.checks.smolvla_vendor_profile import PROFILE_SETTINGS
from learning.common import ContractError, canonical, digest, file_digest, write_json


def template():
    prefix = "tenants/11111111-1111-4111-8111-111111111111/owners/" + "a" * 64 + "/learning"
    return {
        "schema": "physicalai.smolvla-vendor-profile/v1",
        "profile_code_sha256": file_digest(Path("learning/checks/smolvla_vendor_profile.py")),
        "settings": deepcopy(PROFILE_SETTINGS),
        "vendor_specification": {
            "schema": "physicalai.smolvla-vendor-diagnostic/v1",
            "scope": {"tenant_id": "11111111-1111-4111-8111-111111111111", "owner_id": "a" * 64},
            "workspace_id": (
                "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/test-rg"
                "/providers/Microsoft.MachineLearningServices/workspaces/existing-ml"
            ),
            "job_name": "unit-profile-03",
            "compute_name": "existing-single-gpu",
            "managed_identity_client_id": "33333333-3333-4333-8333-333333333333",
            "storage_account_url": "https://teststorage.blob.core.windows.net",
            "container": "artifacts",
            "vendor_prefix": prefix + "/vendor/exact-version",
            "vendor_inventory_sha256": "b" * 64,
            "output_prefix": prefix + "/outputs/profile-03",
            "image": "testregistry.azurecr.io/smol@sha256:" + "c" * 64,
            "image_code_snapshot_sha256": "d" * 64,
            "diagnostic_code_sha256": file_digest(
                Path("learning/checks/smolvla_vendor_diagnostic.py")
            ),
            "execution_timeout_seconds": 600,
        },
    }


def test_short_entry_only_binds_new_immutable_image_to_the_exact_frozen_template(tmp_path):
    from learning.checks.smolvla_profile_entry import materialize_specification

    value = template()
    path = tmp_path / "profile-template.json"
    write_json(path, value)
    expected = deepcopy(value)
    expected["vendor_specification"]["image"] = "testregistry.azurecr.io/smol@sha256:" + "e" * 64
    result = materialize_specification(
        path,
        template_sha256=file_digest(path),
        image=expected["vendor_specification"]["image"],
        configuration_sha256=digest(canonical(expected)),
    )
    assert result == expected
    assert result["settings"] == value["settings"]


@pytest.mark.parametrize("change", ["file", "config-hash", "mutable-image", "settings"])
def test_packaging_cannot_mutate_workload_or_skip_reviewed_hashes(tmp_path, change):
    from learning.checks.smolvla_profile_entry import materialize_specification

    value = template()
    path = tmp_path / "profile-template.json"
    write_json(path, value)
    template_sha = file_digest(path)
    image = "testregistry.azurecr.io/smol@sha256:" + "e" * 64
    expected = deepcopy(value)
    expected["vendor_specification"]["image"] = image
    config_sha = digest(canonical(expected))
    if change == "file":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "config-hash":
        config_sha = "0" * 64
    elif change == "mutable-image":
        image = "testregistry.azurecr.io/smol:latest"
    else:
        value["settings"]["num_steps"] = 4
        path.write_bytes(canonical(value) + b"\n")
        template_sha = file_digest(path)
        expected = deepcopy(value)
        expected["vendor_specification"]["image"] = image
        config_sha = digest(canonical(expected))
    with pytest.raises(ContractError):
        materialize_specification(
            path,
            template_sha256=template_sha,
            image=image,
            configuration_sha256=config_sha,
        )
