from copy import deepcopy

import pytest

from learning.common import ContractError
from learning.smolvla import BACKBONE_ID, BACKBONE_REVISION, MODEL_ID, MODEL_REVISION
from learning.smolvla.prepare import VENDOR_GIT_BLOBS, VENDOR_SHA256

TENANT = "2573db8c-dfe5-4805-9e28-a0859692e705"
OWNER = "2ec908df6936652c12469d5a2f274a3012fbe41a4896a5ab7ef59ecbc2a3f803"
PREFIX = f"tenants/{TENANT}/owners/{OWNER}/learning/vendor/smolvla-20260923-01"


def vendor_report():
    files = []
    for kind, repo, revision in (
        ("model", MODEL_ID, MODEL_REVISION),
        ("backbone", BACKBONE_ID, BACKBONE_REVISION),
    ):
        for name in sorted(set(VENDOR_SHA256[kind]) | set(VENDOR_GIT_BLOBS[kind])):
            files.append(
                {
                    "kind": kind,
                    "name": name,
                    "sha256": VENDOR_SHA256[kind].get(name, "a" * 64),
                    "bytes": 100,
                    "blob": f"{PREFIX}/{kind}/{name}",
                    "repo": repo,
                    "revision": revision,
                }
            )
    return {
        "source": "actual_azure_vendor_asset_staging",
        "scope": "immutable_public_apache_assets_only",
        "customer_data_sent_to_vendor": False,
        "weights_loaded": False,
        "optimizer_steps": 0,
        "files": files,
        "passed": True,
        "manifest_prefix": PREFIX,
    }


def test_completed_vendor_manifest_must_match_exact_scope_and_download_allowlist():
    from learning.contract import Scope
    from learning.smolvla.cloud_assets import validate_vendor_inventory

    files = validate_vendor_inventory(vendor_report(), prefix=PREFIX, scope=Scope(TENANT, OWNER))
    assert len(files) == 20
    assert set(files) == {(item["kind"], item["name"]) for item in vendor_report()["files"]}


@pytest.mark.parametrize(
    "change", ["partial", "duplicate", "owner", "revision", "checksum", "path", "not-passed"]
)
def test_partial_wrong_scope_or_relabelled_vendor_assets_fail_closed(change):
    from learning.contract import Scope
    from learning.smolvla.cloud_assets import validate_vendor_inventory

    report = deepcopy(vendor_report())
    if change == "partial":
        report["files"].pop()
    elif change == "duplicate":
        report["files"].append(report["files"][0])
    elif change == "owner":
        report["manifest_prefix"] = PREFIX.replace(OWNER, "b" * 64)
    elif change == "revision":
        report["files"][0]["revision"] = "main"
    elif change == "checksum":
        next(item for item in report["files"] if item["name"] == "model.safetensors")["sha256"] = (
            "0" * 64
        )
    elif change == "path":
        report["files"][0]["blob"] = PREFIX + "/../private/secret"
    else:
        report["passed"] = False
    with pytest.raises(ContractError):
        validate_vendor_inventory(report, prefix=PREFIX, scope=Scope(TENANT, OWNER))


def test_preparation_output_never_overwrites_inputs_or_escapes_owner():
    from learning.contract import Scope
    from learning.smolvla.cloud_assets import validate_prefixes

    scope = Scope(TENANT, OWNER)
    validate_prefixes(PREFIX, PREFIX.rsplit("/vendor/", 1)[0] + "/prepared/run-01", scope)
    for output in (
        PREFIX,
        PREFIX + "/prepared",
        "public/prepared",
        PREFIX.replace(OWNER, "b" * 64),
    ):
        with pytest.raises(ContractError):
            validate_prefixes(PREFIX, output, scope)
