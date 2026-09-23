from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

from learning.common import (
    ContractError,
    canonical,
    digest,
    integer,
    inventory,
    keys,
    read_json,
    relative_path,
    require,
    safe_path,
    sha256,
    token,
    verify_inventory,
    write_json,
)
from learning.contract import Scope
from learning.offline import OFFLINE_ENV
from learning.train import TrainOptions

AZURE_SDK_VERSION = "1.35.0"
CONFIG_SCHEMA = "physicalai.azure-learning/v1"
PLAN_SCHEMA = "physicalai.azure-learning-plan/v1"
SNAPSHOT_FILES = (
    "learning/__init__.py",
    "learning/common.py",
    "learning/contract.py",
    "learning/capture.py",
    "learning/offline.py",
    "learning/convert.py",
    "learning/train.py",
    "learning/inference.py",
    "learning/evaluation.py",
    "learning/azure.py",
    "learning/components.py",
    "learning/pyproject.toml",
    "learning/uv.lock",
    "learning/.python-version",
    "learning/Dockerfile",
)
CONFIG_KEYS = {
    "schema",
    "kind",
    "subscription_id",
    "tenant_id",
    "owner_id",
    "resource_group",
    "workspace",
    "compute",
    "compute_size",
    "managed_identity_client_id",
    "managed_identity_resource_id",
    "datastore",
    "storage_account_name",
    "storage_account_resource_group",
    "blob_container",
    "output_prefix",
    "retention_days",
    "run_id",
    "environment_image",
    "model_name",
    "model_version",
    "inputs",
    "parameters",
}


def _uuid(value: object, name: str) -> str:
    try:
        require(isinstance(value, str) and str(UUID(value)) == value, f"Invalid {name}")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError(f"Invalid {name}") from exc
    return value


def datastore_prefix(config: dict) -> str:
    return (
        f"azureml://subscriptions/{config['subscription_id']}"
        f"/resourcegroups/{config['resource_group']}/workspaces/{config['workspace']}"
        f"/datastores/{config['datastore']}/paths/"
    )


def validate_config(config: dict) -> None:
    extra = {"compute_tier"} if "compute_tier" in config else set()
    keys(config, CONFIG_KEYS | extra, "Azure learning configuration")
    if extra:
        require(config["compute_tier"] in ("Dedicated", "LowPriority"), "Invalid compute tier")
    require(
        config["schema"] == CONFIG_SCHEMA and config["kind"] in ("train", "gate"),
        "Invalid job kind",
    )
    _uuid(config["subscription_id"], "subscription")
    _uuid(config["managed_identity_client_id"], "managed identity client ID")
    Scope(config["tenant_id"], config["owner_id"]).validate()
    for name in (
        "resource_group",
        "workspace",
        "compute",
        "datastore",
        "storage_account_name",
        "storage_account_resource_group",
        "blob_container",
        "run_id",
        "model_name",
        "model_version",
    ):
        token(config[name], name)
    require(
        config["model_version"].lower() != "latest", "Model version must be immutable and explicit"
    )
    require(
        re.fullmatch(r"Standard_N[CDV][A-Za-z0-9_]+", config["compute_size"]) is not None,
        "Use an explicit approved Azure GPU SKU, not local/serverless compute",
    )
    require(
        re.fullmatch(r"[a-z0-9]{3,24}", config["storage_account_name"]) is not None,
        "Invalid approved storage account",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", config["blob_container"]) is not None,
        "Invalid approved Blob container",
    )
    identity_prefix = f"/subscriptions/{config['subscription_id']}/resourceGroups/"
    identity = config["managed_identity_resource_id"]
    require(
        isinstance(identity, str)
        and identity.startswith(identity_prefix)
        and re.fullmatch(
            re.escape(identity_prefix)
            + r"[A-Za-z0-9_.-]+/providers/Microsoft\.ManagedIdentity/userAssignedIdentities/"
            + r"[A-Za-z0-9_.-]+",
            identity,
        ),
        "Managed identity must be explicit and in the approved subscription",
    )
    require(
        isinstance(config["environment_image"], str)
        and re.fullmatch(
            r"[a-z0-9]{5,50}\.azurecr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}",
            config["environment_image"],
        ),
        "Learning image must be an approved ACR digest, not a tag/public registry",
    )
    relative_path(config["environment_image"].split(".azurecr.io/", 1)[1].split("@", 1)[0])
    scoped_prefix = f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/"
    relative_path(config["output_prefix"])
    require(
        config["output_prefix"].startswith(scoped_prefix), "Outputs must stay in this owner scope"
    )
    integer(config["retention_days"], "output retention days", 1, 365)
    inputs = (
        {"demonstrations": "uri_folder"}
        if config["kind"] == "train"
        else {"model": "uri_folder", "evidence": "uri_folder", "plan": "uri_file"}
    )
    keys(config["inputs"], set(inputs), "job inputs")
    for name, input_type in inputs.items():
        asset = keys(config["inputs"][name], {"name", "version", "uri", "sha256", "type"}, name)
        token(asset["name"], "data asset name")
        token(asset["version"], "data asset version")
        require(asset["version"].lower() != "latest", "Data asset versions must be explicit")
        require(asset["type"] == input_type, "Wrong data asset type")
        sha256(asset["sha256"], "input content checksum")
        require(
            isinstance(asset["uri"], str) and asset["uri"].startswith(datastore_prefix(config)),
            "Input must be in the explicit approved Azure datastore",
        )
        path = asset["uri"][len(datastore_prefix(config)) :]
        relative_path(path)
        require(path.startswith(scoped_prefix), "Input points outside this tenant/owner scope")
    if config["kind"] == "train":
        parameters = keys(
            config["parameters"],
            {
                "steps",
                "batch_size",
                "seed",
                "chunk_size",
                "n_action_steps",
                "timeout_seconds",
                "conversion_timeout_seconds",
                "image_size",
            },
            "training parameters",
        )
        TrainOptions(
            **{
                name: value
                for name, value in parameters.items()
                if name not in ("conversion_timeout_seconds", "image_size")
            }
        ).validate()
        integer(parameters["conversion_timeout_seconds"], "conversion timeout", 1, 3600)
        integer(parameters["image_size"], "image size", 32, 512)
    else:
        keys(config["parameters"], {"timeout_seconds"}, "gate parameters")
        integer(config["parameters"]["timeout_seconds"], "gate timeout", 1, 3600)


def _job_step(config: dict, command: str, timeout: int, inputs: dict, outputs: dict) -> dict:
    return {
        "type": "command",
        "code": "./code",
        "command": command,
        "environment": {"image": config["environment_image"]},
        "environment_variables": {**OFFLINE_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        "compute": f"azureml:{config['compute']}",
        "resources": {"instance_count": 1, "shm_size": "4g"},
        "limits": {"timeout": timeout},
        "identity": {
            "type": "managed",
            "client_id": config["managed_identity_client_id"],
        },
        "inputs": inputs,
        "outputs": outputs,
    }


def build_job(config: dict, snapshot_sha256: str) -> dict:
    validate_config(config)
    sha256(snapshot_sha256)
    output_base = datastore_prefix(config) + config["output_prefix"] + "/" + config["run_id"]
    inputs = {
        name: {
            "type": asset["type"],
            "path": f"azureml:{asset['name']}:{asset['version']}",
            "mode": "ro_mount",
        }
        for name, asset in config["inputs"].items()
    }
    scope_args = (
        f" --tenant-id {config['tenant_id']} --owner-id {config['owner_id']}"
        f" --snapshot-sha256 {snapshot_sha256}"
    )
    tags = {
        "scope_owner": config["owner_id"],
        "code_snapshot_sha256": snapshot_sha256,
        "candidate_model": config["model_name"],
        "candidate_version": config["model_version"],
        "retention_days": str(config["retention_days"]),
        "learning_quality": "unverified",
    }
    if config["kind"] == "train":
        parameters = config["parameters"]
        convert = (
            "python -m learning.components convert --source '${{inputs.raw}}'"
            " --output '${{outputs.converted}}'"
            f" --manifest-sha256 {config['inputs']['demonstrations']['sha256']}"
            f" --image-size {parameters['image_size']}" + scope_args
        )
        train = (
            "python -m learning.components train --source '${{inputs.converted}}'"
            " --output '${{outputs.model}}'"
            f" --model-name {config['model_name']} --model-version {config['model_version']}"
            + "".join(
                f" --{name.replace('_', '-')} {parameters[name]}"
                for name in (
                    "steps",
                    "batch_size",
                    "seed",
                    "chunk_size",
                    "n_action_steps",
                    "timeout_seconds",
                )
            )
            + scope_args
        )
        outputs = {
            "converted": {
                "type": "uri_folder",
                "mode": "rw_mount",
                "path": output_base + "/converted",
            },
            "model": {
                "type": "uri_folder",
                "mode": "rw_mount",
                "path": (
                    f"{output_base}/candidates/{config['model_name']}/{config['model_version']}"
                ),
            },
        }
        jobs = {
            "convert": _job_step(
                config,
                convert,
                parameters["conversion_timeout_seconds"],
                {"raw": "${{parent.inputs.demonstrations}}"},
                {"converted": "${{parent.outputs.converted}}"},
            ),
            "train": _job_step(
                config,
                train,
                parameters["timeout_seconds"],
                {"converted": "${{parent.jobs.convert.outputs.converted}}"},
                {"model": "${{parent.outputs.model}}"},
            ),
        }
    else:
        gate = (
            "python -m learning.components gate --source '${{inputs.model}}'"
            " --evidence '${{inputs.evidence}}' --plan '${{inputs.plan}}'"
            " --output '${{outputs.gate}}'"
            f" --model-sha256 {config['inputs']['model']['sha256']}"
            f" --plan-sha256 {config['inputs']['plan']['sha256']}"
            f" --evidence-sha256 {config['inputs']['evidence']['sha256']}"
            f" --model-name {config['model_name']} --model-version {config['model_version']}"
            + scope_args
        )
        outputs = {
            "gate": {"type": "uri_folder", "mode": "rw_mount", "path": output_base + "/gate"}
        }
        jobs = {
            "validate_evidence": _job_step(
                config,
                gate,
                config["parameters"]["timeout_seconds"],
                {name: "${{parent.inputs." + name + "}}" for name in inputs},
                {"gate": "${{parent.outputs.gate}}"},
            )
        }
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json",
        "type": "pipeline",
        "display_name": f"physicalai-{config['kind']}-{config['run_id']}",
        "experiment_name": "physicalai-policy-learning",
        "tags": tags,
        "settings": {
            "default_datastore": f"azureml:{config['datastore']}",
            "default_compute": f"azureml:{config['compute']}",
            "continue_on_step_failure": False,
            "force_rerun": True,
        },
        "inputs": inputs,
        "outputs": outputs,
        "jobs": jobs,
    }


def verify_snapshot(root: Path, expected_sha256: str) -> None:
    manifest = keys(read_json(root / "snapshot.json"), {"files", "sha256"}, "code snapshot")
    require(
        manifest["sha256"] == sha256(expected_sha256)
        and digest(canonical(manifest["files"])) == expected_sha256,
        "Code snapshot digest mismatch",
    )
    require(set(manifest["files"]) == set(SNAPSHOT_FILES), "Unexpected/missing code snapshot file")
    verify_inventory(root, manifest["files"], exclude={"snapshot.json"})


def create_plan(config: dict, output: Path, *, source_root: Path | None = None) -> str:
    validate_config(config)
    root = source_root or Path(__file__).resolve().parents[1]
    require(not output.exists(), "Plan directory already exists; never overwrite a reviewed plan")
    code = output / "code"
    code.mkdir(parents=True)
    for name in SNAPSHOT_FILES:
        source = safe_path(root, name)
        destination = code.joinpath(*name.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    files = inventory(code)
    snapshot_sha = digest(canonical(files))
    write_json(code / "snapshot.json", {"files": files, "sha256": snapshot_sha})
    verify_snapshot(code, snapshot_sha)
    job = build_job(config, snapshot_sha)
    review = {
        "schema": PLAN_SCHEMA,
        "config": config,
        "code_snapshot_sha256": snapshot_sha,
        "job_sha256": digest(canonical(job)),
    }
    plan_sha = digest(canonical(review))
    write_json(output / "job.json", job)
    write_json(output / "plan.json", {**review, "plan_sha256": plan_sha})
    return plan_sha


def read_plan(root: Path) -> dict:
    plan = keys(
        read_json(root / "plan.json"),
        {"schema", "config", "code_snapshot_sha256", "job_sha256", "plan_sha256"},
        "submission plan",
    )
    require(plan["schema"] == PLAN_SCHEMA, "Unsupported submission plan")
    review = {name: value for name, value in plan.items() if name != "plan_sha256"}
    require(digest(canonical(review)) == sha256(plan["plan_sha256"]), "Plan was changed")
    validate_config(plan["config"])
    verify_snapshot(root / "code", plan["code_snapshot_sha256"])
    job = read_json(root / "job.json")
    require(
        digest(canonical(job)) == plan["job_sha256"]
        and job == build_job(plan["config"], plan["code_snapshot_sha256"]),
        "AML job differs from the reviewed configuration",
    )
    return plan


def _az_json(arguments: list[str]) -> dict:
    result = subprocess.run(
        ["az", *arguments, "--output", "json", "--only-show-errors"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    value = json.loads(result.stdout)
    require(isinstance(value, dict), "Unexpected Azure CLI response")
    return value


def validate_retention(policy: dict, config: dict) -> None:
    prefix = config["blob_container"] + "/" + config["output_prefix"] + "/"
    matched = False
    for rule in policy.get("policy", {}).get("rules", []):
        definition = rule.get("definition", {})
        if rule.get("enabled") is not True:
            continue
        if prefix not in definition.get("filters", {}).get("prefixMatch", []):
            continue
        days = (
            definition.get("actions", {})
            .get("baseBlob", {})
            .get("delete", {})
            .get("daysAfterModificationGreaterThan")
        )
        if type(days) in (float, int) and days == config["retention_days"]:
            matched = True
    require(matched, "Approved scoped Blob output-retention policy is not provisioned")


def validate_compute(compute, config: dict) -> None:
    require(
        compute.type == "amlcompute"
        and compute.size == config["compute_size"]
        and compute.min_instances == 0
        and compute.max_instances == 1
        and compute.enable_node_public_ip is False,
        "Approved single-node GPU cluster/scale-to-zero configuration differs",
    )
    if "compute_tier" in config:
        require(config["compute_tier"] in ("Dedicated", "LowPriority"), "Invalid compute tier")
        actual = getattr(compute.tier, "value", compute.tier)
        require(
            isinstance(actual, str)
            and actual.replace("_", "").lower() == config["compute_tier"].lower(),
            "Actual compute tier differs from explicit approved compute tier",
        )


def validate_managed_network_dependencies(workspace) -> None:
    network = workspace.managed_network
    mode = getattr(network, "isolation_mode", None)
    mode = getattr(mode, "value", mode)
    require(
        isinstance(mode, str) and mode.replace("_", "").lower() == "allowonlyapprovedoutbound",
        "Approved-outbound-only managed network is required",
    )
    rules = network.outbound_rules
    require(isinstance(rules, list), "Managed private endpoint rules have not been provisioned")
    expected = {
        (workspace.storage_account.lower(), "blob"),
        (workspace.storage_account.lower(), "file"),
        (workspace.key_vault.lower(), "vault"),
        (workspace.container_registry.lower(), "registry"),
    }
    active = set()
    for rule in rules:
        kind = getattr(rule.type, "value", rule.type)
        status = getattr(rule.status, "value", rule.status)
        if isinstance(kind, str) and kind.replace("_", "").lower() == "privateendpoint":
            if isinstance(status, str) and status.lower() == "active":
                active.add(
                    (
                        rule.service_resource_id.lower(),
                        rule.subresource_target.lower(),
                    )
                )
    require(
        expected.issubset(active),
        "Associated dependency private endpoints are missing, inactive or unapproved",
    )


def preflight(client, config: dict) -> None:
    account = _az_json(["account", "show", "--subscription", config["subscription_id"]])
    require(
        account.get("id") == config["subscription_id"]
        and account.get("tenantId") == config["tenant_id"]
        and account.get("state") == "Enabled",
        "Explicit subscription/tenant is unavailable; no default subscription fallback",
    )
    workspace = client.workspaces.get(config["workspace"])
    require(
        workspace.managed_network is not None
        and workspace.managed_network.isolation_mode.replace("_", "").lower()
        == "allowonlyapprovedoutbound",
        "Production learning requires approved-outbound-only Azure ML networking",
    )
    compute = client.compute.get(config["compute"])
    validate_compute(compute, config)
    identities = getattr(compute.identity, "user_assigned_identities", None)
    require(isinstance(identities, list), "Compute has no explicit user-assigned identities")
    matching = [
        identity
        for identity in identities
        if isinstance(identity.resource_id, str)
        and identity.resource_id.lower() == config["managed_identity_resource_id"].lower()
    ]
    require(len(matching) == 1, "Job managed identity is not attached to the approved compute")
    require(
        matching[0].client_id == config["managed_identity_client_id"],
        "Managed identity client ID mismatch",
    )
    datastore = client.datastores.get(config["datastore"])
    datastore_type = getattr(datastore.type, "value", datastore.type)
    credentials_type = getattr(datastore.credentials, "type", None)
    credentials_type = getattr(credentials_type, "value", credentials_type)
    require(
        datastore_type == "AzureBlob"
        and datastore.account_name == config["storage_account_name"]
        and datastore.container_name == config["blob_container"]
        and credentials_type == "None",
        "Datastore must be the approved identity-based Azure Blob container",
    )
    for asset in config["inputs"].values():
        registered = client.data.get(asset["name"], version=asset["version"])
        require(
            registered.type == asset["type"] and registered.path == asset["uri"],
            "Registered input data version resolves outside the approved location",
        )
    retention = _az_json(
        [
            "storage",
            "account",
            "management-policy",
            "show",
            "--subscription",
            config["subscription_id"],
            "--resource-group",
            config["storage_account_resource_group"],
            "--account-name",
            config["storage_account_name"],
        ]
    )
    validate_retention(retention, config)


def submit(
    root: Path,
    *,
    approved_plan_sha256: str,
    confirm_billable: bool,
    confirm_code_upload: bool,
    confirm_licenses_reviewed: bool,
) -> str:
    plan = read_plan(root)
    require(
        sha256(approved_plan_sha256) == plan["plan_sha256"],
        "Explicit approval must name this exact reviewed plan checksum",
    )
    require(
        confirm_billable is True
        and confirm_code_upload is True
        and confirm_licenses_reviewed is True,
        "Submission requires separate billable-job, code-upload and license-review confirmations",
    )
    require(version("azure-ai-ml") == AZURE_SDK_VERSION, "Install the locked Azure ML SDK")
    from azure.ai.ml import MLClient, load_job
    from azure.identity import AzureCliCredential

    config = plan["config"]
    job = load_job(source=root / "job.json")
    credential = AzureCliCredential(tenant_id=config["tenant_id"])
    client = MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace"],
    )
    preflight(client, config)
    # Recheck local artifacts immediately before the single authorized write/upload.
    require(
        read_plan(root)["plan_sha256"] == approved_plan_sha256,
        "Approval changed while running the Azure preflight",
    )
    result = client.jobs.create_or_update(job)
    require(isinstance(result.name, str) and bool(result.name), "Azure ML returned no job identity")
    return result.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline-first, explicitly scoped Azure ML v2 jobs."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--approve-plan-sha256")
    parser.add_argument("--confirm-billable", action="store_true")
    parser.add_argument("--confirm-code-upload", action="store_true")
    parser.add_argument("--confirm-licenses-reviewed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.submit:
            require(args.config is None, "Submit the reviewed plan, not a new config")
            require(args.approve_plan_sha256 is not None, "Missing explicit plan approval")
            name = submit(
                args.plan_dir,
                approved_plan_sha256=args.approve_plan_sha256,
                confirm_billable=args.confirm_billable,
                confirm_code_upload=args.confirm_code_upload,
                confirm_licenses_reviewed=args.confirm_licenses_reviewed,
            )
            print(json.dumps({"submitted_job": name}))
        else:
            require(args.config is not None, "Offline planning requires --config")
            require(
                not args.approve_plan_sha256
                and not args.confirm_billable
                and not args.confirm_code_upload
                and not args.confirm_licenses_reviewed,
                "Approval flags require --submit",
            )
            plan_sha = create_plan(read_json(args.config), args.plan_dir)
            print(
                json.dumps(
                    {
                        "submitted": False,
                        "plan_sha256": plan_sha,
                        "job": str(args.plan_dir / "job.json"),
                    }
                )
            )
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"Azure learning operation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
