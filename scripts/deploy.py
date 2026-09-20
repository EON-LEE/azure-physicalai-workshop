"""Explicit, staged Azure deployment. The default is an offline plan, never a write."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.cloud_build import reviewed_source

ROOT = Path(__file__).resolve().parents[1]
PROJECT_TAG = "azure-physicalai-workshop"


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    subscription_id: UUID
    tenant_id: UUID
    operator_object_id: UUID
    spa_client_id: UUID
    api_client_id: UUID
    resource_group: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,80}$")
    prefix: str = Field(pattern=r"^[a-z][a-z0-9-]{2,9}$")
    location: str = Field(pattern=r"^[a-z0-9]+$")
    foundry_location: str = Field(pattern=r"^[a-z0-9]+$")
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_deployment_name: str = Field(pattern=r"^[a-z][a-z0-9-]+$")
    model_sku: str = Field(min_length=1)
    model_capacity: int = Field(ge=1, le=10)
    runtime_profile: Literal["web", "physical"] = "physical"
    bootstrap_runner: Literal["container_instance", "container_app_job"] = "container_instance"
    app_environment_name: str | None = None
    app_subnet_name: Literal["apps", "apps-recovery"] = "apps"
    web_app_name: str | None = None
    source_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    enable_demonstration_capture: bool = False
    gpu_vm_size: str | None = Field(default=None, pattern=r"^Standard_[A-Za-z0-9_]+$")
    ssh_public_key_file: Path | None = None
    licensed_asset_archive: Path | None = None
    franka_usd_relative_path: str = "Franka/franka.usd"
    accept_nvidia_eula: bool = False
    licensed_assets_approved: bool = False
    acknowledged_hourly_budget_usd: float = Field(gt=0)
    shutdown_time_utc: str = Field(pattern=r"^(?:[01][0-9]|2[0-3])[0-5][0-9]$")

    @field_validator(
        "subscription_id", "tenant_id", "operator_object_id", "spa_client_id", "api_client_id"
    )
    @classmethod
    def nonzero_id(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("Replace placeholder IDs with explicitly authorized resources.")
        return value

    @field_validator("franka_usd_relative_path")
    @classmethod
    def relative_usd(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("USD path must be inside the licensed asset archive.")
        if path.suffix not in {".usd", ".usda", ".usdc"}:
            raise ValueError("A USD scene file is required.")
        return value


class AzureCLI:
    def __init__(self, subscription: UUID) -> None:
        self.subscription = str(subscription)

    def call(self, *arguments: str, expect_json: bool = True):
        result = subprocess.run(
            [
                "az",
                *arguments,
                "--subscription",
                self.subscription,
                "--output",
                "json",
                "--only-show-errors",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"Azure CLI {' '.join(arguments[:3])} failed:\n{result.stderr}")
        if not expect_json:
            print(result.stdout, end="")
            return None
        if not result.stdout.strip():
            raise RuntimeError(f"Azure CLI {' '.join(arguments[:3])} returned no JSON result.")
        return json.loads(result.stdout)

    def template(self, name: str, resource_group: str, file: str, values: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="physicalai-deploy-") as directory:
            parameters = Path(directory) / "parameters.json"
            parameters.write_text(
                json.dumps(
                    {
                        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
                        "contentVersion": "1.0.0.0",
                        "parameters": {key: {"value": value} for key, value in values.items()},
                    }
                ),
                encoding="utf-8",
            )
            result = self.call(
                "deployment",
                "group",
                "create",
                "--name",
                name,
                "--resource-group",
                resource_group,
                "--template-file",
                str(ROOT / "infra" / file),
                "--parameters",
                f"@{parameters}",
            )
        return {key: value["value"] for key, value in result["properties"]["outputs"].items()}


def job_execution_status(record: dict) -> str:
    properties = record.get("properties", record)
    status = properties.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("Azure returned no job execution status; readiness is unknown.")
    return status


def preflight(config: Deployment) -> tuple[str | None, str | None]:
    if config.runtime_profile == "web":
        return None, None
    if not config.accept_nvidia_eula or not config.licensed_assets_approved:
        raise ValueError("NVIDIA terms and the right to upload the asset archive require approval.")
    if (
        config.gpu_vm_size is None
        or config.licensed_asset_archive is None
        or config.ssh_public_key_file is None
        or not config.licensed_asset_archive.is_file()
        or not config.ssh_public_key_file.is_file()
    ):
        raise ValueError("The approved asset archive and public SSH key files must exist.")
    if config.licensed_asset_archive.stat().st_size > 4 * 1024**3:
        raise ValueError("Asset archive exceeds 4 GiB.")
    public_key = config.ssh_public_key_file.read_text(encoding="utf-8").strip()
    if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
        raise ValueError("Supply a public SSH key, never a private key.")
    if config.source_commit is None:
        raise ValueError("Physical deployment requires an explicit full source commit.")
    with config.licensed_asset_archive.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    return public_key, checksum


def verify_destination(cli: AzureCLI, config: Deployment) -> str:
    account = cli.call("account", "show")
    if account["id"] != str(config.subscription_id) or account["tenantId"] != str(config.tenant_id):
        raise ValueError(
            "The selected subscription/tenant does not match the approved configuration."
        )
    api = cli.call("ad", "app", "show", "--id", str(config.api_client_id))
    if api.get("api", {}).get("requestedAccessTokenVersion") != 2:
        raise ValueError("The API app registration must issue v2 access tokens.")
    scopes = api.get("api", {}).get("oauth2PermissionScopes", [])
    if not any(
        scope.get("value") == "access_as_user" and scope.get("isEnabled") for scope in scopes
    ):
        raise ValueError("The API app registration must expose enabled access_as_user permission.")
    if f"api://{config.api_client_id}" not in api.get("identifierUris", []):
        raise ValueError("The API identifier URI does not match the configured client ID.")
    roles = cli.call("role", "definition", "list", "--name", "Foundry User")
    if len(roles) != 1:
        raise ValueError(
            "The current Foundry User role must be available; no broader fallback is used."
        )
    return roles[0]["name"]


def deploy(config: Deployment) -> dict:
    public_key, asset_hash = preflight(config)
    cli = AzureCLI(config.subscription_id)
    role_id = verify_destination(cli, config)
    if cli.call("group", "exists", "--name", config.resource_group):
        group = cli.call("group", "show", "--name", config.resource_group)
        tags = group.get("tags") or {}
        if tags.get("project") != PROJECT_TAG or tags.get("physicalaiEnvironment") != config.prefix:
            raise ValueError(
                "Refusing to deploy into a resource group not owned by this environment."
            )
    else:
        cli.call(
            "group",
            "create",
            "--name",
            config.resource_group,
            "--location",
            config.location,
            "--tags",
            f"project={PROJECT_TAG}",
            f"physicalaiEnvironment={config.prefix}",
        )
    revision = uuid4().hex
    foundation = cli.template(
        "physicalai-foundation",
        config.resource_group,
        "foundation.bicep",
        {
            "prefix": config.prefix,
            "location": config.location,
            "foundryLocation": config.foundry_location,
            "modelName": config.model_name,
            "modelVersion": config.model_version,
            "modelDeploymentName": config.model_deployment_name,
            "modelSkuName": config.model_sku,
            "modelCapacity": config.model_capacity,
            "foundryUserRoleDefinitionId": role_id,
            "operatorObjectId": str(config.operator_object_id),
            "deployGpuNetwork": config.runtime_profile == "physical",
            "appEnvironmentName": config.app_environment_name or f"{config.prefix}-apps",
            "appSubnetName": config.app_subnet_name,
        },
    )
    if config.runtime_profile == "physical":
        cli.call(
            "storage",
            "blob",
            "upload",
            "--account-name",
            foundation["storageName"],
            "--container-name",
            "artifacts",
            "--name",
            "assets/franka.tar",
            "--file",
            str(config.licensed_asset_archive.resolve()),
            "--auth-mode",
            "login",
            "--overwrite",
            "true",
        )
    images = {}
    builds = [("api", "Dockerfile")]
    if config.runtime_profile == "physical":
        builds.append(("simulator", "simulation/Dockerfile"))
    for name, dockerfile in builds:
        tag = f"{name}:{revision}"
        with reviewed_source(ROOT) as (source, source_sha256):
            cli.call(
                "acr",
                "build",
                "--registry",
                foundation["registryName"],
                "--image",
                tag,
                "--file",
                dockerfile,
                str(source),
                expect_json=False,
            )
        manifest = cli.call(
            "acr",
            "repository",
            "show",
            "--name",
            foundation["registryName"],
            "--image",
            tag,
        )
        digest = manifest["digest"]
        if re.fullmatch(r"sha256:[a-f0-9]{64}", digest) is None:
            raise ValueError("ACR did not return an immutable image digest.")
        images[name] = f"{foundation['registryServer']}/{name}@{digest}"
    runner_template = (
        "bootstrap-aci.bicep"
        if config.bootstrap_runner == "container_instance"
        else "bootstrap.bicep"
    )
    bootstrap = cli.template(
        "physicalai-bootstrap",
        config.resource_group,
        runner_template,
        {
            "prefix": config.prefix,
            "location": config.location,
            "foundation": foundation,
            "apiImage": images["api"],
        },
    )
    execution = None
    if config.bootstrap_runner == "container_app_job":
        execution = cli.call(
            "containerapp",
            "job",
            "start",
            "--name",
            bootstrap["jobName"],
            "--resource-group",
            config.resource_group,
        )
    deadline = time.monotonic() + 900
    while True:
        if config.bootstrap_runner == "container_instance":
            current = cli.call(
                "container",
                "show",
                "--name",
                bootstrap["containerGroupName"],
                "--resource-group",
                config.resource_group,
            )
            status = current.get("instanceView", {}).get("state")
            if status == "Succeeded":
                states = [
                    container.get("instanceView", {}).get("currentState", {})
                    for container in current["containers"]
                ]
                if not states or any(state.get("exitCode") != 0 for state in states):
                    raise RuntimeError("Bootstrap container lacks a confirmed successful exit.")
        else:
            current = cli.call(
                "containerapp",
                "job",
                "execution",
                "show",
                "--name",
                bootstrap["jobName"],
                "--job-execution-name",
                execution["name"],
                "--resource-group",
                config.resource_group,
            )
            status = job_execution_status(current)
        if status == "Succeeded":
            break
        if status in {"Failed", "Stopped", "Degraded"} or time.monotonic() >= deadline:
            raise RuntimeError(
                f"Azure bootstrap did not succeed: {status}. Inspect its execution logs."
            )
        time.sleep(10)
    runtime_parameters = {
        "prefix": config.prefix,
        "location": config.location,
        "foundation": foundation,
        "apiImage": images["api"],
        "entraTenantId": str(config.tenant_id),
        "entraSpaClientId": str(config.spa_client_id),
        "entraApiClientId": str(config.api_client_id),
        "deploymentRevision": revision,
    }
    if config.runtime_profile == "physical":
        runtime_parameters.update(
            {
                "simulatorImage": images["simulator"],
                "gpuVmSize": config.gpu_vm_size,
                "sshPublicKey": public_key,
                "assetSha256": asset_hash,
                "frankaUsdRelativePath": config.franka_usd_relative_path,
                "acceptNvidiaEula": config.accept_nvidia_eula,
                "shutdownTimeUtc": config.shutdown_time_utc,
                "sourceRevision": config.source_commit,
                "enableDemonstrationCapture": config.enable_demonstration_capture,
            }
        )
    else:
        runtime_parameters["appName"] = config.web_app_name or f"{config.prefix}-web"
    runtime = cli.template(
        "physicalai-runtime",
        config.resource_group,
        "runtime.bicep" if config.runtime_profile == "physical" else "web.bicep",
        runtime_parameters,
    )
    return {
        "subscription_id": str(config.subscription_id),
        "resource_group": config.resource_group,
        "revision": revision,
        "images": images,
        "source_sha256": source_sha256,
        "runtime": runtime,
        "live_verified": False,
        "runtime_profile": config.runtime_profile,
        "gpu_deployed": config.runtime_profile == "physical",
        "next_gate": (
            "Configure SPA redirect URI, then run actual Entra/Foundry/Isaac acceptance tests."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--apply", action="store_true", help="Authorize billable Azure writes.")
    args = parser.parse_args()
    config = Deployment.model_validate_json(args.configuration.read_text(encoding="utf-8"))
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "offline_plan",
                    "azure_resources_changed": False,
                    "subscription_id": str(config.subscription_id),
                    "resource_group": config.resource_group,
                    "stages": [
                        "foundation",
                        "licensed Azure assets",
                        "ACR builds",
                        "Azure bootstrap job",
                        "private GPU and web",
                    ],
                    "warning": (
                        "Budget acknowledgment is not a hard spend cap. "
                        "--apply creates billable resources."
                    ),
                },
                indent=2,
            )
        )
        return
    result = deploy(config)
    output = ROOT / "test-results" / f"deployment-{result['revision']}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
