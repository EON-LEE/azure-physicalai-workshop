from __future__ import annotations

import argparse
import copy
import re
import shutil
from pathlib import Path

from learning.azure import (
    CONFIG_KEYS,
    SNAPSHOT_FILES,
    _uuid,
    datastore_prefix,
    registered_input_matches,
    validate_compute,
    validate_compute_identity,
    validate_managed_network_dependencies,
    validate_retention,
)
from learning.common import (
    canonical,
    digest,
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
from learning.gr00t import POLICY_TYPE
from learning.gr00t.artifacts import UPSTREAM
from learning.gr00t.licensing import require_commercial_model
from learning.gr00t.train import Gr00tTrainOptions, azure_job_identity
from learning.offline import OFFLINE_ENV

CONFIG_SCHEMA = "physicalai.gr00t-azure/v1"
PLAN_SCHEMA = "physicalai.gr00t-azure-plan/v1"
CODE_FILES = SNAPSHOT_FILES + (
    "learning/gr00t/__init__.py",
    "learning/gr00t/dataset.py",
    "learning/gr00t/franka_modality.py",
    "learning/gr00t/artifacts.py",
    "learning/gr00t/source.py",
    "learning/gr00t/inference.py",
    "learning/gr00t/ipc.py",
    "learning/gr00t/train.py",
    "learning/gr00t/azure.py",
    "learning/gr00t/evaluation.py",
    "learning/gr00t/components.py",
    "learning/gr00t/pyproject.toml",
    "learning/gr00t/uv.lock",
    "learning/gr00t/bootstrap.py",
    "learning/gr00t/probe.py",
    "learning/gr00t/prepare.py",
    "learning/gr00t/Dockerfile",
    "learning/gr00t/licensing.py",
    "learning/gr00t/n17.py",
)
STATES = {
    "NotStarted": "submitted",
    "Starting": "submitted",
    "Queued": "submitted",
    "Preparing": "submitted",
    "Provisioning": "submitted",
    "Running": "running",
    "Finalizing": "running",
    "Completed": "succeeded",
    "Failed": "failed",
    "Canceled": "cancelled",
    "CancelRequested": "cancelling",
    "NotResponding": "failed",
    "Paused": "failed",
    "Unknown": "failed",
}


def validate_config(
    config: dict,
    *,
    schema: str = CONFIG_SCHEMA,
    upstream: dict = UPSTREAM,
    input_types: dict | None = None,
    train_options_type=Gr00tTrainOptions,
) -> None:
    keys(
        config,
        CONFIG_KEYS
        | {
            "compute_tier",
            "upstream",
            "specification_sha256",
            "control_profile_sha256",
            "task_sha256",
        },
        "GR00T Azure config",
    )
    require(
        config["schema"] == schema and config["kind"] in ("train", "compare", "bootstrap_compare"),
        "Unsupported GR00T Azure job kind",
    )
    Scope(config["tenant_id"], config["owner_id"]).validate()
    _uuid(config["subscription_id"], "subscription")
    _uuid(config["managed_identity_client_id"], "job managed identity")
    require(config["upstream"] == upstream, "Unpinned policy source/model")
    for name in ("specification_sha256", "control_profile_sha256", "task_sha256"):
        sha256(config[name], name)
    for name in (
        "resource_group",
        "workspace",
        "compute",
        "datastore",
        "run_id",
        "model_name",
        "model_version",
        "storage_account_name",
        "storage_account_resource_group",
        "blob_container",
    ):
        token(config[name], name)
    require(config["model_version"].lower() != "latest", "Model version must be immutable")
    require(
        config["compute_tier"] in ("Dedicated", "LowPriority"), "Explicit compute tier required"
    )
    require(
        re.fullmatch(r"Standard_N[CDV][A-Za-z0-9_]+", config["compute_size"]),
        "Approved GPU SKU required",
    )
    require(
        re.fullmatch(
            re.escape(f"/subscriptions/{config['subscription_id']}/resourceGroups/")
            + r"[A-Za-z0-9_.-]+/providers/Microsoft\.ManagedIdentity/"
            r"userAssignedIdentities/[A-Za-z0-9_.-]+",
            config["managed_identity_resource_id"],
        ),
        "Managed identity is outside the approved subscription",
    )
    require(
        re.fullmatch(
            r"[a-z0-9]{5,50}\.azurecr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}",
            config["environment_image"],
        ),
        "An approved digest-pinned private ACR image is required",
    )
    relative_path(config["environment_image"].split(".azurecr.io/")[1].split("@")[0])
    from learning.common import integer

    integer(config["retention_days"], "retention days", 1, 365)
    prefix = f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/"
    relative_path(config["output_prefix"])
    require(config["output_prefix"].startswith(prefix), "Output escapes tenant/owner scope")
    if config["kind"] == "train":
        expected = {"demonstrations": "uri_folder", "parent_model": "uri_folder"}
    elif config["kind"] == "compare":
        expected = {
            "policy_before": "uri_folder",
            "policy_after": "uri_folder",
            "evidence": "uri_folder",
            "plan": "uri_file",
        }
    else:
        expected = {"candidate": "uri_folder", "evidence": "uri_folder", "plan": "uri_file"}
    if input_types is not None:
        expected = input_types
    keys(config["inputs"], set(expected), "GR00T data inputs")
    for name, input_type in expected.items():
        asset = keys(config["inputs"][name], {"name", "version", "uri", "sha256", "type"}, name)
        token(asset["name"], "data asset")
        token(asset["version"], "data version")
        require(asset["version"].lower() != "latest", "Mutable data version")
        require(asset["type"] == input_type, "Wrong data input type")
        sha256(asset["sha256"], "input manifest checksum")
        require(asset["uri"].startswith(datastore_prefix(config)), "Unapproved data location")
        path = asset["uri"][len(datastore_prefix(config)) :]
        relative_path(path)
        require(path.startswith(prefix), "Input escapes tenant/owner scope")
    if config["kind"] == "train":
        options = train_options_type(
            **keys(
                config["parameters"],
                set(train_options_type.__dataclass_fields__),
                "GR00T parameters",
            )
        )
        options.validate()
        require(
            options.timeout_seconds >= 2,
            "Conversion plus training needs a positive shared time budget",
        )
        require(options.compute_tier == config["compute_tier"], "Training/compute tier mismatch")
    else:
        keys(config["parameters"], {"timeout_seconds"}, "comparison parameters")
        integer(config["parameters"]["timeout_seconds"], "comparison timeout", 1, 3600)


def workspace_id(config: dict) -> str:
    return (
        f"/subscriptions/{config['subscription_id']}/resourceGroups/{config['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{config['workspace']}"
    )


def running_job_binding(client, config: dict, *, snapshot_sha256: str | None = None) -> dict:
    """Bind a component output to its actual running Azure job and approved parent pipeline."""
    if "job_execution" in config:
        from learning.paused.command import running_command_binding

        return running_command_binding(client, config, snapshot_sha256=snapshot_sha256)
    component_id = azure_job_identity(config)
    component = client.jobs.get(component_id.rsplit("/", 1)[1])
    require(
        component.id.lower() == component_id.lower() and component.status == "Running",
        "Azure ML did not confirm this actual running component",
    )
    parent_name = getattr(component, "parent_job_name", None)
    token(parent_name, "actual parent pipeline name")
    parent = client.jobs.get(parent_name)
    require(
        parent.id.lower() == (workspace_id(config) + "/jobs/" + parent_name).lower()
        and all(
            (parent.tags or {}).get(key) == value
            for key, value in {
                "scope_owner": config["owner_id"],
                "scope_tenant": config["tenant_id"],
                "specification_sha256": config["specification_sha256"],
            }.items()
        ),
        "Component parent pipeline owner/specification mismatch",
    )
    if "job_deadline_utc" in config:
        expected = {
            "config_schema": config["schema"],
            "job_deadline_utc": config["job_deadline_utc"],
        }
        if "execution_timing" in config:
            expected.update(
                execution_timing=config["execution_timing"],
                real_time_admission="false",
                criteria_sha256=config["criteria_sha256"],
                frozen_plan_sha256=config["frozen_plan_sha256"],
            )
        if "source_delivery" in config:
            expected.update(
                source_delivery=config["source_delivery"]["mode"],
                static_source_sha256=config["source_delivery"]["static_sha256"],
            )
        require(
            all(
                (job.tags or {}).get(key) == value
                for job in (parent, component)
                for key, value in expected.items()
            ),
            "Actual component/parent job deadline differs from the approved config",
        )
    return {
        "azure_job_id": parent.id,
        "azure_component_job_id": component.id,
        "specification_sha256": config["specification_sha256"],
    }


def job_tags(config: dict, snapshot_sha256: str, *, policy_type: str = POLICY_TYPE) -> dict:
    tags = {
        "scope_tenant": config["tenant_id"],
        "scope_owner": config["owner_id"],
        "specification_sha256": config["specification_sha256"],
        "control_profile_sha256": config["control_profile_sha256"],
        "task_sha256": config["task_sha256"],
        "policy_type": policy_type,
        "code_snapshot_sha256": snapshot_sha256,
        "compute_tier": config["compute_tier"],
        "model_name": config["model_name"],
        "model_version": config["model_version"],
        "retention_days": str(config["retention_days"]),
        "quality_verified": "false",
    }
    if "job_deadline_utc" in config:
        tags.update(config_schema=config["schema"], job_deadline_utc=config["job_deadline_utc"])
    if "execution_timing" in config:
        tags.update(
            execution_timing=config["execution_timing"],
            real_time_admission="false",
            criteria_sha256=config["criteria_sha256"],
            frozen_plan_sha256=config["frozen_plan_sha256"],
        )
    if "source_delivery" in config:
        tags.update(
            source_delivery=config["source_delivery"]["mode"],
            static_source_sha256=config["source_delivery"]["static_sha256"],
        )
    if "job_execution" in config:
        tags.update(
            job_execution="command",
            data_transport="private_blob_mi",
            runtime_config_sha256=digest(canonical(config)),
        )
    return tags


def build_job(
    config: dict,
    snapshot_sha256: str,
    job_name: str,
    *,
    validator=validate_config,
    policy_type: str = POLICY_TYPE,
    command_module: str = "learning.gr00t.components",
    include_backbone: bool = False,
) -> dict:
    validator(config)
    sha256(snapshot_sha256)
    token(job_name, "deterministic job name")
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", job_name), "Invalid AML job name")
    output_prefix = datastore_prefix(config) + config["output_prefix"] + "/" + job_name
    tags = job_tags(config, snapshot_sha256, policy_type=policy_type)

    def step(command: str, inputs: dict, outputs: dict, timeout: int) -> dict:
        return {
            "type": "command",
            "code": "./code",
            "command": command,
            "environment": {"image": config["environment_image"]},
            "environment_variables": {**OFFLINE_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
            "identity": {"type": "managed", "client_id": config["managed_identity_client_id"]},
            "resources": {"instance_count": 1, "shm_size": "8g"},
            "compute": f"azureml:{config['compute']}",
            "limits": {"timeout": timeout},
            "inputs": inputs,
            "outputs": outputs,
            "tags": tags,
        }

    require(
        command_module
        in (
            "learning.gr00t.components",
            "learning.smolvla.components",
            "learning.paused.components",
        ),
        "Arbitrary job entry points are forbidden",
    )
    base_command = f"python -m {command_module}"
    config_arg = f" --runtime-config run-config.json --snapshot-sha256 {snapshot_sha256}"
    if config["kind"] == "train":
        total_timeout = config["parameters"]["timeout_seconds"]
        conversion_timeout = min(600, max(1, total_timeout // 5))
        training_timeout = total_timeout - conversion_timeout
        outputs = {
            name: {"type": "uri_folder", "mode": "rw_mount", "path": f"{output_prefix}/{name}"}
            for name in ("dataset", "model")
        }
        jobs = {
            "convert": step(
                base_command
                + " export --input '${{inputs.raw}}' --output '${{outputs.dataset}}'"
                + config_arg,
                {"raw": "${{parent.inputs.demonstrations}}"},
                {"dataset": "${{parent.outputs.dataset}}"},
                conversion_timeout,
            ),
            "train": step(
                base_command + " train --input '${{inputs.dataset}}'"
                " --parent '${{inputs.parent}}' --output '${{outputs.model}}'"
                + (" --backbone '${{inputs.backbone}}'" if include_backbone else "")
                + config_arg,
                {
                    "dataset": "${{parent.jobs.convert.outputs.dataset}}",
                    "parent": "${{parent.inputs.parent_model}}",
                    **({"backbone": "${{parent.inputs.backbone}}"} if include_backbone else {}),
                },
                {"model": "${{parent.outputs.model}}"},
                training_timeout,
            ),
        }
    elif config["kind"] == "compare":
        outputs = {
            "report": {"type": "uri_folder", "mode": "rw_mount", "path": output_prefix + "/report"}
        }
        jobs = {
            "compare": step(
                base_command
                + " compare --input '${{inputs.evidence}}' --parent '${{inputs.before}}'"
                " --after '${{inputs.after}}' --plan '${{inputs.plan}}'"
                " --output '${{outputs.report}}'" + config_arg,
                {
                    "evidence": "${{parent.inputs.evidence}}",
                    "before": "${{parent.inputs.policy_before}}",
                    "after": "${{parent.inputs.policy_after}}",
                    "plan": "${{parent.inputs.plan}}",
                },
                {"report": "${{parent.outputs.report}}"},
                config["parameters"]["timeout_seconds"],
            )
        }
    else:
        outputs = {
            "report": {"type": "uri_folder", "mode": "rw_mount", "path": output_prefix + "/report"}
        }
        jobs = {
            "bootstrap": step(
                base_command
                + " bootstrap --input '${{inputs.evidence}}' --parent '${{inputs.candidate}}'"
                " --plan '${{inputs.plan}}' --output '${{outputs.report}}'" + config_arg,
                {
                    "evidence": "${{parent.inputs.evidence}}",
                    "candidate": "${{parent.inputs.candidate}}",
                    "plan": "${{parent.inputs.plan}}",
                },
                {"report": "${{parent.outputs.report}}"},
                config["parameters"]["timeout_seconds"],
            )
        }
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json",
        "type": "pipeline",
        "name": job_name,
        "display_name": job_name,
        "experiment_name": f"physicalai-{policy_type}",
        "tags": tags,
        "settings": {
            "default_compute": f"azureml:{config['compute']}",
            "default_datastore": f"azureml:{config['datastore']}",
            "continue_on_step_failure": False,
            "force_rerun": True,
        },
        "inputs": {
            name: {
                "type": value["type"],
                "path": f"azureml:{value['name']}:{value['version']}",
                "mode": "ro_mount",
            }
            for name, value in config["inputs"].items()
        },
        "outputs": outputs,
        "jobs": jobs,
    }


def create_plan(
    config: dict,
    output: Path,
    *,
    deterministic_job_name: str,
    source_root: Path | None = None,
    validator=validate_config,
    job_builder=build_job,
    code_files=CODE_FILES,
    plan_schema=PLAN_SCHEMA,
) -> str:
    validator(config)
    require(not output.exists(), "Never overwrite an approved GR00T plan")
    root = source_root or Path(__file__).resolve().parents[2]
    code = output / "code"
    code.mkdir(parents=True)
    for name in code_files:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(root, name), target)
    write_json(code / "run-config.json", config)
    files = inventory(code)
    snapshot_sha = digest(canonical(files))
    write_json(code / "snapshot.json", {"files": files, "sha256": snapshot_sha})
    job = job_builder(config, snapshot_sha, deterministic_job_name)
    payload = {
        "schema": plan_schema,
        "config": config,
        "job_name": deterministic_job_name,
        "snapshot_sha256": snapshot_sha,
        "job_sha256": digest(canonical(job)),
    }
    plan_sha = digest(canonical(payload))
    write_json(output / "job.json", job)
    write_json(output / "plan.json", {**payload, "plan_sha256": plan_sha})
    return plan_sha


def verify_code(code: Path, expected_sha256: str, *, code_files=CODE_FILES) -> None:
    value = keys(read_json(code / "snapshot.json"), {"files", "sha256"}, "GR00T code snapshot")
    require(
        value["sha256"] == sha256(expected_sha256)
        and digest(canonical(value["files"])) == expected_sha256
        and set(value["files"]) == set(code_files) | {"run-config.json"},
        "Unexpected/modified GR00T code snapshot",
    )
    verify_inventory(code, value["files"], exclude={"snapshot.json"})


def read_plan(
    path: Path,
    *,
    validator=validate_config,
    job_builder=build_job,
    code_files=CODE_FILES,
    plan_schema=PLAN_SCHEMA,
) -> dict:
    value = keys(
        read_json(path / "plan.json"),
        {
            "schema",
            "config",
            "job_name",
            "snapshot_sha256",
            "job_sha256",
            "plan_sha256",
        },
        "GR00T submission plan",
    )
    require(value["schema"] == plan_schema, "Wrong policy submission schema")
    require(
        digest(canonical({key: item for key, item in value.items() if key != "plan_sha256"}))
        == sha256(value["plan_sha256"]),
        "Reviewed GR00T plan changed",
    )
    validator(value["config"])
    verify_code(path / "code", value["snapshot_sha256"], code_files=code_files)
    require(
        read_json(path / "code" / "run-config.json") == value["config"], "Runtime config differs"
    )
    require(
        read_json(path / "job.json")
        == job_builder(value["config"], value["snapshot_sha256"], value["job_name"])
        and digest(canonical(read_json(path / "job.json"))) == value["job_sha256"],
        "Reviewed AML job changed",
    )
    return value


class Gr00tJobs:
    """Trusted worker SDK adapter. The API owns the durable paid-job claim, never this process."""

    policy_type = POLICY_TYPE
    config_validator = staticmethod(validate_config)
    plan_reader = staticmethod(read_plan)

    def __init__(self, client, config: dict, *, storage_client) -> None:
        self.config_validator(config)
        self.client, self.config, self.storage_client = (
            client,
            copy.deepcopy(config),
            storage_client,
        )

    def check_license(self) -> None:
        require_commercial_model(POLICY_TYPE, self.config["upstream"]["model_revision"])

    def check_admission(self) -> None:
        self.check_license()

    def preflight(self) -> None:
        self.check_admission()
        config = self.config
        workspace = self.client.workspaces.get(config["workspace"])
        require(
            workspace.id.lower() == workspace_id(config).lower(),
            "Wrong explicitly scoped AML workspace",
        )
        require(
            workspace.managed_network is not None
            and workspace.managed_network.isolation_mode.replace("_", "").lower()
            == "allowonlyapprovedoutbound"
            and str(
                getattr(workspace.public_network_access, "value", workspace.public_network_access)
            ).lower()
            == "disabled",
            "GR00T requires private approved-outbound-only AML networking",
        )
        require(
            str(workspace.system_datastores_auth_mode).lower() == "identity",
            "AML default system datastores must remain identity-only",
        )
        workspace.managed_network.outbound_rules = list(
            self.client.workspace_outbound_rules.list(workspace_name=config["workspace"])
        )
        validate_managed_network_dependencies(workspace)
        compute = self.client.compute.get(config["compute"])
        validate_compute(compute, config)
        validate_compute_identity(self.client, compute, config)
        datastore = self.client.datastores.get(config["datastore"])
        credential_type = getattr(getattr(datastore.credentials, "type", None), "value", None)
        require(
            getattr(datastore.type, "value", datastore.type) == "AzureBlob"
            and datastore.account_name == config["storage_account_name"]
            and datastore.container_name == config["blob_container"]
            and credential_type == "None",
            "Private identity-based approved datastore is required",
        )
        for asset in config["inputs"].values():
            data = self.client.data.get(asset["name"], version=asset["version"])
            require(
                registered_input_matches(data.type, data.path, asset),
                "Data version/location changed",
            )
        policy = self.storage_client.management_policies.get(
            config["storage_account_resource_group"],
            config["storage_account_name"],
            "default",
        )
        serialized = policy.serialize()
        require(
            isinstance(serialized.get("properties"), dict),
            "Missing actual storage lifecycle properties",
        )
        validate_retention(serialized["properties"], config)

    def _owned(self, job) -> None:
        token(job.name, "actual Azure job name")
        require(
            isinstance(job.id, str)
            and job.id.lower() == (workspace_id(self.config) + "/jobs/" + job.name).lower(),
            "Actual Azure job ID does not belong to the explicitly scoped workspace",
        )
        expected = {
            "scope_owner": self.config["owner_id"],
            "scope_tenant": self.config["tenant_id"],
            "specification_sha256": self.config["specification_sha256"],
            "policy_type": self.policy_type,
        }
        require(
            all((job.tags or {}).get(key) == value for key, value in expected.items()),
            "Named job owner/specification does not match this request",
        )
        require(
            (job.tags or {}).get("job_deadline_utc") == self.config.get("job_deadline_utc")
            and (
                "job_deadline_utc" not in self.config
                or (job.tags or {}).get("config_schema") == self.config["schema"]
            ),
            "Named job deadline does not match this approved config",
        )
        mode_fields = ("execution_timing", "criteria_sha256", "frozen_plan_sha256")
        require(
            all((job.tags or {}).get(key) == self.config.get(key) for key in mode_fields)
            and (job.tags or {}).get("real_time_admission")
            == ("false" if "execution_timing" in self.config else None),
            "Named job mode/criteria/conditions differs from its approved config",
        )
        delivery = self.config.get("source_delivery", {})
        require(
            (job.tags or {}).get("source_delivery") == delivery.get("mode")
            and (job.tags or {}).get("static_source_sha256") == delivery.get("static_sha256"),
            "Named job source delivery differs from its approved config",
        )
        if "job_execution" in self.config:
            from learning.paused.command import validate_job

            validate_job(job, self.config)
        else:
            require(
                not (job.tags or {}).get("job_execution"),
                "A command job cannot be relabelled as pipeline",
            )

    def status(self, job_name: str) -> dict:
        token(job_name, "job name")
        job = self.client.jobs.get(job_name)
        self._owned(job)
        require(job.status in STATES, f"Unknown Azure job status: {job.status}")
        receipt = {
            "job_name": job.name,
            "azure_job_id": job.id,
            "owner_key": self.config["owner_id"],
            "specification_sha256": self.config["specification_sha256"],
            "status": STATES[job.status],
            "azure_status": job.status,
            "optimizer_steps": None,
            "loss": None,
            "quality_verified": False,
        }
        if "job_execution" in self.config:
            receipt["azure_job_type"] = "command"
        return receipt

    def submit(
        self, plan_dir: Path, *, approved_plan_sha256: str, deterministic_job_name: str
    ) -> dict:
        self.check_admission()
        plan = self.plan_reader(plan_dir)
        require(
            plan["plan_sha256"] == sha256(approved_plan_sha256)
            and plan["config"] == self.config
            and plan["job_name"] == deterministic_job_name,
            "Named paid job does not match the exact approved plan",
        )
        from azure.ai.ml import load_job
        from azure.core.exceptions import ResourceNotFoundError

        try:
            existing = self.client.jobs.get(deterministic_job_name)
        except ResourceNotFoundError:
            existing = None
        if existing is not None:
            self._owned(existing)
            require(
                existing.tags.get("plan_sha256") == approved_plan_sha256,
                "Existing job uses another plan",
            )
            return self.status(deterministic_job_name)
        self.preflight()
        require(
            self.plan_reader(plan_dir)["plan_sha256"] == approved_plan_sha256,
            "Plan changed during preflight",
        )
        job = load_job(source=plan_dir / "job.json")
        job.tags["plan_sha256"] = approved_plan_sha256
        self.check_admission()
        self.client.jobs.create_or_update(job)
        return self.status(deterministic_job_name)

    def cancel(self, job_name: str) -> dict:
        current = self.status(job_name)
        requested = current["azure_status"] not in (
            "Completed",
            "Failed",
            "Canceled",
            "CancelRequested",
        )
        if requested:
            self.client.jobs.begin_cancel(job_name, polling=False, retry_total=0)
        return {**self.status(job_name), "cancellation_requested": requested}


def clients_for_managed_identity(config: dict, *, caller_client_id: str, validator=validate_config):
    from azure.ai.ml import MLClient
    from azure.identity import ManagedIdentityCredential
    from azure.mgmt.storage import StorageManagementClient

    validator(config)
    _uuid(caller_client_id, "worker managed identity")
    credential = ManagedIdentityCredential(client_id=caller_client_id)
    client = MLClient(
        credential, config["subscription_id"], config["resource_group"], config["workspace"]
    )
    storage = StorageManagementClient(credential, config["subscription_id"])
    return client, storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local GR00T job plan; no cloud calls.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    print(create_plan(read_json(args.config), args.plan_dir, deterministic_job_name=args.job_name))


if __name__ == "__main__":
    main()
