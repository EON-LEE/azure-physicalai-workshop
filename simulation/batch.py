"""Optional Azure Batch submission surface; importing it never starts a simulator or job."""

from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode, urlsplit
from uuid import UUID

from pydantic import Field, field_validator, model_serializer, model_validator

from apps.api.models import Model, Revision
from learning.common import canonical, digest, parse_json, read_json, relative_path, require, sha256
from simulation.paused_profiles import PausedProfileId

BATCH_SDK_VERSION = "15.1.0"
BATCH_TOKEN_SCOPE = "https://batch.core.windows.net/.default"
GRID_DRIVER = "570.237"
GPU_NAME = "NVIDIA A10-24Q"
HOST_IMAGES = {
    "2204": ("22.04.2026082801", "batch.node.ubuntu 22.04"),
    "2404": ("24.04.2026092501", "batch.node.ubuntu 24.04"),
}
TASK_WALL_SECONDS = 900
WARMUP_WALL_SECONDS = 1800
GRID_INSTALLER_SHA256 = "1a529dd4d173ba3b36f3c284bb54fce5dd39c9e757856980b8ae0b7798e7764b"
PROOF_LIMITS = {
    "inputs/spec.json": 1024**2,
    **{
        f"inputs/{name}.json": 4 * 1024**2
        for name in ("environment", "grant", "criteria", "conditions")
    },
    "preflight.json": 65536,
    "probe.json": 8 * 1024**2,
    "probe.log": 64 * 1024**2,
    "acceptance.json": 65536,
    "acceptance.log": 1024**2,
    "raw-manifest.json": 4 * 1024**2,
}
CONTAINER_OPTIONS = (
    "--entrypoint /isaac-sim/python.sh --cap-drop ALL --security-opt no-new-privileges "
    "--shm-size 2g --tmpfs /data:rw,nosuid,nodev,mode=1777,size=2147483648 "
    "--tmpfs /isaac-sim/.cache:rw,nosuid,nodev,mode=1777,size=2147483648 "
    "--tmpfs /isaac-sim/.nv/ComputeCache:rw,nosuid,nodev,mode=1777,size=536870912 "
    "--tmpfs /isaac-sim/.nvidia-omniverse/logs:rw,nosuid,nodev,mode=1777,size=134217728"
)
PREFLIGHT_COMMAND = (
    "--signal=TERM --kill-after=5s 60s /isaac-sim/python.sh "
    "-m simulation.batch_task preflight --output preflight.json"
)
PREFLIGHT_CONTAINER_OPTIONS = CONTAINER_OPTIONS.replace(
    "--entrypoint /isaac-sim/python.sh", "--entrypoint /usr/bin/timeout", 1
)


class BlobInput(Model):
    name: str = Field(min_length=1, max_length=1024)
    sha256: Revision
    size_bytes: int = Field(gt=0, le=256 * 1024 * 1024, strict=True)

    @field_validator("name")
    @classmethod
    def safe_blob_name(cls, value):
        relative_path(value)
        return value


class BatchPlatform(Model):
    account_url: str = Field(pattern=r"^https://[a-z0-9-]+\.[a-z0-9-]+\.batch\.azure\.com$")
    pool_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    node_identity_resource_id: str = Field(
        pattern=r"^/subscriptions/[a-fA-F0-9-]{36}/resourceGroups/[A-Za-z0-9_.()-]+/"
        r"providers/Microsoft\.ManagedIdentity/userAssignedIdentities/[A-Za-z0-9_-]+$"
    )
    node_identity_client_id: UUID
    tenant_id: UUID
    vm_size: Literal["Standard_NV36ads_A10_v5"]
    image_publisher: Literal["microsoft-dsvm"]
    image_offer: Literal["ubuntu-hpc"]
    image_sku: Literal["2204", "2404"]
    image_version: Literal["22.04.2026082801", "24.04.2026092501"]
    driver_handler_version: Literal["1.14"] | None
    driver_version: Literal["570.237"] = GRID_DRIVER
    driver_installation: Literal["extension", "bootstrap"] = "extension"
    gpu_name: Literal["NVIDIA A10-24Q"] = GPU_NAME
    container_image: str = Field(
        pattern=r"^[a-z0-9]+\.azurecr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$"
    )

    @model_validator(mode="after")
    def exact_host_image(self):
        if self.image_version != HOST_IMAGES[self.image_sku][0]:
            raise ValueError("Host SKU and exact reviewed image version do not match.")
        if self.driver_installation == "bootstrap" and self.image_sku != "2204":
            raise ValueError("Managed driver bootstrap requires the reviewed Ubuntu 22.04 host.")
        if (self.driver_installation == "extension") != (self.driver_handler_version == "1.14"):
            raise ValueError("Only extension installation may declare a handler version.")
        return self

    @property
    def node_agent_sku(self) -> str:
        return HOST_IMAGES[self.image_sku][1]

    @model_serializer(mode="wrap")
    def preserve_extension_wire(self, handler):
        value = handler(self)
        if self.driver_installation == "extension":
            value.pop("driver_installation", None)
        return value


class BatchSimulationSpec(Model):
    schema_version: Literal["physicalai.batch-simulation/v1"] = Field(alias="schema")
    attempt_id: UUID
    previous_attempt_id: UUID | None = None
    owner_id: Revision
    platform: BatchPlatform
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    control_profile_sha256: Revision
    profile_id: PausedProfileId
    storage_account_url: str = Field(pattern=r"^https://[a-z0-9]{3,24}\.blob\.core\.windows\.net$")
    input_container: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
    output_container: Literal["demonstrations"] = "demonstrations"
    environment: BlobInput
    grant: BlobInput
    criteria: BlobInput
    conditions: BlobInput
    criteria_canonical_sha256: Revision
    conditions_canonical_sha256: Revision
    asset_bundle: BlobInput
    asset_root: str = Field(min_length=1, max_length=512)
    isaac_license_preapproved: Literal[True]

    @field_validator("asset_root")
    @classmethod
    def safe_asset_root(cls, value):
        relative_path(value)
        return value

    @model_validator(mode="after")
    def bounded_input_files(self):
        for name in ("environment", "grant", "criteria", "conditions"):
            if getattr(self, name).size_bytes > 4 * 1024 * 1024:
                raise ValueError("Operator input JSON exceeds the existing four-MiB boundary.")
        if self.previous_attempt_id == self.attempt_id:
            raise ValueError("A new approved attempt cannot resume its own physical scene.")
        return self

    @property
    def sha256(self) -> str:
        return digest(canonical(self.model_dump(mode="json", by_alias=True)))

    @property
    def job_id(self) -> str:
        return "sim-" + self.attempt_id.hex

    @property
    def task_id(self) -> str:
        return "episode"

    @property
    def artifact_prefix(self) -> str:
        return f"{self.owner_id}/managed-simulation/{self.attempt_id}"


class PublishedArtifact(Model):
    path: str
    sha256: Revision
    size_bytes: int = Field(alias="bytes", ge=0, strict=True)

    @model_validator(mode="after")
    def bounded_known_artifact(self):
        require(
            self.path in PROOF_LIMITS and self.size_bytes <= PROOF_LIMITS[self.path],
            "Unknown or oversized terminal proof artifact.",
        )
        return self


def validate_completion(spec: BatchSimulationSpec, proof: dict) -> list[PublishedArtifact]:
    require(
        proof.get("schema") == "physicalai.batch-simulation-result/v1"
        and proof.get("attempt_id") == str(spec.attempt_id)
        and proof.get("spec_sha256") == spec.sha256
        and proof.get("job_id") == spec.job_id
        and proof.get("task_id") == spec.task_id
        and proof.get("source_revision") == spec.source_revision
        and proof.get("image") == spec.platform.container_image
        and proof.get("control_profile_sha256") == spec.control_profile_sha256
        and proof.get("profile_id") == spec.profile_id
        and proof.get("previous_attempt_id")
        == (str(spec.previous_attempt_id) if spec.previous_attempt_id else None)
        and proof.get("learning_quality_proven") is False,
        "Terminal proof does not match the immutable managed attempt binding.",
    )
    require(
        type(proof.get("accepted")) is bool
        and proof.get("outcome") in {"accepted", "failed", "incomplete"}
        and proof.get("native_acceptance") in {"accepted", "rejected", "missing"}
        and (proof["accepted"] == (proof["outcome"] == "accepted"))
        and (proof["accepted"] == (proof["native_acceptance"] == "accepted")),
        "Terminal proof has contradictory acceptance fields.",
    )
    raw = proof.get("artifacts")
    require(
        isinstance(raw, list) and 1 <= len(raw) <= len(PROOF_LIMITS), "Missing proof artifacts."
    )
    artifacts = [PublishedArtifact.model_validate(value) for value in raw]
    names = {artifact.path for artifact in artifacts}
    require(
        len(names) == len(artifacts) and "inputs/spec.json" in names,
        "Duplicate or missing immutable specification proof artifact.",
    )
    sha256(proof.get("spec_file_sha256"), "terminal specification-file proof")
    if proof["accepted"]:
        require(set(PROOF_LIMITS) <= names, "Accepted task lacks complete native proof artifacts.")
        require(
            proof.get("physical_status") == "succeeded" and proof.get("capture_status") == "ready",
            "Accepted terminal proof does not represent a complete physical task and capture.",
        )
    return artifacts


def build_job_task(spec: BatchSimulationSpec, spec_url: str, spec_sha256: str):
    require(version("azure-batch") == BATCH_SDK_VERSION, "Install the pinned optional Batch SDK.")
    from azure.batch import models

    parsed = urlsplit(spec_url)
    require(
        spec_url.startswith(f"{spec.storage_account_url}/{spec.input_container}/")
        and not parsed.query
        and not parsed.fragment
        and not parsed.username,
        "The task specification must use the approved private Blob account without SAS.",
    )
    relative_path(parsed.path.lstrip("/"))
    sha256(spec_sha256, "exact specification-file checksum")
    job = models.BatchJobCreateOptions(
        id=spec.job_id,
        display_name="One approved Physical AI episode",
        pool_info=models.BatchPoolInfo(pool_id=spec.platform.pool_id),
        constraints=models.BatchJobConstraints(
            max_wall_clock_time=timedelta(seconds=TASK_WALL_SECONDS), max_task_retry_count=0
        ),
        all_tasks_complete_mode=models.BatchAllTasksCompleteMode.NO_ACTION,
        metadata=[
            models.BatchMetadataItem(name="physicalai_spec_sha256", value=spec.sha256),
            models.BatchMetadataItem(name="physicalai_attempt_id", value=str(spec.attempt_id)),
        ],
    )
    task = models.BatchTaskCreateOptions(
        id=spec.task_id,
        display_name="physicalai-" + spec.sha256[:40],
        command_line=(
            f"-m simulation.batch_task run --spec attempt.json --spec-sha256 {spec_sha256}"
        ),
        required_slots=1,
        constraints=models.BatchTaskConstraints(
            max_wall_clock_time=timedelta(seconds=TASK_WALL_SECONDS),
            max_task_retry_count=0,
            retention_time=timedelta(days=1),
        ),
        container_settings=models.BatchTaskContainerSettings(
            image_name=spec.platform.container_image,
            container_run_options=CONTAINER_OPTIONS,
            working_directory="taskWorkingDirectory",
        ),
        user_identity=models.UserIdentity(
            auto_user=models.AutoUserSpecification(scope="task", elevation_level="nonadmin")
        ),
        resource_files=[
            models.ResourceFile(
                http_url=spec_url,
                file_path="attempt.json",
                file_mode="0400",
                identity_reference=models.BatchNodeIdentityReference(
                    resource_id=spec.platform.node_identity_resource_id
                ),
            )
        ],
        environment_settings=[
            models.EnvironmentSetting(name="PYTHONPATH", value="/app"),
            models.EnvironmentSetting(name="NVIDIA_DRIVER_CAPABILITIES", value="all"),
            models.EnvironmentSetting(
                name="AZURE_CLIENT_ID", value=str(spec.platform.node_identity_client_id)
            ),
        ],
    )
    return job, task


def build_warmup(platform: BatchPlatform, warmup_id: UUID):
    """Queue only bounded non-motion work so autoscale can prepare a node before grant issuance."""
    require(version("azure-batch") == BATCH_SDK_VERSION, "Install the pinned optional Batch SDK.")
    from azure.batch import models

    binding = digest(canonical(platform.model_dump(mode="json")))
    job = models.BatchJobCreateOptions(
        id="warm-" + warmup_id.hex,
        display_name="GPU readiness only; no physics episode",
        pool_info=models.BatchPoolInfo(pool_id=platform.pool_id),
        constraints=models.BatchJobConstraints(
            max_wall_clock_time=timedelta(seconds=WARMUP_WALL_SECONDS), max_task_retry_count=0
        ),
        all_tasks_complete_mode=models.BatchAllTasksCompleteMode.NO_ACTION,
        metadata=[models.BatchMetadataItem(name="physicalai_platform_sha256", value=binding)],
    )
    task = models.BatchTaskCreateOptions(
        id="preflight",
        command_line="-m simulation.batch_task preflight --output preflight.json",
        required_slots=1,
        constraints=models.BatchTaskConstraints(
            max_wall_clock_time=timedelta(seconds=60),
            max_task_retry_count=0,
            retention_time=timedelta(days=1),
        ),
        container_settings=models.BatchTaskContainerSettings(
            image_name=platform.container_image,
            container_run_options=CONTAINER_OPTIONS,
            working_directory="taskWorkingDirectory",
        ),
        user_identity=models.UserIdentity(
            auto_user=models.AutoUserSpecification(scope="task", elevation_level="nonadmin")
        ),
        environment_settings=[
            models.EnvironmentSetting(name="PYTHONPATH", value="/app"),
            models.EnvironmentSetting(name="NVIDIA_DRIVER_CAPABILITIES", value="all"),
        ],
    )
    return job, task


def _same_request(observed, expected, *, job: bool = False) -> None:
    actual, requested = observed.as_dict(), expected.as_dict()
    for wire_name, property_name in (
        ("maxWallClockTime", "max_wall_clock_time"),
        ("retentionTime", "retention_time"),
    ):
        if wire_name in requested.get("constraints", {}):
            require(
                observed.constraints is not None
                and getattr(observed.constraints, property_name)
                == getattr(expected.constraints, property_name),
                "Existing managed job/task has a different immutable duration binding.",
            )
            actual["constraints"][wire_name] = requested["constraints"][wire_name]
    if job:
        require(
            actual.get("onAllTasksComplete") in {"noaction", "terminatejob"},
            "Existing managed job has an unexpected completion policy.",
        )
        actual["onAllTasksComplete"] = requested["onAllTasksComplete"]

    def contains(value, expected):
        if isinstance(expected, dict):
            return isinstance(value, dict) and all(
                contains(value.get(key), item) for key, item in expected.items()
            )
        return value == expected

    require(
        contains(actual, requested),
        "Existing managed job/task has a different immutable attempt binding; do not overwrite it.",
    )


def submit_once(client, spec: BatchSimulationSpec, spec_url: str, spec_sha256: str) -> dict:
    job, task = build_job_task(spec, spec_url, spec_sha256)
    result = submit_requests_once(client, job, task)
    return {**result, "spec_sha256": spec.sha256}


def submit_requests_once(client, job, task) -> dict:
    from azure.batch import models
    from azure.core import MatchConditions
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

    try:
        current = client.get_job(job.id)
    except ResourceNotFoundError:
        try:
            client.create_job(job=job)
        except ResourceExistsError:
            _same_request(client.get_job(job.id), job, job=True)
    else:
        _same_request(current, job, job=True)
    try:
        current = client.get_task(job.id, task.id)
    except ResourceNotFoundError:
        try:
            client.create_task(job.id, task=task)
        except ResourceExistsError:
            _same_request(client.get_task(job.id, task.id), task)
    else:
        _same_request(current, task)
    current = client.get_job(job.id)
    _same_request(current, job, job=True)
    if current.all_tasks_complete_mode != "terminatejob":
        require(current.etag is not None, "Cannot arm completion without the original job ETag.")
        client.update_job(
            job.id,
            job=models.BatchJobUpdateOptions(all_tasks_complete_mode="terminatejob"),
            etag=current.etag,
            match_condition=MatchConditions.IfNotModified,
        )
    return {"job_id": job.id, "task_id": task.id}


def allocation_formula(deadline: str) -> str:
    return (
        "$TargetDedicatedNodes = 0;\n"
        "$samples = $PendingTasks.GetSamplePercent(TimeInterval_Minute * 5);\n"
        "$tasks = $samples < 70 ? max(0, $PendingTasks.GetSample(1)) : "
        "max(0, max($PendingTasks.GetSample(TimeInterval_Minute * 5)));\n"
        f'$TargetLowPriorityNodes = time() < time("{deadline}") ? min(1, $tasks) : 0;\n'
        "$NodeDeallocationOption = terminate;"
    )


def bootstrap_source() -> bytes:
    return Path(__file__).with_name("batch_bootstrap.sh").read_text(encoding="utf-8").encode()


def driver_bootstrap_command(platform: BatchPlatform) -> str:
    source = bootstrap_source()
    checksum = digest(source)
    payload = base64.b64encode(source).decode()
    code = (
        "import base64,hashlib,os,subprocess;"
        f"data=base64.b64decode('{payload}');"
        f"assert hashlib.sha256(data).hexdigest()=='{checksum}';"
        f"os.environ['PHYSICALAI_BOOTSTRAP_SHA256']='{checksum}';"
        f"subprocess.run(['/bin/bash','-s','--','{platform.container_image}'],input=data,check=True)"
    )
    return (
        '/usr/bin/timeout --signal=TERM --kill-after=10s 900s /usr/bin/python3 -u -c "' + code + '"'
    )


def _management_account_id(platform: BatchPlatform) -> str:
    resource_group = "/".join(platform.node_identity_resource_id.split("/")[:5])
    account_name = urlsplit(platform.account_url).hostname.split(".")[0]
    return f"{resource_group}/providers/Microsoft.Batch/batchAccounts/{account_name}"


def validate_regional_capacity(platform: BatchPlatform, account: dict, catalog: dict) -> dict:
    hostname = urlsplit(platform.account_url).hostname
    region = hostname.split(".")[1]
    properties = account.get("properties", {})
    require(
        account.get("id", "").lower() == _management_account_id(platform).lower()
        and account.get("location", "").lower() == region
        and properties.get("accountEndpoint", "").lower() == hostname
        and properties.get("poolAllocationMode") == "BatchService"
        and properties.get("provisioningState") == "Succeeded",
        "Regional capacity evidence does not match the approved Batch service account.",
    )
    require(
        isinstance(catalog.get("value"), list) and not catalog.get("nextLink"),
        "A complete bounded regional SKU catalog is required.",
    )
    matching = [
        item
        for item in catalog["value"]
        if item.get("name", "").lower() == platform.vm_size.lower()
    ]
    require(
        len(matching) == 1, "The exact GPU SKU is absent or duplicated in the regional catalog."
    )
    values = matching[0].get("capabilities", [])
    require(
        isinstance(values, list) and len({item["name"] for item in values}) == len(values),
        "Regional SKU capability metadata is ambiguous.",
    )
    capabilities = {item["name"]: item["value"] for item in values}
    require(
        capabilities.get("LowPriorityCapable") == "True",
        f"{platform.vm_size} in {region} is not LowPriorityCapable; quota alone is insufficient.",
    )
    require(
        capabilities.get("vCPUs") == "36" and capabilities.get("GPUs") == "1",
        "The regional SKU is not the approved single-GPU 36-core renderer.",
    )
    quota = properties.get("lowPriorityCoreQuota")
    require(
        type(quota) is int and quota >= 36, "The Batch Spot core quota cannot admit one GPU node."
    )
    return {
        "region": region,
        "vm_size": platform.vm_size,
        "low_priority_core_quota": quota,
        "can_request_one_node": True,
        "capacity_guaranteed": False,
        "physical_episode_started": False,
    }


def _management_document(url: str, credential) -> dict:
    import requests

    require(
        urlsplit(url).scheme == "https" and urlsplit(url).hostname == "management.azure.com",
        "Capacity admission only reads Azure Resource Manager.",
    )
    token = credential.get_token("https://management.azure.com/.default")
    deadline = time.monotonic() + 30
    with requests.get(
        url,
        headers={"Authorization": "Bearer " + token.token},
        timeout=(5, 20),
        allow_redirects=False,
        stream=True,
    ) as response:
        response.raise_for_status()
        require(response.status_code == 200, "Capacity admission requires an actual ARM readback.")
        body = bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk)
            require(
                len(body) <= 2 * 1024**2 and time.monotonic() < deadline,
                "Regional admission read exceeded its size or time budget.",
            )
    return parse_json(bytes(body))


def check_regional_capacity(platform: BatchPlatform, credential) -> dict:
    subscription = platform.node_identity_resource_id.split("/")[2]
    region = urlsplit(platform.account_url).hostname.split(".")[1]
    account = _management_document(
        "https://management.azure.com"
        + _management_account_id(platform)
        + "?api-version=2025-06-01",
        credential,
    )
    query = urlencode(
        {
            "api-version": "2025-06-01",
            "maxresults": 100,
            "$filter": "familyName eq 'StandardNVADSA10v5Family'",
        }
    )
    catalog = _management_document(
        f"https://management.azure.com/subscriptions/{subscription}/providers/"
        f"Microsoft.Batch/locations/{region}/virtualMachineSkus?{query}",
        credential,
    )
    return validate_regional_capacity(platform, account, catalog)


def validate_pool(pool, platform: BatchPlatform) -> None:
    require(
        pool.vm_size.lower() == platform.vm_size.lower()
        and pool.task_slots_per_node == 1
        and pool.current_dedicated_nodes == 0
        and pool.current_low_priority_nodes in (0, 1)
        and pool.target_dedicated_nodes == 0
        and pool.target_low_priority_nodes in (0, 1)
        and pool.enable_auto_scale is True,
        "The approved zero-to-one LowPriority GPU pool bounds differ.",
    )
    image = pool.virtual_machine_configuration.image_reference
    require(
        (image.publisher, image.offer, image.sku, image.version)
        == (
            platform.image_publisher,
            platform.image_offer,
            platform.image_sku,
            platform.image_version,
        )
        and pool.virtual_machine_configuration.node_agent_sku_id == platform.node_agent_sku,
        "The actual managed pool image differs from the approved exact version.",
    )
    identity_ids = (
        {identity.resource_id for identity in pool.identity.user_assigned_identities}
        if pool.identity is not None
        else set()
    )
    require(
        identity_ids == {platform.node_identity_resource_id}
        and pool.virtual_machine_configuration.container_configuration.container_image_names
        == [platform.container_image],
        "The managed pool identity or prefetched container differs from the approved platform.",
    )
    metadata = {item.name: item.value for item in pool.metadata or []}
    cutoff_text = metadata.get("physicalaiAllocationDeadline", "")
    cutoff = datetime.fromisoformat(cutoff_text.replace("Z", "+00:00"))
    now = datetime.now(UTC)
    require(
        cutoff.tzinfo is not None and now < cutoff <= now + timedelta(minutes=60),
        "The original managed allocation window expired or exceeds 60 minutes; no renewal.",
    )
    require(
        isinstance(pool.auto_scale_formula, str)
        and "".join(pool.auto_scale_formula.split())
        == "".join(allocation_formula(cutoff_text).split())
        and pool.auto_scale_evaluation_interval == timedelta(minutes=5),
        "The actual autoscale formula differs from the bounded approved zero-to-one window.",
    )
    network = pool.network_configuration
    require(
        network is not None
        and bool(network.subnet_id)
        and network.public_ip_address_configuration is not None
        and network.public_ip_address_configuration.ip_address_provisioning_type
        in ("nopublicipaddresses", "NoPublicIPAddresses"),
        "Managed simulator nodes must not have public IPs.",
    )
    start = pool.start_task
    bootstrap = platform.driver_installation == "bootstrap"
    require(
        start is not None and start.wait_for_success is True and start.max_task_retry_count == 0,
        "The pool lacks the reviewed bounded preflight StartTask.",
    )
    extensions = pool.virtual_machine_configuration.extensions or []
    if bootstrap:
        require(
            not extensions
            and start.command_line == driver_bootstrap_command(platform)
            and start.container_settings is None
            and start.user_identity.auto_user.scope == "pool"
            and start.user_identity.auto_user.elevation_level == "admin",
            "The driver bootstrap differs from its reviewed script or has competing extensions.",
        )
    else:
        require(
            start.command_line == PREFLIGHT_COMMAND
            and start.container_settings.image_name == platform.container_image
            and start.container_settings.container_run_options == PREFLIGHT_CONTAINER_OPTIONS
            and start.user_identity.auto_user.scope == "task"
            and start.user_identity.auto_user.elevation_level == "nonadmin",
            "The pool lacks the reviewed non-admin bounded preflight StartTask.",
        )
        require(len(extensions) == 1, "The pool must use only the reviewed NVIDIA extension.")
        validate_driver_extension(extensions[0], platform)


def validate_driver_extension(
    extension, platform: BatchPlatform, *, allow_redacted_settings: bool = False
) -> None:
    expected_settings = {
        "driverVersion": platform.driver_version,
        "installCUDA": False,
        "updateOS": False,
    }
    settings = extension.as_dict().get("settings")
    require(
        extension.name == "nvidia-grid"
        and extension.publisher == "Microsoft.HpcCompute"
        and extension.type == "NvidiaGpuDriverLinux"
        and extension.type_handler_version == platform.driver_handler_version
        and extension.auto_upgrade_minor_version is False
        and extension.enable_automatic_upgrade is False
        and (
            settings == expected_settings
            or (
                allow_redacted_settings
                and settings == {"length": len(canonical(expected_settings))}
            )
        ),
        "The actual GRID extension differs from the approved fixed driver/handler configuration.",
    )


def inspect_platform(client, platform: BatchPlatform) -> dict:
    pool = client.get_pool(platform.pool_id)
    validate_pool(pool, platform)
    require(
        pool.current_low_priority_nodes == 1 and pool.target_low_priority_nodes == 1,
        "The approved LowPriority GPU node is not ready; no episode was submitted.",
    )
    nodes = []
    for node in client.list_nodes(platform.pool_id, max_results=2):
        nodes.append(node)
        require(len(nodes) <= 1, "The managed simulator pool has more than one compute node.")
    require(len(nodes) == 1, "No managed simulator node is available.")
    node = nodes[0]
    require(
        node.state == "idle"
        and node.is_dedicated is False
        and node.running_tasks_count == 0
        and node.start_task_info is not None
        and node.start_task_info.exit_code == 0,
        "The managed GPU node has not completed its bounded hardware preflight.",
    )
    if platform.driver_installation == "extension":
        extension = client.get_node_extension(platform.pool_id, node.id, "nvidia-grid")
        require(
            str(extension.provisioning_state).lower() == "succeeded",
            "The actual GRID extension has not succeeded at the approved handler version.",
        )
        # The node endpoint redacts settings; validate_pool already checked the full configuration.
        validate_driver_extension(extension.vm_extension, platform, allow_redacted_settings=True)
    payload = bytearray()
    for chunk in client.download_node_file(
        platform.pool_id, node.id, "startup/wd/preflight.json", ocp_range="bytes=0-65536"
    ):
        payload.extend(chunk)
        require(len(payload) <= 65536, "Node preflight proof exceeds its bounded size.")
    from learning.common import parse_json
    from simulation.batch_task import validate_gpu_evidence

    proof = parse_json(bytes(payload))
    validate_gpu_evidence(proof)
    if platform.driver_installation == "bootstrap":
        require(
            proof.get("bootstrap_sha256") == digest(bootstrap_source())
            and proof.get("driver_installer_sha256") == GRID_INSTALLER_SHA256,
            "The GPU proof does not bind the reviewed driver bootstrap and installer.",
        )
    observed = datetime.fromisoformat(proof["observed_at_utc"].replace("Z", "+00:00"))
    require(
        observed.tzinfo is not None
        and node.start_task_info.start_time <= observed <= datetime.now(UTC),
        "Node preflight evidence is stale or future-dated.",
    )
    return {
        "ready": True,
        "node_id": node.id,
        "pool_id": platform.pool_id,
        "hardware": proof,
        "physical_episode_started": False,
    }


def read_status(client, spec: BatchSimulationSpec, read_completion) -> dict:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        task = client.get_task(spec.job_id, spec.task_id)
    except ResourceNotFoundError:
        return {"outcome": "not_submitted", "physical_task_success": False}
    info = task.execution_info
    proof = read_completion()
    result = {
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "batch_state": str(task.state),
        "exit_code": info.exit_code if info else None,
        "requeue_count": info.requeue_count if info else None,
        "outcome": "incomplete",
        "physical_task_success": False,
    }
    if proof is None:
        return result
    validate_completion(spec, proof)
    accepted = proof["accepted"]
    result.update(
        outcome="accepted" if accepted else proof["outcome"],
        physical_task_success=accepted,
        native_proof=proof,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "plan",
            "warmup-plan",
            "capacity-check",
            "warmup",
            "preflight",
            "submit",
            "status",
        ),
    )
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--warmup-id", type=UUID)
    parser.add_argument("--spec-url")
    parser.add_argument("--confirm-submission", action="store_true")
    args = parser.parse_args()
    warming = args.operation in {"warmup-plan", "warmup"}
    if warming or args.operation in {"capacity-check", "preflight"}:
        require(args.platform is not None, "An approved platform file is required.")
        platform = BatchPlatform.model_validate(read_json(args.platform, max_bytes=1024**2))
        spec = None
    else:
        require(args.spec is not None, "An approved attempt specification is required.")
        spec = BatchSimulationSpec.model_validate(read_json(args.spec, max_bytes=1024**2))
        platform = spec.platform
    if warming:
        require(args.warmup_id is not None, "A unique approved warmup attempt ID is required.")
    if args.operation in {"plan", "submit"}:
        require(args.spec_url, "An immutable MI-readable specification URL is required.")
    if args.operation == "plan":
        job, task = build_job_task(spec, args.spec_url, digest(args.spec.read_bytes()))
        print(
            json.dumps(
                {
                    "job": job.as_dict(),
                    "task": task.as_dict(),
                    "after_task_created_patch": {"onAllTasksComplete": "terminatejob"},
                },
                indent=2,
            )
        )
        return
    if args.operation == "warmup-plan":
        job, task = build_warmup(platform, args.warmup_id)
        print(
            json.dumps(
                {
                    "job": job.as_dict(),
                    "task": task.as_dict(),
                    "after_task_created_patch": {"onAllTasksComplete": "terminatejob"},
                },
                indent=2,
            )
        )
        return
    from azure.batch import BatchClient
    from azure.identity import DefaultAzureCredential

    with DefaultAzureCredential(exclude_interactive_browser_credential=True) as credential:
        if args.operation in {"capacity-check", "warmup", "submit"}:
            capacity = check_regional_capacity(platform, credential)
            if args.operation == "capacity-check":
                print(json.dumps(capacity, indent=2))
                return
        with BatchClient(
            endpoint=platform.account_url,
            credential=credential,
            credential_scopes=[BATCH_TOKEN_SCOPE],
            retry_total=0,
        ) as client:
            if args.operation == "preflight":
                result = inspect_platform(client, platform)
            elif args.operation == "warmup":
                require(
                    args.confirm_submission,
                    "Explicit non-motion warmup submission approval is required.",
                )
                validate_pool(client.get_pool(platform.pool_id), platform)
                result = submit_requests_once(client, *build_warmup(platform, args.warmup_id))
            elif args.operation == "submit":
                require(
                    args.confirm_submission,
                    "Explicit managed job submission confirmation is required.",
                )
                inspect_platform(client, spec.platform)
                result = submit_once(client, spec, args.spec_url, digest(args.spec.read_bytes()))
            else:
                from simulation.batch_task import PrivateArtifacts

                with PrivateArtifacts(spec, credential) as artifacts:
                    result = read_status(client, spec, artifacts.read_completion)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
