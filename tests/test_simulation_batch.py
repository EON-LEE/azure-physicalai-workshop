"""CPU Batch SDK and idempotency tests; no account, node, GPU or paid task is created."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from azure.core.exceptions import ResourceNotFoundError

from simulation.batch import BatchSimulationSpec, build_job_task, read_status, submit_once


@pytest.fixture
def batch_sdk():
    return pytest.importorskip("azure.batch")


@pytest.fixture
def spec():
    reference = {"name": "approved/item.json", "sha256": "b" * 64, "size_bytes": 128}
    return BatchSimulationSpec.model_validate(
        {
            "schema": "physicalai.batch-simulation/v1",
            "attempt_id": "11111111-1111-4111-8111-111111111111",
            "owner_id": "a" * 64,
            "platform": {
                "account_url": "https://unit.eastus2.batch.azure.com",
                "pool_id": "sim-unit",
                "node_identity_resource_id": "/subscriptions/22222222-2222-4222-8222-222222222222/"
                "resourceGroups/unit/providers/Microsoft.ManagedIdentity/userAssignedIdentities/sim",
                "node_identity_client_id": "33333333-3333-4333-8333-333333333333",
                "tenant_id": "44444444-4444-4444-8444-444444444444",
                "vm_size": "Standard_NV36ads_A10_v5",
                "image_publisher": "microsoft-dsvm",
                "image_offer": "ubuntu-hpc",
                "image_sku": "2404",
                "image_version": "24.04.2026092501",
                "driver_handler_version": "1.14",
                "container_image": "unit.azurecr.io/physicalai-simulator@sha256:" + "c" * 64,
            },
            "source_revision": "d" * 40,
            "control_profile_sha256": "e" * 64,
            "profile_id": "franka-position-hold-10hz-paused-v2",
            "storage_account_url": "https://unitstorage.blob.core.windows.net",
            "input_container": "artifacts",
            "environment": {**reference, "name": "approved/environment.json"},
            "grant": {**reference, "name": "approved/grant.json"},
            "criteria": {**reference, "name": "approved/criteria.json"},
            "conditions": {**reference, "name": "approved/conditions.json"},
            "criteria_canonical_sha256": "b" * 64,
            "conditions_canonical_sha256": "b" * 64,
            "asset_bundle": {**reference, "name": "approved/franka.tar", "size_bytes": 100000},
            "asset_root": "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "isaac_license_preapproved": True,
        }
    )


def spec_url(spec):
    return spec.storage_account_url + "/artifacts/approved/batch-attempt.json"


def test_sdk_payload_is_one_digest_pinned_job_task_with_no_retries(spec, batch_sdk):
    from azure.batch import models

    job, task = build_job_task(spec, spec_url(spec), spec.sha256)
    assert isinstance(job, models.BatchJobCreateOptions)
    assert isinstance(task, models.BatchTaskCreateOptions)
    job, task = job.as_dict(), task.as_dict()
    assert job["id"] == spec.job_id and task["id"] == "episode"
    assert job["poolInfo"] == {"poolId": spec.platform.pool_id}
    assert "maxParallelTasks" not in job
    assert "allowTaskPreemption" not in job
    assert job["constraints"]["maxTaskRetryCount"] == 0
    assert task["constraints"]["maxTaskRetryCount"] == 0
    assert job["constraints"]["maxWallClockTime"].startswith("PT15M")
    assert task["constraints"]["maxWallClockTime"].startswith("PT15M")
    assert job["onAllTasksComplete"] == "noaction"
    assert task["requiredSlots"] == 1
    assert task["containerSettings"]["imageName"] == spec.platform.container_image
    options = task["containerSettings"]["containerRunOptions"]
    assert "--entrypoint /isaac-sim/python.sh" in options
    assert "--runtime" not in options
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"} in task["environmentSettings"]
    assert {"name": "HOME", "value": "/data"} in task["environmentSettings"]
    assert {"name": "XDG_CACHE_HOME", "value": "/data/.cache"} in task["environmentSettings"]
    assert "/isaac-sim/kit/cache:rw,nosuid,nodev,mode=1777,size=2147483648" in options
    assert "/isaac-sim/kit/data:rw,nosuid,nodev,mode=1777,size=268435456" in options
    assert "--gpus" not in options and "--privileged" not in options
    assert "simulation.batch_task" in task["commandLine"]
    assert "bootstrap_tls" not in task["commandLine"]
    resource = task["resourceFiles"][0]
    assert resource["httpUrl"] == spec_url(spec)
    assert resource["identityReference"]["resourceId"] == spec.platform.node_identity_resource_id
    assert task["userIdentity"] == {"autoUser": {"scope": "task", "elevationLevel": "nonadmin"}}
    assert all("sig=" not in json.dumps(value) for value in (job, task))


def test_warmup_uses_single_task_pool_bounds_without_account_gated_job_properties(spec, batch_sdk):
    from simulation.batch import build_warmup

    job, task = build_warmup(spec.platform, UUID("11111111-1111-4111-8111-111111111111"))
    assert "maxParallelTasks" not in job.as_dict()
    assert "allowTaskPreemption" not in job.as_dict()
    assert task.required_slots == 1
    assert "--runtime" not in task.container_settings.container_run_options
    assert "nvidia-ctk runtime configure --runtime=docker --set-as-default" in (
        job.job_preparation_task.command_line
    )
    assert "--kill-after=5s 30s" in job.job_preparation_task.command_line
    assert job.job_preparation_task.wait_for_success is True
    assert job.job_preparation_task.user_identity.auto_user.elevation_level == "admin"
    assert any(
        item.name == "NVIDIA_VISIBLE_DEVICES" and item.value == "all"
        for item in task.environment_settings
    )
    assert job.pool_info.pool_id == spec.platform.pool_id


@pytest.mark.parametrize(
    "change",
    [
        {"image_version": "latest"},
        {"vm_size": "Standard_NC24ads_A100_v4"},
        {"container_image": "unit.azurecr.io/physicalai-simulator:latest"},
        {"driver_version": "latest"},
        {"driver_handler_version": "1.14.0.6"},
        {"image_sku": "2204"},
    ],
)
def test_unreviewed_renderer_platforms_and_mutable_images_are_rejected(spec, change):
    value = spec.model_dump(mode="json", by_alias=True)
    value["platform"].update(change)
    with pytest.raises(ValueError):
        BatchSimulationSpec.model_validate(value)


class Client:
    def __init__(self):
        self.jobs, self.tasks, self.writes = {}, {}, []

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise ResourceNotFoundError("not present")
        from azure.batch import models

        return models.BatchJob(self.jobs[job_id].as_dict() | {"eTag": '"unit"'})

    def get_task(self, job_id, task_id):
        if (job_id, task_id) not in self.tasks:
            raise ResourceNotFoundError("not present")
        return self.tasks[job_id, task_id]

    def create_job(self, *, job):
        self.writes.append(("job", job.id))
        self.jobs[job.id] = job

    def create_task(self, job_id, *, task):
        self.writes.append(("task", task.id))
        self.tasks[job_id, task.id] = task

    def update_job(self, job_id, *, job, **kwargs):
        assert self.tasks, "Do not terminate a job before adding its single task."
        assert kwargs["etag"] == '"unit"'
        self.writes.append(("arm-termination", job_id))
        self.jobs[job_id].all_tasks_complete_mode = job.all_tasks_complete_mode


def test_empty_job_cannot_terminate_before_its_single_task_is_added(spec, batch_sdk):
    job, _ = build_job_task(spec, spec_url(spec), spec.sha256)
    assert job.as_dict()["onAllTasksComplete"] == "noaction"
    client = Client()
    submit_once(client, spec, spec_url(spec), spec.sha256)
    assert [write[0] for write in client.writes] == ["job", "task", "arm-termination"]


def test_repeated_submission_reconciles_without_duplicate_job_or_episode(spec, batch_sdk):
    client = Client()
    first = submit_once(client, spec, spec_url(spec), spec.sha256)
    second = submit_once(client, spec, spec_url(spec), spec.sha256)
    assert first["job_id"] == second["job_id"] == spec.job_id
    assert first["task_id"] == second["task_id"] == spec.task_id
    assert client.writes == [
        ("job", spec.job_id),
        ("task", spec.task_id),
        ("arm-termination", spec.job_id),
    ]


def test_service_duration_format_does_not_change_existing_job_authority(spec, batch_sdk):
    from azure.batch import models

    from simulation.batch import build_warmup, submit_requests_once

    class ServiceDurationClient(Client):
        def get_job(self, job_id):
            result = super().get_job(job_id).as_dict()
            result["constraints"]["maxWallClockTime"] = "PT30M"
            return models.BatchJob(result)

        def get_task(self, job_id, task_id):
            result = super().get_task(job_id, task_id).as_dict()
            result["constraints"]["maxWallClockTime"] = "PT1M"
            return models.BatchTask(result)

    client = ServiceDurationClient()
    job, task = build_warmup(spec.platform, spec.attempt_id)
    submit_requests_once(client, job, task)
    submit_requests_once(client, job, task)
    assert [item[0] for item in client.writes] == ["job", "task", "arm-termination"]


@pytest.mark.parametrize("duration", ["PT16M", "PT14M", None])
def test_reconciliation_rejects_changed_or_missing_duration(spec, batch_sdk, duration):
    from azure.batch import models

    from simulation.batch import _same_request

    job, _ = build_job_task(spec, spec_url(spec), spec.sha256)
    wire = job.as_dict()
    wire["constraints"]["maxWallClockTime"] = duration
    with pytest.raises(ValueError, match="binding|duration"):
        _same_request(models.BatchJob(wire), job, job=True)


def test_reused_attempt_id_cannot_change_the_source_or_physical_input(spec, batch_sdk):
    client = Client()
    submit_once(client, spec, spec_url(spec), spec.sha256)
    altered = spec.model_copy(update={"source_revision": "f" * 40})
    with pytest.raises(ValueError, match="binding|different"):
        submit_once(client, altered, spec_url(spec), altered.sha256)
    assert len(client.writes) == 3


def test_polling_completed_batch_task_without_terminal_proof_is_only_incomplete(spec):
    client = Client()
    client.tasks[spec.job_id, spec.task_id] = SimpleNamespace(
        state="completed",
        execution_info=SimpleNamespace(exit_code=0, retry_count=0, requeue_count=1),
    )
    result = read_status(client, spec, lambda: None)
    assert result["outcome"] == "incomplete"
    assert result["physical_task_success"] is False
    assert client.writes == []


def test_success_shaped_completion_without_native_artifacts_cannot_qualify(spec):
    client = Client()
    client.tasks[spec.job_id, spec.task_id] = SimpleNamespace(
        state="completed", execution_info=SimpleNamespace(exit_code=0, requeue_count=0)
    )
    proof = {
        "schema": "physicalai.batch-simulation-result/v1",
        "attempt_id": str(spec.attempt_id),
        "spec_sha256": spec.sha256,
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "source_revision": spec.source_revision,
        "image": spec.platform.container_image,
        "control_profile_sha256": spec.control_profile_sha256,
        "profile_id": spec.profile_id,
        "accepted": True,
        "outcome": "accepted",
        "native_acceptance": "accepted",
        "learning_quality_proven": False,
        "artifacts": [],
    }
    with pytest.raises(ValueError, match="proof|artifact"):
        read_status(client, spec, lambda: proof)


def test_warmup_has_no_grant_or_simulator_and_uses_existing_pool_without_resize(spec, batch_sdk):
    from simulation.batch import build_warmup, submit_requests_once

    job, task = build_warmup(spec.platform, spec.attempt_id)
    wire = task.as_dict()
    assert wire["id"] == "preflight"
    assert "simulation.batch_task preflight" in wire["commandLine"]
    assert "paused_probe" not in wire["commandLine"]
    assert "resourceFiles" not in wire
    assert wire["constraints"]["maxWallClockTime"].startswith("PT01M")
    assert wire["constraints"]["maxTaskRetryCount"] == 0
    client = Client()
    submit_requests_once(client, job, task)
    submit_requests_once(client, job, task)
    assert client.writes == [
        ("job", "warm-" + spec.attempt_id.hex),
        ("task", "preflight"),
        ("arm-termination", "warm-" + spec.attempt_id.hex),
    ]


@pytest.fixture
def platform_client(spec, batch_sdk):
    from azure.batch import models
    from test_batch_simulation_task import gpu

    now = datetime.now(UTC)
    platform = spec.platform
    from simulation.batch import PREFLIGHT_COMMAND, PREFLIGHT_CONTAINER_OPTIONS, allocation_formula

    cutoff = (now + timedelta(minutes=30)).isoformat()
    extension_value = {
        "name": "nvidia-grid",
        "publisher": "Microsoft.HpcCompute",
        "type": "NvidiaGpuDriverLinux",
        "typeHandlerVersion": platform.driver_handler_version,
        "autoUpgradeMinorVersion": False,
        "enableAutomaticUpgrade": False,
        "settings": {
            "driverVersion": platform.driver_version,
            "installCUDA": False,
            "updateOS": False,
        },
    }
    pool = models.BatchPool(
        {
            "vmSize": platform.vm_size,
            "taskSlotsPerNode": 1,
            "enableAutoScale": True,
            "currentDedicatedNodes": 0,
            "currentLowPriorityNodes": 1,
            "targetDedicatedNodes": 0,
            "targetLowPriorityNodes": 1,
            "autoScaleFormula": allocation_formula(cutoff),
            "autoScaleEvaluationInterval": "PT5M",
            "identity": {
                "type": "userAssigned",
                "userAssignedIdentities": [
                    {
                        "resourceId": platform.node_identity_resource_id,
                    }
                ],
            },
            "virtualMachineConfiguration": {
                "nodeAgentSKUId": "batch.node.ubuntu 24.04",
                "imageReference": {
                    "publisher": platform.image_publisher,
                    "offer": platform.image_offer,
                    "sku": platform.image_sku,
                    "version": platform.image_version,
                },
                "containerConfiguration": {"containerImageNames": [platform.container_image]},
                "extensions": [extension_value],
            },
            "networkConfiguration": {
                "subnetId": "/subscriptions/unit/resourceGroups/unit/providers/Microsoft.Network/"
                "virtualNetworks/unit/subnets/private",
                "publicIPAddressConfiguration": {"provision": "nopublicipaddresses"},
            },
            "metadata": [
                {
                    "name": "physicalaiAllocationDeadline",
                    "value": cutoff,
                }
            ],
            "startTask": {
                "commandLine": PREFLIGHT_COMMAND,
                "waitForSuccess": True,
                "maxTaskRetryCount": 0,
                "containerSettings": {
                    "imageName": platform.container_image,
                    "containerRunOptions": PREFLIGHT_CONTAINER_OPTIONS,
                },
                "userIdentity": {"autoUser": {"scope": "task", "elevationLevel": "nonadmin"}},
            },
        }
    )
    node = models.BatchNode(
        {
            "id": "unit-node",
            "state": "idle",
            "isDedicated": False,
            "runningTasksCount": 0,
            "startTaskInfo": {
                "exitCode": 0,
                "startTime": (now - timedelta(seconds=10)).isoformat(),
            },
        }
    )
    extension = models.BatchNodeVMExtension(
        {
            "provisioningState": "Succeeded",
            "vmExtension": extension_value,
        }
    )
    downloads = []

    def download(pool_id, node_id, path, **kwargs):
        downloads.append((pool_id, node_id, path, kwargs))
        yield json.dumps(gpu() | {"observed_at_utc": now.isoformat()}).encode()

    client = SimpleNamespace(
        get_pool=lambda pool_id: pool,
        list_nodes=lambda *args, **kwargs: iter([node]),
        get_node_extension=lambda *args: extension,
        download_node_file=download,
    )
    return SimpleNamespace(
        client=client, pool=pool, node=node, extension=extension, downloads=downloads
    )


def test_pool_preflight_consumes_actual_lowercase_sdk_enum_and_startup_file(spec, platform_client):
    from simulation.batch import inspect_platform

    assert inspect_platform(platform_client.client, spec.platform)["ready"] is True
    assert platform_client.downloads[0][2] == "startup/wd/preflight.json"


def test_explicit_ubuntu22_host_requires_its_matching_agent(spec, platform_client):
    from simulation.batch import BatchPlatform, inspect_platform

    platform = BatchPlatform.model_validate(
        {
            **spec.platform.model_dump(mode="json"),
            "image_sku": "2204",
            "image_version": "22.04.2026082801",
        }
    )
    vm = platform_client.pool.virtual_machine_configuration
    vm.image_reference.sku = platform.image_sku
    vm.image_reference.version = platform.image_version
    with pytest.raises(ValueError, match="image"):
        inspect_platform(platform_client.client, platform)
    vm.node_agent_sku_id = "batch.node.ubuntu 22.04"
    assert inspect_platform(platform_client.client, platform)["ready"] is True


def test_node_settings_redaction_requires_full_pool_settings_and_actual_gpu_proof(
    spec, platform_client
):
    from simulation.batch import inspect_platform

    platform_client.extension.vm_extension["settings"] = {"length": 64}
    assert inspect_platform(platform_client.client, spec.platform)["ready"] is True
    platform_client.pool.virtual_machine_configuration.extensions[0]["settings"] = {"length": 64}
    with pytest.raises(ValueError, match="GRID extension"):
        inspect_platform(platform_client.client, spec.platform)


def test_driver_bootstrap_preserves_physics_user_and_requires_exact_script_and_gpu_proof(
    spec, platform_client
):
    from azure.batch import models

    from learning.common import digest
    from simulation.batch import (
        GRID_INSTALLER_SHA256,
        BatchPlatform,
        bootstrap_source,
        driver_bootstrap_command,
        inspect_platform,
    )

    assert "driver_installation" not in spec.platform.model_dump(mode="json")
    platform = BatchPlatform.model_validate(
        {
            **spec.platform.model_dump(mode="json"),
            "image_sku": "2204",
            "image_version": "22.04.2026082801",
            "driver_installation": "bootstrap",
            "driver_handler_version": None,
        }
    )
    vm = platform_client.pool.virtual_machine_configuration
    vm.image_reference.sku, vm.image_reference.version = platform.image_sku, platform.image_version
    vm.node_agent_sku_id = platform.node_agent_sku
    vm.extensions = []
    command = driver_bootstrap_command(platform)
    assert "--kill-after=10s 900s" in command
    assert digest(bootstrap_source()) in command
    platform_client.pool.start_task = models.BatchStartTask(
        {
            "commandLine": command,
            "waitForSuccess": True,
            "maxTaskRetryCount": 0,
            "userIdentity": {"autoUser": {"scope": "pool", "elevationLevel": "admin"}},
        }
    )
    download = platform_client.client.download_node_file

    def with_bootstrap(*args, **kwargs):
        proof = json.loads(b"".join(download(*args, **kwargs)))
        proof.update(
            bootstrap_sha256=digest(bootstrap_source()),
            driver_installer_sha256=GRID_INSTALLER_SHA256,
        )
        yield json.dumps(proof).encode()

    with pytest.raises(ValueError, match="bootstrap and installer"):
        inspect_platform(platform_client.client, platform)
    platform_client.client.download_node_file = with_bootstrap
    assert inspect_platform(platform_client.client, platform)["ready"] is True
    platform_client.pool.start_task.command_line = command + " extra"
    with pytest.raises(ValueError, match="bootstrap differs"):
        inspect_platform(platform_client.client, platform)
    _, task = build_job_task(spec, spec_url(spec), spec.sha256)
    assert task.user_identity.auto_user.elevation_level == "nonadmin"


def test_driver_cleanup_preserves_container_runtime_packages():
    import re
    import subprocess
    from pathlib import Path

    source = Path("simulation/batch_bootstrap.sh").read_text()
    subprocess.run(["bash", "-n"], input=source, text=True, check=True)
    expression = re.search(r"awk '([^']+)'", source, re.DOTALL).group(1)
    packages = [
        "nvidia-driver-570",
        "nvidia-kernel-common-570",
        "libnvidia-compute-570:amd64",
        "nvidia-container-toolkit",
        "nvidia-container-toolkit-base",
        "libnvidia-container1:amd64",
        "libnvidia-container-tools",
        "nvidia-docker2",
        "docker.io",
    ]
    result = subprocess.run(
        ["awk", expression],
        input="".join(name + "\tinstalled\n" for name in packages),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == packages[:3]
    assert "no-check-for-alternate-installs" not in source
    assert "sha256sum --check --status" in source
    assert 'sh "$driver" --check' in source
    assert "NEEDRESTART_MODE=l" in source


def test_batch_overlay_allows_non_admin_traversal_without_world_write():
    from pathlib import Path

    recipe = Path("simulation/Dockerfile.code").read_text()
    assert "RUN chmod a+rx /isaac-sim" in recipe
    assert "chmod 777" not in recipe


def test_pool_preflight_accepts_actual_arm_created_service_enum_casing(spec, platform_client):
    from simulation.batch import inspect_platform

    platform_client.pool.network_configuration.public_ip_address_configuration["provision"] = (
        "NoPublicIPAddresses"
    )
    assert inspect_platform(platform_client.client, spec.platform)["ready"] is True


@pytest.mark.parametrize("provision", ["batchmanaged", "UserManaged", "", None])
def test_pool_preflight_still_rejects_public_or_missing_ip_configuration(
    spec, platform_client, provision
):
    from simulation.batch import inspect_platform

    platform_client.pool.network_configuration.public_ip_address_configuration["provision"] = (
        provision
    )
    with pytest.raises(ValueError, match="public IPs"):
        inspect_platform(platform_client.client, spec.platform)


def test_autoscale_that_can_later_allocate_more_nodes_is_rejected_before_warmup(
    spec, platform_client
):
    from simulation import batch

    assert hasattr(batch, "validate_pool"), (
        "Warmup must validate zero-node configuration before queuing."
    )
    pool = platform_client.pool
    pool.current_low_priority_nodes = 0
    pool.target_low_priority_nodes = 0
    pool.auto_scale_formula = "$TargetLowPriorityNodes = 64;"
    with pytest.raises(ValueError, match="autoscale"):
        batch.validate_pool(pool, spec.platform)


def test_status_rejects_foreign_completion_proof_instead_of_trusting_batch_exit_zero(spec):
    client = Client()
    client.tasks[spec.job_id, spec.task_id] = SimpleNamespace(
        state="completed",
        execution_info=SimpleNamespace(exit_code=0, retry_count=0, requeue_count=0),
    )
    proof = {"attempt_id": str(UUID(int=1)), "spec_sha256": "f" * 64, "accepted": True}
    with pytest.raises(ValueError, match="proof|binding"):
        read_status(client, spec, lambda: proof)
    assert client.writes == []
