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
                "driver_handler_version": "1.14.0.6",
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
    assert job["maxParallelTasks"] == 1
    assert job["constraints"]["maxTaskRetryCount"] == 0
    assert task["constraints"]["maxTaskRetryCount"] == 0
    assert job["constraints"]["maxWallClockTime"].startswith("PT15M")
    assert task["constraints"]["maxWallClockTime"].startswith("PT15M")
    assert job["onAllTasksComplete"] == "noaction"
    assert task["requiredSlots"] == 1
    assert task["containerSettings"]["imageName"] == spec.platform.container_image
    options = task["containerSettings"]["containerRunOptions"]
    assert "--entrypoint /isaac-sim/python.sh" in options
    assert "--gpus" not in options and "--privileged" not in options
    assert "simulation.batch_task" in task["commandLine"]
    assert "bootstrap_tls" not in task["commandLine"]
    resource = task["resourceFiles"][0]
    assert resource["httpUrl"] == spec_url(spec)
    assert resource["identityReference"]["resourceId"] == spec.platform.node_identity_resource_id
    assert task["userIdentity"] == {"autoUser": {"scope": "task", "elevationLevel": "nonadmin"}}
    assert all("sig=" not in json.dumps(value) for value in (job, task))


@pytest.mark.parametrize(
    "change",
    [
        {"image_version": "latest"},
        {"vm_size": "Standard_NC24ads_A100_v4"},
        {"container_image": "unit.azurecr.io/physicalai-simulator:latest"},
        {"driver_version": "latest"},
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
