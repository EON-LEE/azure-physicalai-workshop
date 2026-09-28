"""Offline real native-guard fixtures; no pool, task, credential or cloud call is created."""

import copy
import json
import socket
import ssl
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_simulation_batch import batch_sdk as batch_sdk
from test_simulation_batch import platform_client as platform_client
from test_simulation_batch import spec as spec

from learning.common import ContractError, canonical, digest
from learning.paused import pool_preparation as prep


@pytest.fixture
def scenario(spec, platform_client, batch_sdk):
    from azure.batch import models
    from simulation.batch import (
        GRID_INSTALLER_SHA256,
        BatchPlatform,
        allocation_formula,
        bootstrap_source,
        driver_bootstrap_command,
    )

    platform = BatchPlatform.model_validate(
        {
            **spec.platform.model_dump(mode="json"),
            "image_sku": "2204",
            "image_version": "22.04.2026082801",
            "driver_installation": "bootstrap",
            "driver_handler_version": None,
        }
    )
    pool = platform_client.pool
    vm = pool.virtual_machine_configuration
    vm.image_reference.sku, vm.image_reference.version = platform.image_sku, platform.image_version
    vm.node_agent_sku_id, vm.extensions = platform.node_agent_sku, []
    pool.start_task = models.BatchStartTask(
        {
            "commandLine": driver_bootstrap_command(platform),
            "waitForSuccess": True,
            "maxTaskRetryCount": 0,
            "userIdentity": {"autoUser": {"scope": "pool", "elevationLevel": "admin"}},
        }
    )
    start = datetime.now(UTC) - timedelta(minutes=5)

    def stamp(value):
        return value.isoformat().replace("+00:00", "Z")

    end = stamp(start + timedelta(minutes=55))
    resource = {
        "identity": {
            "type": "UserAssigned",
            "userAssignedIdentities": {
                platform.node_identity_resource_id: {},
            },
        },
        "properties": {
            "vmSize": platform.vm_size,
            "taskSlotsPerNode": 1,
            "interNodeCommunication": "Disabled",
            "taskSchedulingPolicy": {"nodeFillType": "Pack"},
            "deploymentConfiguration": {
                "virtualMachineConfiguration": {
                    "imageReference": vm.image_reference.as_dict(),
                    "nodeAgentSkuId": platform.node_agent_sku,
                    "containerConfiguration": {
                        "type": "DockerCompatible",
                        "containerImageNames": [platform.container_image],
                    },
                    "extensions": [],
                }
            },
            "networkConfiguration": {
                "subnetId": pool.network_configuration.subnet_id,
                "publicIPAddressConfiguration": {"provision": "NoPublicIPAddresses"},
            },
            "startTask": {
                "commandLine": driver_bootstrap_command(platform),
                "waitForSuccess": True,
                "maxTaskRetryCount": 0,
                "userIdentity": {"autoUser": {"scope": "Pool", "elevationLevel": "Admin"}},
            },
            "scaleSettings": {
                "autoScale": {"formula": allocation_formula(end), "evaluationInterval": "PT5M"}
            },
            "metadata": [{"name": "physicalaiAllocationDeadline", "value": end}],
        },
    }
    config = {
        "operation_id": str(uuid4()),
        "warmup_id": str(uuid4()),
        "scope": {"tenant_id": str(platform.tenant_id), "owner_id": spec.owner_id},
        "platform": platform.model_dump(mode="json"),
        "pool": resource,
        "allocation_start_utc": stamp(start),
        "preparation_deadline_utc": stamp(start + timedelta(minutes=35)),
        "pool_deadline_utc": end,
        "overall_start_utc": stamp(start),
        "overall_deadline_utc": stamp(start + timedelta(hours=4)),
        "budget_usd": 10,
        "observer_client_id": "55555555-5555-4555-8555-555555555555",
        "storage_account_url": "https://unitstorage.blob.core.windows.net",
        "artifact_container": "artifacts",
    }
    plan = prep.make_plan(config)
    pool.auto_scale_formula = plan["pool"]["properties"]["scaleSettings"]["autoScale"]["formula"]
    pool.metadata = [
        models.BatchMetadataItem(**item) for item in plan["pool"]["properties"]["metadata"]
    ]
    downloads = platform_client.client.download_node_file

    def files(*args, **kwargs):
        if args[2] != "startup/wd/preflight.json":
            return iter([b"unit setup log; not live evidence\n"])
        value = json.loads(b"".join(downloads(*args, **kwargs)))
        value.update(
            bootstrap_sha256=digest(bootstrap_source()),
            driver_installer_sha256=GRID_INSTALLER_SHA256,
        )
        return iter([canonical(value)])

    platform_client.client.download_node_file = files
    platform_client.client.list_jobs = lambda **kwargs: iter([])
    arm_value = copy.deepcopy(plan["pool"])
    arm_value.update(id=prep.pool_resource_id(plan), name=platform.pool_id, etag='"original"')
    arm_value["properties"].update(currentLowPriorityNodes=1, currentDedicatedNodes=0)

    class Arm:
        def __init__(self):
            self.value, self.patches, self.error = arm_value, [], None
            self.patch_error, self.commit_before_error = None, False

        def get(self):
            if self.error:
                raise self.error
            return copy.deepcopy(self.value)

        def patch(self, body, *, etag):
            self.patches.append((copy.deepcopy(body), etag))
            assert etag == self.value["etag"]
            if not self.patch_error or self.commit_before_error:
                self.value["properties"].update(copy.deepcopy(body["properties"]))
                self.value["etag"] = '"transitioned"'
                pool.auto_scale_formula = body["properties"]["scaleSettings"]["autoScale"][
                    "formula"
                ]
            if self.patch_error:
                raise self.patch_error

    return SimpleNamespace(
        config=config,
        plan=plan,
        arm=Arm(),
        batch=platform_client.client,
        pool=pool,
        node=platform_client.node,
        platform=platform,
    )


def test_plan_is_closed_fixed_cutoff_no_job_and_no_new_clock(scenario):
    plan = scenario.plan
    assert plan["mode"] == "bounded_pool_preparation"
    assert plan["cleanup_deadline_utc"] == (
        datetime.fromisoformat(plan["pool_deadline_utc"]) + timedelta(minutes=5)
    ).isoformat().replace("+00:00", "Z")
    formula = plan["pool"]["properties"]["scaleSettings"]["autoScale"]["formula"]
    assert "$PendingTasks" not in formula and "$CurrentLowPriorityNodes" not in formula
    assert "$TargetDedicatedNodes = 0" in formula and "? 1 : 0" in formula
    for value in (-1, 0, 60, 2100, 3300, 50000):
        now = datetime.fromisoformat(plan["allocation_start_utc"]) + timedelta(seconds=value)
        assert prep.preparation_target(plan, now) in (0, 1)
    assert (
        prep.preparation_target(plan, datetime.fromisoformat(plan["preparation_deadline_utc"])) == 0
    )
    assert plan["pool_deadline_utc"] == scenario.config["pool_deadline_utc"]
    assert not scenario.arm.patches


@pytest.mark.parametrize(
    "change", ["more-than-55", "overall-extension", "unknown-field", "weak-cutoff"]
)
def test_invalid_or_unbounded_plan_is_rejected(scenario, change):
    config = copy.deepcopy(scenario.config)
    if change == "more-than-55":
        config["pool_deadline_utc"] = (
            (datetime.fromisoformat(config["allocation_start_utc"]) + timedelta(minutes=56))
            .isoformat()
            .replace("+00:00", "Z")
        )
    elif change == "overall-extension":
        config["overall_deadline_utc"] = (
            (datetime.fromisoformat(config["overall_start_utc"]) + timedelta(hours=4, seconds=1))
            .isoformat()
            .replace("+00:00", "Z")
        )
    elif change == "unknown-field":
        config["force"] = True
    else:
        config["preparation_deadline_utc"] = "latest"
    with pytest.raises(ContractError):
        prep.make_plan(config)


def test_real_native_guards_reject_prep_and_accept_exact_transition_same_deadline(
    scenario, tmp_path
):
    from simulation.batch import inspect_platform, validate_pool

    with pytest.raises(ContractError, match="autoscale formula"):
        validate_pool(scenario.pool, scenario.platform)
    with pytest.raises(ContractError, match="autoscale formula"):
        inspect_platform(scenario.batch, scenario.platform)
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    assert evidence["state"] == "ready" and not scenario.arm.patches
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    assert result["state"] == "canonical_observed"
    verified = prep.verify_transition(
        scenario.plan, evidence, result, arm=scenario.arm, batch=scenario.batch
    )
    assert verified["state"] == "canonical_verified"
    assert scenario.arm.patches[0][1] == '"original"'
    validate_pool(scenario.pool, scenario.platform)
    assert inspect_platform(scenario.batch, scenario.platform)["ready"] is True
    assert scenario.plan["pool_deadline_utc"] == scenario.config["pool_deadline_utc"]
    assert {item.name: item.value for item in scenario.pool.metadata}[
        "physicalaiAllocationDeadline"
    ] == scenario.plan["pool_deadline_utc"]


@pytest.mark.parametrize(
    "error", [socket.gaierror("dns"), ssl.SSLError("tls"), TimeoutError("timeout")]
)
def test_local_transport_failure_is_unknown_with_zero_patch_or_task_creation(
    scenario, tmp_path, error
):
    scenario.arm.error = error
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    assert evidence["state"] == "unknown" and evidence["errors"]
    with pytest.raises(ContractError):
        prep.transition(
            scenario.plan,
            evidence,
            arm=scenario.arm,
            journal=tmp_path / "intent.json",
            approved_plan_sha256=scenario.plan["plan_sha256"],
        )
    assert scenario.arm.patches == [] and not (tmp_path / "intent.json").exists()


@pytest.mark.parametrize(
    "change", ["starting", "bad-start", "missing-proof", "busy", "stale", "tampered"]
)
def test_not_ready_missing_or_changed_evidence_never_advances(scenario, tmp_path, change):
    from azure.core.exceptions import ResourceNotFoundError

    if change == "starting":
        scenario.node.state = "starting"
    elif change == "bad-start":
        scenario.node.start_task_info.exit_code = 1
    elif change == "missing-proof":

        def missing(*args, **kwargs):
            raise ResourceNotFoundError("file unavailable")

        scenario.batch.download_node_file = missing
    elif change == "busy":
        scenario.node.running_tasks_count = 1
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    if change == "stale":
        evidence["observed_at_utc"] = "2020-01-01T00:00:00Z"
    elif change == "tampered":
        evidence["plan_sha256"] = "0" * 64
    with pytest.raises(ContractError):
        prep.transition(
            scenario.plan,
            evidence,
            arm=scenario.arm,
            journal=tmp_path / "intent.json",
            approved_plan_sha256=scenario.plan["plan_sha256"],
        )
    assert not scenario.arm.patches


@pytest.mark.parametrize("committed", [False, True])
def test_ambiguous_patch_only_reconciles_known_state_without_repeating_or_renewing(
    scenario, tmp_path, committed
):
    scenario.arm.patch_error = TimeoutError("acknowledgement unknown")
    scenario.arm.commit_before_error = committed
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    journal = tmp_path / "transition.json"
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=journal,
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    assert result["state"] == ("canonical_observed" if committed else "preparation_unchanged")
    assert result["patch_outcome"] == "unknown"
    assert len(scenario.arm.patches) == 1
    with pytest.raises(ContractError):
        prep.transition(
            scenario.plan,
            evidence,
            arm=scenario.arm,
            journal=journal,
            approved_plan_sha256=scenario.plan["plan_sha256"],
        )
    assert len(scenario.arm.patches) == 1


def test_log_records_are_bounded_and_missing_files_are_explicit(scenario):
    from azure.core.exceptions import ResourceNotFoundError

    real = scenario.batch.download_node_file

    def download(pool, node, path, **kwargs):
        if path == "startup/stderr.txt":
            raise ResourceNotFoundError("404")
        return real(pool, node, path, **kwargs)

    scenario.batch.download_node_file = download
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    assert evidence["logs"]["startup/stderr.txt"]["state"] == "unavailable"
    assert evidence["logs"]["startup/stdout.txt"]["sha256"]
    assert evidence["logs"]["startup/wd/preflight.json"]["sha256"]
    assert evidence["node"]["start_task_info"]["exitCode"] == 0
    assert not scenario.arm.patches


def test_new_or_foreign_node_after_operator_transition_cannot_start_warmup(scenario, tmp_path):
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    changed = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    scenario.node.id = "replacement-node-not-the-observed-node"
    with pytest.raises(ContractError, match="original verified node"):
        prep.verify_transition(
            scenario.plan, evidence, changed, arm=scenario.arm, batch=scenario.batch
        )


def test_native_guard_is_not_bypassed_when_sdk_still_reports_preparation_formula(
    scenario, tmp_path
):
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    scenario.pool.auto_scale_formula = scenario.plan["pool"]["properties"]["scaleSettings"][
        "autoScale"
    ]["formula"]
    with pytest.raises(ContractError, match="autoscale formula"):
        prep.verify_transition(
            scenario.plan, evidence, result, arm=scenario.arm, batch=scenario.batch
        )


def test_ordinary_warmup_is_one_shot_and_retains_original_job_task_limits(scenario, tmp_path):
    from test_simulation_batch import Client

    client = Client()
    for name in ("get_job", "get_task", "create_job", "create_task", "update_job"):
        setattr(scenario.batch, name, getattr(client, name))
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    args = dict(
        arm=scenario.arm,
        batch=scenario.batch,
        journal=tmp_path / "warmup.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    assert client.writes == []
    submitted = prep.submit_warmup_once(scenario.plan, evidence, result, **args)
    assert [verb for verb, _ in client.writes] == ["job", "task", "arm-termination"]
    job = client.jobs[submitted["job_id"]]
    task = client.tasks[submitted["job_id"], submitted["task_id"]]
    assert job.constraints.max_wall_clock_time == timedelta(minutes=30)
    assert task.constraints.max_wall_clock_time == timedelta(seconds=60)
    assert task.user_identity.auto_user.elevation_level == "nonadmin"
    assert task.required_slots == 1 and submitted["physical_grant_issued"] is False
    with pytest.raises(ContractError, match="already exists|already requested"):
        prep.submit_warmup_once(scenario.plan, evidence, result, **args)
    assert len(client.writes) == 3


def test_unknown_warmup_submission_cannot_be_replayed_with_original_journal(
    scenario, tmp_path, monkeypatch
):
    from azure.core.exceptions import ResourceNotFoundError

    from simulation import batch as native

    def absent(*args):
        raise ResourceNotFoundError("No original warmup yet")

    scenario.batch.get_job = absent
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    calls = []

    def failed(*args):
        calls.append(True)
        raise TimeoutError("Create acknowledgement unknown")

    monkeypatch.setattr(native, "submit_requests_once", failed)
    args = dict(
        arm=scenario.arm,
        batch=scenario.batch,
        journal=tmp_path / "warmup.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    with pytest.raises(TimeoutError):
        prep.submit_warmup_once(scenario.plan, evidence, result, **args)
    with pytest.raises(ContractError, match="already requested"):
        prep.submit_warmup_once(scenario.plan, evidence, result, **args)
    assert calls == [True]


def test_exact_noncanonical_json_bytes_remain_preserved_through_native_verification(
    scenario, tmp_path
):
    original = scenario.batch.download_node_file

    def spaced(pool_id, node_id, path, **kwargs):
        raw = b"".join(original(pool_id, node_id, path, **kwargs))
        if path.endswith("preflight.json"):
            raw = json.dumps(json.loads(raw), indent=2).encode()
        return iter([raw])

    scenario.batch.download_node_file = spaced
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    assert (
        prep.verify_transition(
            scenario.plan, evidence, result, arm=scenario.arm, batch=scenario.batch
        )["state"]
        == "canonical_verified"
    )


@pytest.mark.parametrize(
    "change", ["new-vm-setting", "new-credential", "new-start-environment", "scale-mode"]
)
def test_preparation_does_not_expand_approved_platform_or_credentials(scenario, change):
    config = copy.deepcopy(scenario.config)
    properties = config["pool"]["properties"]
    if change == "new-vm-setting":
        properties["deploymentConfiguration"]["virtualMachineConfiguration"]["userData"] = (
            "unreviewed"
        )
    elif change == "new-credential":
        properties["deploymentConfiguration"]["virtualMachineConfiguration"][
            "containerConfiguration"
        ]["containerRegistries"] = [
            {
                "registryServer": "unit.azurecr.io",
                "userName": "new",
                "password": "not-an-actual-secret",
            },
        ]
    elif change == "new-start-environment":
        properties["startTask"]["environmentSettings"] = [
            {"name": "BASH_ENV", "value": "unreviewed"}
        ]
    else:
        properties["scaleSettings"] = {"fixedScale": {"targetLowPriorityNodes": 1}}
    with pytest.raises(ContractError):
        prep.make_plan(config)


def test_observer_transport_cannot_patch_even_with_existing_credentials(scenario):
    from learning.paused.pool_preparation_cli import PoolArm

    credential = SimpleNamespace(
        get_token=lambda *args: pytest.fail("Read-only transport requested mutation credentials")
    )
    transport = PoolArm(scenario.plan, credential)
    with pytest.raises(ContractError, match="Observer transport"):
        transport.patch({"properties": {}}, etag='"original"')


def test_expired_preparation_cannot_use_a_fresh_proof_to_renew_the_clock(scenario, tmp_path):
    old = copy.deepcopy(scenario.config)
    old["preparation_deadline_utc"] = (
        (datetime.now(UTC) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    )
    plan = prep.make_plan(old)
    with pytest.raises(ContractError, match="expired"):
        prep.transition(
            plan,
            {},
            arm=scenario.arm,
            journal=tmp_path / "intent.json",
            approved_plan_sha256=plan["plan_sha256"],
        )
    assert not scenario.arm.patches and not (tmp_path / "intent.json").exists()


def test_ambiguous_patch_reconciliation_does_not_accept_an_extended_cutoff(scenario, tmp_path):
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    result = prep.transition(
        scenario.plan,
        evidence,
        arm=scenario.arm,
        journal=tmp_path / "transition.json",
        approved_plan_sha256=scenario.plan["plan_sha256"],
    )
    scenario.arm.value["properties"]["metadata"][0]["value"] = "2030-01-01T00:00:00Z"
    with pytest.raises(ContractError):
        prep.reconcile_transition(scenario.plan, result, arm=scenario.arm)
    assert len(scenario.arm.patches) == 1


def test_ready_proof_cannot_be_forged_by_rehashing_a_failed_node(scenario, tmp_path):
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    evidence["node"]["start_task_info"]["exitCode"] = 1
    evidence["evidence_sha256"] = digest(
        canonical({name: value for name, value in evidence.items() if name != "evidence_sha256"})
    )
    with pytest.raises(ContractError, match="successful StartTask"):
        prep.transition(
            scenario.plan,
            evidence,
            arm=scenario.arm,
            journal=tmp_path / "intent.json",
            approved_plan_sha256=scenario.plan["plan_sha256"],
        )
    assert not scenario.arm.patches


def test_dummy_or_preexisting_job_is_not_a_valid_preparation_trigger(scenario, tmp_path):
    scenario.batch.list_jobs = lambda **kwargs: iter(
        [
            SimpleNamespace(pool_info=SimpleNamespace(pool_id=scenario.platform.pool_id)),
        ]
    )
    evidence = prep.observe(scenario.plan, arm=scenario.arm, batch=scenario.batch)
    assert evidence["state"] == "rejected"
    with pytest.raises(ContractError):
        prep.transition(
            scenario.plan,
            evidence,
            arm=scenario.arm,
            journal=tmp_path / "intent.json",
            approved_plan_sha256=scenario.plan["plan_sha256"],
        )
    assert not scenario.arm.patches


def test_node_logs_are_preserved_before_ready_marker_and_publication_failure_stops(
    scenario, tmp_path
):
    from learning.paused.pool_preparation_cli import watch

    published = []

    def record(name, value):
        published.append((name, value["evidence_sha256"]))

    result = watch(
        scenario.plan,
        arm=scenario.arm,
        batch=scenario.batch,
        output=tmp_path / "complete",
        publish=record,
    )
    assert result["state"] == "ready"
    assert [name for name, _ in published] == ["sample-0000.json", "result.json"]
    assert (tmp_path / "complete" / "result.json").is_file()

    def fail(name, value):
        raise TimeoutError("Private publication readback unavailable")

    with pytest.raises(TimeoutError):
        watch(
            scenario.plan,
            arm=scenario.arm,
            batch=scenario.batch,
            output=tmp_path / "incomplete",
            publish=fail,
        )
    assert not (tmp_path / "incomplete" / "result.json").exists()
    assert not scenario.arm.patches


def test_changed_reviewed_source_is_rejected_before_observation_or_mutation(scenario):
    changed = copy.deepcopy(scenario.plan)
    changed["source_sha256"]["simulation/batch.py"] = "0" * 64
    with pytest.raises(ContractError, match="source changed"):
        prep.observe(changed, arm=scenario.arm, batch=scenario.batch)
    assert not scenario.arm.patches


def test_cli_plan_is_offline_and_cannot_allocate_or_acquire_credentials(
    scenario, tmp_path, monkeypatch, capsys
):
    import sys

    import azure.identity

    from learning.common import read_json, write_json
    from learning.paused.pool_preparation_cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("Offline plan attempted credential acquisition")

    monkeypatch.setattr(azure.identity, "ManagedIdentityCredential", forbidden)
    monkeypatch.setattr(azure.identity, "AzureCliCredential", forbidden)
    config = tmp_path / "config.json"
    write_json(config, scenario.config)
    output = tmp_path / "offline"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pool-preparation",
            "plan",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )
    main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["cloud_calls"] == 0 and receipt["execution_authorized"] is False
    plan = read_json(output / "plan.json")
    assert read_json(output / "pool.json") == plan["pool"]
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert scenario.arm.patches == []


def test_operator_transport_uses_only_exact_pool_patch_and_original_etag(scenario, monkeypatch):
    from urllib import request

    from simulation.batch import allocation_formula

    from learning.paused.pool_preparation_cli import PoolArm

    scopes, calls = [], []

    def token(scope):
        scopes.append(scope)
        return SimpleNamespace(token="offline-transport-fixture-not-a-credential")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            return b'{"properties":{"provisioningState":"Succeeded"}}'

    def send(value, *, timeout):
        calls.append(value)
        assert 0 < timeout <= 20
        return Response()

    monkeypatch.setattr(request, "urlopen", send)
    transport = PoolArm(scenario.plan, SimpleNamespace(get_token=token), allow_transition=True)
    body = {
        "properties": {
            "scaleSettings": {
                "autoScale": {
                    "formula": allocation_formula(scenario.plan["pool_deadline_utc"]),
                    "evaluationInterval": "PT5M",
                }
            }
        }
    }
    transport.patch(body, etag='"original"')
    assert len(calls) == 1 and calls[0].get_method() == "PATCH"
    assert calls[0].full_url == (
        "https://management.azure.com"
        + prep.pool_resource_id(scenario.plan)
        + "?api-version=2025-06-01"
    )
    assert calls[0].get_header("If-match") == '"original"'
    assert calls[0].data == canonical(body)
    assert scopes == ["https://management.azure.com/.default"]


def test_preparation_does_not_change_the_existing_training_or_model_payload():
    from learning.smolvla.embedded_source import static_code_files

    files = static_code_files(direct=True)
    assert len(files) == 74
    assert not any("pool_preparation" in name for name in files)


def test_operator_transport_cannot_expand_a_transition_into_an_arbitrary_pool_change(scenario):
    from learning.paused.pool_preparation_cli import PoolArm

    credential = SimpleNamespace(
        get_token=lambda *args: pytest.fail("Unapproved PATCH acquired credentials")
    )
    transport = PoolArm(scenario.plan, credential, allow_transition=True)
    with pytest.raises(ContractError, match="exact original canonical formula"):
        transport.patch({"properties": {"vmSize": "unapproved"}}, etag='"original"')
