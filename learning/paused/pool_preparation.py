"""Explicit non-motion preparation; ordinary Batch guards and job clocks are never rewritten."""

from __future__ import annotations

import base64
import copy
import os
import re
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from uuid import UUID

from azure.core.exceptions import AzureError, ResourceNotFoundError

from learning.common import (
    ContractError,
    canonical,
    digest,
    file_digest,
    finite,
    keys,
    parse_json,
    require,
    sha256,
)
from learning.deadlines import JobDeadline, validate_deadline

SCHEMA = "physicalai.batch-pool-preparation/v1"
OBSERVATION_SCHEMA = "physicalai.batch-pool-preparation-observation/v1"
TRANSITION_SCHEMA = "physicalai.batch-pool-preparation-transition/v1"
LOG_PATHS = ("startup/stdout.txt", "startup/stderr.txt", "startup/wd/preflight.json")
MAX_LOG_BYTES = 65536
CONFIG_KEYS = {
    "operation_id",
    "warmup_id",
    "scope",
    "platform",
    "pool",
    "allocation_start_utc",
    "preparation_deadline_utc",
    "pool_deadline_utc",
    "overall_start_utc",
    "overall_deadline_utc",
    "budget_usd",
    "observer_client_id",
    "storage_account_url",
    "artifact_container",
}
POOL_KEYS = {
    "vmSize",
    "deploymentConfiguration",
    "networkConfiguration",
    "startTask",
    "taskSlotsPerNode",
    "interNodeCommunication",
    "taskSchedulingPolicy",
}
SOURCE_FILES = (
    "learning/paused/pool_preparation.py",
    "learning/paused/pool_preparation_cli.py",
    "learning/common.py",
    "learning/deadlines.py",
    "simulation/batch.py",
    "simulation/batch_task.py",
    "simulation/batch_bootstrap.sh",
)
READ_ERRORS = (OSError, AzureError)


def _native():
    from simulation import batch

    return batch


def _sources() -> dict:
    root = Path(__file__).resolve().parents[2]
    return {name: file_digest(root / name) for name in SOURCE_FILES}


def _state(value):
    return getattr(value, "value", value)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _seal(value: dict, field: str) -> dict:
    return {**value, field: digest(canonical(value))}


def _metadata(pool: dict) -> dict:
    items = pool["properties"]["metadata"]
    require(isinstance(items, list) and len(items) <= 32, "Invalid pool metadata")
    value = {item["name"]: item["value"] for item in items}
    require(len(value) == len(items), "Duplicate pool metadata")
    return value


def _etag(value) -> str:
    require(
        isinstance(value, str)
        and 1 <= len(value) <= 256
        and not value.startswith("W/")
        and "\r" not in value
        and "\n" not in value,
        "An actual strong pool ETag is required",
    )
    return value


def preparation_formula(start: str, cutoff: str) -> str:
    validate_deadline(start)
    validate_deadline(cutoff)
    return (
        "$TargetDedicatedNodes = 0;\n"
        f'$TargetLowPriorityNodes = time() < time("{start}") ? 0 : '
        f'(time() < time("{cutoff}") ? 1 : 0);\n'
        "$NodeDeallocationOption = terminate;"
    )


def preparation_target(plan: dict, now: datetime) -> int:
    return int(
        validate_deadline(plan["allocation_start_utc"])
        <= now
        < validate_deadline(plan["preparation_deadline_utc"])
    )


def pool_resource_id(plan: dict) -> str:
    platform = _native().BatchPlatform.model_validate(plan["platform"])
    return _native()._management_account_id(platform) + "/pools/" + platform.pool_id


def _pool_blueprint(config: dict, platform) -> dict:
    pool = copy.deepcopy(config["pool"])
    keys(pool, {"identity", "properties"}, "approved pool resource")
    properties = pool["properties"]
    require(
        set(properties)
        in (
            POOL_KEYS | {"metadata", "scaleSettings"},
            POOL_KEYS | {"metadata", "scaleSettings", "displayName"},
        ),
        "Unexpected pool resource fields",
    )
    require(
        properties["vmSize"].lower() == platform.vm_size.lower()
        and properties["taskSlotsPerNode"] == 1
        and properties["interNodeCommunication"] == "Disabled",
        "Preparation is one LowPriority A10/task slot, never a different compute tier",
    )
    vm = properties["deploymentConfiguration"]["virtualMachineConfiguration"]
    require(
        set(vm).issubset(
            {
                "imageReference",
                "nodeAgentSkuId",
                "containerConfiguration",
                "extensions",
                "osDisk",
            }
        ),
        "Unreviewed VM preparation configuration",
    )
    if "osDisk" in vm:
        require(
            vm["osDisk"].get("diskSizeGB") == 128
            and set(vm["osDisk"]).issubset({"diskSizeGB", "caching"}),
            "Preparation cannot enlarge or replace the approved OS disk",
        )
    container = vm["containerConfiguration"]
    require(
        set(container).issubset({"type", "containerImageNames", "containerRegistries"})
        and container["type"] == "DockerCompatible",
        "Unreviewed container preparation settings",
    )
    for registry in container.get("containerRegistries", []):
        keys(registry, {"registryServer", "identityReference"}, "existing MI registry")
        require(
            registry["registryServer"] == platform.container_image.split("/", 1)[0]
            and registry["identityReference"] == {"resourceId": platform.node_identity_resource_id},
            "Registry preparation must not introduce passwords or other credentials",
        )
    require(
        vm["imageReference"]
        == {
            "publisher": platform.image_publisher,
            "offer": platform.image_offer,
            "sku": platform.image_sku,
            "version": platform.image_version,
        }
        and vm["nodeAgentSkuId"] == platform.node_agent_sku
        and vm["containerConfiguration"]["containerImageNames"] == [platform.container_image]
        and not vm.get("extensions"),
        "Preparation must retain the exact qualified host and container",
    )
    require(
        pool["identity"]["type"] == "UserAssigned"
        and set(pool["identity"]["userAssignedIdentities"]) == {platform.node_identity_resource_id},
        "Preparation may only reuse the exact existing node identity",
    )
    start = properties["startTask"]
    keys(
        start,
        {"commandLine", "waitForSuccess", "maxTaskRetryCount", "userIdentity"},
        "unchanged bootstrap StartTask",
    )
    require(
        start["commandLine"] == _native().driver_bootstrap_command(platform)
        and start["waitForSuccess"] is True
        and start["maxTaskRetryCount"] == 0
        and start["userIdentity"] == {"autoUser": {"scope": "Pool", "elevationLevel": "Admin"}}
        and "containerSettings" not in start,
        "Existing bounded driver bootstrap may not change",
    )
    network = properties["networkConfiguration"]
    require(
        set(network).issubset(
            {
                "subnetId",
                "publicIPAddressConfiguration",
                "dynamicVnetAssignmentScope",
                "enableAcceleratedNetworking",
            }
        ),
        "Preparation cannot add endpoint or network settings",
    )
    require(
        network["subnetId"]
        and network["publicIPAddressConfiguration"]["provision"] == "NoPublicIPAddresses",
        "Preparation cannot open public node networking",
    )
    metadata = _metadata(pool)
    require(
        metadata["physicalaiAllocationDeadline"] == config["pool_deadline_utc"],
        "The original pool cutoff must already be bound in the approved resource",
    )
    require(
        metadata.get("physicalaiPreparationOperation", config["operation_id"])
        == config["operation_id"],
        "Cannot rebind an existing preparation operation",
    )
    metadata["physicalaiPreparationOperation"] = config["operation_id"]
    properties["metadata"] = [
        {"name": key, "value": value} for key, value in sorted(metadata.items())
    ]
    preparation_settings = {
        "autoScale": {
            "formula": preparation_formula(
                config["allocation_start_utc"], config["preparation_deadline_utc"]
            ),
            "evaluationInterval": "PT5M",
        }
    }
    ordinary_settings = {
        "autoScale": {
            "formula": _native().allocation_formula(config["pool_deadline_utc"]),
            "evaluationInterval": "PT5M",
        }
    }
    require(
        properties["scaleSettings"] in (ordinary_settings, preparation_settings),
        "Unknown scale mode must not be silently converted into preparation",
    )
    properties["scaleSettings"] = preparation_settings
    return pool


def make_plan(config: dict) -> dict:
    keys(config, CONFIG_KEYS, "pool preparation configuration")
    require(
        str(UUID(config["operation_id"])) == config["operation_id"]
        and str(UUID(config["warmup_id"])) == config["warmup_id"],
        "New explicit operation and warmup UUIDs are required",
    )
    from learning.contract import Scope

    scope = Scope(**keys(config["scope"], {"tenant_id", "owner_id"}, "scope"))
    scope.validate()
    platform = _native().BatchPlatform.model_validate(config["platform"])
    require(
        str(platform.tenant_id) == scope.tenant_id
        and platform.image_sku == "2204"
        and platform.driver_installation == "bootstrap",
        "Only the scoped, qualified Ubuntu 22.04 bootstrap preparation is supported",
    )
    times = {
        name: validate_deadline(config[name])
        for name in (
            "allocation_start_utc",
            "preparation_deadline_utc",
            "pool_deadline_utc",
            "overall_start_utc",
            "overall_deadline_utc",
        )
    }
    cleanup = times["pool_deadline_utc"] + timedelta(minutes=5)
    require(
        times["overall_start_utc"]
        <= times["allocation_start_utc"]
        < times["preparation_deadline_utc"]
        <= times["pool_deadline_utc"]
        <= times["allocation_start_utc"] + timedelta(minutes=55)
        and cleanup
        <= times["overall_deadline_utc"]
        <= times["overall_start_utc"] + timedelta(hours=4),
        "Preparation/pool/cleanup exceed the original 55+5 minute or overall four-hour bounds",
    )
    require(
        str(UUID(config["observer_client_id"])) == config["observer_client_id"],
        "The existing observer MI client ID must be explicit",
    )
    require(
        re.fullmatch(
            r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", config["storage_account_url"]
        )
        and re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", config["artifact_container"]),
        "Evidence must use the approved Blob account/container without SAS or SharedKey",
    )
    require(
        0 < finite(config["budget_usd"], "explicit budget") <= 10,
        "Preparation cannot increase the approved cost ceiling",
    )
    value = {
        **copy.deepcopy(config),
        "schema": SCHEMA,
        "mode": "bounded_pool_preparation",
        "pool": _pool_blueprint(config, platform),
        "cleanup_deadline_utc": _stamp(cleanup),
        "source_sha256": _sources(),
    }
    return _seal(value, "plan_sha256")


def validate_plan(plan: dict) -> None:
    keys(
        plan,
        CONFIG_KEYS | {"schema", "mode", "cleanup_deadline_utc", "source_sha256", "plan_sha256"},
        "preparation plan",
    )
    require(
        plan == make_plan({key: plan[key] for key in CONFIG_KEYS}),
        "Preparation plan, fixed cutoff or reviewed source changed",
    )


def _pool_phase(plan: dict, observed: dict) -> str:
    require(
        observed["id"].lower() == pool_resource_id(plan).lower()
        and observed["name"] == plan["platform"]["pool_id"],
        "Observed another pool",
    )
    expected, actual = plan["pool"]["properties"], observed["properties"]
    require(
        all(actual.get(key) == expected[key] for key in POOL_KEYS)
        and observed["identity"]["type"] == "UserAssigned"
        and {key.lower() for key in observed["identity"]["userAssignedIdentities"]}
        == {key.lower() for key in plan["pool"]["identity"]["userAssignedIdentities"]}
        and _metadata(observed) == _metadata(plan["pool"]),
        "Actual preparation pool configuration/identity/cutoff differs from its plan",
    )
    require(
        actual["currentDedicatedNodes"] == 0 and actual["currentLowPriorityNodes"] in (0, 1),
        "Preparation exceeded its one LowPriority node ceiling",
    )
    settings = actual["scaleSettings"]["autoScale"]
    require(settings["evaluationInterval"] == "PT5M", "Autoscale evaluation bound changed")
    formula = settings["formula"]
    if formula == expected["scaleSettings"]["autoScale"]["formula"]:
        return "preparation"
    require(
        formula == _native().allocation_formula(plan["pool_deadline_utc"]),
        "Unknown allocation formula or renewed deadline",
    )
    return "canonical"


def _read_logs(batch, pool_id: str, node_id: str) -> dict:
    result = {}
    for path in LOG_PATHS:
        try:
            body = bytearray()
            for part in batch.download_node_file(pool_id, node_id, path, ocp_range="bytes=0-65536"):
                body.extend(part)
                require(
                    len(body) <= MAX_LOG_BYTES + 1, "Log endpoint exceeded requested byte range"
                )
            truncated = len(body) > MAX_LOG_BYTES
            captured = bytes(body[:MAX_LOG_BYTES])
            result[path] = {
                "state": "available",
                "bytes": len(captured),
                "sha256": digest(captured),
                "base64": base64.b64encode(captured).decode(),
                "truncated": truncated,
                "requested_range": "bytes=0-65536",
            }
        except (OSError, AzureError) as error:
            result[path] = {
                "state": "unavailable",
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
            }
    return result


def _no_jobs(batch, pool_id: str) -> None:
    jobs = list(islice(batch.list_jobs(max_results=1000), 1001))
    require(
        len(jobs) <= 1000
        and not any(job.pool_info is not None and job.pool_info.pool_id == pool_id for job in jobs),
        "Preparation pool already has a job; no dummy, warmup or physical job may be present",
    )


def _ready_proof(plan: dict, node: dict, logs: dict, now: datetime) -> dict:
    require(
        node["state"] == "idle"
        and node["is_dedicated"] is False
        and node["running_tasks"] == 0
        and node["start_task_info"]
        and node["start_task_info"].get("exitCode") == 0
        and not node["errors"],
        "Node evidence is not idle with a successful StartTask",
    )
    require(set(logs) == set(LOG_PATHS), "Missing bounded node log inventory")
    for value in logs.values():
        require(value["state"] == "available", "Node log/proof is unavailable")
        body = base64.b64decode(value["base64"], validate=True)
        require(
            len(body) == value["bytes"] <= MAX_LOG_BYTES and digest(body) == value["sha256"],
            "Captured node log bytes changed",
        )
    value = logs["startup/wd/preflight.json"]
    require(not value["truncated"], "GPU preflight exceeds its original bound")
    proof = parse_json(base64.b64decode(value["base64"], validate=True))
    from simulation.batch_task import validate_gpu_evidence

    validate_gpu_evidence(proof)
    require(
        proof.get("bootstrap_sha256") == digest(_native().bootstrap_source())
        and proof.get("driver_installer_sha256") == _native().GRID_INSTALLER_SHA256,
        "GPU proof does not bind the reviewed bootstrap/driver",
    )
    started = datetime.fromisoformat(node["start_task_info"]["startTime"].replace("Z", "+00:00"))
    observed = datetime.fromisoformat(proof["observed_at_utc"].replace("Z", "+00:00"))
    require(started <= observed <= now, "GPU proof is stale or future-dated")
    return proof


def observe(plan: dict, *, arm, batch, clock=None) -> dict:
    """Only GET/list/download. It has no transition, termination, allocation or task fallback."""
    validate_plan(plan)
    clock = clock or (lambda: datetime.now(UTC))
    now = clock()
    result = {
        "schema": OBSERVATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "operation_id": plan["operation_id"],
        "observed_at_utc": _stamp(now),
        "state": "unknown",
        "phase": None,
        "pool_etag": None,
        "node": None,
        "logs": {},
        "errors": [],
        "cloud_mutations": 0,
    }
    try:
        require(
            validate_deadline(plan["allocation_start_utc"])
            <= now
            < validate_deadline(plan["pool_deadline_utc"]),
            "Original pool window is not active",
        )
        current = arm.get()
        result["phase"], result["pool_etag"] = _pool_phase(plan, current), _etag(current["etag"])
        pool_id = plan["platform"]["pool_id"]
        actual_pool = batch.get_pool(pool_id)
        require(
            actual_pool.current_dedicated_nodes == actual_pool.target_dedicated_nodes == 0
            and actual_pool.current_low_priority_nodes in (0, 1)
            and actual_pool.target_low_priority_nodes in (0, 1),
            "Unexpected actual node targets",
        )
        _no_jobs(batch, pool_id)
        nodes = list(islice(batch.list_nodes(pool_id, max_results=2), 3))
        require(len(nodes) <= 1, "More than one node in preparation")
        result["state"] = "not_ready"
        if nodes:
            node = nodes[0]
            result["node"] = {
                "id": node.id,
                "state": _state(node.state),
                "is_dedicated": node.is_dedicated,
                "running_tasks": node.running_tasks_count,
                "start_task_info": node.start_task_info.as_dict() if node.start_task_info else None,
                "errors": [item.as_dict() for item in (getattr(node, "errors", None) or [])],
                "raw": node.as_dict(),
            }
            result["logs"] = _read_logs(batch, pool_id, node.id)
            if (
                result["node"]["state"] in ("starttaskfailed", "unusable", "preempted")
                or result["node"]["errors"]
                or (node.start_task_info and node.start_task_info.exit_code not in (None, 0))
            ):
                result["state"] = "rejected"
                result["errors"].append("Actual node/StartTask failed; no transition or retry")
            elif any(value["state"] != "available" for value in result["logs"].values()):
                result["state"] = "unknown"
                result["errors"].append("Required node log/proof read is unavailable")
            elif (
                result["node"]["state"] == "idle"
                and node.is_dedicated is False
                and node.running_tasks_count == 0
                and node.start_task_info
                and node.start_task_info.exit_code == 0
                and not result["node"]["errors"]
                and actual_pool.current_low_priority_nodes
                == actual_pool.target_low_priority_nodes
                == 1
            ):
                _ready_proof(plan, result["node"], result["logs"], now)
                result["state"] = "ready"
        if result["phase"] == "preparation" and now >= validate_deadline(
            plan["preparation_deadline_utc"]
        ):
            result["state"] = "expired"
    except READ_ERRORS as error:
        result["state"] = "unknown"
        result["errors"].append({"type": type(error).__name__, "message": str(error)[:500]})
    except (ContractError, KeyError, TypeError, ValueError) as error:
        result["state"] = "rejected"
        result["errors"].append({"type": type(error).__name__, "message": str(error)[:500]})
    return _seal(result, "evidence_sha256")


def _evidence(plan: dict, evidence: dict, *, clock=None, fresh=True) -> None:
    now = (clock or (lambda: datetime.now(UTC)))()
    require(
        evidence["evidence_sha256"]
        == digest(
            canonical({key: value for key, value in evidence.items() if key != "evidence_sha256"})
        )
        and evidence["plan_sha256"] == plan["plan_sha256"]
        and evidence["operation_id"] == plan["operation_id"]
        and evidence["schema"] == OBSERVATION_SCHEMA
        and evidence["state"] == "ready"
        and not evidence["errors"],
        "Only complete, plan-bound ready node evidence can advance",
    )
    _ready_proof(plan, evidence["node"], evidence["logs"], now)
    if fresh:
        require(
            0 <= (now - validate_deadline(evidence["observed_at_utc"])).total_seconds() <= 120,
            "Preparation evidence is stale or future-dated",
        )
    require(
        now < validate_deadline(plan["pool_deadline_utc"]) - timedelta(seconds=660),
        "Insufficient original window for ordinary readiness and an unshortened physical grant",
    )


def _intent(path: Path, value: dict) -> None:
    require(
        not path.exists() and not path.is_symlink() and not path.parent.is_symlink(),
        "Operation already requested; reconcile the original intent instead of retrying",
    )
    body = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def reconcile_transition(plan: dict, intent: dict, *, arm) -> dict:
    validate_plan(plan)
    require(
        intent["plan_sha256"] == plan["plan_sha256"]
        and intent["operation_id"] == plan["operation_id"],
        "Transition intent was rebound",
    )
    result = {
        **intent,
        "schema": TRANSITION_SCHEMA,
        "state": "unknown",
        "observed_at_utc": _stamp(datetime.now(UTC)),
        "cloud_mutations": 0,
    }
    try:
        current = arm.get()
        phase = _pool_phase(plan, current)
        actual_etag = _etag(current["etag"])
        result["pool_etag"] = actual_etag
        if phase == "canonical":
            require(
                actual_etag != intent["original_pool_etag"],
                "Transition did not change its actual ETag",
            )
            result["state"] = "canonical_observed"
        else:
            require(
                actual_etag == intent["original_pool_etag"], "Unexpected concurrent pool mutation"
            )
            result["state"] = "preparation_unchanged"
    except READ_ERRORS as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    return result


def transition(
    plan: dict, evidence: dict, *, arm, journal: Path, approved_plan_sha256: str
) -> dict:
    validate_plan(plan)
    deadline = JobDeadline(plan["preparation_deadline_utc"])
    deadline.check()
    require(plan["plan_sha256"] == sha256(approved_plan_sha256), "Unapproved preparation plan")
    _evidence(plan, evidence)
    require(
        evidence["phase"] == "preparation"
        and datetime.now(UTC) < validate_deadline(plan["preparation_deadline_utc"]),
        "Preparation expired or already transitioned; no renewal",
    )
    try:
        current = arm.get()
    except READ_ERRORS as error:
        raise ContractError("Operator read is UNKNOWN; no transition request issued") from error
    require(
        _pool_phase(plan, current) == "preparation"
        and _etag(current["etag"]) == evidence["pool_etag"],
        "Original evidence ETag no longer identifies the actual preparation pool",
    )
    deadline.check()
    patch = {
        "properties": {
            "scaleSettings": {
                "autoScale": {
                    "formula": _native().allocation_formula(plan["pool_deadline_utc"]),
                    "evaluationInterval": "PT5M",
                }
            }
        }
    }
    intent = {
        "operation_id": plan["operation_id"],
        "plan_sha256": plan["plan_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "original_pool_etag": current["etag"],
        "requested_at_utc": _stamp(datetime.now(UTC)),
        "patch": patch,
    }
    _intent(journal, intent)
    outcome = "acknowledged"
    error = None
    try:
        deadline.check()
        arm.patch(patch, etag=current["etag"])
    except READ_ERRORS as failure:
        outcome = "unknown"
        error = {"type": type(failure).__name__, "message": str(failure)[:500]}
    result = reconcile_transition(plan, intent, arm=arm)
    result.update(patch_outcome=outcome, patch_error=error, patch_requests=1)
    return result


def verify_transition(plan: dict, evidence: dict, transition_result: dict, *, arm, batch) -> dict:
    validate_plan(plan)
    _evidence(plan, evidence, fresh=False)
    require(
        transition_result["plan_sha256"] == plan["plan_sha256"]
        and transition_result["evidence_sha256"] == evidence["evidence_sha256"],
        "Transition no longer binds the original ready evidence",
    )
    current = reconcile_transition(plan, transition_result, arm=arm)
    require(current["state"] == "canonical_observed", "Canonical transition not confirmed")
    platform = _native().BatchPlatform.model_validate(plan["platform"])
    _no_jobs(batch, platform.pool_id)
    # Never rewrite a preparation pool object to satisfy these existing native guards.
    _native().validate_pool(batch.get_pool(platform.pool_id), platform)
    ready = _native().inspect_platform(batch, platform)
    original_proof = parse_json(
        base64.b64decode(
            evidence["logs"]["startup/wd/preflight.json"]["base64"],
            validate=True,
        )
    )
    require(
        ready["node_id"] == evidence["node"]["id"]
        and digest(canonical(ready["hardware"])) == digest(canonical(original_proof)),
        "The original verified node or bootstrap proof changed",
    )
    return {**current, "state": "canonical_verified", "native_readiness": ready}


def submit_warmup_once(
    plan: dict,
    evidence: dict,
    transition_result: dict,
    *,
    arm,
    batch,
    journal: Path,
    approved_plan_sha256: str,
) -> dict:
    require(
        not journal.exists() and not journal.is_symlink(),
        "Warmup already requested; reconcile its original UUID, never retry submission",
    )
    require(plan["plan_sha256"] == sha256(approved_plan_sha256), "Unapproved preparation operation")
    verified = verify_transition(plan, evidence, transition_result, arm=arm, batch=batch)
    platform = _native().BatchPlatform.model_validate(plan["platform"])
    job, task = _native().build_warmup(platform, UUID(plan["warmup_id"]))
    require(
        job.constraints.max_wall_clock_time == timedelta(minutes=30)
        and task.constraints.max_wall_clock_time == timedelta(seconds=60),
        "Original ordinary warmup bounds changed",
    )
    try:
        batch.get_job(job.id)
    except ResourceNotFoundError:
        pass
    else:
        raise ContractError("Original warmup already exists; read its state, never resubmit")
    _evidence(plan, evidence, fresh=False)
    _intent(
        journal,
        {
            "plan_sha256": plan["plan_sha256"],
            "operation_id": plan["operation_id"],
            "warmup_job_id": job.id,
            "task_id": task.id,
            "requested_at_utc": _stamp(datetime.now(UTC)),
            "transition": verified,
        },
    )
    result = _native().submit_requests_once(batch, job, task)
    return {
        **result,
        "state": "warmup_submitted",
        "ordinary_preflight_verified": False,
        "physical_grant_issued": False,
        "original_pool_deadline_utc": plan["pool_deadline_utc"],
    }
