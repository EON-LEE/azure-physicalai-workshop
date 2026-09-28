"""Post-hoc verification of operator-native training, never an API training authorization."""

import io
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from azure.core.exceptions import AzureError
from pydantic import AwareDatetime, Field

from apps.api.errors import Problem, unavailable
from apps.api.learning_models import (
    DatasetVersion,
    ExternalImportReference,
    ExternalTrainingImport,
    ExternalTrainingResult,
    Frozen,
    PolicyCandidate,
    fingerprint,
)
from apps.api.models import Revision, utcnow
from apps.learning_worker.candidate_provenance import (
    VerifiedCommandJob,
    command_config,
    read_command_result,
    verify_candidate_record,
)
from apps.learning_worker.managed_reports import _version
from learning.common import (
    canonical,
    digest,
    file_digest,
    keys,
    parse_json,
    read_json,
    safe_path,
    utc,
)


class ExternalImportCompletion(Frozen):
    schema_version: Literal["physicalai.external-native-training-request/v1"] = Field(
        alias="schema"
    )
    import_id: UUID
    project_sha256: Revision
    created_at: AwareDatetime
    native_job_name: UUID
    approval_sha256: Revision
    claim_sha256: Revision
    configuration_file_sha256: Revision
    specification_file_sha256: Revision
    result_sha256: Revision
    transfer_sha256: Revision
    model_sha256: Revision


@dataclass(frozen=True)
class ExternalContext:
    reference: ExternalImportReference
    request: ExternalImportCompletion
    prefix: str
    native_prefix: str
    config: dict
    specification: dict
    approval: dict
    claim: dict
    approval_modified: datetime
    claim_modified: datetime


def prefix(project_id, import_id):
    return f"projects/{UUID(str(project_id))}/external-native-training/{UUID(str(import_id))}"


def _read(registry, actor, name, expected=None, *, max_bytes=1024**2):
    content, modified = registry.import_document(actor, name, max_bytes=max_bytes)
    if expected is not None and digest(content) != expected:
        raise Problem(409, "external_import_digest", "Original native document bytes changed.")
    try:
        value = parse_json(content)
    except ValueError as exc:
        raise Problem(409, "external_import_document", "Invalid original native JSON.") from exc
    if not isinstance(value, dict):
        raise Problem(409, "external_import_document", "Expected an original native document.")
    return value, content, modified


def context(registry, actor, project, import_id, *, expected=None):
    base = prefix(project.id, import_id)
    value, content, modified = _read(registry, actor, f"{base}/completion.json")
    request = ExternalImportCompletion.model_validate(value)
    reference = ExternalImportReference(
        import_id=import_id,
        project_id=project.id,
        owner_key=actor.owner_key,
        completion_sha256=digest(content),
    )
    if (
        request.import_id != import_id
        or project.owner_key != actor.owner_key
        or project.tenant_id != actor.tenant_id
        or project.actor_id != actor.object_id
        or project.execution_timing != "paused_simulation"
        or project.policy_type != "smolvla"
        or request.project_sha256 != fingerprint(project.public())
        or request.created_at >= modified + timedelta(seconds=1)
        or modified > utcnow()
        or (expected is not None and expected != reference)
    ):
        raise Problem(403, "external_import_scope", "Post-hoc import owner or project differs.")
    config, _, _ = _read(
        registry, actor, f"{base}/files/run-config.json", request.configuration_file_sha256
    )
    specification, _, _ = _read(
        registry, actor, f"{base}/files/specification.json", request.specification_file_sha256
    )
    native = f"native-operator/{request.native_job_name}"
    approval, _, approved_at = _read(
        registry, actor, f"{native}/approval.json", request.approval_sha256
    )
    claim, _, claimed_at = _read(registry, actor, f"{native}/claim.json", request.claim_sha256)
    if not command_config(config):
        raise Problem(
            409, "external_import_command", "Only original standalone commands can import."
        )
    try:
        required = {
            "schema",
            "scope",
            "job_name",
            "job_deadline_utc",
            "maximum_gpu_nodes",
            "maximum_cost_usd",
            "price_is_not_actual_billing",
            "public_list_price_usd_per_hour",
            "plan_blob",
            "plan_archive_sha256",
            "plan_sha256",
            "configuration_sha256",
            "global_target_steps",
            "checkpoint_steps",
            "resume",
            "environment_image",
            "source_delivery",
            "job_execution",
            "image_qualification_sha256",
            "image_qualification_blob",
        }
        keys(
            approval,
            required
            | ({"previous_failed_attempt"} if "previous_failed_attempt" in approval else set()),
            "native approval",
        )
        keys(
            claim,
            {
                "scope",
                "job_name",
                "job_type",
                "approval_sha256",
                "plan_sha256",
                "configuration_sha256",
                "job_deadline_utc",
                "claimed_at_utc",
            },
            "native claim",
        )
        approved_cost = Decimal(str(approval["maximum_cost_usd"]))
        hourly = Decimal(str(approval["public_list_price_usd_per_hour"]))
        claimed = utc(claim["claimed_at_utc"].replace("+00:00", "Z"))
        deadline = utc(config["job_deadline_utc"])
    except (ValueError, KeyError, ArithmeticError, AttributeError) as exc:
        raise Problem(
            409, "external_import_approval", "Retained native approval is incomplete."
        ) from exc
    scope = {"tenant_id": str(actor.tenant_id), "owner_id": actor.owner_key}
    same = {
        "scope": scope,
        "job_name": str(request.native_job_name),
        "job_deadline_utc": config["job_deadline_utc"],
        "maximum_gpu_nodes": 1,
        "environment_image": config["environment_image"],
        "source_delivery": config["source_delivery"],
        "job_execution": config["job_execution"],
    }
    if (
        approval["schema"] != "physicalai.native-training-approval/v1"
        or config["run_id"] != str(request.native_job_name)
        or config["tenant_id"] != str(actor.tenant_id)
        or config["owner_id"] != actor.owner_key
        or any(
            approval.get(name) != item or specification.get(name) != item
            for name, item in same.items()
        )
        or specification.get("schema") != "physicalai.native-bootstrap-authorization/v1"
        or specification.get("id") != str(request.native_job_name)
        or specification.get("parameters") != config["parameters"]
        or specification.get("inputs") != config["inputs"]
        or specification.get("learning_quality_claim") is not False
        or specification.get("image_qualification_sha256") != approval["image_qualification_sha256"]
        or specification.get("maximum_cost_usd") != approval["maximum_cost_usd"]
        or specification.get("previous_failed_attempt") != approval.get("previous_failed_attempt")
        or digest(canonical(specification)) != config["specification_sha256"]
        or digest(canonical(config)) != approval["configuration_sha256"]
        or approval["global_target_steps"] != config["parameters"]["max_steps"]
        or approval["checkpoint_steps"] != config["parameters"]["checkpoint_steps"]
        or approval["resume"] != config["checkpointing"]["resume"]
        or approval["price_is_not_actual_billing"] is not True
        or type(approval["maximum_gpu_nodes"]) is not int
        or not approved_cost.is_finite()
        or not 0 < approved_cost <= project.budget.maximum_cost_usd
        or not hourly.is_finite()
        or hourly <= 0
        or any(
            claim.get(name) != item
            for name, item in {
                "scope": scope,
                "job_name": str(request.native_job_name),
                "job_type": "command",
                "approval_sha256": request.approval_sha256,
                "plan_sha256": approval["plan_sha256"],
                "configuration_sha256": approval["configuration_sha256"],
                "job_deadline_utc": config["job_deadline_utc"],
            }.items()
        )
        or not approved_at + timedelta(seconds=1) <= claimed < deadline
        or claimed >= claimed_at + timedelta(seconds=1)
        or claimed_at + timedelta(seconds=1) > request.created_at
        or Decimal(str((deadline - approved_at).total_seconds())) * hourly / 3600 > approved_cost
        or config["parameters"]["timeout_seconds"] > project.budget.training_seconds
        or config["parameters"]["max_steps"] > project.budget.optimizer_steps
        or any(
            config[name] != getattr(project, name)
            for name in (
                "control_profile_sha256",
                "criteria_sha256",
                "frozen_plan_sha256",
            )
        )
        or config["task_sha256"]
        != fingerprint(
            {
                "task_id": project.task_id,
                "instruction": project.instruction,
                "goal_id": project.goal_station_id,
            }
        )
        or f"https://{config['storage_account_name']}.blob.core.windows.net"
        != registry.client.url.rstrip("/")
        or config["blob_container"] != registry.container.container_name
        or approval["plan_blob"] != registry.key(actor, f"{native}/plan.tar.gz")
        or approval["image_qualification_blob"]
        != registry.key(actor, f"{native}/image-qualification.json")
    ):
        raise Problem(
            409, "external_import_approval", "Native approval, claim or source authority differs."
        )
    return ExternalContext(
        reference,
        request,
        base,
        native,
        config,
        specification,
        approval,
        claim,
        approved_at,
        claimed_at,
    )


def input_locations(registry, actor, value):
    from learning.paused.blob_transfer import input_prefix

    config = value.config
    output = f"{config['output_prefix']}/{config['run_id']}"
    return (
        registry.key(actor, f"{value.prefix}/files/"),
        registry.key(actor, f"{value.native_prefix}/"),
        *(input_prefix(config, name).rstrip("/") + "/" for name in config["inputs"]),
        output + "/model/",
        output + "/transfer/",
        *(() if config["checkpointing"]["resume"] is not None else (output + "/dataset/",)),
    )


def _inventory(verifier, actor, value):
    registry, config = verifier.registry, value.config
    locations = [
        (location, False)
        for location in input_locations(registry, actor, value)
        if location != registry.key(actor, f"{value.native_prefix}/")
    ] + [
        (registry.key(actor, f"{value.native_prefix}/{name}"), True)
        for name in ("approval.json", "claim.json", "plan.tar.gz", "image-qualification.json")
    ]
    return fingerprint(
        {
            location: verifier._blob_inventory(
                registry.client.url.rstrip("/"), config["blob_container"], location, exact=exact
            )
            for location, exact in locations
        }
    )


def _plan(registry, actor, value, directory):
    from learning.common import relative_path
    from learning.smolvla.azure import read_plan

    content, modified = registry.import_document(
        actor, f"{value.native_prefix}/plan.tar.gz", max_bytes=32 * 1024**2
    )
    if (
        digest(content) != value.approval["plan_archive_sha256"]
        or modified > value.approval_modified
    ):
        raise Problem(
            409, "external_import_plan", "The retained pre-execution plan archive differs."
        )
    _, _, qualified_at = _read(
        registry,
        actor,
        f"{value.native_prefix}/image-qualification.json",
        value.approval["image_qualification_sha256"],
        max_bytes=16 * 1024**2,
    )
    if qualified_at > value.approval_modified:
        raise Problem(
            409, "external_import_plan", "Image qualification was not retained before approval."
        )
    seen, size = set(), 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for member in archive:
            path = str(relative_path(member.name.rstrip("/")))
            if path in seen or not (path == "plan" or path.startswith("plan/")):
                raise Problem(
                    409,
                    "external_import_plan",
                    "Plan archive contains a duplicate or foreign path.",
                )
            seen.add(path)
            if len(seen) > 1024 or not (member.isfile() or member.isdir()):
                raise Problem(
                    409, "external_import_plan", "Plan archive contains unapproved entries."
                )
            size += member.size
            if size > 16 * 1024**2:
                raise Problem(
                    409, "external_import_plan", "Plan archive exceeds its unpacked bound."
                )
            target = directory.joinpath(*path.split("/"))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise Problem(409, "external_import_plan", "Original plan member is missing.")
            with source, target.open("xb") as destination:
                destination.write(source.read(member.size + 1))
    plan = read_plan(directory / "plan")
    if (
        plan["config"] != value.config
        or plan["job_name"] != str(value.request.native_job_name)
        or plan["plan_sha256"] != value.approval["plan_sha256"]
    ):
        raise Problem(409, "external_import_plan", "Retained native plan/config/source differs.")
    return plan


def _job(verifier, actor, value, plan):
    from learning.gr00t.azure import workspace_id
    from learning.paused.command import validate_command_job

    config = value.config
    try:
        client = verifier._metadata_client(config)
        job = client.jobs.get(str(value.request.native_job_name))
        observed = validate_command_job(
            client,
            config,
            job,
            snapshot_sha256=plan["snapshot_sha256"],
            expected_status="Completed",
        )
    except AzureError as exc:
        raise unavailable("Original external Azure ML command") from exc
    except ValueError as exc:
        raise Problem(
            409, "external_import_job", "Actual original command does not match native approval."
        ) from exc
    created = getattr(getattr(job, "creation_context", None), "created_at", None)
    if (
        not isinstance(created, datetime)
        or created.tzinfo is None
        or value.claim_modified + timedelta(seconds=1) > created
        or not created < utc(config["job_deadline_utc"])
        or not created < value.request.created_at
        or observed
        != {
            "azure_job_id": workspace_id(config) + "/jobs/" + str(value.request.native_job_name),
            "azure_job_type": "command",
            "specification_sha256": config["specification_sha256"],
        }
        or (job.tags or {}).get("plan_sha256") != plan["plan_sha256"]
    ):
        raise Problem(
            409, "external_import_job", "Actual creation/claim/plan chronology is not proven."
        )
    return VerifiedCommandJob(
        actor.owner_key,
        observed["azure_job_id"],
        config["specification_sha256"],
        digest(canonical(config)),
        plan["snapshot_sha256"],
    ), created


def _download(verifier, config, location, target):
    verifier._download_prefix(
        f"https://{config['storage_account_name']}.blob.core.windows.net",
        config["blob_container"],
        location.rstrip("/") + "/",
        target,
    )


def verify_payload(verifier, actor, project, value, directory, proof):
    from learning.paused.blob_transfer import input_prefix
    from learning.paused.capture import validate_dataset
    from learning.paused.command_artifacts import validate_parent
    from learning.paused.dataset import validate_conversion
    from learning.smolvla.artifacts import validate_backbone
    from learning.smolvla.train import TrainOptions

    config = value.config
    roots = {}
    for name in ("demonstrations", "parent_model", "backbone"):
        roots[name] = directory / name
        _download(verifier, config, input_prefix(config, name), roots[name])
    raw = validate_dataset(
        roots["demonstrations"],
        expected_scope=verifier._scope(actor),
        expected_manifest_sha256=config["inputs"]["demonstrations"]["sha256"],
        require_live=True,
        require_demonstrations=True,
    )
    parent = validate_parent(
        roots["parent_model"],
        expected_scope=verifier._scope(actor),
        expected_model_sha256=config["inputs"]["parent_model"]["sha256"],
        for_inference=False,
    )
    validate_backbone(
        roots["backbone"],
        scope=verifier._scope(actor),
        expected_sha256=config["inputs"]["backbone"]["sha256"],
    )
    output = f"{config['output_prefix']}/{config['run_id']}"
    roots["converted"] = directory / "converted"
    conversion_prefix = (
        input_prefix(config, "converted_dataset")
        if config["checkpointing"]["resume"]
        else output + "/dataset"
    )
    _download(verifier, config, conversion_prefix, roots["converted"])
    conversion_root = roots["converted"] / "dataset"
    converted = validate_conversion(conversion_root, verifier._scope(actor))
    roots["model"] = directory / "model"
    _download(verifier, config, output + "/model", roots["model"])
    result = read_command_result(roots["model"] / "result.json")
    result_blob = verifier.registry.import_blob(
        actor,
        (output + "/model/result.json").removeprefix(verifier.registry.key(actor, "")),
        max_bytes=1024**2,
    )
    transfer, _, transfer_time = _read(
        verifier.registry,
        actor,
        (output + "/transfer/completion.json").removeprefix(verifier.registry.key(actor, "")),
        value.request.transfer_sha256,
    )
    result_bytes, result_time = result_blob.content, result_blob.modified_at
    if (
        digest(result_bytes) != value.request.result_sha256
        or file_digest(roots["model"] / "result.json") != value.request.result_sha256
        or result_time + timedelta(seconds=1) > utc(config["job_deadline_utc"])
        or result_time >= value.request.created_at
        or transfer_time + timedelta(seconds=1) > utc(config["job_deadline_utc"])
        or transfer_time < result_time
        or transfer.get("schema") != "physicalai.smolvla-command-transfer/v1"
        or transfer.get("azure_job_id") != proof.azure_job_id
        or transfer.get("azure_job_type") != "command"
        or transfer.get("specification_sha256") != proof.specification_sha256
        or transfer.get("optimizer_steps") != result["optimizer_steps"]
        or transfer.get("learning_quality_verified") is not False
    ):
        raise Problem(
            409, "external_import_result", "Original result bytes or publication deadline differs."
        )
    marker = transfer.get("model_result", {})
    if (
        marker.get("prefix") != output + "/model"
        or marker.get("marker") != "result.json"
        or marker.get("marker_sha256") != value.request.result_sha256
        or marker.get("readback_verified") is not True
        or marker.get("etag") != result_blob.etag
    ):
        raise Problem(409, "external_import_result", "Original readback transfer receipt differs.")
    candidate_root = safe_path(roots["model"], result["candidate"], must_exist=False)
    model = verifier._model(
        actor,
        candidate_root,
        value.request.model_sha256,
        "smolvla",
        execution_timing="paused_simulation",
        verified_command=proof,
    )
    training, options = model["training"], TrainOptions(**config["parameters"])
    raw_manifest = raw.manifest
    episodes = raw_manifest["episodes"]
    cohort = config.get("training_cohort")
    expected_seeds = (
        set(range(10001, 10021))
        if cohort is None
        else set(range(11001, 11021))
        if cohort == {"schema": "physicalai.smolvla-training-cohort/v1", "kind": "p1_additional20"}
        else set()
    )
    selected = {
        (item.environment_id, item.revision, item.seed, item.split)
        for item in project.teaching_cases
    }
    raw_lineage = {
        (item["episode_id"], item["environment_id"], item["revision"], item["seed"])
        for item in episodes
    }
    lineage = {
        tuple(item[name] for name in ("episode_id", "environment_id", "revision", "seed"))
        for item in training["episodes"]
    }
    previous_lineage = {
        tuple(item[name] for name in ("episode_id", "environment_id", "revision", "seed"))
        for item in (parent["training"]["episodes"] if parent["training"] else [])
    }
    if (
        len(episodes) != 20
        or {item["seed"] for item in episodes} != expected_seeds
        or any(
            item["split"] != "train"
            or (item["environment_id"], item["revision"], item["seed"], "train") not in selected
            for item in episodes
        )
        or lineage != raw_lineage | previous_lineage
        or any(item["seed"] in project.evaluation_plan.seeds for item in training["episodes"])
        or training["raw_manifest_sha256"] != config["inputs"]["demonstrations"]["sha256"]
        or training["parent_model_sha256"] != config["inputs"]["parent_model"]["sha256"]
        or (
            config["checkpointing"]["resume"] is None
            and training["parent_weights_sha256"] != parent["weights_sha256"]
        )
        or (
            config["checkpointing"]["resume"] is None
            and parent["weights_sha256"] == model["weights_sha256"]
        )
        or model["backbone_manifest_sha256"] != config["inputs"]["backbone"]["sha256"]
        or parent["backbone_manifest_sha256"] != config["inputs"]["backbone"]["sha256"]
        or training["config_sha256"] != digest(canonical(asdict(options)))
        or training["code_snapshot_sha256"] != proof.snapshot_sha256
        or training["specification_sha256"] != config["specification_sha256"]
        or result["specification_sha256"] != config["specification_sha256"]
        or result["azure_job_id"] != proof.azure_job_id
        or result["model_manifest_sha256"] != value.request.model_sha256
        or training["optimizer_steps"] != result["optimizer_steps"]
        or training["checkpoint_step"] > options.max_steps
        or result["candidate"] != f"candidates/step-{training['checkpoint_step']:06d}"
        or training["resume_mode"] != options.resume_mode
        or training["conversion_sha256"] != file_digest(conversion_root / "conversion.json")
        or converted["raw_manifest_sha256"] != raw.manifest_sha256
        or converted.get("test_only") is not False
        or verifier._model_timing(model) != project.timing_fields()
        or model["task"]
        != {
            "task_id": project.task_id,
            "instruction": project.instruction,
            "goal_id": project.goal_station_id,
        }
        or any(
            fingerprint(item["control_profile"]) != project.control_profile_sha256
            or item["criteria_sha256"] != project.criteria_sha256
            or item["frozen_plan_sha256"] != project.frozen_plan_sha256
            for item in (raw_manifest, parent, converted)
        )
        or sorted(
            tuple(
                item[name]
                for name in ("episode_id", "environment_id", "revision", "seed", "frame_count")
            )
            for item in episodes
        )
        != sorted(
            tuple(
                item[name]
                for name in ("episode_id", "environment_id", "revision", "seed", "frame_count")
            )
            for item in converted["episodes"]
        )
    ):
        raise Problem(
            409, "external_import_lineage", "Actual native data/model/optimizer lineage differs."
        )
    resume = config["checkpointing"]["resume"]
    if resume is not None:
        from learning.paused.command import checkpoint_origin
        from learning.smolvla.checkpoint_runner import make_binding, validate_resume_checkpoint

        roots["resume"] = directory / "resume"
        _download(verifier, config, input_prefix(config, "resume_checkpoint"), roots["resume"])
        source = read_json(roots["resume"] / "checkpoint.json")
        binding = make_binding(config, parent, converted, training["conversion_sha256"], {})
        binding["runtime_sha256"] = source["binding"]["runtime_sha256"]
        checkpoint = validate_resume_checkpoint(
            roots["resume"],
            config=config,
            expected_binding=binding,
            current_origin=checkpoint_origin(
                {
                    "azure_job_id": proof.azure_job_id,
                    "specification_sha256": proof.specification_sha256,
                },
                config=config,
                snapshot_sha256=proof.snapshot_sha256,
            ),
        )
        if (
            checkpoint["origin"]["azure_job_id"] != resume["source_azure_job_id"]
            or checkpoint["origin"].get("azure_job_type") != "command"
            or checkpoint["step"] != resume["step"]
            or training.get("resume_from_checkpoint_sha256") != resume["checkpoint_sha256"]
            or training.get("resume_from_job_id") != resume["source_azure_job_id"]
            or config["inputs"]["converted_dataset"]["sha256"] != training["conversion_sha256"]
        ):
            raise Problem(
                409, "external_import_resume", "Original external resume provenance differs."
            )
    return raw, model, candidate_root, roots["demonstrations"]


def _result(actor, project, value, proof, created, raw, model):
    now = utcnow()
    import_id = value.reference.import_id
    candidate_id = uuid5(NAMESPACE_URL, f"{actor.owner_key}:external-candidate:{import_id}")
    dataset_id = uuid5(
        NAMESPACE_URL, f"{actor.owner_key}:external-dataset:{project.id}:{raw.manifest_sha256}"
    )
    metadata = {
        "owner_key": actor.owner_key,
        "actor_id": actor.object_id,
        "tenant_id": actor.tenant_id,
        "created_at": now,
        "updated_at": now,
        "fingerprint": value.reference.completion_sha256,
    }
    dataset = DatasetVersion(
        **metadata,
        id=dataset_id,
        project_id=project.id,
        artifact_id=dataset_id,
        manifest_sha256=raw.manifest_sha256,
        episode_ids=tuple(UUID(item.metadata["episode_id"]) for item in raw.episodes),
        seeds=tuple(item.metadata["seed"] for item in raw.episodes),
        human_teleop_count=sum(
            item.metadata["demonstration"]["kind"] == "human_teleop" for item in raw.episodes
        ),
        reference_controller_count=sum(
            item.metadata["demonstration"]["kind"] == "reference_controller"
            for item in raw.episodes
        ),
        learned_policy_count=sum(
            item.metadata["demonstration"]["kind"] == "learned" for item in raw.episodes
        ),
        evaluation_plan_sha256=project.evaluation_plan.sha256,
        **project.timing_fields(),
    )
    candidate = PolicyCandidate(
        **metadata,
        id=candidate_id,
        project_id=project.id,
        dataset_id=dataset.id,
        training_run_id=None,
        training_origin="external_native_import",
        external_import_id=import_id,
        parent_release_id=None,
        pretrained_artifact_id=None,
        policy_type="smolvla",
        model_sha256=value.request.model_sha256,
        manifest_sha256=value.request.model_sha256,
        parent_model_sha256=model["training"]["parent_model_sha256"],
        processor_sha256=model["processor_sha256"],
        artifact_id=candidate_id,
        optimizer_steps=model["training"]["optimizer_steps"],
        azure_job_id=proof.azure_job_id,
        source_commit=model["upstream"]["source_commit"],
        model_revision=model["upstream"]["model_revision"],
        **project.timing_fields(),
    )
    record = ExternalTrainingImport(
        **metadata,
        id=import_id,
        project_id=project.id,
        imported_at=now,
        native_job_name=value.request.native_job_name,
        azure_job_id=proof.azure_job_id,
        native_created_at=created,
        job_deadline_utc=utc(value.config["job_deadline_utc"]),
        approval_sha256=value.request.approval_sha256,
        plan_sha256=value.approval["plan_sha256"],
        plan_archive_sha256=value.approval["plan_archive_sha256"],
        configuration_sha256=proof.configuration_sha256,
        specification_sha256=proof.specification_sha256,
        image_qualification_sha256=value.approval["image_qualification_sha256"],
        environment_image=value.config["environment_image"],
        managed_identity_client_id=value.config["managed_identity_client_id"],
        code_snapshot_sha256=proof.snapshot_sha256,
        static_source_sha256=value.config["source_delivery"]["static_sha256"],
        completion_sha256=value.reference.completion_sha256,
        result_sha256=value.request.result_sha256,
        transfer_sha256=value.request.transfer_sha256,
        model_sha256=value.request.model_sha256,
        parent_model_sha256=candidate.parent_model_sha256,
        backbone_sha256=model["backbone_manifest_sha256"],
        raw_manifest_sha256=raw.manifest_sha256,
        candidate_id=candidate.id,
        dataset_id=dataset.id,
        optimizer_steps=candidate.optimizer_steps,
        **project.timing_fields(),
    )
    return ExternalTrainingResult(record=record, candidate=candidate, dataset=dataset)


def complete(verifier, actor, work):
    value = context(
        verifier.registry, actor, work.project, work.target_id, expected=work.external_import
    )
    if verifier.budget is None:
        raise unavailable("Bounded external-native import processor")
    if verifier.registry.get(actor, f"{value.prefix}/verified.json") is not None:
        return verified_result(verifier, actor, work.project, work.external_import)[0]
    before = _inventory(verifier, actor, value)
    with TemporaryDirectory(prefix="physicalai-external-training-") as temporary:
        directory = Path(temporary)
        try:
            plan = _plan(verifier.registry, actor, value, directory)
            proof, created = _job(verifier, actor, value, plan)
            raw, model, candidate_root, raw_root = verify_payload(
                verifier, actor, work.project, value, directory, proof
            )
            verifier.budget.check()
            if _inventory(verifier, actor, value) != before:
                raise Problem(
                    409, "external_import_changed", "Original inputs changed during verification."
                )
            result = _result(actor, work.project, value, proof, created, raw, model)
            verifier.registry.upload(
                actor,
                result.dataset.artifact_id,
                raw_root,
                {
                    "manifest_sha256": result.dataset.manifest_sha256,
                    "role": "sealed_dataset",
                },
            )
            verifier.registry.upload(
                actor,
                result.candidate.artifact_id,
                candidate_root,
                {
                    "manifest_sha256": result.candidate.manifest_sha256,
                    "role": "trained_candidate",
                    "training_origin": "external_native_import",
                    "external_import_id": str(work.target_id),
                    "project_id": str(work.project.id),
                },
            )
            verifier.registry.put(
                actor,
                f"{value.prefix}/verified.json",
                {
                    "reference": value.reference.model_dump(mode="json"),
                    "project": work.project.model_dump(mode="json"),
                    "verifier": _version(),
                    "inventory": before,
                    "published": _published_inventory(verifier, actor, result),
                    "result": result.model_dump(mode="json"),
                },
            )
            return result
        except (ValueError, OSError, tarfile.TarError) as exc:
            raise Problem(
                503, "external_import_invalid", "Original native training verification failed."
            ) from exc


def verified_result(verifier, actor, project, reference):
    value = context(verifier.registry, actor, project, reference.import_id, expected=reference)
    certificate = verifier.registry.get(actor, f"{value.prefix}/verified.json")
    if not isinstance(certificate, dict) or set(certificate) != {
        "reference",
        "project",
        "verifier",
        "inventory",
        "result",
        "published",
    }:
        raise unavailable("Verified external native training receipt")
    result = ExternalTrainingResult.model_validate(certificate["result"])
    if (
        certificate["reference"] != reference.model_dump(mode="json")
        or certificate["project"] != project.model_dump(mode="json")
        or certificate["verifier"] != _version()
        or certificate["inventory"] != _inventory(verifier, actor, value)
        or result.record.id != reference.import_id
        or result.record.completion_sha256 != reference.completion_sha256
        or result.record.project_id != project.id
        or result.record.owner_key != actor.owner_key
        or result.record.imported_at < value.request.created_at
    ):
        raise Problem(409, "external_import_certificate", "External import provenance changed.")
    proof, created = _job(
        verifier,
        actor,
        value,
        {
            "snapshot_sha256": result.record.code_snapshot_sha256,
            "plan_sha256": value.approval["plan_sha256"],
        },
    )
    expected = {
        "native_job_name": value.request.native_job_name,
        "azure_job_id": proof.azure_job_id,
        "native_created_at": created,
        "job_deadline_utc": utc(value.config["job_deadline_utc"]),
        "approval_sha256": value.request.approval_sha256,
        "plan_sha256": value.approval["plan_sha256"],
        "plan_archive_sha256": value.approval["plan_archive_sha256"],
        "configuration_sha256": proof.configuration_sha256,
        "specification_sha256": proof.specification_sha256,
        "image_qualification_sha256": value.approval["image_qualification_sha256"],
        "environment_image": value.config["environment_image"],
        "managed_identity_client_id": UUID(value.config["managed_identity_client_id"]),
        "static_source_sha256": value.config["source_delivery"]["static_sha256"],
        "result_sha256": value.request.result_sha256,
        "transfer_sha256": value.request.transfer_sha256,
        "model_sha256": value.request.model_sha256,
        "parent_model_sha256": value.config["inputs"]["parent_model"]["sha256"],
        "backbone_sha256": value.config["inputs"]["backbone"]["sha256"],
        "raw_manifest_sha256": value.config["inputs"]["demonstrations"]["sha256"],
    }
    if any(getattr(result.record, name) != item for name, item in expected.items()):
        raise Problem(409, "external_import_certificate", "Cached native provenance differs.")
    if certificate["published"] != _published_inventory(verifier, actor, result):
        raise Problem(409, "external_import_certificate", "Published artifact inventory changed.")
    metadata, _, _ = _read(
        verifier.registry,
        actor,
        f"artifacts/{result.candidate.artifact_id}/files/model.json",
        value.request.model_sha256,
        max_bytes=4 * 1024**2,
    )
    verify_candidate_record(result.candidate, metadata)
    for record, name in ((result.candidate, "model.json"), (result.dataset, "manifest.json")):
        index = verifier.registry.artifact_index(actor, record.artifact_id)
        if (
            index.get("manifest_sha256") != record.manifest_sha256
            or index.get("files", {}).get(name) != record.manifest_sha256
        ):
            raise Problem(409, "external_import_certificate", "Verified external artifact changed.")
    return result, proof


def _published_inventory(verifier, actor, result):
    registry = verifier.registry
    return fingerprint(
        {
            str(record.artifact_id): {
                "index": registry.artifact_index(actor, record.artifact_id),
                "files": verifier._blob_inventory(
                    registry.client.url.rstrip("/"),
                    registry.container.container_name,
                    registry.key(actor, f"artifacts/{record.artifact_id}/files/"),
                ),
            }
            for record in (result.candidate, result.dataset)
        }
    )


def registered_import(verifier, actor, record, index, project=None):
    from apps.api.learning_models import LearningProject

    try:
        project_id, import_id = UUID(index["project_id"]), UUID(index["external_import_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Problem(
            409, "external_import_certificate", "External artifact identity is invalid."
        ) from exc
    if (
        index.get("training_origin") != "external_native_import"
        or index.get("role") != "trained_candidate"
        or any(
            name in index
            for name in (
                "training_execution",
                "azure_job_id",
                "azure_job_type",
                "azure_pipeline_job_id",
                "azure_component_job_id",
            )
        )
    ):
        raise Problem(409, "external_import_certificate", "External artifact has mixed provenance.")
    certificate = verifier.registry.get(actor, f"{prefix(project_id, import_id)}/verified.json")
    if certificate is None:
        raise unavailable("Verified external model import")
    original_project = LearningProject.model_validate(certificate["project"])
    if project is not None and original_project != project:
        raise Problem(409, "external_import_scope", "Post-hoc associated project changed.")
    reference = ExternalImportReference.model_validate(certificate["reference"])
    result, proof = verified_result(verifier, actor, original_project, reference)
    candidate = result.candidate
    if (
        record.artifact_id != candidate.artifact_id
        or record.model_sha256 != candidate.model_sha256
        or index.get("manifest_sha256") != candidate.manifest_sha256
        or (isinstance(record, PolicyCandidate) and record != candidate)
        or (not isinstance(record, PolicyCandidate) and record.candidate_id != candidate.id)
    ):
        raise Problem(409, "external_import_certificate", "Registered external candidate changed.")
    return result, proof


def registered_model(verifier, actor, record, root, index, project=None):
    result, proof = registered_import(verifier, actor, record, index, project)
    candidate = result.candidate
    model = verifier._model(
        actor,
        root,
        candidate.manifest_sha256,
        "smolvla",
        execution_timing="paused_simulation",
        verified_command=proof,
    )
    verify_candidate_record(candidate, model)
    return model
