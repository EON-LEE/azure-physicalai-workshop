"""CPU codec/pixel/model doubles for post-hoc imports; not GPU training or vendor weights."""

import copy
import io
import shutil
import tarfile
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from apps.api.artifact_models import ArtifactWork
from apps.api.learning_models import LearningProject, fingerprint
from apps.api.models import utcnow
from apps.learning_worker.artifact_operations import ArtifactBudget
from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.external_training import ExternalImportCompletion, context, prefix
from apps.learning_worker.registry import BlobRegistry
from learning.checks.fixtures import PROVENANCE
from learning.common import canonical, digest, file_digest, inventory, read_json
from learning.contract import DemonstrationSource, EpisodeSpec, Scope
from learning.paused import PausedControlProfile
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter, assemble_dataset
from learning.paused.dataset import convert_dataset
from learning.smolvla import UPSTREAM, prepare
from learning.smolvla.azure import build_job, create_plan
from learning.smolvla.embedded_source import static_inventory
from learning.smolvla.train import TrainOptions
from tests.runtime_support import ACTOR
from tests.test_command_candidate_provenance import command_fixture
from tests.test_worker_deadlines import ConditionalBlobs


class ImportBlobs(ConditionalBlobs):
    container_name = "artifacts"

    def __init__(self):
        super().__init__()
        self.modified = {}

    def _upload(self, name, data, **kwargs):
        body = data.read() if hasattr(data, "read") else data
        receipt = super()._upload(name, body, **kwargs)
        self.modified[name] = utcnow()
        return receipt

    def download_blob(self, name):
        result = super().download_blob(name)
        body = result.readall()
        result.chunks = lambda: iter((body,))
        result.properties.last_modified = self.modified[name]
        return result


def make_fixture(directory, monkeypatch):
    from tests.learning.test_conversion_training import mock_converter
    from tests.learning.test_paused_capture import shifted_sample

    mock_converter.__wrapped__(monkeypatch)
    base = command_fixture(directory / "initial")
    now = utcnow().replace(microsecond=0)
    job_name, import_id = str(uuid4()), uuid4()
    profile = PausedControlProfile(**base.model["control_profile"])
    task = base.model["task"]
    roots = []
    cases = []
    for index in range(20):
        episode_id = str(uuid4())
        case = {
            "case_id": f"cpu-train-{index}",
            "environment_id": f"cpu-cell-{index}",
            "revision": f"{index + 1:064x}",
            "seed": 10001 + index,
            "split": "train",
        }
        cases.append(case)
        writer = PausedEpisodeWriter(
            directory / f"episode-{index}",
            dataset_id=f"cpu-source-{index}",
            scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
            episode=EpisodeSpec(
                episode_id, case["environment_id"], case["revision"], case["seed"], "train"
            ),
            provenance=replace(
                PROVENANCE,
                source_kind="isaac_sim",
                capture_host="azure_gpu",
                gpu_model="CPU codec fixture, not hardware proof",
            ),
            profile=profile,
            demonstration=DemonstrationSource("reference_controller", **task),
            budget=PausedEpisodeBudget(1_000_000_000, 601_000_000_000, 60, 1860),
            purpose="demonstration",
            criteria_sha256=base.spec.project.criteria_sha256,
            frozen_plan_sha256=base.spec.project.frozen_plan_sha256,
        )
        for offset in range(2):
            sample = shifted_sample(offset, terminal=offset == 1)
            observation = replace(
                sample.observation,
                scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
                episode_id=episode_id,
                environment_id=case["environment_id"],
                revision=case["revision"],
            )
            writer.append(replace(sample, observation=observation))
        writer.finalize()
        roots.append(writer.root)
    raw_root = directory / "raw"
    assemble_dataset(
        roots,
        raw_root,
        dataset_id="native-cpu-train20",
        expected_scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
        require_live=True,
    )
    raw = read_json(raw_root / "manifest.json")
    converted_root = directory / "converted" / "dataset"
    convert_dataset(
        raw_root,
        converted_root,
        expected_scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
        expected_manifest_sha256=file_digest(raw_root / "manifest.json"),
        expected_criteria_sha256=base.spec.project.criteria_sha256,
        expected_frozen_plan_sha256=base.spec.project.frozen_plan_sha256,
    )
    # Native validators still check every fixture file; only the vendor-file oracle is injected.
    parent_root, backbone_root = directory / "parent", directory / "backbone"
    shutil.copytree(base.root, parent_root)
    (backbone_root / "assets").mkdir(parents=True)
    for name, content in (
        ("config.json", b"{}"),
        ("processor_config.json", b"{}"),
        ("model.safetensors", b"CPU-only backbone bytes, not released weights"),
    ):
        (backbone_root / "assets" / name).write_bytes(content)
    (backbone_root / "backbone.json").write_bytes(
        canonical(
            {
                "schema": "physicalai.smolvla-backbone/v1",
                "scope": raw["scope"],
                "upstream": UPSTREAM,
                "files": inventory(backbone_root),
            }
        )
    )
    parent = read_json(parent_root / "model.json")
    (parent_root / "checkpoint" / "model.safetensors").write_bytes(
        b"CPU-only untrained parent bytes"
    )
    parent["checkpoint_files"] = inventory(parent_root / "checkpoint")
    parent["weights_sha256"] = parent["checkpoint_files"]["model.safetensors"]
    parent.update(
        schema="physicalai.smolvla-checkpoint/v2",
        role="pretrained",
        training=None,
        backbone_manifest_sha256=file_digest(backbone_root / "backbone.json"),
    )
    parent.pop("training_execution")
    (parent_root / "model.json").write_bytes(canonical(parent))
    vendor_inventory = {
        "model": inventory(parent_root / "checkpoint"),
        "backbone": inventory(backbone_root / "assets"),
    }

    def verify_fixture_vendor(root, *, kind):
        assert inventory(root) == vendor_inventory[kind], (
            "All test vendor bytes must remain unchanged."
        )
        return vendor_inventory[kind]

    monkeypatch.setattr(prepare, "verify_vendor_files", verify_fixture_vendor)
    config = copy.deepcopy(base.config)
    config["run_id"] = job_name
    config["job_deadline_utc"] = (now - timedelta(seconds=300)).isoformat().replace("+00:00", "Z")
    config["parameters"]["timeout_seconds"] = 600
    config["parameters"].update(max_steps=1000, checkpoint_steps=100)
    from learning.azure import datastore_prefix
    from learning.gr00t.azure import workspace_id

    config["source_delivery"]["static_sha256"] = digest(
        canonical(static_inventory(Path(__file__).resolve().parents[1], direct=True))
    )
    for name, root, manifest in (
        ("demonstrations", raw_root, "manifest.json"),
        ("parent_model", parent_root, "model.json"),
        ("backbone", backbone_root, "backbone.json"),
    ):
        config["inputs"][name].update(
            sha256=file_digest(root / manifest),
            uri=datastore_prefix(config)
            + f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/"
            + f"learning/native-inputs/{job_name}/{name}",
        )
    native = f"native-operator/{job_name}"
    namespace = f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/learning/"
    qualified = {"fixture": "CPU-only source qualification", "image": config["environment_image"]}
    qualification_sha = digest(canonical(qualified))
    specification = {
        "schema": "physicalai.native-bootstrap-authorization/v1",
        "id": job_name,
        "job_name": job_name,
        "scope": raw["scope"],
        "maximum_cost_usd": 5,
        "maximum_gpu_nodes": 1,
        "learning_quality_claim": False,
        "job_deadline_utc": config["job_deadline_utc"],
        "parameters": config["parameters"],
        "inputs": config["inputs"],
        "environment_image": config["environment_image"],
        "source_delivery": config["source_delivery"],
        "job_execution": config["job_execution"],
        "image_qualification_sha256": qualification_sha,
    }
    config["specification_sha256"] = digest(canonical(specification))
    plan_root = directory / "frozen-plan"
    plan_sha = create_plan(config, plan_root, deterministic_job_name=job_name)
    plan = read_json(plan_root / "plan.json")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        tar.add(plan_root, arcname="plan")
    archive_bytes = archive.getvalue()
    approval = {
        "schema": "physicalai.native-training-approval/v1",
        "scope": raw["scope"],
        "job_name": job_name,
        "job_deadline_utc": config["job_deadline_utc"],
        "maximum_gpu_nodes": 1,
        "maximum_cost_usd": 5,
        "price_is_not_actual_billing": True,
        "public_list_price_usd_per_hour": 0.6,
        "plan_blob": namespace + native + "/plan.tar.gz",
        "plan_archive_sha256": digest(archive_bytes),
        "plan_sha256": plan_sha,
        "configuration_sha256": digest(canonical(config)),
        "global_target_steps": config["parameters"]["max_steps"],
        "checkpoint_steps": config["parameters"]["checkpoint_steps"],
        "resume": None,
        "environment_image": config["environment_image"],
        "source_delivery": config["source_delivery"],
        "job_execution": config["job_execution"],
        "image_qualification_sha256": qualification_sha,
        "image_qualification_blob": namespace + native + "/image-qualification.json",
    }
    claim = {
        "scope": raw["scope"],
        "job_name": job_name,
        "job_type": "command",
        "approval_sha256": digest(canonical(approval)),
        "plan_sha256": plan_sha,
        "configuration_sha256": digest(canonical(config)),
        "job_deadline_utc": config["job_deadline_utc"],
        "claimed_at_utc": (now - timedelta(seconds=990)).isoformat(),
    }
    job_id = workspace_id(config) + "/jobs/" + job_name
    model = base.model
    model.update(backbone_manifest_sha256=config["inputs"]["backbone"]["sha256"])
    model["training"].update(
        azure_job_id=job_id,
        parent_model_sha256=file_digest(parent_root / "model.json"),
        parent_weights_sha256=parent["weights_sha256"],
        raw_manifest_sha256=file_digest(raw_root / "manifest.json"),
        conversion_sha256=file_digest(converted_root / "conversion.json"),
        config_sha256=digest(canonical(asdict(TrainOptions(**config["parameters"])))),
        code_snapshot_sha256=plan["snapshot_sha256"],
        specification_sha256=config["specification_sha256"],
        episodes=[
            {name: item[name] for name in ("episode_id", "environment_id", "revision", "seed")}
            for item in raw["episodes"]
        ],
        optimizer_steps=1000,
        checkpoint_step=1000,
        cumulative_optimizer_steps=1000,
    )
    canonical_root = base.output / "candidates" / "step-001000"
    base.root.rename(canonical_root)
    base.root = canonical_root
    (base.root / "model.json").write_bytes(canonical(model))
    result = base.result | {
        "azure_job_id": job_id,
        "specification_sha256": config["specification_sha256"],
        "model_manifest_sha256": file_digest(base.root / "model.json"),
        "optimizer_steps": 1000,
        "candidate": "candidates/step-001000",
    }
    (base.output / "result.json").write_bytes(canonical(result))
    project = LearningProject.model_validate(
        base.spec.project.model_dump()
        | {
            "teaching_cases": cases,
            "created_at": now - timedelta(seconds=50),
            "updated_at": now - timedelta(seconds=50),
            "budget": base.spec.project.budget.model_dump() | {"optimizer_steps": 1000},
        }
    )
    request = ExternalImportCompletion(
        schema="physicalai.external-native-training-request/v1",
        import_id=import_id,
        project_sha256=fingerprint(project.public()),
        created_at=now - timedelta(seconds=20),
        native_job_name=job_name,
        approval_sha256=digest(canonical(approval)),
        claim_sha256=digest(canonical(claim)),
        configuration_file_sha256=digest(canonical(config)),
        specification_file_sha256=digest(canonical(specification)),
        result_sha256=digest(canonical(result)),
        transfer_sha256="f" * 64,
        model_sha256=file_digest(base.root / "model.json"),
    )
    registry = BlobRegistry.__new__(BlobRegistry)
    registry.budget = None
    registry.container = ImportBlobs()
    registry.container.container_name = config["blob_container"]
    registry.client = SimpleNamespace(
        url=f"https://{config['storage_account_name']}.blob.core.windows.net", close=lambda: None
    )
    base_prefix = prefix(project.id, import_id)

    def put(name, body, seconds):
        registry.container.upload_blob(name=name, data=body, overwrite=False)
        registry.container.modified[name] = now - timedelta(seconds=seconds)

    for suffix, body, age in (
        (f"{base_prefix}/files/run-config.json", canonical(config), 20),
        (f"{base_prefix}/files/specification.json", canonical(specification), 20),
        (f"{native}/approval.json", canonical(approval), 1000),
        (f"{native}/claim.json", canonical(claim), 989),
        (f"{native}/plan.tar.gz", archive_bytes, 1001),
        (f"{native}/image-qualification.json", canonical(qualified), 1002),
    ):
        put(namespace + suffix, body, age)
    from learning.paused.blob_transfer import input_prefix

    for location, root in (
        (input_prefix(config, "demonstrations"), raw_root),
        (input_prefix(config, "parent_model"), parent_root),
        (input_prefix(config, "backbone"), backbone_root),
        (config["output_prefix"] + "/" + job_name + "/dataset", converted_root.parent),
        (config["output_prefix"] + "/" + job_name + "/model", base.output),
    ):
        for path in root.rglob("*"):
            if path.is_file():
                put(location + "/" + path.relative_to(root).as_posix(), path.read_bytes(), 400)
    output = config["output_prefix"] + "/" + job_name
    result_etag = registry.container.items[output + "/model/result.json"][1]
    transfer = {
        "schema": "physicalai.smolvla-command-transfer/v1",
        "azure_job_id": job_id,
        "azure_job_type": "command",
        "specification_sha256": config["specification_sha256"],
        "optimizer_steps": result["optimizer_steps"],
        "learning_quality_verified": False,
        "model_result": {
            "prefix": output + "/model",
            "marker": "result.json",
            "marker_sha256": digest(canonical(result)),
            "readback_verified": True,
            "etag": result_etag,
            "file_count": sum(path.is_file() for path in base.output.rglob("*")),
            "bytes": sum(path.stat().st_size for path in base.output.rglob("*") if path.is_file()),
        },
    }
    put(output + "/transfer/completion.json", canonical(transfer), 399)
    request = request.model_copy(update={"transfer_sha256": digest(canonical(transfer))})
    put(
        namespace + f"{base_prefix}/completion.json",
        canonical(request.model_dump(mode="json", by_alias=True)),
        19,
    )
    built = build_job(config, plan["snapshot_sha256"], job_name)
    job = SimpleNamespace(
        **built,
        id=job_id,
        status="Completed",
        parent_job_name=None,
        creation_context=SimpleNamespace(created_at=now - timedelta(seconds=985)),
    )
    for name in ("resources", "limits", "environment", "identity"):
        setattr(job, name, SimpleNamespace(**built[name]))
    job.tags["plan_sha256"] = plan_sha
    calls = []

    def read_job(name):
        assert name == job_name
        calls.append(name)
        return job

    verifier = VerifiedArtifacts(
        registry, None, registry.client.url, "demonstrations", allowed_policy_types=("smolvla",)
    )
    verifier._metadata_client = lambda _: SimpleNamespace(jobs=SimpleNamespace(get=read_job))
    ctx = context(registry, ACTOR, project, import_id)
    work = ArtifactWork(
        id=import_id,
        target_id=import_id,
        actor=ACTOR,
        project=project,
        operation="external_training",
        created_at=now,
        deadline=now + timedelta(seconds=1800),
        max_bytes=20 * 1024**3,
        max_files=100000,
        external_import=ctx.reference,
    )
    return SimpleNamespace(
        registry=registry,
        verifier=verifier,
        context=ctx,
        work=work,
        project=project,
        config=config,
        job=job,
        job_id=job_id,
        request=request,
        raw_root=raw_root,
        model=model,
        result=result,
        root=base.root,
        calls=calls,
        prepared_vendor=vendor_inventory,
    )


def connect_fixture(sample, monkeypatch):
    from apps.learning_worker import artifact_task, artifacts

    class Blobs:
        def __init__(self, account, **kwargs):
            assert account == sample.registry.client.url

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get_container_client(self, name):
            assert name == sample.registry.container.container_name
            return sample.registry.container

    monkeypatch.setattr(artifacts, "BlobServiceClient", Blobs)
    monkeypatch.setattr(artifact_task, "BlobServiceClient", Blobs)
    budget = ArtifactBudget(
        sample.work.deadline, max_bytes=sample.work.max_bytes, max_files=sample.work.max_files
    )
    sample.verifier.budget = sample.registry.budget = budget


def clone_fixture(original, monkeypatch):
    registry = BlobRegistry.__new__(BlobRegistry)
    registry.budget = None
    registry.client = original.registry.client
    registry.container = ImportBlobs()
    registry.container.container_name = original.registry.container.container_name
    registry.container.items = copy.deepcopy(original.registry.container.items)
    registry.container.modified = dict(original.registry.container.modified)
    registry.container.version = original.registry.container.version
    verifier = VerifiedArtifacts(
        registry, None, registry.client.url, "demonstrations", allowed_policy_types=("smolvla",)
    )
    sample = SimpleNamespace(
        **(
            vars(original)
            | {
                "registry": registry,
                "verifier": verifier,
                "job": copy.deepcopy(original.job),
                "calls": [],
            }
        )
    )

    def get(name):
        sample.calls.append(name)
        assert name == str(sample.request.native_job_name)
        return sample.job

    verifier._metadata_client = lambda _: SimpleNamespace(jobs=SimpleNamespace(get=get))
    connect_fixture(sample, monkeypatch)
    return sample
