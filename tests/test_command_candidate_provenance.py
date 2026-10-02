"""CPU-only provenance codecs; no model execution or cloud job is started."""

import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import DatasetVersion, LearningProject, TrainingParent, TrainingRun
from apps.api.learning_ports import JobSpecification
from apps.api.models import utcnow
from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.candidate_provenance import (
    CommandTrainingResult,
    VerifiedCommandJob,
    command_config,
    command_snapshot,
    read_command_result,
)
from learning.common import canonical, digest, file_digest, read_json
from tests.learning.test_paused_model_artifacts import fixture
from tests.runtime_support import ACTOR
from tests.test_paused_learning_artifacts import project


def pipeline_fixture(tmp_path):
    output = tmp_path / "output"
    root = output / "candidates" / "step-1"
    fixture(root)
    model = read_json(root / "model.json")
    current = project()
    model["scope"] = {"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key}
    model["task"] = {
        "task_id": current.task_id,
        "instruction": current.instruction,
        "goal_id": current.goal_station_id,
    }
    (root / "model.json").write_bytes(canonical(model))
    training = model["training"]
    job_id = training["azure_pipeline_job_id"]
    result = {
        "azure_job_id": job_id,
        "azure_component_job_id": training["azure_job_id"],
        "specification_sha256": training["specification_sha256"],
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "criteria_sha256": current.criteria_sha256,
        "frozen_plan_sha256": current.frozen_plan_sha256,
        "optimizer_steps": 1,
        "candidate": "candidates/step-1",
        "model_manifest_sha256": file_digest(root / "model.json"),
        "learning_quality_verified": False,
    }
    (output / "result.json").write_bytes(canonical(result))
    from learning.common import digest

    config = {
        "control_profile_sha256": current.control_profile_sha256,
        "task_sha256": digest(canonical(model["task"])),
    }
    spec = SimpleNamespace(
        project=current,
        run=SimpleNamespace(
            id=uuid4(),
            parent_release_id=uuid4(),
            pretrained_artifact_id=None,
            backend_job_name="pipeline",
            specification_sha256=training["specification_sha256"],
        ),
        training_parent=None,
        baseline=SimpleNamespace(model_sha256=training["parent_model_sha256"]),
        dataset=SimpleNamespace(id=uuid4(), manifest_sha256=training["raw_manifest_sha256"]),
    )
    uploads = []
    registry = SimpleNamespace(
        upload=lambda *args: uploads.append(args),
        get=lambda *args: None,
        put=lambda *args: True,
    )
    verifier = VerifiedArtifacts(
        registry,
        None,
        "https://unused.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )

    def download(_actor, _spec, _name, destination):
        shutil.copytree(output, destination)
        return config

    verifier._output = download
    return SimpleNamespace(
        output=output,
        root=root,
        model=model,
        result=result,
        config=config,
        spec=spec,
        verifier=verifier,
        uploads=uploads,
        job_id=job_id,
    )


def test_legacy_pipeline_candidate_keeps_real_distinct_root_and_component(tmp_path):
    sample = pipeline_fixture(tmp_path)
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert candidate.azure_job_id == sample.job_id
    assert candidate.optimizer_steps == 1
    assert len(sample.uploads) == 1


@pytest.mark.parametrize(
    "change", ["same-id", "command-result", "command-training", "foreign-component"]
)
def test_pipeline_path_rejects_spoofed_or_mixed_command_hierarchy(tmp_path, change):
    sample = pipeline_fixture(tmp_path)
    if change == "same-id":
        sample.model["training"]["azure_job_id"] = sample.job_id
        sample.result["azure_component_job_id"] = sample.job_id
    elif change == "command-result":
        sample.result.update(
            schema="physicalai.smolvla-command-training-result/v1", azure_job_type="command"
        )
    elif change == "command-training":
        sample.model["training"]["azure_job_type"] = "command"
    else:
        component = sample.model["training"]["azure_job_id"].replace(
            "/workspaces/unit/", "/workspaces/foreign/"
        )
        sample.model["training"]["azure_job_id"] = component
        sample.result["azure_component_job_id"] = component
    (sample.root / "model.json").write_bytes(canonical(sample.model))
    sample.result["model_manifest_sha256"] = file_digest(sample.root / "model.json")
    (sample.output / "result.json").write_bytes(canonical(sample.result))
    with pytest.raises(Problem):
        sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert sample.uploads == []


def command_result():
    return {
        "schema": "physicalai.smolvla-command-training-result/v1",
        "azure_job_id": (
            "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/unit/"
            "providers/Microsoft.MachineLearningServices/workspaces/unit/jobs/learning-abcd"
        ),
        "azure_job_type": "command",
        "specification_sha256": "a" * 64,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "optimizer_steps": 1,
        "candidate": "candidates/step-000001",
        "model_manifest_sha256": "f" * 64,
        "learning_quality_verified": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"azure_component_job_id": "/fake/component"},
        {"azure_pipeline_job_id": "/fake/pipeline"},
        {"azure_job_type": "pipeline"},
        {"schema": "physicalai.smolvla-checkpoint/v2"},
        {"optimizer_steps": True},
        {"real_time_admission": 0},
        {"learning_quality_verified": True},
        {"learning_quality_verified": 0},
        {"candidate": "../checkpoint"},
    ],
)
def test_command_result_is_closed_and_never_promotes_quality_or_invents_parent(change):
    value = command_result()
    assert CommandTrainingResult.model_validate(value).azure_job_type == "command"
    with pytest.raises(ValidationError):
        CommandTrainingResult.model_validate(value | change)


def test_command_result_reader_preserves_strict_original_json(tmp_path):
    path = tmp_path / "result.json"
    value = command_result()
    path.write_bytes(canonical(value) + b"\n")
    assert read_command_result(path) == value
    path.write_bytes(
        canonical(value).replace(
            b'"azure_job_type":"command"', b'"azure_job_type":"pipeline","azure_job_type":"command"'
        )
    )
    with pytest.raises(Problem):
        read_command_result(path)
    path.write_bytes(b" " * (1024**2 + 1))
    with pytest.raises(Problem):
        read_command_result(path)


@pytest.mark.parametrize("selector", [None, {}, {"kind": "command"}, {"kind": "pipeline"}])
def test_present_but_invalid_command_selector_never_falls_back_to_a_pipeline(selector):
    assert command_config({}) is False
    with pytest.raises(Problem):
        command_config({"job_execution": selector})


def test_command_snapshot_includes_the_actual_direct_command_source_inventory():
    from learning.smolvla.embedded_source import static_inventory, verify_static_binding
    from tests.learning.test_direct_command_jobs import command_config as native_command_config

    root = Path(__file__).resolve().parents[1]
    config = native_command_config()
    files = static_inventory(root, direct=True)
    assert "learning/paused/command_artifacts.py" in files
    assert "learning/paused/command.py" in files
    assert "learning/paused/command_model.py" in files
    assert "learning/paused/blob_transfer.py" in files
    config["source_delivery"]["static_sha256"] = digest(canonical(files))
    verify_static_binding(config, root)
    expected = digest(canonical({**files, "run-config.json": digest(canonical(config) + b"\n")}))
    assert command_snapshot(config) == expected


@pytest.mark.parametrize("mode", ["mapping", "foreign-owner", "legacy-mode", "train-only"])
def test_unverified_context_cannot_select_the_new_native_validator(tmp_path, mode, monkeypatch):
    sample = pipeline_fixture(tmp_path)
    from apps.learning_worker import artifacts

    proof = VerifiedCommandJob(ACTOR.owner_key, sample.job_id, "a" * 64, "b" * 64, "c" * 64)
    if mode == "mapping":
        proof = {"owner_key": ACTOR.owner_key}
    elif mode == "foreign-owner":
        proof = VerifiedCommandJob("f" * 64, sample.job_id, "a" * 64, "b" * 64, "c" * 64)
    monkeypatch.setattr(
        artifacts.importlib,
        "import_module",
        lambda *_: pytest.fail("Unverified routing must fail before native module selection."),
    )
    with pytest.raises(Problem) as failure:
        sample.verifier._model(
            ACTOR,
            sample.root,
            sample.result["model_manifest_sha256"],
            "smolvla",
            execution_timing=None if mode == "legacy-mode" else "paused_simulation",
            inference=mode != "train-only",
            verified_command=proof,
        )
    assert failure.value.code == "candidate_job_mismatch"


@pytest.mark.parametrize(
    "index_fields",
    [
        {"training_execution": "azureml_command"},
        {"azure_job_type": "command"},
        {"azure_job_id": "/a/job"},
        {"training_execution": "azureml_command", "azure_job_type": "pipeline"},
    ],
)
def test_partial_registered_command_provenance_never_selects_a_validator(
    tmp_path, index_fields, monkeypatch
):
    sample = pipeline_fixture(tmp_path)
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    monkeypatch.setattr(
        sample.verifier,
        "_model",
        lambda *_args, **_kwargs: pytest.fail(
            "Partial command registry metadata is not authority."
        ),
    )
    with pytest.raises(Problem) as failure:
        sample.verifier.registered_model(
            ACTOR,
            candidate,
            sample.root,
            {
                "manifest_sha256": candidate.manifest_sha256,
                "role": "trained_candidate",
                **index_fields,
            },
            execution_timing="paused_simulation",
        )
    assert failure.value.code == "candidate_job_mismatch"


def command_fixture(tmp_path, *, full_state=False):
    from learning.gr00t.azure import datastore_prefix, workspace_id
    from learning.smolvla.azure import build_job
    from learning.smolvla.embedded_source import static_inventory
    from learning.smolvla.train import TrainOptions
    from tests.learning.test_direct_command_jobs import command_config as native_command_config

    sample = pipeline_fixture(tmp_path)
    current = sample.spec.project
    config = native_command_config()
    config["source_delivery"]["static_sha256"] = digest(
        canonical(static_inventory(Path(__file__).resolve().parents[1], direct=True))
    )
    job_name = f"learning-{uuid4().hex}"
    now = utcnow()
    metadata = {
        "owner_key": ACTOR.owner_key,
        "actor_id": ACTOR.object_id,
        "tenant_id": ACTOR.tenant_id,
        "created_at": now - timedelta(hours=1),
        "updated_at": now - timedelta(hours=1),
        "fingerprint": "a" * 64,
    }
    parent_id, run_id, dataset_id = uuid4(), uuid4(), uuid4()
    current = LearningProject.model_validate(
        current.model_dump()
        | {
            "project_kind": "bootstrap",
            "baseline_release_id": None,
            "pretrained_artifact_id": parent_id,
        }
    )
    parent = TrainingParent(
        **metadata,
        id=parent_id,
        artifact_id=parent_id,
        policy_type="smolvla",
        model_sha256=sample.model["training"]["parent_model_sha256"],
        processor_sha256=sample.model["processor_sha256"],
        source_commit=sample.model["upstream"]["source_commit"],
        model_revision=sample.model["upstream"]["model_revision"],
        registered_by=ACTOR.object_id,
        **current.timing_fields(),
    )
    dataset = DatasetVersion(
        **metadata,
        id=dataset_id,
        project_id=current.id,
        artifact_id=dataset_id,
        manifest_sha256=sample.model["training"]["raw_manifest_sha256"],
        episode_ids=(uuid4(),),
        seeds=(10001,),
        human_teleop_count=0,
        reference_controller_count=1,
        learned_policy_count=0,
        evaluation_plan_sha256=current.evaluation_plan.sha256,
        **current.timing_fields(),
    )
    config.update(
        run_id=job_name,
        tenant_id=str(ACTOR.tenant_id),
        owner_id=ACTOR.owner_key,
        job_deadline_utc=(now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
        output_prefix=f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/learning/outputs",
        control_profile_sha256=current.control_profile_sha256,
        task_sha256=digest(canonical(sample.model["task"])),
        job_execution={
            "schema": "physicalai.smolvla-command-execution/v1",
            "kind": "command",
            "data_transport": "private_blob_mi",
        },
    )
    config["parameters"].update(max_steps=2, checkpoint_steps=1)
    for name, checksum in (
        ("parent_model", parent.model_sha256),
        ("demonstrations", dataset.manifest_sha256),
        ("backbone", sample.model["backbone_manifest_sha256"]),
    ):
        config["inputs"][name].update(
            sha256=checksum,
            uri=datastore_prefix(config)
            + f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/{name}",
        )
    if full_state:
        source = workspace_id(config) + "/jobs/previous-command"
        config["parameters"].update(resume_mode="full_state", max_steps=11)
        config["checkpointing"]["resume"] = {
            "checkpoint_sha256": "6" * 64,
            "source_azure_job_id": source,
            "source_azure_job_type": "command",
            "step": 10,
        }
        for name, checksum in (("resume_checkpoint", "6" * 64), ("converted_dataset", "7" * 64)):
            config["inputs"][name] = {
                **config["inputs"]["demonstrations"],
                "name": name,
                "sha256": checksum,
                "uri": datastore_prefix(config)
                + f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/{name}",
            }
    job_id = workspace_id(config) + "/jobs/" + job_name
    run = TrainingRun(
        **metadata,
        id=run_id,
        project_id=current.id,
        policy_type="smolvla",
        status="submitted",
        backend_job_name=job_name,
        azure_job_id=job_id,
        deadline=now,
        job_deadline_utc=now - timedelta(seconds=1),
        approved_cost_usd="10",
        specification_sha256=config["specification_sha256"],
        dataset_id=dataset.id,
        parent_release_id=None,
        pretrained_artifact_id=parent.id,
        optimizer_steps=1 if full_state else 2,
        **current.timing_fields(),
    )
    specification = JobSpecification(
        owner_key=ACTOR.owner_key,
        project=current,
        run=run,
        training_parent=parent,
        dataset=dataset,
    )
    snapshot = command_snapshot(config)
    model = sample.model
    model.update(schema="physicalai.smolvla-checkpoint/v3", training_execution="azureml_command")
    model["training"].pop("azure_pipeline_job_id")
    model["training"].update(
        azure_job_id=job_id,
        azure_job_type="command",
        code_snapshot_sha256=snapshot,
        config_sha256=digest(canonical(asdict(TrainOptions(**config["parameters"])))),
        specification_sha256=config["specification_sha256"],
        optimizer_steps=1 if full_state else 2,
        checkpoint_step=1 if full_state else 2,
        cumulative_optimizer_steps=1 if full_state else 2,
        episodes=[
            {
                "episode_id": str(dataset.episode_ids[0]),
                "environment_id": current.environment_id,
                "revision": current.revision,
                "seed": dataset.seeds[0],
            }
        ],
    )
    model["training"]["gpu"]["device_count"] = 1
    if full_state:
        model["training"].update(
            resume_mode="full_state",
            checkpoint_step=11,
            cumulative_optimizer_steps=11,
            resume_from_job_id=config["checkpointing"]["resume"]["source_azure_job_id"],
            resume_from_job_type="command",
            resume_from_checkpoint_sha256="6" * 64,
            resume_from_checkpoint_step=10,
            resume_from_cumulative_optimizer_steps=10,
            optimizer_state_restored=True,
            bitwise_continuation_claimed=False,
            conversion_sha256="7" * 64,
        )
    (sample.root / "model.json").write_bytes(canonical(model))
    candidate_path = "candidates/step-000011" if full_state else "candidates/step-000002"
    sample.root.rename(sample.output / candidate_path)
    sample.root = sample.output / candidate_path
    result = command_result() | {
        "azure_job_id": job_id,
        "specification_sha256": run.specification_sha256,
        "model_manifest_sha256": file_digest(sample.root / "model.json"),
        "candidate": candidate_path,
        "optimizer_steps": 1 if full_state else 2,
    }
    (sample.output / "result.json").write_bytes(canonical(result))
    job = build_job(config, snapshot, job_name)
    job.update(
        id=job_id,
        name=job_name,
        status="Completed",
        parent_job_name=None,
    )
    job["compute"] = "azureml:" + config["compute"]
    root = SimpleNamespace(**job)
    root.identity = SimpleNamespace(**job["identity"])
    root.environment = SimpleNamespace(**job["environment"])
    root.resources = SimpleNamespace(**job["resources"])
    root.limits = SimpleNamespace(**job["limits"])
    calls, saved = [], {}

    def get(name):
        calls.append(name)
        assert name == job_name, "Only the actual approved root command can be read."
        return root

    def download(_actor, _specification, _name, destination):
        shutil.copytree(sample.output, destination)
        return config

    registry = sample.verifier.registry
    registry.job_configuration = lambda *args: config
    registry.job = lambda actor, name: specification if name == job_name else None
    indexes = {}

    def upload(actor, artifact_id, root, metadata):
        sample.uploads.append((actor, artifact_id, root, metadata))
        indexes[artifact_id] = {
            **metadata,
            "owner_key": actor.owner_key,
            "files": {"model.json": file_digest(root / "model.json")},
        }

    def registered_download(actor, artifact_id, destination, **kwargs):
        shutil.copytree(sample.root, destination)
        return indexes[artifact_id]

    registry.upload = upload
    registry.download = registered_download
    registry.get = lambda actor, key: saved.get(key)
    registry.put = lambda actor, key, value: saved.setdefault(key, value)
    sample.verifier._output = download
    sample.verifier._metadata_client = lambda value: SimpleNamespace(jobs=SimpleNamespace(get=get))

    def result_download(account, container, key, destination):
        assert account == f"https://{config['storage_account_name']}.blob.core.windows.net"
        assert container == config["blob_container"]
        assert key == f"{config['output_prefix']}/{job_name}/model/result.json"
        shutil.copyfile(sample.output / "result.json", destination)

    sample.verifier._download_file = result_download
    sample.config, sample.spec, sample.model, sample.result, sample.job_id = (
        config,
        specification,
        model,
        result,
        job_id,
    )
    sample.job, sample.get_calls, sample.saved, sample.indexes = root, calls, saved, indexes
    return sample


@pytest.mark.parametrize("full_state", [False, True])
def test_actual_native_command_candidate_has_only_one_verified_job_identity(tmp_path, full_state):
    sample = command_fixture(tmp_path, full_state=full_state)
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert candidate.azure_job_id == sample.job_id
    assert candidate.optimizer_steps == (1 if full_state else 2)
    assert sample.get_calls == [sample.spec.run.backend_job_name]
    sample.verifier.verify_candidate(ACTOR, sample.spec.project, sample.spec.run, candidate)
    assert sample.get_calls == [sample.spec.run.backend_job_name] * 2
    assert "azure_pipeline_job_id" not in sample.model["training"]
    assert "azure_component_job_id" not in sample.result
    assert sample.model["schema"] == "physicalai.smolvla-checkpoint/v3"
    assert sample.indexes[candidate.artifact_id]["training_execution"] == "azureml_command"


@pytest.mark.parametrize(
    "change",
    [
        "job-type",
        "job-name",
        "job-id",
        "job-parent",
        "owner",
        "tenant",
        "specification",
        "source",
        "configuration",
        "deadline",
        "compute",
        "identity",
        "image",
        "code",
        "inputs",
        "outputs",
        "status",
    ],
)
def test_actual_command_root_mismatch_never_publishes_a_candidate(tmp_path, change):
    sample = command_fixture(tmp_path)
    job = sample.job
    tag = {
        "owner": "scope_owner",
        "tenant": "scope_tenant",
        "specification": "specification_sha256",
        "source": "static_source_sha256",
        "configuration": "runtime_config_sha256",
        "deadline": "job_deadline_utc",
    }
    if change in tag:
        job.tags[tag[change]] = "different"
    elif change == "job-type":
        job.type = "pipeline"
    elif change == "job-name":
        job.name = "not-the-approved-job"
    elif change == "job-id":
        job.id = sample.job_id.replace("/workspaces/", "/foreign-workspaces/")
    elif change == "job-parent":
        job.parent_job_name = "invented-parent"
    elif change == "compute":
        job.compute = "azureml:unapproved"
    elif change == "identity":
        job.identity.client_id = str(uuid4())
    elif change == "image":
        job.environment.image = "foreign.azurecr.io/unapproved@sha256:" + "f" * 64
    elif change == "code":
        job.code = "azureml:unapproved-code:1"
    elif change == "inputs":
        job.inputs = {"raw": {"path": "azureml:unapproved:1"}}
    elif change == "outputs":
        job.outputs = {"custom": {"path": "azureml://unapproved/output"}}
    else:
        job.status = "Running"
    with pytest.raises(Problem):
        sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert sample.uploads == []
    assert sample.get_calls == [sample.spec.run.backend_job_name]


@pytest.mark.parametrize(
    "field",
    [
        "azure_job_type",
        "azure_job_id",
        "specification_sha256",
        "code_snapshot_sha256",
        "config_sha256",
        "raw_manifest_sha256",
        "parent_model_sha256",
    ],
)
def test_rehashed_command_model_cannot_change_original_lineage(tmp_path, field):
    sample = command_fixture(tmp_path)
    replacement = (
        "pipeline"
        if field == "azure_job_type"
        else (sample.job_id + "-different" if field == "azure_job_id" else "f" * 64)
    )
    sample.model["training"][field] = replacement
    (sample.root / "model.json").write_bytes(canonical(sample.model))
    sample.result["model_manifest_sha256"] = file_digest(sample.root / "model.json")
    (sample.output / "result.json").write_bytes(canonical(sample.result))
    with pytest.raises(Problem):
        sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert sample.uploads == []
    assert sample.get_calls == [sample.spec.run.backend_job_name]


def test_command_output_cannot_be_reinterpreted_by_removing_explicit_job_selector(tmp_path):
    sample = command_fixture(tmp_path)
    sample.config.pop("job_execution")
    with pytest.raises(Problem):
        sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert sample.uploads == []


def test_recorded_command_candidate_cannot_skip_original_result_or_job_reverification(tmp_path):
    sample = command_fixture(tmp_path)
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    sample.result["azure_component_job_id"] = sample.job_id
    (sample.output / "result.json").write_bytes(canonical(sample.result))
    with pytest.raises(Problem):
        sample.verifier.verify_candidate(ACTOR, sample.spec.project, sample.spec.run, candidate)
    assert sample.get_calls == [sample.spec.run.backend_job_name] * 2


def test_v3_validator_is_not_selected_without_explicit_registered_command_authority(tmp_path):
    sample = command_fixture(tmp_path)
    from learning.contract import Scope
    from learning.paused.artifacts import validate_model

    checksum = file_digest(sample.root / "model.json")
    with pytest.raises(ValueError):
        validate_model(
            sample.root,
            expected_scope=Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
            expected_model_sha256=checksum,
        )
    with pytest.raises(Problem):
        sample.verifier._model(
            ACTOR,
            sample.root,
            checksum,
            "smolvla",
            execution_timing="paused_simulation",
        )
    assert sample.get_calls == []
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    sample.indexes[candidate.artifact_id].pop("training_execution")
    with pytest.raises(Problem):
        sample.verifier.verify_candidate(ACTOR, sample.spec.project, sample.spec.run, candidate)
    assert sample.get_calls == [sample.spec.run.backend_job_name]


def test_command_worker_approval_keeps_exact_run_name_and_default_admission_block(tmp_path):
    from apps.learning_worker.backend import PolicyLearningWorker

    sample = command_fixture(tmp_path)
    deadline = utcnow() + timedelta(hours=1)
    run = sample.spec.run.model_copy(update={"deadline": deadline, "job_deadline_utc": deadline})
    specification = sample.spec.model_copy(update={"run": run})
    sample.config["job_deadline_utc"] = deadline.isoformat().replace("+00:00", "Z")
    sample.config["parameters"]["timeout_seconds"] = 1800
    approval = {
        "expires_at": deadline.isoformat(),
        "maximum_cost_usd": "10.00",
        "gpu_hourly_usd": "1.00",
        "config": sample.config,
    }
    sample.verifier.registry.approved_plan = lambda *_: approval
    worker = PolicyLearningWorker(
        sample.verifier.registry,
        sample.verifier,
        uuid4(),
        allowed_policy_types=("smolvla",),
        sdk_factory=lambda *_: pytest.fail("This test never initializes Azure clients."),
    )
    with pytest.raises(Problem) as failure:
        worker.preflight(ACTOR, specification)
    assert failure.value.code == "paused_learning_unavailable"
    assert worker._configuration(ACTOR, specification) == sample.config
    sample.config["run_id"] = "not-the-approved-command"
    with pytest.raises(Problem) as failure:
        worker._configuration(ACTOR, specification)
    assert failure.value.code == "worker_plan_mismatch"


def test_candidate_accepts_real_sdk_command_root_without_code_inputs_or_synthetic_parent(
    tmp_path, monkeypatch
):
    sdk = pytest.importorskip("azure.ai.ml")
    from azure.ai.ml._restclient.arm_ml_service.models import UriFolderJobOutput
    from azure.ai.ml.operations import JobOperations

    from learning.common import write_json
    from learning.gr00t.azure import workspace_id
    from learning.smolvla.azure import build_job

    def no_network(*_args, **_kwargs):
        pytest.fail("Offline SDK consumer verification must never send a network request.")

    monkeypatch.setattr("azure.core.pipeline.transport.RequestsTransport.send", no_network)
    sample = command_fixture(tmp_path)
    config = sample.config
    path = tmp_path / "command-job.json"
    write_json(path, build_job(config, command_snapshot(config), config["run_id"]))
    parsed = sdk.load_job(source=path)
    assert (
        parsed.type == "command"
        and parsed.code is None
        and not parsed.inputs
        and not parsed.outputs
    )
    environment_id = workspace_id(config) + "/environments/owned-image/versions/1"
    resolutions = []

    def resolve(asset, *, azureml_type):
        resolutions.append(azureml_type)
        assert azureml_type == "environments"
        assert asset.image == config["environment_image"]
        return environment_id

    JobOperations._resolve_arm_id_for_command_job(
        SimpleNamespace(
            _resolve_compute_id=lambda *_: workspace_id(config) + "/computes/" + config["compute"]
        ),
        parsed,
        resolve,
    )
    wire = parsed._to_job()._to_rest_object()
    wire.id, wire.name = sample.job_id, config["run_id"]
    wire.properties.status = "Completed"
    assert (
        wire.properties.code_id is None
        and not wire.properties.inputs
        and not wire.properties.outputs
    )
    wire.properties.outputs = {
        "default": UriFolderJobOutput(
            uri="azureml://datastores/workspaceartifactstore/ExperimentRun/dcid."
            + config["run_id"],
            mode="ReadWriteMount",
        )
    }
    jobs, environments = [], []

    def read_job_wire(name):
        jobs.append(name)
        assert name == config["run_id"]
        return wire

    def read_environment(name, *, version):
        environments.append((name, version))
        return SimpleNamespace(image=config["environment_image"])

    client = sdk.MLClient(
        SimpleNamespace(get_token=no_network),
        config["subscription_id"],
        config["resource_group"],
        config["workspace"],
    )
    monkeypatch.setattr(client.jobs, "_get_job", read_job_wire)
    monkeypatch.setattr(client.environments, "get", read_environment)
    actual = client.jobs.get(config["run_id"])
    assert actual.compute == config["compute"]
    assert actual.environment == "owned-image:1"
    sample.verifier._metadata_client = lambda _: client
    candidate = sample.verifier.completed_candidate(ACTOR, sample.spec, sample.job_id)
    assert candidate.azure_job_id == actual.id and candidate.optimizer_steps == 2
    assert resolutions == ["environments"]
    assert jobs == [config["run_id"]] * 2 and environments == [("owned-image", "1")]
