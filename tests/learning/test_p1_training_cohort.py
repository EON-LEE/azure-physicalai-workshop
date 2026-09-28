"""Explicit P1 admission fixtures. No additional demonstrations or training are claimed."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import SCOPE, overwrite
from learning.common import ContractError, canonical, digest, file_digest
from learning.gr00t.azure import workspace_id
from learning.smolvla import azure
from tests.learning.test_direct_command_jobs import command_config, live_job
from tests.learning.test_direct_command_provenance import command_model
from tests.learning.test_paused_contract import profile

SELECTOR = {"schema": "physicalai.smolvla-training-cohort/v1", "kind": "p1_additional20"}


def p1_config():
    config = command_config()
    config.update(
        training_cohort=dict(SELECTOR),
        run_id="p1-additional-command",
        control_profile_sha256=profile().sha256,
    )
    config["parameters"]["resume_mode"] = "weights_only"
    return config


def p0_parent(root, config):
    parent = command_model(root)
    parent["training"].update(
        azure_job_id=workspace_id(config) + "/jobs/p0-completed-command",
        episodes=[
            {
                "episode_id": f"p0-episode-{seed}",
                "environment_id": f"p0-environment-{seed}",
                "revision": "a" * 64,
                "seed": seed,
            }
            for seed in range(10001, 10021)
        ],
        optimizer_steps=1000,
        cumulative_optimizer_steps=1000,
        checkpoint_step=1000,
        ancestor_model_sha256s=["a" * 64],
        optimizer_state_restored=False,
        bitwise_continuation_claimed=False,
    )
    overwrite(root / "model.json", parent)
    config["inputs"]["parent_model"]["sha256"] = file_digest(root / "model.json")
    config["inputs"]["backbone"]["sha256"] = parent["backbone_manifest_sha256"]
    config["task_sha256"] = digest(canonical(parent["task"]))
    return parent


def p1_manifest(config, parent):
    return {
        "scope": asdict(SCOPE),
        "control_profile": parent["control_profile"],
        "criteria_sha256": parent["criteria_sha256"],
        "frozen_plan_sha256": parent["frozen_plan_sha256"],
        "episodes": [
            {
                "episode_id": f"p1-episode-{seed}",
                "environment_id": f"p1-environment-{seed}",
                "revision": "b" * 64,
                "seed": seed,
                "split": "train",
                "demonstration": {"kind": "reference_controller", **parent["task"]},
            }
            for seed in range(11001, 11021)
        ],
    }


def test_default_p0_config_job_bytes_and_static_allowlist_are_unchanged():
    config = command_config()
    assert "training_cohort" not in config
    assert (
        digest(canonical(config))
        == "eb230dd558c785ff6cc489913eb74afcce2de6e2dc0a5124ba637dce1f9ddf9e"
    )
    job = azure.build_job(config, "b" * 64, config["run_id"])
    assert (
        digest(canonical(job)) == "d2860f80ff1d37250d353e315f30d1903400bd69835f34508459ed646f1ec570"
    )
    assert (
        digest(canonical(list(azure.code_files(config))))
        == "4cdc449ba55219ec91261d8d3dd0489fd0845a63243b4cfe66debe3c21daba54"
    )
    assert not any(name.startswith("training_cohort") for name in job["tags"])


def test_p1_is_explicit_in_config_image_command_snapshot_and_job_tags():
    config = p1_config()
    azure.validate_config(config)
    job = azure.build_job(config, "b" * 64, config["run_id"])
    assert job["type"] == "command"
    assert not {"code", "inputs", "outputs", "jobs"} & job.keys()
    assert job["tags"]["training_cohort"] == "p1_additional20"
    assert job["tags"]["training_cohort_schema"] == SELECTOR["schema"]
    assert job["tags"]["runtime_config_sha256"] == digest(canonical(config))
    assert config["checkpointing"]["resume"] is None
    assert azure.code_files(config) == azure.code_files(command_config())


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "extra",
        "null",
        "list",
        "no-direct",
        "no-image",
        "realtime",
        "pipeline",
        "new-optimizer-mode",
        "full-state",
        "checkpoint-resume",
        "no-checkpointing",
    ],
)
def test_p1_selector_is_closed_and_only_explicit_weights_warm_start(change):
    config = p1_config()
    if change == "unknown":
        config["training_cohort"]["kind"] = "any_train"
    elif change == "extra":
        config["training_cohort"]["seeds"] = list(range(30001, 30021))
    elif change == "null":
        config["training_cohort"] = None
    elif change == "list":
        config["training_cohort"] = ["p1_additional20"]
    elif change == "no-direct":
        config.pop("job_execution")
    elif change == "no-image":
        config.pop("source_delivery")
    elif change == "realtime":
        config.pop("execution_timing")
    elif change == "pipeline":
        config["job_execution"]["kind"] = "pipeline"
    elif change == "new-optimizer-mode":
        config["parameters"]["resume_mode"] = "new"
    elif change == "full-state":
        config["parameters"]["resume_mode"] = "full_state"
    elif change == "checkpoint-resume":
        config["checkpointing"]["resume"] = {
            "checkpoint_sha256": "a" * 64,
            "step": 100,
            "source_azure_job_id": workspace_id(config) + "/jobs/old",
            "source_azure_job_type": "command",
        }
    else:
        config["checkpointing"] = None
    with pytest.raises(ContractError):
        azure.validate_config(config)


def test_p1_binding_accepts_only_disjoint_additional_twenty_and_real_p0(tmp_path):
    from learning.paused.command import validate_p1_training
    from learning.paused.command_artifacts import validate_model

    config = p1_config()
    parent = p0_parent(tmp_path, config)
    parent = validate_model(
        tmp_path,
        expected_scope=SCOPE,
        expected_model_sha256=config["inputs"]["parent_model"]["sha256"],
        for_inference=False,
    )
    validate_p1_training(config, p1_manifest(config, parent), parent)


@pytest.mark.parametrize(
    "change",
    [
        "p0-seeds",
        "heldout-seed",
        "missing-episode",
        "duplicate-seed",
        "duplicate-id",
        "learned-source",
        "relabelled-old-ids",
        "wrong-task",
        "wrong-profile",
        "wrong-criteria",
        "wrong-conditions",
        "foreign-owner",
        "foreign-workspace",
        "same-root-job",
        "pretrained",
        "pipeline-parent",
        "parent-p1-lineage",
        "parent-missing-episode",
        "parent-same-raw",
    ],
)
def test_p1_rejects_unapproved_data_parent_and_asymmetric_lineage(tmp_path, change):
    from learning.paused.command import validate_p1_training

    config = p1_config()
    parent = p0_parent(tmp_path, config)
    raw = p1_manifest(config, parent)
    if change == "p0-seeds":
        for item in raw["episodes"]:
            item["seed"] -= 1000
    elif change == "heldout-seed":
        raw["episodes"][0]["seed"] = 30001
    elif change == "missing-episode":
        raw["episodes"].pop()
    elif change == "duplicate-seed":
        raw["episodes"][0]["seed"] = raw["episodes"][1]["seed"]
    elif change == "duplicate-id":
        raw["episodes"][0]["episode_id"] = raw["episodes"][1]["episode_id"]
    elif change == "learned-source":
        raw["episodes"][0]["demonstration"]["kind"] = "learned"
    elif change == "relabelled-old-ids":
        raw["episodes"][0]["episode_id"] = parent["training"]["episodes"][0]["episode_id"]
    elif change == "wrong-task":
        raw["episodes"][0]["demonstration"]["instruction"] = "Another task"
    elif change == "wrong-profile":
        raw["control_profile"] = {**raw["control_profile"], "servo_sha256": "0" * 64}
    elif change == "wrong-criteria":
        raw["criteria_sha256"] = "0" * 64
    elif change == "wrong-conditions":
        parent["frozen_plan_sha256"] = "0" * 64
    elif change == "foreign-owner":
        parent["scope"] = {**parent["scope"], "owner_id": "0" * 64}
    elif change == "foreign-workspace":
        parent["training"]["azure_job_id"] = workspace_id(config) + "-other/jobs/p0"
    elif change == "same-root-job":
        parent["training"]["azure_job_id"] = workspace_id(config) + "/jobs/" + config["run_id"]
    elif change == "pretrained":
        parent["role"] = "pretrained"
        parent["training"] = None
    elif change == "pipeline-parent":
        parent["schema"] = "physicalai.smolvla-checkpoint/v2"
        parent["training"].pop("azure_job_type")
    elif change == "parent-p1-lineage":
        parent["training"]["episodes"][0]["seed"] = 11001
    elif change == "parent-missing-episode":
        parent["training"]["episodes"].pop()
    else:
        parent["training"]["raw_manifest_sha256"] = config["inputs"]["demonstrations"]["sha256"]
    with pytest.raises(ContractError):
        validate_p1_training(config, raw, parent)


def test_p1_current_job_cannot_lose_selector_tags_after_root_get():
    from learning.paused.command import validate_command_job

    config = p1_config()
    job = live_job(config)
    validate_command_job(
        SimpleNamespace(), config, job, snapshot_sha256="b" * 64, expected_status="Running"
    )
    job.tags.pop("training_cohort")
    with pytest.raises(ContractError):
        validate_command_job(
            SimpleNamespace(), config, job, snapshot_sha256="b" * 64, expected_status="Running"
        )


def test_p0_full_state_resume_remains_a_distinct_unmodified_variant():
    from tests.learning.test_checkpoint_job_plan import checkpoint_config

    config = checkpoint_config(resume=True)
    config["parameters"]["resume_mode"] = "full_state"
    original = copy.deepcopy(config)
    azure.validate_config(config)
    assert config == original and "training_cohort" not in config


def test_missing_selector_does_not_admit_additional_training_data(tmp_path):
    from learning.paused.command import validate_training_episodes

    config = p1_config()
    parent = p0_parent(tmp_path, config)
    episodes = p1_manifest(config, parent)["episodes"]
    config.pop("training_cohort")
    with pytest.raises(ContractError, match="twenty TRAIN seeds"):
        validate_training_episodes(config, episodes)


@pytest.mark.parametrize(
    "change", [None, "running", "pipeline", "other-id", "source", "scope", "cohort"]
)
def test_p1_reads_real_completed_parent_with_original_source_tags(tmp_path, change):
    from learning.paused.command import validate_p1_parent_job

    config = p1_config()
    parent = p0_parent(tmp_path, config)
    original = command_config()
    original["run_id"] = "p0-completed-command"
    original["control_profile_sha256"] = config["control_profile_sha256"]
    original["task_sha256"] = config["task_sha256"]
    original["specification_sha256"] = parent["training"]["specification_sha256"]
    job = live_job(original)
    job.status = "Completed"
    job.tags["code_snapshot_sha256"] = parent["training"]["code_snapshot_sha256"]
    if change == "running":
        job.status = "Running"
    elif change == "pipeline":
        job.parent_job_name = "fake-pipeline"
    elif change == "other-id":
        job.id += "-other"
    elif change == "source":
        job.tags["code_snapshot_sha256"] = "0" * 64
    elif change == "scope":
        job.tags["scope_owner"] = "0" * 64
    elif change == "cohort":
        job.tags["training_cohort"] = "p1_additional20"
    calls = []

    def get(name):
        calls.append(name)
        return job

    client = SimpleNamespace(jobs=SimpleNamespace(get=get))
    if change is None:
        validate_p1_parent_job(client, config, parent)
    else:
        with pytest.raises(ContractError):
            validate_p1_parent_job(client, config, parent)
    assert calls == ["p0-completed-command"]


def test_new_reader_verifies_original_p0_code_instead_of_current_worker_inventory(tmp_path):
    from learning.common import read_json
    from learning.smolvla.embedded_source import prepare_context, static_inventory

    context = tmp_path / "old-image"
    prepare_context(context, direct=True)
    archived_source = context / "source"
    old_file = archived_source / "learning" / "paused" / "command.py"
    old_file.write_bytes(old_file.read_bytes() + b"\n# historical source fixture; never executed\n")
    config = command_config()
    config["source_delivery"]["static_sha256"] = digest(
        canonical(static_inventory(archived_source, direct=True))
    )
    plan_dir = tmp_path / "original-plan"
    approved = azure.create_plan(
        config,
        plan_dir,
        deterministic_job_name=config["run_id"],
        source_root=archived_source,
    )
    assert file_digest(plan_dir / "code" / "learning" / "paused" / "command.py") == file_digest(
        old_file
    )
    plan = azure.read_plan(plan_dir)
    assert plan["plan_sha256"] == approved and plan["config"] == config
    snapshot = read_json(plan_dir / "code" / "snapshot.json")
    assert snapshot["sha256"] == plan["snapshot_sha256"]
    assert "training_cohort" not in plan["config"]
    with pytest.raises(ContractError, match="Static image source"):
        from learning.smolvla.embedded_source import verify_static_binding

        verify_static_binding(config, Path(__file__).resolve().parents[2])


@pytest.mark.parametrize("change", [None, "unknown", "resume"])
def test_image_bootstrap_admits_only_the_closed_additional_cohort(tmp_path, change):
    import base64

    from learning.common import read_json
    from learning.smolvla.embedded_source import prepare_context
    from learning.smolvla.image_bootstrap import ImageSourceError, materialize_source

    context = tmp_path / "image"
    receipt = prepare_context(context, direct=True)
    config = p1_config()
    config["source_delivery"] = read_json(context / "source-delivery.json")
    if change == "unknown":
        config["training_cohort"]["kind"] = "arbitrary"
    elif change == "resume":
        config["parameters"]["resume_mode"] = "full_state"
    payload = canonical(config) + b"\n"
    files = read_json(context / "static-manifest.json")["files"]
    snapshot = digest(canonical({**files, "run-config.json": digest(payload)}))
    args = dict(
        source_root=context / "source",
        manifest_path=context / "static-manifest.json",
        config_base64=base64.b64encode(payload).decode(),
        config_sha256=digest(payload),
        static_sha256=receipt["static_sha256"],
        snapshot_sha256=snapshot,
        destination=tmp_path / "task",
    )
    if change is None:
        assert materialize_source(**args) == config
        azure.verify_code(args["destination"], snapshot, config=config)
    else:
        with pytest.raises(ImageSourceError, match="cohort authority"):
            materialize_source(**args)
        assert not args["destination"].exists()
