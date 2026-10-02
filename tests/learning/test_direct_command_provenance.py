"""Schema fixtures and mocked training boundaries, never actual optimizer/quality evidence."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from learning.checks.fixtures import SCOPE, overwrite
from learning.common import ContractError, file_digest, read_json
from learning.paused.command import CommandDeadline
from tests.learning.test_direct_command_jobs import command_config, live_job
from tests.learning.test_paused_model_artifacts import fixture
from tests.learning.test_training_checkpoints import binding, native_bundle, origin


def command_model(root):
    fixture(root)
    value = read_json(root / "model.json")
    value["schema"] = "physicalai.smolvla-checkpoint/v3"
    value["training_execution"] = "azureml_command"
    value["training"]["azure_job_type"] = "command"
    value["training"].pop("azure_pipeline_job_id")
    value["training"]["episodes"] = [
        {
            "episode_id": "codec-episode",
            "environment_id": "codec-environment",
            "revision": "a" * 64,
            "seed": 10001,
        },
    ]
    value["training"]["gpu"]["device_count"] = 1
    overwrite(root / "model.json", value)
    return value


def command_origin():
    value = origin()
    value.pop("azure_pipeline_job_id")
    value["azure_job_type"] = "command"
    return value


def test_command_candidate_has_explicit_one_job_provenance_and_same_physical_validator(tmp_path):
    from learning.paused.artifacts import validate_model as legacy_model
    from learning.paused.command_artifacts import validate_model
    from learning.smolvla.artifacts import validate_model as realtime_model

    expected = command_model(tmp_path)
    checksum = file_digest(tmp_path / "model.json")
    assert (
        validate_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum) == expected
    )
    with pytest.raises(ContractError):
        realtime_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)
    with pytest.raises(ContractError):
        legacy_model(tmp_path, expected_scope=SCOPE, expected_model_sha256=checksum)


@pytest.mark.parametrize(
    "change",
    [
        "fake-parent",
        "fake-component",
        "wrong-type",
        "missing-type",
        "wrong-execution",
        "downgrade-v2",
        "pretrained",
        "zero-updates",
        "wrong-criteria",
        "wrong-source",
    ],
)
def test_command_model_cannot_invent_hierarchy_or_bypass_native_training_evidence(tmp_path, change):
    from learning.paused.command_artifacts import validate_model

    value = command_model(tmp_path)
    if change == "fake-parent":
        value["training"]["azure_pipeline_job_id"] = value["training"]["azure_job_id"]
    elif change == "fake-component":
        value["training"]["azure_component_job_id"] = value["training"]["azure_job_id"]
    elif change == "wrong-type":
        value["training"]["azure_job_type"] = "batch"
    elif change == "missing-type":
        value["training"].pop("azure_job_type")
    elif change == "wrong-execution":
        value["training_execution"] = "pipeline"
    elif change == "downgrade-v2":
        value["schema"] = "physicalai.smolvla-checkpoint/v2"
        value.pop("training_execution")
    elif change == "pretrained":
        value["role"] = "pretrained"
        value["training"] = None
    elif change == "zero-updates":
        value["training"]["optimizer_steps"] = 0
    elif change == "wrong-criteria":
        value["training"]["criteria_sha256"] = "0" * 64
    else:
        value["training"]["code_snapshot_sha256"] = ""
    overwrite(tmp_path / "model.json", value)
    with pytest.raises(ContractError):
        validate_model(
            tmp_path,
            expected_scope=SCOPE,
            expected_model_sha256=file_digest(tmp_path / "model.json"),
        )


def test_command_checkpoint_v2_preserves_native_safe_full_state_without_fake_parent(tmp_path):
    from learning.smolvla.checkpoints import seal_checkpoint, validate_checkpoint

    root = native_bundle(tmp_path / "checkpoint")
    checksum = seal_checkpoint(
        root,
        step=10,
        binding=binding(),
        origin=command_origin(),
        state_kind="full_state",
    )
    value = validate_checkpoint(
        root, expected_sha256=checksum, expected_binding=binding(), require_full_state=True
    )
    assert value["schema"] == "physicalai.smolvla-training-checkpoint/v2"
    assert value["origin"]["azure_job_type"] == "command"
    assert "azure_pipeline_job_id" not in value["origin"]
    assert value["origin"]["test_only"] is True
    assert value["learning_quality_verified"] is False


@pytest.mark.parametrize("change", ["v1", "fake-parent", "fake-component", "bad-type"])
def test_command_checkpoint_cannot_relabel_or_invent_a_parent(tmp_path, change):
    from learning.smolvla.checkpoints import seal_checkpoint, validate_checkpoint

    root = native_bundle(tmp_path / "checkpoint")
    seal_checkpoint(
        root, step=10, binding=binding(), origin=command_origin(), state_kind="full_state"
    )
    value = read_json(root / "checkpoint.json")
    if change == "v1":
        value["schema"] = "physicalai.smolvla-training-checkpoint/v1"
    elif change == "fake-parent":
        value["origin"]["azure_pipeline_job_id"] = value["origin"]["azure_job_id"]
    elif change == "fake-component":
        value["origin"]["azure_component_job_id"] = value["origin"]["azure_job_id"]
    else:
        value["origin"]["azure_job_type"] = "pipeline"
    overwrite(root / "checkpoint.json", value)
    with pytest.raises(ContractError):
        validate_checkpoint(
            root, expected_sha256=file_digest(root / "checkpoint.json"), expected_binding=binding()
        )


def test_new_command_resume_does_not_synthesize_source_pipeline(tmp_path):
    from learning.smolvla.azure import validate_config
    from learning.smolvla.checkpoint_runner import validate_resume_checkpoint
    from learning.smolvla.checkpoints import seal_checkpoint

    config = command_config()
    config["parameters"]["resume_mode"] = "weights_only"
    root = native_bundle(tmp_path / "checkpoint")
    checksum = seal_checkpoint(
        root, step=10, binding=binding(), origin=command_origin(), state_kind="full_state"
    )
    config["checkpointing"]["resume"] = {
        "checkpoint_sha256": checksum,
        "step": 10,
        "source_azure_job_id": command_origin()["azure_job_id"],
        "source_azure_job_type": "command",
    }
    config["inputs"].update(
        {
            name: {**config["inputs"]["demonstrations"], "name": name, "sha256": checksum}
            for name in ("resume_checkpoint", "converted_dataset")
        }
    )
    validate_config(config)
    new_origin = {
        **command_origin(),
        "azure_job_id": command_origin()["azure_job_id"] + "-new",
        "specification_sha256": "0" * 64,
    }
    value = validate_resume_checkpoint(
        root, config=config, expected_binding=binding(), current_origin=new_origin
    )
    assert value["origin"]["azure_job_type"] == "command"
    for changed in (
        command_origin(),
        {**new_origin, "azure_pipeline_job_id": new_origin["azure_job_id"]},
    ):
        with pytest.raises(ContractError):
            validate_resume_checkpoint(
                root, config=config, expected_binding=binding(), current_origin=changed
            )
    config["checkpointing"]["resume"]["source_azure_pipeline_job_id"] = new_origin["azure_job_id"]
    with pytest.raises(ContractError):
        validate_config(config)


def test_consumer_checks_completed_root_without_readmitting_expired_authority():
    from learning.paused.command import validate_command_job

    config = command_config()
    config["job_deadline_utc"] = "2020-01-01T00:00:00Z"
    job = live_job(config)
    job.status = "Completed"
    result = validate_command_job(
        SimpleNamespace(),
        config,
        job,
        snapshot_sha256="b" * 64,
        expected_status="Completed",
    )
    assert result["azure_job_id"] == job.id
    assert result["azure_job_type"] == "command"
    with pytest.raises(ContractError):
        validate_command_job(
            SimpleNamespace(),
            config,
            job,
            snapshot_sha256="0" * 64,
            expected_status="Completed",
        )


@pytest.mark.parametrize(
    "change",
    ["command", "instance-count", "timeout", "distribution", "source", "output", "image"],
)
def test_completed_root_is_not_accepted_on_tag_only_provenance(change):
    from learning.paused.command import validate_command_job

    config = command_config()
    job = live_job(config)
    job.status = "Completed"
    if change == "command":
        job.command = "python -c 'print(1)'"
    elif change == "instance-count":
        job.resources.instance_count = 2
    elif change == "timeout":
        job.limits.timeout += 1
    elif change == "distribution":
        job.distribution = "PyTorch"
    elif change == "source":
        job.tags["code_snapshot_sha256"] = "0" * 64
    elif change == "output":
        job.outputs = {"model": object()}
    else:
        job.environment.image = "mutable:image"
    with pytest.raises(ContractError):
        validate_command_job(
            SimpleNamespace(),
            config,
            job,
            snapshot_sha256="b" * 64,
            expected_status="Completed",
        )


def test_command_monotonic_budget_includes_transfer_and_publication_not_only_optimizer():
    config = command_config()
    clock = [datetime(2029, 12, 31, 23, 50, tzinfo=UTC)]
    monotonic = [10.0]
    config["parameters"]["timeout_seconds"] = 120
    deadline = CommandDeadline(config, clock=lambda: clock[0], monotonic=lambda: monotonic[0])
    assert deadline.check() == 120
    monotonic[0] += 100
    clock[0] -= timedelta(hours=1)
    assert deadline.check() == 20
    monotonic[0] += 21
    with pytest.raises(ContractError, match="deadline"):
        deadline.check()


def test_command_timeout_cannot_extend_original_absolute_deadline():
    config = command_config()
    clock = datetime(2029, 12, 31, 23, 59, tzinfo=UTC)
    deadline = CommandDeadline(config, clock=lambda: clock, monotonic=lambda: 10)
    assert deadline.check() == 60


def test_pipeline_resume_config_stays_separate_from_command_type():
    from learning.smolvla.azure import validate_config
    from tests.learning.test_checkpoint_job_plan import checkpoint_config

    original = checkpoint_config(resume=True)
    validate_config(original)
    changed = copy.deepcopy(original)
    changed["checkpointing"]["resume"].pop("source_azure_pipeline_job_id")
    changed["checkpointing"]["resume"]["source_azure_job_type"] = "command"
    with pytest.raises(ContractError, match="provenance"):
        validate_config(changed)
