import pytest

from learning.common import ContractError
from learning.gr00t.train import Gr00tTrainOptions
from learning.smolvla.train import TrainOptions
from tests.learning.test_checkpoint_job_plan import checkpoint_config
from tests.learning.test_training_checkpoints import binding, native_bundle, origin, seal


def test_full_state_is_explicitly_supported_only_for_native_smol_resume():
    TrainOptions(resume_mode="full_state").validate()
    with pytest.raises(ContractError):
        Gr00tTrainOptions(resume_mode="full_state").validate()


def test_full_state_job_requires_complete_checkpoint_and_remaining_optimizer_steps():
    from learning.smolvla.azure import build_job, validate_config

    value = checkpoint_config(resume=True)
    value["parameters"]["resume_mode"] = "full_state"
    validate_config(value)
    job = build_job(value, "b" * 64, "new-full-state-job")
    assert set(job["jobs"]) == {"train"}
    assert "--resume-checkpoint" in job["jobs"]["train"]["command"]
    value["checkpointing"]["resume"]["step"] = value["parameters"]["max_steps"]
    with pytest.raises(ContractError, match="remaining"):
        validate_config(value)


def test_full_state_refuses_weights_only_published_checkpoint(tmp_path):
    from learning.smolvla.checkpoint_runner import validate_resume_checkpoint

    root = native_bundle(tmp_path / "weights", full=False)
    checksum = seal(root, kind="weights_only")
    config = checkpoint_config(resume=True)
    config["parameters"]["resume_mode"] = "full_state"
    config["checkpointing"]["resume"].update(
        checkpoint_sha256=checksum,
        source_azure_job_id=origin()["azure_job_id"],
        source_azure_pipeline_job_id=origin()["azure_pipeline_job_id"],
    )
    current = {
        **origin(),
        "azure_job_id": origin()["azure_job_id"] + "-new",
        "azure_pipeline_job_id": origin()["azure_pipeline_job_id"] + "-new",
        "specification_sha256": "9" * 64,
    }
    with pytest.raises(ContractError, match="full.state"):
        validate_resume_checkpoint(
            root, config=config, expected_binding=binding(), current_origin=current
        )


@pytest.mark.parametrize(
    "changed", ["training_config_sha256", "training_code_sha256", "runtime_sha256"]
)
def test_full_state_does_not_allow_the_weights_only_config_runtime_exceptions(tmp_path, changed):
    from learning.smolvla.checkpoint_runner import validate_resume_checkpoint

    root = native_bundle(tmp_path / "state")
    checksum = seal(root)
    config = checkpoint_config(resume=True)
    config["parameters"]["resume_mode"] = "full_state"
    config["checkpointing"]["resume"].update(
        checkpoint_sha256=checksum,
        source_azure_job_id=origin()["azure_job_id"],
        source_azure_pipeline_job_id=origin()["azure_pipeline_job_id"],
    )
    current = {
        **origin(),
        "azure_job_id": origin()["azure_job_id"] + "-new",
        "azure_pipeline_job_id": origin()["azure_pipeline_job_id"] + "-new",
        "specification_sha256": "9" * 64,
    }
    with pytest.raises(ContractError, match="binding"):
        validate_resume_checkpoint(
            root,
            config=config,
            expected_binding={**binding(), changed: "0" * 64},
            current_origin=current,
        )


def test_full_state_cli_resumes_native_config_without_policy_path_override(tmp_path):
    from learning.smolvla.train import full_state_training_command

    command = full_state_training_command(
        tmp_path / "original", tmp_path / "dataset", tmp_path / "backbone", tmp_path / "new-output"
    )
    assert "--resume=true" in command
    assert any(
        arg.startswith("--config_path=") and arg.endswith("pretrained_model/train_config.json")
        for arg in command
    )
    assert not any(arg.startswith("--policy.path=") for arg in command)
    assert not any(arg.startswith("--steps=") for arg in command)
    assert any(arg.startswith("--output_dir=") for arg in command)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "missing-source-sha",
        "missing-source-step",
        "no-new-updates",
        "wrong-checkpoint-step",
        "wrong-cumulative",
        "missing-source-cumulative",
        "restoration-not-verified",
        "missing-source-job",
        "same-source-job",
    ],
)
def test_candidate_full_state_lineage_requires_exact_positive_new_update_counts(tmp_path, change):
    from learning.checks.fixtures import SCOPE, overwrite
    from learning.common import file_digest, read_json
    from learning.paused.artifacts import validate_model
    from tests.learning.test_paused_model_artifacts import fixture

    fixture(tmp_path)
    value = read_json(tmp_path / "model.json")
    training = value["training"]
    training.update(
        resume_mode="full_state",
        optimizer_steps=3,
        checkpoint_step=13,
        cumulative_optimizer_steps=23,
        resume_from_checkpoint_step=10,
        resume_from_cumulative_optimizer_steps=20,
        resume_from_checkpoint_sha256="8" * 64,
        optimizer_state_restored=True,
        bitwise_continuation_claimed=False,
        resume_from_job_id=training["azure_job_id"] + "-previous",
        resume_from_pipeline_job_id=training["azure_pipeline_job_id"] + "-previous",
    )
    if change == "missing-source-sha":
        training.pop("resume_from_checkpoint_sha256")
    elif change == "missing-source-step":
        training.pop("resume_from_checkpoint_step")
    elif change == "no-new-updates":
        training["optimizer_steps"] = 0
    elif change == "wrong-checkpoint-step":
        training["checkpoint_step"] = 14
    elif change == "wrong-cumulative":
        training["cumulative_optimizer_steps"] = 24
    elif change == "missing-source-cumulative":
        training.pop("resume_from_cumulative_optimizer_steps")
    elif change == "restoration-not-verified":
        training["optimizer_state_restored"] = False
    elif change == "missing-source-job":
        training.pop("resume_from_pipeline_job_id")
    elif change == "same-source-job":
        training["resume_from_job_id"] = training["azure_job_id"]
    overwrite(tmp_path / "model.json", value)
    if change is None:
        validate_model(
            tmp_path,
            expected_scope=SCOPE,
            expected_model_sha256=file_digest(tmp_path / "model.json"),
        )
    else:
        with pytest.raises(ContractError):
            validate_model(
                tmp_path,
                expected_scope=SCOPE,
                expected_model_sha256=file_digest(tmp_path / "model.json"),
            )


@pytest.mark.parametrize(
    "change",
    ["missing-gaussian", "missing-sampler", "wrong-step", "workers", "amp", "precision-bool"],
)
def test_incomplete_full_state_metadata_fails_before_optional_torch_import(tmp_path, change):
    from learning.common import write_json
    from learning.smolvla.checkpoint_state import continuation_metadata

    value = {
        "schema": "physicalai.smolvla-continuation/v1",
        "step": 10,
        "sampler": {
            "schema": "physicalai.smolvla-data-order/v1",
            "dataset_size": 10,
            "indices_sha256": "a" * 64,
            "batch_size": 2,
            "epoch": 1,
            "cursor": 10,
            "batches": 10,
        },
        "rng": {
            "python_version": 3,
            "python_gaussian": 0.123456789012345,
            "numpy_algorithm": "MT19937",
            "numpy_position": 7,
            "numpy_has_gaussian": 1,
            "numpy_gaussian": 0.123456789012345,
            "cuda_device_count": 0,
        },
        "precision": {
            "matmul_precision": "highest",
            "matmul_tf32": False,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "deterministic_algorithms": True,
            "default_dtype": "torch.float32",
        },
        "num_workers": 0,
        "world_size": 1,
        "mixed_precision": "no",
    }
    if change == "missing-gaussian":
        value["rng"].pop("python_gaussian")
    elif change == "missing-sampler":
        value["sampler"].pop("cursor")
    elif change == "wrong-step":
        value["step"] = 11
    elif change == "workers":
        value["num_workers"] = 2
    elif change == "amp":
        value["mixed_precision"] = "bf16"
    else:
        value["precision"]["matmul_tf32"] = 1
    (tmp_path / "training_state").mkdir()
    write_json(tmp_path / "training_state" / "continuation.json", value)
    with pytest.raises(ContractError):
        continuation_metadata(tmp_path, 10)
