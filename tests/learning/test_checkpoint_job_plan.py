from dataclasses import asdict

import pytest

from learning.common import ContractError
from learning.smolvla.azure import build_job, create_plan, read_plan, validate_config
from learning.smolvla.checkpoint_runner import POLICY_SCHEMA
from learning.smolvla.checkpoints import DEFAULT_LIMITS
from tests.learning.test_paused_azure import config


def checkpoint_config(*, resume=False):
    value = config()
    value["checkpointing"] = {
        "schema": POLICY_SCHEMA,
        "limits": asdict(DEFAULT_LIMITS),
        "resume": None,
    }
    if resume:
        value["parameters"]["resume_mode"] = "weights_only"
        source = (
            "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/unit/"
            "providers/Microsoft.MachineLearningServices/workspaces/unit/jobs/"
        )
        value["checkpointing"]["resume"] = {
            "checkpoint_sha256": "a" * 64,
            "source_azure_job_id": source + "previous-component",
            "source_azure_pipeline_job_id": source + "previous-pipeline",
            "step": 10,
        }
        asset = value["inputs"]["demonstrations"]
        value["inputs"].update(
            {
                name: {**asset, "name": name, "sha256": "a" * 64, "uri": asset["uri"] + "/" + name}
                for name in ("resume_checkpoint", "converted_dataset")
            }
        )
    return value


def test_intermediate_checkpoint_publisher_code_is_in_exact_reviewed_snapshot(tmp_path):
    value = checkpoint_config()
    checksum = create_plan(value, tmp_path / "plan", deterministic_job_name="save-intermediate")
    saved = read_plan(tmp_path / "plan")
    assert saved["plan_sha256"] == checksum
    assert (tmp_path / "plan" / "code" / "learning" / "smolvla" / "checkpoint_runner.py").is_file()
    assert set(build_job(value, "b" * 64, "save-intermediate")["jobs"]) == {"convert", "train"}


def test_resume_reuses_exact_converted_dataset_without_reconversion_or_same_job_retry():
    value = checkpoint_config(resume=True)
    job = build_job(value, "b" * 64, "new-authorized-resume")
    assert set(job["jobs"]) == {"train"}
    train = job["jobs"]["train"]
    assert train["inputs"]["dataset"] == "${{parent.inputs.converted_dataset}}"
    assert train["inputs"]["resume"] == "${{parent.inputs.resume_checkpoint}}"
    assert "--resume-checkpoint" in train["command"]
    assert train["limits"]["timeout"] == value["parameters"]["timeout_seconds"]


@pytest.mark.parametrize(
    "change", ["checkpoint-hash", "missing-data", "wrong-mode", "too-many", "too-many-bytes"]
)
def test_partial_or_unbounded_checkpoint_plan_is_rejected(change):
    value = checkpoint_config(resume=True)
    if change == "checkpoint-hash":
        value["inputs"]["resume_checkpoint"]["sha256"] = "0" * 64
    elif change == "missing-data":
        value["inputs"].pop("converted_dataset")
    elif change == "wrong-mode":
        value["parameters"]["resume_mode"] = "new"
    elif change == "too-many":
        value["parameters"]["max_steps"] = 100000
    else:
        value["checkpointing"]["limits"]["max_total_bytes"] = 2**50
    with pytest.raises(ContractError):
        validate_config(value)
