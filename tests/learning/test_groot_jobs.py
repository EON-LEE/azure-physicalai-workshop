from dataclasses import asdict
from types import SimpleNamespace

import pytest

from learning.checks.groot_fixtures import PROFILE, TASK
from learning.common import ContractError, canonical, digest
from learning.gr00t.artifacts import UPSTREAM, task_contract

pytest_plugins = ("learning.checks.pytest_fixtures",)


@pytest.fixture
def config(azure_config):
    from learning.gr00t.train import Gr00tTrainOptions

    result = {**azure_config}
    result.update(
        schema="physicalai.gr00t-azure/v1",
        compute_tier="LowPriority",
        parameters=asdict(
            Gr00tTrainOptions(max_steps=100, checkpoint_steps=10, compute_tier="LowPriority")
        ),
        upstream=UPSTREAM,
        specification_sha256="a" * 64,
        control_profile_sha256=PROFILE.sha256,
        task_sha256=digest(canonical(task_contract(TASK))),
    )
    result["inputs"] = {
        **azure_config["inputs"],
        "parent_model": {
            **azure_config["inputs"]["demonstrations"],
            "name": "franka-p0",
            "uri": azure_config["inputs"]["demonstrations"]["uri"] + "/parent",
        },
    }
    return result


def test_groot_job_plan_is_bound_private_offline_and_spot_resilient(config):
    from learning.gr00t.azure import build_job

    job = build_job(config, "e" * 64, "groot-run-1")
    assert job["name"] == "groot-run-1"
    assert set(job["jobs"]) == {"convert", "train"}
    assert job["tags"]["policy_type"] == "gr00t_n1_5"
    assert job["tags"]["specification_sha256"] == config["specification_sha256"]
    assert job["outputs"]["model"]["mode"] == "rw_mount"
    assert "learning.gr00t.components train" in job["jobs"]["train"]["command"]
    assert "--parent" in job["jobs"]["train"]["command"]
    assert job["jobs"]["train"]["identity"]["client_id"] == config["managed_identity_client_id"]
    assert job["settings"]["continue_on_step_failure"] is False


@pytest.mark.parametrize("change", ["tier", "parent-uri", "mutable-model", "ACT", "no-checkpoints"])
def test_groot_plan_refuses_implicit_or_incompatible_resources(config, change):
    from learning.gr00t.azure import validate_config

    if change == "tier":
        config.pop("compute_tier")
    elif change == "parent-uri":
        config["inputs"]["parent_model"]["uri"] = "https://huggingface.co/model"
    elif change == "mutable-model":
        config["upstream"] = {**UPSTREAM, "model_revision": "main"}
    elif change == "ACT":
        config["upstream"] = {**UPSTREAM, "model_id": "act"}
    else:
        config["parameters"]["checkpoint_steps"] = config["parameters"]["max_steps"]
    with pytest.raises(ContractError):
        validate_config(config)


def test_named_job_reconciliation_never_resubmits_or_reads_another_owner(config):
    from learning.gr00t.azure import Gr00tJobs, workspace_id

    job = SimpleNamespace(
        name="groot-run-1",
        id=workspace_id(config) + "/jobs/groot-run-1",
        status="Running",
        tags={
            "scope_owner": config["owner_id"],
            "scope_tenant": config["tenant_id"],
            "specification_sha256": config["specification_sha256"],
            "policy_type": "gr00t_n1_5",
        },
    )
    backend = Gr00tJobs(
        SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job)),
        config,
        storage_client=SimpleNamespace(),
    )
    status = backend.status("groot-run-1")
    assert status["status"] == "running" and status["optimizer_steps"] is None
    job.tags["scope_owner"] = "b" * 64
    with pytest.raises(ContractError, match="owner"):
        backend.status("groot-run-1")


def test_unknown_remote_job_status_is_not_success(config):
    from learning.gr00t.azure import Gr00tJobs, workspace_id

    job = SimpleNamespace(
        name="groot-run-1",
        id=workspace_id(config) + "/jobs/groot-run-1",
        status="FutureUnknownStatus",
        tags={
            "scope_owner": config["owner_id"],
            "scope_tenant": config["tenant_id"],
            "specification_sha256": config["specification_sha256"],
            "policy_type": "gr00t_n1_5",
        },
    )
    backend = Gr00tJobs(
        SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job)),
        config,
        storage_client=SimpleNamespace(),
    )
    with pytest.raises(ContractError, match="status"):
        backend.status("groot-run-1")
