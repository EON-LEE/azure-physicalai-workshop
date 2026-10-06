from types import SimpleNamespace

import pytest

from learning.common import ContractError
from learning.gr00t.azure import job_tags, workspace_id
from learning.smolvla import azure
from tests.learning.test_checkpoint_job_plan import checkpoint_config


def command_config():
    value = checkpoint_config()
    value["compute_size"] = "Standard_NC24ads_A100_v4"
    value["output_prefix"] = (
        f"tenants/{value['tenant_id']}/owners/{value['owner_id']}/learning/outputs"
    )
    value["run_id"] = "direct-p0"
    value["source_delivery"] = {
        "schema": "physicalai.smolvla-source-delivery/v1",
        "mode": "image_embedded",
        "static_sha256": "a" * 64,
    }
    value["job_execution"] = {
        "schema": "physicalai.smolvla-command-execution/v1",
        "kind": "command",
        "data_transport": "private_blob_mi",
    }
    from learning.azure import datastore_prefix

    for name, asset in value["inputs"].items():
        asset["uri"] = (
            datastore_prefix(value)
            + f"tenants/{value['tenant_id']}/owners/{value['owner_id']}/"
            + "learning/inputs/"
            + name
        )
    return value


def test_actual_command_shape_has_no_code_or_custom_service_data_preparation():
    value = command_config()
    result = azure.build_job(value, "b" * 64, value["run_id"])
    assert result["type"] == "command"
    assert not {"code", "inputs", "outputs", "jobs", "settings"} & result.keys()
    assert result["compute"] == "azureml:" + value["compute"]
    assert result["resources"]["instance_count"] == 1
    assert result["identity"] == {
        "type": "managed",
        "client_id": value["managed_identity_client_id"],
    }
    assert result["environment"]["image"] == value["environment_image"]
    assert result["limits"]["timeout"] == value["parameters"]["timeout_seconds"]
    assert "command-train" in result["command"]
    assert result["tags"]["job_execution"] == "command"
    assert result["tags"]["data_transport"] == "private_blob_mi"
    assert result["tags"]["runtime_config_sha256"]


@pytest.mark.parametrize("tier", ["LowPriority", "Dedicated"])
def test_command_accepts_approved_one_a100_tiers(tier):
    value = command_config()
    value["compute_tier"] = tier
    value["parameters"]["compute_tier"] = tier
    result = azure.build_job(value, "b" * 64, value["run_id"])
    assert result["tags"]["job_execution"] == "command"


@pytest.mark.parametrize(
    "compute_size,compute_tier",
    [
        ("Standard_NC24ads_A100_v4", "Standard"),
        ("Standard_NC24ads_A100_v4", "Spot"),
        ("Standard_NC6s_v3", "LowPriority"),
        ("Standard_NC24ads_A100_v4", "dedicated"),
    ],
)
def test_command_rejects_unapproved_tier_or_size(compute_size, compute_tier):
    value = command_config()
    value["compute_size"] = compute_size
    value["compute_tier"] = compute_tier
    with pytest.raises(ContractError):
        azure.build_job(value, "b" * 64, value["run_id"])


@pytest.mark.parametrize(
    "change",
    ["unknown", "extra", "no-image", "no-checkpoints", "legacy-mode", "other-name"],
)
def test_command_is_only_an_explicit_closed_new_authority(change):
    value = command_config()
    if change == "unknown":
        value["job_execution"]["kind"] = "batch"
    elif change == "extra":
        value["job_execution"]["allow_shared_keys"] = True
    elif change == "no-image":
        value.pop("source_delivery")
    elif change == "no-checkpoints":
        value.pop("checkpointing")
    elif change == "legacy-mode":
        value.pop("execution_timing")
    else:
        value["run_id"] = "another-job"
    with pytest.raises(ContractError):
        azure.build_job(value, "b" * 64, "direct-p0")


def live_job(value):
    from learning.offline import OFFLINE_ENV
    from learning.smolvla.embedded_source import embedded_command

    return SimpleNamespace(
        id=workspace_id(value) + "/jobs/" + value["run_id"],
        name=value["run_id"],
        type="command",
        status="Running",
        parent_job_name=None,
        code=None,
        inputs={},
        outputs={},
        compute=workspace_id(value) + "/computes/" + value["compute"],
        identity=SimpleNamespace(type="managed", client_id=value["managed_identity_client_id"]),
        environment=SimpleNamespace(image=value["environment_image"]),
        environment_variables={**OFFLINE_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        command=embedded_command(value, "b" * 64, stage="command-train"),
        resources=SimpleNamespace(instance_count=1),
        limits=SimpleNamespace(timeout=value["parameters"]["timeout_seconds"]),
        distribution=None,
        tags=job_tags(value, "b" * 64, policy_type="smolvla"),
    )


def test_running_command_has_one_actual_job_identity_and_no_fabricated_parent(monkeypatch):
    from learning.paused.command import running_command_binding

    value = command_config()
    job = live_job(value)
    monkeypatch.setenv("AZUREML_RUN_ID", value["run_id"])
    calls = []

    def get(name):
        calls.append(name)
        return job

    result = running_command_binding(
        SimpleNamespace(jobs=SimpleNamespace(get=get)),
        value,
        snapshot_sha256="b" * 64,
    )
    assert result["azure_job_id"] == job.id
    assert result["azure_job_type"] == "command"
    assert result["specification_sha256"] == value["specification_sha256"]
    assert "azure_pipeline_job_id" not in result and "azure_component_job_id" not in result
    assert calls == [value["run_id"]]


@pytest.mark.parametrize("representation", ["name", "azureml", "arm"])
def test_command_accepts_exact_approved_compute_readback_forms(representation):
    from learning.paused.command import validate_command_job

    value = command_config()
    job = live_job(value)
    job.compute = {
        "name": value["compute"],
        "azureml": "azureml:" + value["compute"],
        "arm": workspace_id(value) + "/computes/" + value["compute"],
    }[representation]
    result = validate_command_job(
        SimpleNamespace(), value, job, snapshot_sha256="b" * 64, expected_status="Running"
    )
    assert result["azure_job_id"] == job.id


@pytest.mark.parametrize("change", ["other-name", "other-workspace", "suffix", "case", "none"])
def test_compute_readback_compatibility_does_not_admit_other_resources(change):
    from learning.paused.command import validate_job

    value = command_config()
    job = live_job(value)
    job.compute = {
        "other-name": value["compute"] + "-other",
        "other-workspace": workspace_id(value) + "-other/computes/" + value["compute"],
        "suffix": value["compute"] + "/",
        "case": value["compute"].upper(),
        "none": None,
    }[change]
    with pytest.raises(ContractError, match="compute"):
        validate_job(job, value)


def test_bare_compute_does_not_hide_a_foreign_job_workspace():
    from learning.paused.command import validate_job

    value = command_config()
    job = live_job(value)
    job.compute = value["compute"]
    job.id = workspace_id(value) + "-other/jobs/" + value["run_id"]
    with pytest.raises(ContractError, match="standalone command"):
        validate_job(job, value)


@pytest.mark.parametrize("representation", ["arm", "versioned-name"])
def test_command_resolves_exact_versioned_environment_from_public_sdk_get(representation):
    from learning.paused.command import validate_command_job

    value = command_config()
    job = live_job(value)
    job.environment = (
        workspace_id(value) + "/environments/native-image/versions/1"
        if representation == "arm"
        else "native-image:1"
    )
    calls = []

    def get(name, *, version):
        calls.append((name, version))
        return SimpleNamespace(image=value["environment_image"])

    validate_command_job(
        SimpleNamespace(environments=SimpleNamespace(get=get)),
        value,
        job,
        snapshot_sha256="b" * 64,
        expected_status="Running",
    )
    assert calls == [("native-image", "1")]


@pytest.mark.parametrize(
    "reference",
    [
        "native-image",
        "native-image@latest",
        "native-image:1:extra",
        "native-image:1?query=1",
        "native-image:1/other",
        "https://unapproved.example/native:1",
        "azureml://registries/other/environments/native-image/versions/1",
    ],
)
def test_invalid_environment_references_fail_before_an_environment_read(reference):
    from learning.paused.command import validate_command_job

    value = command_config()
    job = live_job(value)
    job.environment = reference
    with pytest.raises(ContractError):
        validate_command_job(
            SimpleNamespace(
                environments=SimpleNamespace(
                    get=lambda *args, **kwargs: pytest.fail("Unapproved environment read")
                )
            ),
            value,
            job,
            snapshot_sha256="b" * 64,
            expected_status="Running",
        )


def test_foreign_environment_arm_id_is_not_reduced_to_a_local_name():
    from learning.paused.command import validate_command_job

    value = command_config()
    job = live_job(value)
    job.environment = workspace_id(value) + "-other/environments/native-image/versions/1"
    with pytest.raises(ContractError):
        validate_command_job(
            SimpleNamespace(
                environments=SimpleNamespace(
                    get=lambda *args, **kwargs: pytest.fail("Foreign environment read")
                )
            ),
            value,
            job,
            snapshot_sha256="b" * 64,
            expected_status="Running",
        )


def test_short_environment_reference_still_requires_the_exact_approved_image():
    from learning.paused.command import validate_command_job

    value = command_config()
    job = live_job(value)
    job.environment = "native-image:1"
    with pytest.raises(ContractError, match="image digest"):
        validate_command_job(
            SimpleNamespace(
                environments=SimpleNamespace(
                    get=lambda *args, **kwargs: SimpleNamespace(image="unapproved:latest")
                )
            ),
            value,
            job,
            snapshot_sha256="b" * 64,
            expected_status="Running",
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing-run",
        "different-run",
        "parent",
        "pipeline-type",
        "nonrunning",
        "identity",
        "image",
        "compute",
        "config-hash",
        "static-hash",
        "owner",
        "source-code",
        "registered-input",
    ],
)
def test_command_runtime_rejects_unverified_native_identity(monkeypatch, change):
    from learning.paused.command import running_command_binding

    value = command_config()
    job = live_job(value)
    monkeypatch.setenv("AZUREML_RUN_ID", value["run_id"])
    if change == "missing-run":
        monkeypatch.delenv("AZUREML_RUN_ID")
    elif change == "different-run":
        monkeypatch.setenv("AZUREML_RUN_ID", "other-job")
    elif change == "parent":
        job.parent_job_name = "invented-parent"
    elif change == "pipeline-type":
        job.type = "pipeline"
    elif change == "nonrunning":
        job.status = "Completed"
    elif change == "identity":
        job.identity.client_id = "00000000-0000-0000-0000-000000000000"
    elif change == "image":
        job.environment.image = value["environment_image"].replace("@sha256:", ":mutable-")
    elif change == "compute":
        job.compute = "azureml:other"
    elif change == "config-hash":
        job.tags["runtime_config_sha256"] = "0" * 64
    elif change == "static-hash":
        job.tags["static_source_sha256"] = "0" * 64
    elif change == "owner":
        job.tags["scope_owner"] = "0" * 64
    elif change == "source-code":
        job.code = "azureml:unexpected-code:1"
    else:
        job.inputs = {"data": "unexpected-data-mount"}
    with pytest.raises(ContractError):
        running_command_binding(
            SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job)),
            value,
            snapshot_sha256="b" * 64,
        )


def test_old_pipeline_graph_is_not_relabelled_as_command():
    value = checkpoint_config()
    result = azure.build_job(value, "b" * 64, "legacy-pipeline")
    assert result["type"] == "pipeline" and set(result["jobs"]) == {"convert", "train"}
    assert "job_execution" not in result["tags"]
    assert all(step["code"] == "./code" for step in result["jobs"].values())
