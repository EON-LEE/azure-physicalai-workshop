from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from learning import azure
from learning.checks.fixtures import overwrite
from learning.common import ContractError, read_json

pytest_plugins = ("learning.checks.pytest_fixtures",)


@pytest.fixture
def source_snapshot(tmp_path):
    root = tmp_path / "source"
    for name in azure.SNAPSHOT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test-only snapshot contents: {name}\n")
    (root / ".env").write_text("NOT_A_REAL_SECRET=must_not_be_uploaded")
    (root / "tests").mkdir()
    (root / "tests" / "private.json").write_text("private fixture, excluded")
    return root


def test_plans_are_offline_deterministic_and_allowlist_only(
    azure_config, source_snapshot, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        azure, "version", lambda _: pytest.fail("Imported cloud SDK during planning")
    )
    first, second = tmp_path / "first", tmp_path / "second"
    first_sha = azure.create_plan(azure_config, first, source_root=source_snapshot)
    second_sha = azure.create_plan(azure_config, second, source_root=source_snapshot)
    assert first_sha == second_sha
    plan = azure.read_plan(first)
    assert plan["plan_sha256"] == first_sha
    assert not (first / "code" / ".env").exists()
    assert not (first / "code" / "tests").exists()
    assert set(read_json(first / "code" / "snapshot.json")["files"]) == set(azure.SNAPSHOT_FILES)
    assert not any("checks" in name or "tests" in name for name in azure.SNAPSHOT_FILES)
    assert (first / "job.json").read_bytes() == (second / "job.json").read_bytes()


def test_training_pipeline_has_actual_components_scoped_io_and_finite_limits(azure_config):
    job = azure.build_job(azure_config, "a" * 64)
    assert set(job["jobs"]) == {"convert", "train"}
    assert job["settings"]["continue_on_step_failure"] is False
    assert job["settings"]["force_rerun"] is True
    assert job["inputs"]["demonstrations"]["path"] == "azureml:approved-demonstrations:1"
    assert (
        job["jobs"]["train"]["inputs"]["converted"] == "${{parent.jobs.convert.outputs.converted}}"
    )
    for step in job["jobs"].values():
        assert step["identity"]["client_id"] == azure_config["managed_identity_client_id"]
        assert set(step["identity"]) == {"type", "client_id"}
        assert step["resources"]["instance_count"] == 1
        assert step["limits"]["timeout"] <= 1800
        assert step["environment_variables"]["HF_HUB_OFFLINE"] == "1"
        assert step["environment"]["image"] == azure_config["environment_image"]
        assert step["command"].startswith("python -m learning.components")
        assert step["compute"] == "azureml:approved-gpu"
    for output in job["outputs"].values():
        assert output["path"].startswith(azure.datastore_prefix(azure_config))
        assert azure_config["output_prefix"] in output["path"]
    assert job["tags"]["learning_quality"] == "unverified"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subscription_id", ""),
        ("tenant_id", "default"),
        ("owner_id", "raw-oid"),
        ("compute", "local; shell"),
        ("compute_size", "local"),
        ("managed_identity_client_id", "default"),
        ("environment_image", "approvedacr.azurecr.io/learning:latest"),
        ("environment_image", "docker.io/public/image@sha256:" + "a" * 64),
        ("environment_image", "approvedacr.azurecr.io/../image@sha256:" + "a" * 64),
        ("model_version", "latest"),
        ("retention_days", 0),
        ("output_prefix", "../outside"),
        ("output_prefix", "other-tenant/models"),
        ("managed_identity_resource_id", "/subscriptions/other/resourceGroups/rg"),
    ],
)
def test_unapproved_or_implicit_azure_configuration_is_rejected(azure_config, field, value):
    azure_config[field] = value
    with pytest.raises(ContractError):
        azure.validate_config(azure_config)


@pytest.mark.parametrize(
    "uri",
    [
        "https://public.example/data",
        "http://localhost:8080/data",
        "C:/private/observations",
        "azureml:input@latest",
    ],
)
def test_remote_or_local_unapproved_input_paths_are_forbidden(azure_config, uri):
    azure_config["inputs"]["demonstrations"]["uri"] = uri
    with pytest.raises(ContractError):
        azure.validate_config(azure_config)


def test_same_datastore_does_not_authorize_another_owner(azure_config):
    uri = azure_config["inputs"]["demonstrations"]["uri"]
    azure_config["inputs"]["demonstrations"]["uri"] = uri.replace(
        azure_config["owner_id"], "b" * 64
    )
    with pytest.raises(ContractError, match="tenant/owner"):
        azure.validate_config(azure_config)


@pytest.mark.parametrize("kind", ["code", "job", "extra-code", "plan"])
def test_altered_reviewed_plans_are_rejected(azure_config, source_snapshot, tmp_path, kind):
    root = tmp_path / "reviewed"
    azure.create_plan(azure_config, root, source_root=source_snapshot)
    if kind == "code":
        (root / "code" / "learning" / "train.py").write_text("changed")
    elif kind == "extra-code":
        (root / "code" / "secret.txt").write_text("not an approved snapshot input")
    elif kind == "job":
        job = read_json(root / "job.json")
        job["jobs"]["train"]["limits"]["timeout"] = 999999
        overwrite(root / "job.json", job)
    else:
        plan = read_json(root / "plan.json")
        plan["config"]["subscription_id"] = "9" * 36
        overwrite(root / "plan.json", plan)
    with pytest.raises(ContractError):
        azure.read_plan(root)


@pytest.mark.parametrize("missing", ["approval", "billable", "upload", "licenses"])
def test_submission_guards_fail_before_sdk_import(
    azure_config, source_snapshot, tmp_path, monkeypatch, missing
):
    root = tmp_path / "reviewed"
    approval = azure.create_plan(azure_config, root, source_root=source_snapshot)
    monkeypatch.setattr(
        azure, "version", lambda _: pytest.fail("SDK imported without authorization")
    )
    with pytest.raises(ContractError):
        azure.submit(
            root,
            approved_plan_sha256="0" * 64 if missing == "approval" else approval,
            confirm_billable=missing != "billable",
            confirm_code_upload=missing != "upload",
            confirm_licenses_reviewed=missing != "licenses",
        )


def test_one_approved_submission_uses_explicit_client_scope(
    azure_config, source_snapshot, tmp_path, monkeypatch
):
    root = tmp_path / "reviewed"
    approval = azure.create_plan(azure_config, root, source_root=source_snapshot)
    calls = []
    sdk = ModuleType("azure.ai.ml")
    identity = ModuleType("azure.identity")
    job = object()
    client = SimpleNamespace(
        jobs=SimpleNamespace(
            create_or_update=lambda submitted: (
                calls.append(submitted) or SimpleNamespace(name="aml-job")
            )
        )
    )
    sdk.load_job = lambda source: job

    def factory(**kwargs):
        assert kwargs["subscription_id"] == azure_config["subscription_id"]
        assert kwargs["resource_group_name"] == azure_config["resource_group"]
        assert kwargs["workspace_name"] == azure_config["workspace"]
        assert kwargs["credential"] == ("explicit-cli", azure_config["tenant_id"])
        return client

    sdk.MLClient = factory
    identity.AzureCliCredential = lambda tenant_id: ("explicit-cli", tenant_id)
    monkeypatch.setitem(sys.modules, "azure.ai.ml", sdk)
    monkeypatch.setitem(sys.modules, "azure.identity", identity)
    monkeypatch.setattr(azure, "version", lambda name: azure.AZURE_SDK_VERSION)
    monkeypatch.setattr(azure, "preflight", lambda actual_client, config: None)
    assert (
        azure.submit(
            root,
            approved_plan_sha256=approval,
            confirm_billable=True,
            confirm_code_upload=True,
            confirm_licenses_reviewed=True,
        )
        == "aml-job"
    )
    assert calls == [job]


def retention_policy(config):
    return {
        "policy": {
            "rules": [
                {
                    "enabled": True,
                    "definition": {
                        "filters": {
                            "prefixMatch": [
                                config["blob_container"] + "/" + config["output_prefix"] + "/"
                            ]
                        },
                        "actions": {
                            "baseBlob": {"delete": {"daysAfterModificationGreaterThan": 7}}
                        },
                    },
                }
            ]
        }
    }


@pytest.mark.parametrize("change", ["missing", "disabled", "days", "prefix"])
def test_output_retention_requires_a_real_scoped_policy(azure_config, change):
    policy = retention_policy(azure_config)
    azure.validate_retention(policy, azure_config)
    rule = policy["policy"]["rules"][0]
    if change == "missing":
        policy = {}
    elif change == "disabled":
        rule["enabled"] = False
    elif change == "days":
        rule["definition"]["actions"]["baseBlob"]["delete"]["daysAfterModificationGreaterThan"] = (
            365
        )
    else:
        rule["definition"]["filters"]["prefixMatch"] = ["learning/other-owner"]
    with pytest.raises(ContractError, match="retention"):
        azure.validate_retention(policy, azure_config)


def test_gate_pipeline_is_explicit_evidence_validation_not_fake_isaac_rollouts(azure_config):
    config = copy.deepcopy(azure_config)
    asset = config["inputs"]["demonstrations"]
    config["kind"] = "gate"
    config["parameters"] = {"timeout_seconds": 300}
    config["inputs"] = {
        name: {
            **asset,
            "name": name,
            "type": "uri_file" if name == "plan" else "uri_folder",
            "uri": asset["uri"] + "/" + name,
        }
        for name in ("model", "evidence", "plan")
    }
    job = azure.build_job(config, "a" * 64)
    assert set(job["jobs"]) == {"validate_evidence"}
    command = job["jobs"]["validate_evidence"]["command"]
    assert "learning.components gate" in command
    assert "--model-sha256" in command and "--plan-sha256" in command
    assert "--evidence-sha256" in command


def test_no_cli_flags_can_silently_submit_a_plan(
    azure_config, source_snapshot, tmp_path, monkeypatch, capsys
):
    config_file = tmp_path / "approved.json"
    overwrite(config_file, azure_config)
    monkeypatch.setattr(azure, "create_plan", lambda config, output: "a" * 64)
    monkeypatch.setattr(
        azure, "submit", lambda *args, **kwargs: pytest.fail("Unexpected submission")
    )
    assert azure.main(["--config", str(config_file), "--plan-dir", str(tmp_path / "plan")]) == 0
    assert '"submitted": false' in capsys.readouterr().out
    assert (
        azure.main(
            [
                "--config",
                str(config_file),
                "--plan-dir",
                str(tmp_path / "plan"),
                "--confirm-billable",
            ]
        )
        == 2
    )
