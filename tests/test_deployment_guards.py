import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from scripts.deploy import AzureCLI, Deployment, deploy, job_execution_status, main, preflight


def configuration(tmp_path):
    key = tmp_path / "public.pub"
    key.write_text("ssh-ed25519 SYNTHETIC_TEST_PUBLIC_KEY")
    assets = tmp_path / "assets.tar"
    assets.write_bytes(b"synthetic-test-data")
    return {
        "subscription_id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "operator_object_id": str(uuid4()),
        "spa_client_id": str(uuid4()),
        "api_client_id": str(uuid4()),
        "resource_group": "rg-test-factory",
        "prefix": "factory",
        "location": "testregion",
        "foundry_location": "testregion",
        "model_name": "test-model",
        "model_version": "test-version",
        "model_deployment_name": "inspection",
        "model_sku": "test-sku",
        "model_capacity": 1,
        "gpu_vm_size": "Standard_TestOnly",
        "ssh_public_key_file": str(key),
        "licensed_asset_archive": str(assets),
        "franka_usd_relative_path": "Franka/franka.usd",
        "accept_nvidia_eula": True,
        "licensed_assets_approved": True,
        "acknowledged_hourly_budget_usd": 10,
        "shutdown_time_utc": "2300",
        "source_commit": "a" * 40,
    }


def test_offline_plan_never_calls_azure(tmp_path, monkeypatch, capsys):
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(configuration(tmp_path)))
    monkeypatch.setattr("sys.argv", ["deploy", str(path)])
    monkeypatch.setattr(
        AzureCLI, "call", lambda *args, **kwargs: pytest.fail("No Azure calls allowed.")
    )
    main()
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "offline_plan"
    assert output["azure_resources_changed"] is False


@pytest.mark.parametrize("field", ["subscription_id", "tenant_id", "api_client_id"])
def test_zero_placeholder_ids_fail_before_deployment(tmp_path, field):
    values = configuration(tmp_path)
    values[field] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValidationError):
        Deployment.model_validate(values)


@pytest.mark.parametrize("field", ["accept_nvidia_eula", "licensed_assets_approved"])
def test_license_approval_is_required_before_writes(tmp_path, field):
    values = configuration(tmp_path)
    values[field] = False
    with pytest.raises(ValueError, match="approval"):
        preflight(Deployment.model_validate(values))


def test_web_profile_does_not_require_or_imply_nvidia_approval(tmp_path):
    values = configuration(tmp_path)
    values.update(runtime_profile="web", accept_nvidia_eula=False, licensed_assets_approved=False)
    for field in ("gpu_vm_size", "licensed_asset_archive", "ssh_public_key_file"):
        values.pop(field)
    assert preflight(Deployment.model_validate(values)) == (None, None)


def test_existing_deployment_defaults_remain_basic_but_learning_can_pin_premium(tmp_path):
    values = configuration(tmp_path)
    assert Deployment.model_validate(values).registry_sku == "Basic"
    values["registry_sku"] = "Premium"
    assert Deployment.model_validate(values).registry_sku == "Premium"
    values["registry_sku"] = "unrecognized"
    with pytest.raises(ValidationError):
        Deployment.model_validate(values)


def test_premium_registry_choice_reaches_foundation_without_cloud_writes(tmp_path, monkeypatch):
    values = configuration(tmp_path)
    values.update(runtime_profile="web", registry_sku="Premium")
    observed = {}

    class FoundationReached(Exception):
        pass

    class RecordingCLI:
        def __init__(self, subscription):
            assert subscription == Deployment.model_validate(values).subscription_id

        def call(self, *args):
            if args == ("group", "exists", "--name", values["resource_group"]):
                return True
            if args == ("group", "show", "--name", values["resource_group"]):
                return {
                    "tags": {
                        "project": "azure-physicalai-workshop",
                        "physicalaiEnvironment": values["prefix"],
                    }
                }
            pytest.fail(f"Unexpected Azure operation: {args}")

        def template(self, name, group, file, parameters):
            assert file == "foundation.bicep"
            observed.update(parameters)
            raise FoundationReached

    monkeypatch.setattr("scripts.deploy.AzureCLI", RecordingCLI)
    monkeypatch.setattr("scripts.deploy.verify_destination", lambda *args: "test-foundry-role")
    with pytest.raises(FoundationReached):
        deploy(Deployment.model_validate(values))
    assert observed["registrySku"] == "Premium"


def test_private_key_is_not_accepted_as_a_public_key(tmp_path):
    values = configuration(tmp_path)
    Path(values["ssh_public_key_file"]).write_text("-----BEGIN PRIVATE KEY-----")
    with pytest.raises(ValueError, match="public SSH"):
        preflight(Deployment.model_validate(values))


def test_cli_always_pins_the_subscription_and_preserves_streaming_build_logs(monkeypatch, capsys):
    subscription = uuid4()
    captured = []

    def run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="ACR build log, not JSON\n", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    assert AzureCLI(subscription).call("acr", "build", expect_json=False) is None
    assert captured[0][captured[0].index("--subscription") + 1] == str(subscription)
    assert "ACR build log" in capsys.readouterr().out


def test_empty_expected_json_is_an_error(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="no JSON"):
        AzureCLI(uuid4()).call("group", "show")


@pytest.mark.parametrize(
    "record",
    [
        {"properties": {"status": "Failed"}},
        {"status": "Failed"},
    ],
)
def test_job_status_supports_both_core_and_extension_cli_shapes(record):
    assert job_execution_status(record) == "Failed"


def test_job_status_does_not_turn_missing_metadata_into_success():
    with pytest.raises(ValueError):
        job_execution_status({"name": "started-but-not-verified"})
