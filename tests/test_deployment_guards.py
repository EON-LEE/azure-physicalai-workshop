import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from scripts.deploy import AzureCLI, Deployment, main, preflight


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
