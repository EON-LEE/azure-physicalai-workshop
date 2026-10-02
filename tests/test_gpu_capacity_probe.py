import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from scripts.gpu_capacity_probe import PURPOSE, cleanup_owned, owned_resource, parse_pci_log, probe


def pci_record():
    return {
        "kind": "physicalai_gpu_pci_probe",
        "nvidia_display_devices": [
            {"pci": "0000:01:00.0", "vendor": "0x10de", "device": "0x2bb5", "class": "0x030200"}
        ],
        "nvidia_application_executed": False,
        "isaac_verified": False,
    }


@pytest.mark.parametrize("encoded", [False, True])
def test_pci_marker_parses_plain_and_json_wrapped_azure_logs(encoded):
    text = "boot output\nPHYSICALAI_GPU_PROBE " + json.dumps(pci_record()) + "\n"
    assert parse_pci_log(json.dumps(text) if encoded else text) == pci_record()


def test_missing_or_invalid_gpu_evidence_is_not_a_success():
    assert parse_pci_log("normal boot, no marker") is None
    assert parse_pci_log('PHYSICALAI_GPU_PROBE {"kind":"physicalai_gpu_pci_probe"}') is None
    invalid = pci_record()
    invalid["nvidia_display_devices"][0]["vendor"] = "0x1234"
    assert parse_pci_log("PHYSICALAI_GPU_PROBE " + json.dumps(invalid)) is None


def test_owned_resource_rejects_a_different_probe_before_modification():
    cli = SimpleNamespace(
        command=lambda args: SimpleNamespace(
            returncode=0, stdout=json.dumps({"tags": {"purpose": PURPOSE, "probeId": "other"}})
        )
    )
    with pytest.raises(ValueError, match="exact probe"):
        owned_resource(cli, "/owned-id", "this-probe")


def test_cleanup_refuses_resources_with_different_ownership_tags():
    calls = []

    class CLI:
        def command(self, args, **kwargs):
            calls.append(args)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"tags": {"purpose": PURPOSE, "probeId": "another-probe"}}),
            )

    ids = {kind: f"/exact/{kind}" for kind in ("vm", "nic", "network", "nsg", "shutdown")}
    with pytest.raises(ValueError):
        cleanup_owned(CLI(), ids, "owned-probe")
    assert calls == [["resource", "show", "--ids", "/exact/vm"]]


@pytest.mark.parametrize("price", [0, -1, 1.01, float("nan"), float("inf")])
def test_price_limit_is_checked_before_any_cloud_call(tmp_path, price):
    with pytest.raises(ValueError, match="price"):
        probe(
            None,
            "rg",
            "fai-spot-test",
            "westus2",
            "Standard_NC24lds_xl_RTXPRO6000BSE_v6",
            price,
            tmp_path / "result.json",
        )


def test_unreviewed_gpu_cannot_be_substituted(tmp_path):
    with pytest.raises(ValueError):
        probe(
            None,
            "rg",
            "fai-spot-test",
            "westus2",
            "Standard_NC24ads_A100_v4",
            0.3,
            tmp_path / "result.json",
        )


def test_actual_running_and_pci_are_recorded_then_vm_is_deallocated(tmp_path):
    output = tmp_path / "report.json"
    operations = []

    class CLI:
        subscription = str(UUID(int=1))

        def read(self, args):
            operations.append(args)
            if args[:2] == ["group", "show"]:
                return {
                    "id": "/subscriptions/test/resourceGroups/rg",
                    "tags": {"project": "azure-physicalai-workshop"},
                }
            stopped = any(item[:2] == ["vm", "deallocate"] for item in operations)
            return {
                "instanceView": {
                    "statuses": [
                        {"code": "PowerState/deallocated" if stopped else "PowerState/running"}
                    ]
                }
            }

        def command(self, args, **kwargs):
            operations.append(args)
            if args[:2] == ["vm", "show"]:
                return SimpleNamespace(returncode=1, stderr="ResourceNotFound", stdout="")
            if args[:2] == ["resource", "show"]:
                identifier = json.loads(output.read_text())["probe_id"]
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"tags": {"purpose": PURPOSE, "probeId": identifier}}),
                )
            if args[:3] == ["vm", "boot-diagnostics", "get-boot-log"]:
                return SimpleNamespace(
                    returncode=0, stdout="PHYSICALAI_GPU_PROBE " + json.dumps(pci_record())
                )
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    result = probe(
        CLI(), "rg", "fai-spot-test", "westus2", "Standard_NC24lds_xl_RTXPRO6000BSE_v6", 0.3, output
    )
    assert result["allocation_confirmed"] is True
    assert result["gpu_pci_confirmed"] is True
    assert result["deallocated"] is True
    assert result["isaac_executed"] is False
    assert result["license_accepted"] is False
