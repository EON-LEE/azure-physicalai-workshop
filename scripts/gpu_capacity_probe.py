"""Opt-in real Spot allocation; inspect actual state, then deallocate or clean owned failures."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SIZES = (
    "Standard_NC24lds_xl_RTXPRO6000BSE_v6",
    "Standard_NC8as_T4_v3",
    "Standard_NV36ads_A10_v5",
)
PURPOSE = "gpu-capacity-probe"


class ProbeCLI:
    def __init__(self, subscription: UUID) -> None:
        self.subscription = str(subscription)

    def command(self, arguments: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "az",
                *arguments,
                "--subscription",
                self.subscription,
                "--output",
                "json",
                "--only-show-errors",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def read(self, arguments: list[str]) -> dict:
        result = self.command(arguments)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout)


def parse_pci_log(log: str) -> dict | None:
    try:
        decoded = json.loads(log)
        if isinstance(decoded, str):
            log = decoded
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        if "PHYSICALAI_GPU_PROBE " not in line:
            continue
        payload = line.split("PHYSICALAI_GPU_PROBE ", 1)[1]
        try:
            record, _ = decoder.raw_decode(payload)
        except ValueError:
            continue
        devices = record.get("nvidia_display_devices")
        if (
            record.get("kind") == "physicalai_gpu_pci_probe"
            and isinstance(devices, list)
            and record.get("nvidia_application_executed") is False
            and record.get("isaac_verified") is False
            and all(
                isinstance(device, dict)
                and device.get("vendor") == "0x10de"
                and str(device.get("class", "")).startswith("0x03")
                and re.fullmatch(r"0x[0-9a-f]{4}", str(device.get("device", "")))
                for device in devices
            )
        ):
            return record
    return None


def owned_resource(cli: ProbeCLI, resource_id: str, probe_id: str) -> dict | None:
    result = cli.command(["resource", "show", "--ids", resource_id])
    if result.returncode:
        if "ResourceNotFound" in result.stderr or "ResourceGroupNotFound" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    resource = json.loads(result.stdout)
    tags = resource.get("tags") or {}
    if tags.get("purpose") != PURPOSE or tags.get("probeId") != probe_id:
        raise ValueError("Refusing to modify a resource outside this exact probe.")
    return resource


def cleanup_owned(cli: ProbeCLI, ids: dict[str, str], probe_id: str) -> None:
    for kind in ("vm", "nic", "network", "nsg", "shutdown"):
        if owned_resource(cli, ids[kind], probe_id) is None:
            continue
        deleted = cli.command(["resource", "delete", "--ids", ids[kind]], timeout=600)
        if deleted.returncode:
            raise RuntimeError(deleted.stderr[-4000:])


def probe(
    cli: ProbeCLI,
    resource_group: str,
    name: str,
    location: str,
    size: str,
    max_price: float,
    output: Path,
) -> dict:
    if size not in ALLOWED_SIZES or not 0 < max_price <= 1:
        raise ValueError(
            "Choose a reviewed GPU and a positive maximum compute price <= USD 1/hour."
        )
    if re.fullmatch(r"fai-spot-[a-z0-9-]{1,15}", name) is None:
        raise ValueError("Probe names must be dedicated fai-spot-* names of at most 24 characters.")
    group = cli.read(["group", "show", "--name", resource_group])
    if (group.get("tags") or {}).get("project") != "azure-physicalai-workshop":
        raise ValueError("The target group is not the approved Physical AI group.")
    scope = group["id"]
    ids = {
        "vm": f"{scope}/providers/Microsoft.Compute/virtualMachines/{name}",
        "nic": f"{scope}/providers/Microsoft.Network/networkInterfaces/{name}-nic",
        "network": f"{scope}/providers/Microsoft.Network/virtualNetworks/{name}-vnet",
        "nsg": f"{scope}/providers/Microsoft.Network/networkSecurityGroups/{name}-nsg",
        "shutdown": f"{scope}/providers/Microsoft.DevTestLab/schedules/shutdown-computevm-{name}",
    }
    existing = cli.command(["vm", "show", "--resource-group", resource_group, "--name", name])
    if existing.returncode == 0:
        raise ValueError(
            "A probe VM with this name already exists; inspect it rather than replacing it."
        )
    if "ResourceNotFound" not in existing.stderr:
        raise RuntimeError(existing.stderr.strip())
    identifier = str(uuid4())
    now = datetime.now(UTC)
    report = {
        "probe_id": identifier,
        "subscription_id": cli.subscription,
        "resource_group": resource_group,
        "name": name,
        "region": location,
        "sku": size,
        "priority": "Spot",
        "max_compute_usd_per_hour": max_price,
        "started_at": now.isoformat(),
        "allocation_confirmed": False,
        "gpu_pci_confirmed": False,
        "isaac_executed": False,
        "license_accepted": False,
        "resources": ids,
        "status": "starting",
    }

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_key = (
        key.public_key()
        .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
        .decode("ascii")
    )
    del key
    values = {
        "probeName": name,
        "location": location,
        "vmSize": size,
        "sshPublicKey": public_key,
        "maxPrice": str(max_price),
        "shutdownTimeUtc": (now + timedelta(hours=1)).strftime("%H%M"),
        "probeId": identifier,
    }
    with tempfile.TemporaryDirectory(prefix="physicalai-gpu-probe-") as temporary:
        parameters = Path(temporary) / "parameters.json"
        parameters.write_text(
            json.dumps(
                {
                    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
                    "contentVersion": "1.0.0.0",
                    "parameters": {key: {"value": value} for key, value in values.items()},
                }
            )
        )
        try:
            deployment = cli.command(
                [
                    "deployment",
                    "group",
                    "create",
                    "--name",
                    name,
                    "--resource-group",
                    resource_group,
                    "--template-file",
                    str(ROOT / "infra" / "gpu-capacity-probe.bicep"),
                    "--parameters",
                    f"@{parameters}",
                ],
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            report.update(
                status="deployment_timeout", error="Azure deployment exceeded 30 minutes."
            )
            try:
                cancelled = cli.command(
                    [
                        "deployment",
                        "group",
                        "cancel",
                        "--name",
                        name,
                        "--resource-group",
                        resource_group,
                    ]
                )
                report["deployment_cancellation_requested"] = cancelled.returncode == 0
                cleanup_owned(cli, ids, identifier)
                report["failed_resources_cleaned"] = True
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                report["cleanup_error"] = str(exc)[-4000:]
            save()
            raise
    if deployment.returncode:
        report.update(status="allocation_failed", error=deployment.stderr[-12000:])
        save()
        try:
            cleanup_owned(cli, ids, identifier)
            report["failed_resources_cleaned"] = True
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            report["cleanup_error"] = str(exc)[-4000:]
        save()
        return report
    try:
        owned_resource(cli, ids["vm"], identifier)
        state = cli.read(
            ["vm", "get-instance-view", "--resource-group", resource_group, "--name", name]
        )
        statuses = state.get("instanceView", state).get("statuses", [])
        report["instance_statuses"] = statuses
        report["allocation_confirmed"] = any(
            item.get("code") == "PowerState/running" for item in statuses
        )
        report["status"] = "allocated" if report["allocation_confirmed"] else "vm_not_running"
        save()
        if report["allocation_confirmed"]:
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                result = cli.command(
                    [
                        "vm",
                        "boot-diagnostics",
                        "get-boot-log",
                        "--resource-group",
                        resource_group,
                        "--name",
                        name,
                    ],
                    timeout=60,
                )
                if result.returncode == 0:
                    record = parse_pci_log(result.stdout)
                    if record is not None:
                        report["hardware_probe"] = record
                        report["gpu_pci_confirmed"] = bool(record.get("nvidia_display_devices"))
                        save()
                        break
                else:
                    report["boot_log_error"] = result.stderr[-2000:]
                    save()
                time.sleep(15)
    finally:
        if owned_resource(cli, ids["vm"], identifier) is not None:
            stopped = cli.command(
                ["vm", "deallocate", "--resource-group", resource_group, "--name", name],
                timeout=600,
            )
            report["deallocation_requested"] = stopped.returncode == 0
            if stopped.returncode:
                report["deallocation_error"] = stopped.stderr[-4000:]
            else:
                state = cli.read(
                    ["vm", "get-instance-view", "--resource-group", resource_group, "--name", name]
                )
                statuses = state.get("instanceView", state).get("statuses", [])
                report["deallocated"] = any(
                    item.get("code") == "PowerState/deallocated" for item in statuses
                )
        report["finished_at"] = datetime.now(UTC).isoformat()
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", required=True, type=UUID)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--size", choices=ALLOWED_SIZES, required=True)
    parser.add_argument("--max-price", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "sku": args.size,
                    "region": args.region,
                    "max_compute_usd_per_hour": args.max_price,
                    "resources_changed": False,
                }
            )
        )
        return
    result = probe(
        ProbeCLI(args.subscription),
        args.resource_group,
        args.name,
        args.region,
        args.size,
        args.max_price,
        args.output,
    )
    print(json.dumps(result, indent=2))
    passed = all(
        result.get(key) for key in ("allocation_confirmed", "gpu_pci_confirmed", "deallocated")
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
