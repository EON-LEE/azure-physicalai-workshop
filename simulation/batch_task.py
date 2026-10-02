"""One approved Batch episode with durable anti-reexecution and manifest-last proof."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError

from learning.common import canonical, digest, file_digest, parse_json, read_json, require
from simulation.batch import (
    GPU_NAME,
    GRID_DRIVER,
    PROOF_LIMITS,
    TASK_WALL_SECONDS,
    BatchSimulationSpec,
    BlobInput,
    validate_completion,
)

RAY_TRACING_EXTENSIONS = frozenset(
    {
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_deferred_host_operations",
    }
)
MAX_LOG_BYTES = PROOF_LIMITS["probe.log"]


def run_bounded(command, *, timeout: float, cwd: Path | str, log) -> subprocess.CompletedProcess:
    """All descendants belong to this Linux process group, including the Isaac python.sh wrapper."""
    require(
        timeout > 0 and os.name == "posix", "Native Batch tasks require bounded Linux execution."
    )
    with subprocess.Popen(
        command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, process_group=0
    ) as process:
        try:
            return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
        except BaseException:
            # Cancellation/timeout must stop motion before any failure receipt can be published.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
            raise
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def vulkan_ray_tracing_devices() -> list[dict]:
    """Enumerate actual Vulkan device capabilities; do not infer RTX from CUDA."""

    class ApplicationInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_uint32),
            ("pNext", ctypes.c_void_p),
            ("applicationName", ctypes.c_char_p),
            ("applicationVersion", ctypes.c_uint32),
            ("engineName", ctypes.c_char_p),
            ("engineVersion", ctypes.c_uint32),
            ("apiVersion", ctypes.c_uint32),
        ]

    class InstanceInfo(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_uint32),
            ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32),
            ("applicationInfo", ctypes.POINTER(ApplicationInfo)),
            ("layerCount", ctypes.c_uint32),
            ("layerNames", ctypes.c_void_p),
            ("extensionCount", ctypes.c_uint32),
            ("extensionNames", ctypes.c_void_p),
        ]

    class Extension(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 256), ("version", ctypes.c_uint32)]

    loader = ctypes.CDLL("libvulkan.so.1")
    loader.vkCreateInstance.argtypes = [
        ctypes.POINTER(InstanceInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    loader.vkCreateInstance.restype = ctypes.c_int32
    loader.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    loader.vkEnumeratePhysicalDevices.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    loader.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    loader.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    loader.vkEnumerateDeviceExtensionProperties.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(Extension),
    ]
    loader.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int32
    app = ApplicationInfo(0, None, b"physicalai-batch-preflight", 1, None, 0, (1 << 22) | (2 << 12))
    info = InstanceInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    require(
        loader.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance)) == 0,
        "Vulkan instance creation failed.",
    )
    devices = []
    try:
        count = ctypes.c_uint32()
        require(
            loader.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) == 0,
            "Vulkan enumeration failed.",
        )
        require(1 <= count.value <= 8, "Unexpected Vulkan device count.")
        handles = (ctypes.c_void_p * count.value)()
        require(
            loader.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles) == 0,
            "Vulkan enumeration changed.",
        )
        for handle in handles:
            # VkPhysicalDeviceProperties begins with five uint32s and a 256-byte name.
            properties = ctypes.create_string_buffer(4096)
            loader.vkGetPhysicalDeviceProperties(handle, properties)
            vendor = ctypes.c_uint32.from_buffer(properties, 8).value
            name = properties.raw[20:276].split(b"\0", 1)[0].decode("utf-8", errors="strict")
            extension_count = ctypes.c_uint32()
            require(
                loader.vkEnumerateDeviceExtensionProperties(
                    handle, None, ctypes.byref(extension_count), None
                )
                == 0
                and extension_count.value <= 2048,
                "Vulkan extension enumeration failed or exceeds its bound.",
            )
            extensions = (Extension * extension_count.value)()
            require(
                loader.vkEnumerateDeviceExtensionProperties(
                    handle, None, ctypes.byref(extension_count), extensions
                )
                == 0,
                "Vulkan extension enumeration changed.",
            )
            supported = {extension.name.decode("ascii") for extension in extensions}
            if vendor == 0x10DE:
                devices.append(
                    {
                        "name": name,
                        "vendor_id": vendor,
                        "ray_tracing_extensions": sorted(RAY_TRACING_EXTENSIONS & supported),
                    }
                )
    finally:
        loader.vkDestroyInstance(instance, None)
    return devices


def validate_gpu_evidence(evidence: dict) -> None:
    require(
        evidence.get("schema") == "physicalai.batch-gpu-preflight/v1"
        and evidence.get("gpu_count") == 1
        and evidence.get("gpu_name") == GPU_NAME
        and evidence.get("driver_version") == GRID_DRIVER
        and evidence.get("simulation_app_started") is False,
        "The actual single A10/GRID renderer differs from the approved platform.",
    )
    devices = evidence.get("vulkan_devices", [])
    require(
        len(devices) == 1
        and devices[0].get("vendor_id") == 0x10DE
        and "A10" in devices[0].get("name", "")
        and RAY_TRACING_EXTENSIONS <= set(devices[0].get("ray_tracing_extensions", [])),
        "Actual Vulkan ray-tracing capabilities are unavailable; a CUDA GPU is not sufficient.",
    )


def gpu_preflight() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 2, "Exactly one actual GPU is required.")
    evidence = {
        "schema": "physicalai.batch-gpu-preflight/v1",
        "gpu_count": len(rows),
        "gpu_name": rows[0][0].strip(),
        "driver_version": rows[0][1].strip(),
        "vulkan_devices": vulkan_ray_tracing_devices(),
        "simulation_app_started": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }
    validate_gpu_evidence(evidence)
    return evidence


class PrivateArtifacts:
    def __init__(self, spec: BatchSimulationSpec, credential) -> None:
        from azure.storage.blob import BlobServiceClient

        self.spec = spec
        self.client = BlobServiceClient(
            spec.storage_account_url,
            credential=credential,
            connection_timeout=5,
            read_timeout=15,
            retry_total=1,
        )

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *args):
        return self.client.__exit__(*args)

    def _output(self, name):
        return self.client.get_blob_client(
            self.spec.output_container, self.spec.artifact_prefix + "/" + name
        )

    def claim(self, value: dict) -> bool:
        try:
            self._output("claim.json").upload_blob(canonical(value), overwrite=False)
        except ResourceExistsError:
            return False
        return True

    def download(self, reference: BlobInput, path: Path) -> None:
        blob = self.client.get_blob_client(self.spec.input_container, reference.name)
        require(
            blob.get_blob_properties().size == reference.size_bytes, "Approved input size changed."
        )
        received = 0
        with path.open("xb") as stream:
            for chunk in blob.download_blob().chunks():
                received += len(chunk)
                require(received <= reference.size_bytes, "Input exceeds its approved size.")
                stream.write(chunk)
        require(
            received == reference.size_bytes and file_digest(path) == reference.sha256,
            "Approved input checksum changed.",
        )

    def verify_asset_reference(self) -> None:
        blob = self.client.get_blob_client(self.spec.input_container, self.spec.asset_bundle.name)
        require(
            blob.get_blob_properties().size == self.spec.asset_bundle.size_bytes,
            "The approved asset archive size differs.",
        )

    def _read_bounded(self, blob, limit: int) -> bytes:
        require(blob.get_blob_properties().size <= limit, "Private proof exceeds its size limit.")
        value = bytearray()
        for chunk in blob.download_blob().chunks():
            value.extend(chunk)
            require(len(value) <= limit, "Private proof grew beyond its size limit.")
        return bytes(value)

    def verify_raw_manifest(self, receipt: dict, local_manifest: Path) -> None:
        episode_id = str(UUID(receipt["episode_id"]))
        name = f"{self.spec.owner_id}/{episode_id}/manifest.json"
        expected = f"{self.spec.storage_account_url}/{self.spec.output_container}/{name}"
        require(
            receipt.get("manifest_uri") == expected, "Raw manifest escaped the private owner scope."
        )
        blob = self.client.get_blob_client(self.spec.output_container, name)
        payload = self._read_bounded(blob, PROOF_LIMITS["raw-manifest.json"])
        require(
            digest(payload) == receipt["manifest_sha256"] == file_digest(local_manifest),
            "The private uploaded raw manifest differs from the validated local bytes.",
        )

    def publish(self, name: str, path: Path) -> dict:
        from azure.storage.blob import ContentSettings

        size = path.stat().st_size
        require(
            name in PROOF_LIMITS and size <= PROOF_LIMITS[name],
            "Task proof/log exceeds the bounded publication limit.",
        )
        with path.open("rb") as stream:
            self._output(name).upload_blob(
                stream,
                overwrite=False,
                content_settings=ContentSettings(
                    content_type="application/json" if name.endswith(".json") else "text/plain"
                ),
            )
        return {"path": name, "bytes": size, "sha256": file_digest(path)}

    def finish(self, value: dict) -> None:
        from azure.storage.blob import ContentSettings

        self._output("completion.json").upload_blob(
            canonical(value),
            overwrite=False,
            content_settings=ContentSettings(content_type="application/json"),
        )

    def validate_evidence(self, documents: dict[str, bytes]) -> dict:
        return validate_success_evidence(self.spec, documents)

    def read_completion(self) -> dict | None:
        try:
            proof = parse_json(self._read_bounded(self._output("completion.json"), 1024**2))
        except ResourceNotFoundError:
            return None
        artifacts = validate_completion(self.spec, proof)
        documents = {}
        for artifact in artifacts:
            blob = self._output(artifact.path)
            require(
                blob.get_blob_properties().size == artifact.size_bytes,
                "Published artifact size changed.",
            )
            checksum = hashlib.sha256()
            received = 0
            payload = bytearray()
            for chunk in blob.download_blob().chunks():
                received += len(chunk)
                require(received <= artifact.size_bytes, "Published artifact grew.")
                checksum.update(chunk)
                if artifact.path.endswith(".json"):
                    payload.extend(chunk)
            require(
                received == artifact.size_bytes and checksum.hexdigest() == artifact.sha256,
                "Published artifact checksum changed.",
            )
            if artifact.path.endswith(".json"):
                documents[artifact.path] = bytes(payload)
        require(
            digest(documents["inputs/spec.json"]) == proof["spec_file_sha256"]
            and type(self.spec).model_validate(parse_json(documents["inputs/spec.json"])).sha256
            == self.spec.sha256,
            "Stored task specification differs from the original attempt.",
        )
        if proof["accepted"]:
            native_receipt = self.validate_evidence(documents)
            receipt = proof["raw_manifest"]
            require(
                receipt == native_receipt, "Terminal receipt references a different native episode."
            )
            episode_id = str(UUID(receipt["episode_id"]))
            expected = (
                f"{self.spec.storage_account_url}/{self.spec.output_container}/"
                f"{self.spec.owner_id}/{episode_id}/manifest.json"
            )
            require(receipt["manifest_uri"] == expected, "Raw manifest escaped its private scope.")
            blob = self.client.get_blob_client(
                self.spec.output_container, f"{self.spec.owner_id}/{episode_id}/manifest.json"
            )
            require(
                digest(self._read_bounded(blob, PROOF_LIMITS["raw-manifest.json"]))
                == digest(documents["raw-manifest.json"])
                == receipt["manifest_sha256"],
                "Remote raw manifest changed after terminal publication.",
            )
        return proof


def run_episode(
    spec: BatchSimulationSpec,
    spec_path: Path,
    store,
    *,
    directory: Path,
    runner=None,
    evidence_validator=None,
) -> dict:
    started = time.monotonic()
    result = {
        "schema": "physicalai.batch-simulation-result/v1",
        "attempt_id": str(spec.attempt_id),
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "previous_attempt_id": str(spec.previous_attempt_id) if spec.previous_attempt_id else None,
        "spec_sha256": spec.sha256,
        "spec_file_sha256": file_digest(spec_path),
        "source_revision": spec.source_revision,
        "image": spec.platform.container_image,
        "control_profile_sha256": spec.control_profile_sha256,
        "profile_id": spec.profile_id,
        "outcome": "incomplete",
        "accepted": False,
        "native_acceptance": "missing",
        "learning_quality_proven": False,
    }
    claim = {
        **result,
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "batch_node_id": os.environ.get("AZ_BATCH_NODE_ID"),
        "physical_state_resume": False,
    }
    if not store.claim(claim):
        return {
            **result,
            "reason": "Attempt already claimed; Batch requeue cannot start more physics.",
        }
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    inputs = directory / "inputs"
    inputs.mkdir(mode=0o700)
    paths = {}
    try:
        for name in ("environment", "grant", "criteria", "conditions"):
            paths[name] = inputs / (name + ".json")
            store.download(getattr(spec, name), paths[name])
        store.verify_asset_reference()
        verdict = (run_native if runner is None else runner)(
            spec, paths, directory, started + TASK_WALL_SECONDS - 30
        )
        require(
            type(verdict.get("accepted")) is bool
            and verdict.get("outcome") in {"accepted", "failed", "incomplete"},
            "The native task adapter returned no explicit outcome.",
        )
        result.update(verdict)
        if result["accepted"]:
            store.verify_raw_manifest(result["raw_manifest"], directory / "raw-manifest.json")
            native_receipt = (evidence_validator or validate_success_evidence)(
                spec,
                {
                    **{f"inputs/{name}.json": path.read_bytes() for name, path in paths.items()},
                    **{
                        name: (directory / name).read_bytes()
                        for name in (
                            "preflight.json",
                            "probe.json",
                            "acceptance.json",
                            "raw-manifest.json",
                        )
                    },
                },
            )
            require(
                result["raw_manifest"] == native_receipt,
                "Terminal receipt references a different native episode.",
            )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, AzureError) as exc:
        result.update(
            outcome="incomplete",
            accepted=False,
            native_acceptance="missing",
            failure_type=type(exc).__name__,
            failure=str(exc)[:2048],
        )
    artifacts = [store.publish("inputs/spec.json", spec_path)]
    for name, path in paths.items():
        if path.exists():
            artifacts.append(store.publish("inputs/" + name + ".json", path))
    for name in (
        "preflight.json",
        "probe.json",
        "probe.log",
        "acceptance.json",
        "acceptance.log",
        "raw-manifest.json",
    ):
        path = directory / name
        if path.is_file():
            artifacts.append(store.publish(name, path))
    result.update(artifacts=artifacts, elapsed_wall_seconds=time.monotonic() - started)
    require(
        time.monotonic() < started + TASK_WALL_SECONDS,
        "The original task publication budget expired.",
    )
    validate_completion(spec, result)
    # Publish last: node loss or a failed artifact upload cannot leave terminal success proof.
    store.finish(result)
    return result


def validate_success_evidence(spec: BatchSimulationSpec, documents: dict[str, bytes]) -> dict:
    from apps.api.models import EnvironmentRecord
    from learning.paused.contract import RAW_SCHEMA
    from simulation.paused_acceptance import validate_attempt

    for name in ("environment", "grant", "criteria", "conditions"):
        require(
            digest(documents[f"inputs/{name}.json"]) == getattr(spec, name).sha256,
            "Terminal evidence changed an approved input.",
        )
    environment = EnvironmentRecord.model_validate(parse_json(documents["inputs/environment.json"]))
    report = parse_json(documents["probe.json"])
    native = validate_attempt(report, environment=environment, mode="reference-task")
    acceptance = parse_json(documents["acceptance.json"])
    manifest = parse_json(documents["raw-manifest.json"])
    receipt = report["capture"]["receipt"]
    validate_gpu_evidence(parse_json(documents["preflight.json"]))
    require(
        acceptance == {**native, "manifest_sha256": digest(documents["raw-manifest.json"])}
        and receipt["manifest_sha256"] == acceptance["manifest_sha256"]
        and receipt["episode_id"] == report["command_id"]
        and report["source_revision"] == spec.source_revision
        and report["simulator_image_digest"] == spec.platform.container_image.split("@", 1)[1]
        and report["control_profile_sha256"] == spec.control_profile_sha256
        and report["criteria_sha256"] == spec.criteria_canonical_sha256
        and report["frozen_plan_sha256"] == spec.conditions_canonical_sha256,
        "Native acceptance, physical report, source or raw manifest proof differs.",
    )
    episodes = manifest.get("episodes")
    require(
        isinstance(episodes, list)
        and len(episodes) == 1
        and manifest.get("schema") == RAW_SCHEMA
        and manifest.get("scope")
        == {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id}
        and manifest.get("execution_timing") == "paused_simulation"
        and manifest.get("real_time_admission") is False
        and digest(canonical(manifest["control_profile"])) == spec.control_profile_sha256
        and manifest.get("criteria_sha256") == spec.criteria_canonical_sha256
        and manifest.get("frozen_plan_sha256") == spec.conditions_canonical_sha256,
        "Raw manifest has a different owner, profile, task or episode count.",
    )
    episode = episodes[0]
    require(
        episode["episode_id"] == report["command_id"]
        and episode["environment_id"] == environment.environment_id
        and episode["revision"] == environment.revision
        and episode["frame_count"] == native["actual_intervals"]
        and episode["provenance"]["code_revision"] == spec.source_revision
        and episode["provenance"]["robot_asset_sha256"] == spec.asset_bundle.sha256
        and episode["provenance"]["simulator_image_digest"]
        == spec.platform.container_image.split("@", 1)[1]
        and episode["demonstration"]["kind"] == "reference_controller",
        "Raw episode provenance differs from the accepted native report.",
    )
    return receipt


def configure_native_environment(spec: BatchSimulationSpec) -> None:
    os.environ.update(
        {
            "SOURCE_REVISION": spec.source_revision,
            "SIMULATOR_IMAGE": spec.platform.container_image,
            "ISAAC_SIM_VERSION": "6.0.0",
            "ENTRA_TENANT_ID": str(spec.platform.tenant_id),
            "AZURE_CLIENT_ID": str(spec.platform.node_identity_client_id),
            "STORAGE_ACCOUNT_URL": spec.storage_account_url,
            "STORAGE_CONTAINER": spec.input_container,
            "DEMONSTRATION_CONTAINER": spec.output_container,
            "CAPTURE_ENABLED": "true",
            "USE_BAKED_REFERENCE_ASSETS": "false",
            "FRANKA_ASSET_SHA256": spec.asset_bundle.sha256,
            "FRANKA_ASSET_BLOB": spec.asset_bundle.name,
            "FRANKA_USD_RELATIVE_PATH": spec.asset_root,
            "ACCEPT_EULA": "Y",
        }
    )


def run_native(spec, paths, directory, deadline) -> dict:
    from apps.api.models import EnvironmentRecord
    from simulation.paused_configuration import OperatorPausedAuthority, paused_servo_sha256
    from simulation.paused_profiles import paused_profile

    def remaining():
        value = deadline - time.monotonic()
        require(value > 0, "The original Batch task deadline expired.")
        return value

    configure_native_environment(spec)
    profile = paused_profile(paused_servo_sha256(), spec.profile_id)
    require(
        profile.sha256 == spec.control_profile_sha256,
        "The installed runtime source/profile differs.",
    )
    environment = EnvironmentRecord.model_validate(read_json(paths["environment"]))
    authority = OperatorPausedAuthority.load(
        paths["grant"],
        spec.grant.sha256,
        environment=environment,
        profile=profile,
        criteria=paths["criteria"],
        criteria_sha256=spec.criteria_canonical_sha256,
        conditions=paths["conditions"],
        conditions_sha256=spec.conditions_canonical_sha256,
    )
    require(authority.grant.authorization.owner == spec.owner_id, "Operator grant owner differs.")
    remaining()
    (directory / "preflight.json").write_bytes(canonical(gpu_preflight()) + b"\n")
    command = [
        "/isaac-sim/python.sh",
        "-m",
        "simulation.paused_probe",
        "--environment-record",
        str(paths["environment"]),
        "--operator-grant",
        str(paths["grant"]),
        "--grant-sha256",
        spec.grant.sha256,
        "--criteria",
        str(paths["criteria"]),
        "--criteria-sha256",
        spec.criteria_canonical_sha256,
        "--conditions",
        str(paths["conditions"]),
        "--conditions-sha256",
        spec.conditions_canonical_sha256,
        "--mode",
        "reference-task",
        "--output",
        str(directory / "probe.json"),
        "--confirm-isolated-simulator",
    ]
    with (directory / "probe.log").open("xb") as log:
        process = run_bounded(command, cwd="/app", log=log, timeout=remaining())
    report_path = directory / "probe.json"
    if not report_path.is_file():
        return {
            "accepted": False,
            "outcome": "incomplete",
            "native_acceptance": "missing",
            "probe_exit_code": process.returncode,
        }
    report = read_json(
        report_path, max_bytes=(8 if spec.profile_id.endswith("-v2") else 4) * 1024**2
    )
    capture = report.get("capture") or {}
    receipt = capture.get("receipt") or {}
    verdict = {
        "accepted": False,
        "outcome": "failed",
        "native_acceptance": "rejected",
        "probe_exit_code": process.returncode,
        "physical_status": report.get("physical_status"),
        "capture_status": capture.get("status"),
        "raw_manifest": receipt if receipt else None,
    }
    if capture.get("status") != "ready" or receipt.get("status") != "uploaded":
        (directory / "acceptance.json").write_bytes(canonical(verdict) + b"\n")
        return verdict
    episode_id = str(UUID(receipt["episode_id"]))
    require(report.get("command_id") == episode_id, "Capture and physical command differ.")
    dataset_root = Path("/data/demonstrations") / spec.owner_id / episode_id
    manifest = dataset_root / "manifest.json"
    require(file_digest(manifest) == receipt["manifest_sha256"], "Raw manifest checksum differs.")
    expected_uri = (
        f"{spec.storage_account_url}/{spec.output_container}/"
        f"{spec.owner_id}/{episode_id}/manifest.json"
    )
    require(receipt.get("manifest_uri") == expected_uri, "Capture used another output scope.")
    (directory / "raw-manifest.json").write_bytes(manifest.read_bytes())
    with (directory / "acceptance.log").open("xb") as log:
        validation = run_bounded(
            [
                "/isaac-sim/python.sh",
                "-m",
                "simulation.paused_acceptance",
                "--report",
                str(report_path),
                "--environment-record",
                str(paths["environment"]),
                "--dataset-root",
                str(dataset_root),
                "--tenant-id",
                str(spec.platform.tenant_id),
                "--owner",
                spec.owner_id,
                "--source-revision",
                spec.source_revision,
                "--image-digest",
                spec.platform.container_image.split("@", 1)[1],
                "--profile-sha256",
                spec.control_profile_sha256,
                "--criteria-sha256",
                spec.criteria_canonical_sha256,
                "--conditions-sha256",
                spec.conditions_canonical_sha256,
                "--mode",
                "reference-task",
            ],
            cwd="/app",
            log=log,
            timeout=min(120, remaining()),
        )
    require(
        (directory / "acceptance.log").stat().st_size <= PROOF_LIMITS["acceptance.log"],
        "Native acceptance output exceeds its bound.",
    )
    lines = [
        line.removeprefix("PHYSICALAI_PAUSED_ACCEPTANCE ")
        for line in (directory / "acceptance.log").read_text().splitlines()
        if line.startswith("PHYSICALAI_PAUSED_ACCEPTANCE ")
    ]
    require(len(lines) == 1, "Native acceptance emitted no unambiguous proof.")
    accepted = parse_json(lines[0])
    (directory / "acceptance.json").write_bytes(canonical(accepted) + b"\n")
    passed = (
        process.returncode == validation.returncode == 0
        and accepted.get("accepted") is True
        and report.get("physical_status") == "succeeded"
        and report.get("physical_task_success") is True
    )
    verdict.update(
        accepted=passed,
        outcome="accepted" if passed else "failed",
        native_acceptance="accepted" if passed else "rejected",
    )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, default=Path("preflight.json"))
    run = subcommands.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    if args.operation == "preflight":
        evidence = gpu_preflight()
        with args.output.open("xb") as stream:
            stream.write(canonical(evidence) + b"\n")
        print("Batch GPU preflight passed; no simulator episode was started.")
        return
    require(file_digest(args.spec) == args.spec_sha256, "Batch specification checksum mismatch.")
    spec = BatchSimulationSpec.model_validate(read_json(args.spec, max_bytes=1024**2))
    require(
        os.environ.get("AZ_BATCH_JOB_ID") == spec.job_id
        and os.environ.get("AZ_BATCH_TASK_ID") == spec.task_id
        and os.environ.get("AZ_BATCH_POOL_ID") == spec.platform.pool_id,
        "The actual Batch task identity differs from the approved attempt.",
    )
    directory = Path("/data/managed-simulation") / str(spec.attempt_id)
    from azure.identity import ManagedIdentityCredential

    with ManagedIdentityCredential(
        client_id=str(spec.platform.node_identity_client_id)
    ) as credential:
        with PrivateArtifacts(spec, credential) as store:
            result = run_episode(spec, args.spec, store, directory=directory)
    print(json.dumps({key: result[key] for key in ("attempt_id", "outcome", "accepted")}))
    if result["accepted"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
