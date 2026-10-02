"""One-shot vendor CUDA compatibility check. Fixtures are NOT Franka observations or admission."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import os
import re
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    keys,
    parse_json,
    read_json,
    require,
    safe_path,
    sha256,
    token,
)
from learning.contract import Scope
from learning.offline import enforce_offline
from learning.smolvla import UPSTREAM
from learning.smolvla.cloud_assets import validate_prefixes, validate_vendor_inventory
from learning.smolvla.prepare import _source_license, verify_vendor_files

SCHEMA = "physicalai.smolvla-vendor-diagnostic/v1"
CAMERA_KEYS = tuple(f"observation.images.camera{index}" for index in (1, 2, 3))


def validate_vendor_configuration(value: dict) -> None:
    require(
        value.get("type") == "smolvla"
        and value.get("n_obs_steps") == 1
        and value.get("chunk_size") == value.get("n_action_steps") == 50
        and value.get("max_state_dim") == value.get("max_action_dim") == 32
        and value.get("empty_cameras") == 0
        and value.get("adapt_to_pi_aloha") is False
        and value.get("use_delta_joint_actions_aloha") is False,
        "Diagnostic requires the actual published SmolVLA architecture, not a robot adaptation",
    )
    expected = {
        "observation.state": {"type": "STATE", "shape": [6]},
        **{name: {"type": "VISUAL", "shape": [3, 256, 256]} for name in CAMERA_KEYS},
    }
    require(
        value.get("input_features") == expected
        and value.get("output_features") == {"action": {"type": "ACTION", "shape": [6]}},
        "Vendor diagnostic must retain six published dimensions and three camera slots",
    )


def validate_specification(spec: dict) -> None:
    keys(
        spec,
        {
            "schema",
            "scope",
            "workspace_id",
            "job_name",
            "compute_name",
            "managed_identity_client_id",
            "storage_account_url",
            "container",
            "vendor_prefix",
            "vendor_inventory_sha256",
            "output_prefix",
            "image",
            "image_code_snapshot_sha256",
            "diagnostic_code_sha256",
            "execution_timeout_seconds",
        },
        "one-shot vendor diagnostic",
    )
    require(spec["schema"] == SCHEMA, "Unknown vendor diagnostic schema")
    scope = Scope(**keys(spec["scope"], {"tenant_id", "owner_id"}, "diagnostic scope"))
    validate_prefixes(spec["vendor_prefix"], spec["output_prefix"], scope)
    prefix = f"tenants/{scope.tenant_id}/owners/{scope.owner_id}/learning/"
    require(
        spec["vendor_prefix"].startswith(prefix + "vendor/")
        and spec["output_prefix"].startswith(prefix + "outputs/"),
        "Diagnostic inputs/output must use the approved vendor/retained-output paths",
    )
    from learning.azure import _uuid

    _uuid(spec["managed_identity_client_id"], "explicit compute MI")
    require(
        re.fullmatch(
            r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
            r"Microsoft.MachineLearningServices/workspaces/[A-Za-z0-9_.-]+",
            spec["workspace_id"],
        ),
        "Explicit existing AML workspace identity is required",
    )
    token(spec["job_name"], "one-shot job name")
    token(spec["compute_name"], "existing compute name")
    require(
        re.fullmatch(
            r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", spec["storage_account_url"]
        ),
        "Explicit private Azure Blob endpoint required",
    )
    require(spec["container"] == "artifacts", "Use the approved existing artifact container")
    require(
        re.fullmatch(r"[a-z0-9]+\.azurecr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}", spec["image"]),
        "Diagnostic image must be immutable",
    )
    for name in ("vendor_inventory_sha256", "image_code_snapshot_sha256", "diagnostic_code_sha256"):
        sha256(spec[name], name)
    integer(spec["execution_timeout_seconds"], "one-shot execution limit", 60, 600)


def initial_report(spec: dict) -> dict:
    validate_specification(spec)
    return {
        "schema": "physicalai.smolvla-vendor-compatibility-proof/v1",
        "purpose": "vendor_weight_and_native_runtime_compatibility_only",
        "source": "actual_azure_ml",
        "scope": spec["scope"],
        "azure_job_id": spec["workspace_id"] + "/jobs/" + spec["job_name"],
        "image": spec["image"],
        "image_code_snapshot_sha256": spec["image_code_snapshot_sha256"],
        "diagnostic_code_sha256": spec["diagnostic_code_sha256"],
        "upstream": UPSTREAM,
        "vendor_inventory_sha256": spec["vendor_inventory_sha256"],
        "observation_source": "fixture",
        "test_only": True,
        "declared_state_dimensions": 6,
        "declared_action_dimensions": 6,
        "declared_camera_count": 3,
        "optimizer_steps": 0,
        "actuator_calls": 0,
        "franka_adaptation_verified": False,
        "latency_admission_verified": False,
        "learning_quality_verified": False,
        "ready_for_live_execution": False,
        "vendor_weights_loaded": False,
        "actual_forward_calls": 0,
        "passed": False,
        "phase": "starting",
        "limitations": [
            "Six-dimensional state and three camera fixtures, not simulator observations.",
            "Published vendor normalization, not nine-DOF Franka dataset normalization.",
            "No optimizer, actuator, accepted profile, physical rollout or task-quality proof.",
            "Diagnostic latency excludes renderer, IPC and the actual complete control cycle.",
        ],
    }


def _verify_code(spec: dict) -> None:
    require(
        file_digest(Path(__file__)) == spec["diagnostic_code_sha256"],
        "Diagnostic code checksum mismatch",
    )
    snapshot = read_json(Path("/work/code-manifest.json"))
    require(
        snapshot.get("sha256") == spec["image_code_snapshot_sha256"]
        and digest(canonical(snapshot.get("files"))) == spec["image_code_snapshot_sha256"],
        "Dependency image contains different reviewed learning code",
    )
    for name, checksum in snapshot["files"].items():
        require(
            file_digest(safe_path(Path("/work"), name)) == checksum, "Reviewed image code changed"
        )


def _download_private_assets(container, spec: dict, root: Path, within_budget) -> dict:
    from azure.core import MatchConditions

    source = container.get_blob_client(spec["vendor_prefix"] + "/vendor-inventory.json")
    properties = source.get_blob_properties()
    require(properties.size <= 256 * 1024, "Oversized vendor inventory")
    raw = source.download_blob(
        etag=properties.etag, match_condition=MatchConditions.IfNotModified
    ).readall()
    require(
        digest(raw) == spec["vendor_inventory_sha256"], "Reviewed private vendor inventory changed"
    )
    entries = validate_vendor_inventory(
        parse_json(raw), prefix=spec["vendor_prefix"], scope=Scope(**spec["scope"])
    )
    for (kind, name), entry in entries.items():
        within_budget()
        blob = container.get_blob_client(entry["blob"])
        metadata = blob.get_blob_properties()
        require(
            metadata.size == entry["bytes"], "Private vendor size differs from immutable inventory"
        )
        path = safe_path(root, f"{kind}/{name}", must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        checksum, size = hashlib.sha256(), 0
        with path.open("xb") as output:
            for chunk in blob.download_blob(
                etag=metadata.etag,
                match_condition=MatchConditions.IfNotModified,
                max_concurrency=2,
            ).chunks():
                within_budget()
                size += len(chunk)
                require(size <= entry["bytes"], "Vendor file exceeded the approved byte limit")
                checksum.update(chunk)
                output.write(chunk)
        require(
            size == entry["bytes"] and checksum.hexdigest() == entry["sha256"],
            "Actual private vendor file checksum mismatch",
        )
    return {kind: verify_vendor_files(root / kind, kind=kind) for kind in ("model", "backbone")}


def fixture_observation():
    import torch

    # Deliberately unbatched test tensors. The vendor processor adds the native batch dimension.
    state = torch.linspace(-0.2, 0.2, 6, dtype=torch.float32)
    pixels = (
        torch.arange(3 * 256 * 256, dtype=torch.float32).reshape(3, 256, 256).remainder(256) / 255
    )
    return {
        "observation.state": state,
        **{
            name: torch.roll(pixels, shifts=index, dims=2) for index, name in enumerate(CAMERA_KEYS)
        },
        "task": "Vendor compatibility fixture only. No robot action or physical observation.",
    }


def _measure(root: Path, report: dict, within_budget) -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    importlib.import_module("lerobot.policies.smolvla.processor_smolvla")
    require(
        version("lerobot") == "0.4.4" and version("transformers") == "4.57.3",
        "Diagnostic runtime version differs from the reviewed dependency image",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one actual CUDA GPU"
    )
    _source_license()
    device = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": device.name,
        "total_memory_bytes": device.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nvidia_smi": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip(),
    }
    original = read_json(root / "model" / "config.json")
    validate_vendor_configuration(original)
    config = PreTrainedConfig.from_pretrained(root / "model", local_files_only=True)
    require(isinstance(config, SmolVLAConfig), "Loaded configuration is not native SmolVLA")
    config.vlm_model_name = str((root / "backbone").resolve())
    config.device, config.push_to_hub = "cuda", False
    report["config_overrides"] = {
        "device": "cuda",
        "push_to_hub": False,
        "vlm_model_name": "hash-verified local backbone directory",
    }
    within_budget()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    report["phase"] = "loading_actual_vendor_weights"
    start = time.monotonic()
    policy = SmolVLAPolicy.from_pretrained(
        root / "model",
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    policy.requires_grad_(False)
    torch.cuda.synchronize()
    require(next(policy.parameters()).is_cuda, "Actual policy weights are not resident on CUDA")
    report["vendor_weights_loaded"] = True
    report["load_seconds"] = time.monotonic() - start
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["parameters"] = sum(parameter.numel() for parameter in policy.parameters())
    within_budget()
    report["phase"] = "loading_published_six_dimensional_processors"
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(root / "model"),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str((root / "backbone").resolve())},
        },
    )
    observation = fixture_observation()
    report["fixture_input_shapes"] = {
        name: list(value.shape) for name, value in observation.items() if name != "task"
    }
    report["phase"] = "actual_native_vendor_forward"
    results = []
    report["forward_calls"] = results
    for index in range(3):
        within_budget()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        with torch.inference_mode():
            batch = preprocessor(copy.deepcopy(observation))
            require(
                tuple(batch["observation.state"].shape) == (1, 6), "Vendor state dimension changed"
            )
            actions = postprocessor(policy.predict_action_chunk(batch))
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
        require(
            tuple(actions.shape) == (1, 50, 6),
            "Native vendor output shape differs from published config",
        )
        require(bool(actions.isfinite().all()), "Actual vendor forward produced nonfinite actions")
        values = actions.detach().float().cpu()
        results.append(
            {
                "call": index + 1,
                "shape": list(values.shape),
                "latency_ms": elapsed_ms,
                "finite": True,
                "output_tensor_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                "first_action_fixture_only": values[0, 0].tolist(),
            }
        )
        report["actual_forward_calls"] = len(results)
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        within_budget()
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["phase"] = "completed"
    report["passed"] = True


def run(spec: dict) -> dict:
    validate_specification(spec)
    enforce_offline()
    _verify_code(spec)
    require(
        os.environ.get("AZUREML_RUN_ID") == spec["job_name"], "Not the actual approved Azure ML run"
    )
    report = initial_report(spec)
    report["started_at_utc"] = datetime.now(UTC).isoformat()
    started = time.monotonic()
    deadline = started + spec["execution_timeout_seconds"] - 30

    def within_budget() -> None:
        require(
            time.monotonic() < deadline, "Bounded vendor diagnostic execution deadline exceeded"
        )

    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    with (
        ManagedIdentityCredential(client_id=spec["managed_identity_client_id"]) as credential,
        BlobServiceClient(
            spec["storage_account_url"],
            credential=credential,
            connection_timeout=10,
            read_timeout=30,
            retry_total=0,
        ) as client,
        tempfile.TemporaryDirectory(prefix="smolvla-vendor-compatibility-only-") as temporary,
    ):
        container = client.get_container_client(spec["container"])
        require(
            next(iter(container.list_blobs(name_starts_with=spec["output_prefix"] + "/")), None)
            is None,
            "Diagnostic output already exists; never overwrite or auto-retry",
        )
        try:
            root = Path(temporary)
            report["phase"] = "reading_verified_private_vendor_assets"
            report["verified_assets"] = _download_private_assets(
                container, spec, root, within_budget
            )
            report["asset_read_seconds"] = time.monotonic() - started
            _measure(root, report, within_budget)
        except Exception as exc:
            # Preserve the job's actual failure, then re-raise rather than returning success.
            report["passed"] = False
            report["failure_type"] = type(exc).__name__
            report["failure"] = str(exc)[:3000]
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            content = canonical(report) + b"\n"
            name = spec["output_prefix"] + "/vendor-compatibility.json"
            container.upload_blob(
                name,
                content,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/json"),
            )
            print(
                "PHYSICALAI_SMOL_VENDOR_COMPATIBILITY "
                + canonical(
                    {
                        "proof_blob": name,
                        "proof_sha256": digest(content),
                        "passed": report["passed"],
                        "phase": report["phase"],
                        "vendor_weights_loaded": report["vendor_weights_loaded"],
                        "actual_forward_calls": report["actual_forward_calls"],
                        "test_only": True,
                        "optimizer_steps": 0,
                        "learning_quality_verified": False,
                    }
                ).decode(),
                flush=True,
            )
    return report


if __name__ == "__main__":
    encoded = os.environ["PHYSICALAI_VENDOR_DIAGNOSTIC_SPEC"]
    require(len(encoded) <= 32768, "Oversized diagnostic specification")
    run(parse_json(base64.b64decode(encoded, validate=True)))
