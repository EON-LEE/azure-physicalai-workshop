"""Pinned native LeRobot save-boundary integration; never a replacement training loop."""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from unittest.mock import patch

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
)
from learning.deadlines import JobDeadline
from learning.smolvla.checkpoints import (
    CONTINUATION_SCHEMA,
    FULL_STATE_FILES,
    CheckpointLimits,
    publish_checkpoint,
    seal_checkpoint,
    validate_checkpoint,
)

POLICY_SCHEMA = "physicalai.smolvla-checkpointing/v1"
CONTEXT_SCHEMA = "physicalai.smolvla-checkpoint-run/v2"
NATIVE_SOURCE_SHA256 = {
    "lerobot/utils/train_utils.py": (
        "ed0f839e6caf51ff6f79a1c8f750eb1a144f83c40a92366a5d32da0921e5b82b"
    ),
    "lerobot/utils/random_utils.py": (
        "3f3b373ad8ae7dd240c13eddc10c226836e31bd68bf0cd1765132599e482a17d"
    ),
    "lerobot/optim/optimizers.py": (
        "9bce2b9e5651b9a4f72aecbb8f54ef3a1143021217d261972bd505126e94f97b"
    ),
    "lerobot/optim/schedulers.py": (
        "47af5671a9d9c79ecdb5101e92e6b9f127331127549497047f691258f10b1020"
    ),
    "lerobot/configs/train.py": "996a5787b8e2141f56a6163b45b7131be63606091d5ec4f5b7c3a8bae1aa4e63",
    "lerobot/scripts/lerobot_train.py": (
        "827a2b7e74341f1d89d5c508d62ce52592bb9cab0c74fe8a3de38e9f03886410"
    ),
    "lerobot/datasets/sampler.py": (
        "0202ce63101050dc8b28f00cf68b9c28bf66d9c5c2f9e05d1e7c9e24d4905d19"
    ),
}


def validate_policy(value: dict, *, parameters: dict) -> CheckpointLimits:
    keys(value, {"schema", "limits", "resume"}, "checkpoint policy")
    require(value["schema"] == POLICY_SCHEMA, "Unknown checkpoint publication policy")
    limits = CheckpointLimits(
        **keys(value["limits"], set(CheckpointLimits.__dataclass_fields__), "checkpoint limits")
    )
    limits.validate()
    integer(parameters["max_steps"], "training horizon", 1, 100000)
    integer(parameters["checkpoint_steps"], "checkpoint interval", 1, parameters["max_steps"])
    require(
        (parameters["max_steps"] + parameters["checkpoint_steps"] - 1)
        // parameters["checkpoint_steps"]
        <= limits.max_checkpoints,
        "Checkpoint frequency exceeds the explicit storage/count budget",
    )
    resume = value["resume"]
    if resume is not None:
        command = resume.get("source_azure_job_type") == "command"
        keys(
            resume,
            {"checkpoint_sha256", "source_azure_job_id", "step"}
            | ({"source_azure_job_type"} if command else {"source_azure_pipeline_job_id"}),
            "explicit checkpoint resume",
        )
        sha256(resume["checkpoint_sha256"])
        integer(resume["step"], "original checkpoint step", 1, 100000)
        require(
            parameters["resume_mode"] in ("weights_only", "full_state"),
            "Choose explicit full_state or weights_only restart",
        )
        if parameters["resume_mode"] == "full_state":
            require(
                resume["step"] < parameters["max_steps"],
                "Full-state job has no remaining optimizer steps",
            )
    return limits


def native_runtime(*, environment_image: str | None = None) -> dict:
    require(version("lerobot") == "0.4.4", "Checkpoint runner requires pinned LeRobot 0.4.4")
    package = distribution("lerobot")
    for name, checksum in NATIVE_SOURCE_SHA256.items():
        require(
            file_digest(Path(package.locate_file(name))) == checksum,
            f"Pinned native checkpoint/trainer source changed: {name}",
        )
    import torch

    driver = None
    if torch.cuda.is_available():
        drivers = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
        require(
            drivers and len(set(drivers)) == 1 and re.fullmatch(r"\d+(?:\.\d+)+", drivers[0]),
            "Unverified or inconsistent actual GPU driver version",
        )
        driver = drivers[0]
    return {
        "environment_image": environment_image,
        "python_version": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "lerobot",
                "torch",
                "torchvision",
                "numpy",
                "safetensors",
                "transformers",
                "accelerate",
            )
        },
        "native_source_sha256": NATIVE_SOURCE_SHA256,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "compute_capability": list(torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": driver,
    }


def training_code_sha256(config: dict) -> str:
    from learning.smolvla.azure import code_files

    root = Path(__file__).resolve().parents[2]
    names = code_files(config)
    return digest(canonical({name: file_digest(root / name) for name in names}))


def make_binding(
    config: dict, parent: dict, converted: dict, conversion_sha256: str, runtime: dict
) -> dict:
    parameters = {
        key: value
        for key, value in config["parameters"].items()
        if key not in ("resume_mode", "timeout_seconds", "compute_tier")
    }
    return {
        "scope": parent["scope"],
        "raw_manifest_sha256": converted["raw_manifest_sha256"],
        "conversion_sha256": conversion_sha256,
        "parent_model_sha256": config["inputs"]["parent_model"]["sha256"],
        "parent_weights_sha256": parent["weights_sha256"],
        "control_profile_sha256": config["control_profile_sha256"],
        "task_sha256": config["task_sha256"],
        "criteria_sha256": config.get("criteria_sha256"),
        "frozen_plan_sha256": config.get("frozen_plan_sha256"),
        "training_config_sha256": digest(canonical(parameters)),
        "training_code_sha256": training_code_sha256(config),
        "runtime_sha256": digest(canonical(runtime)),
    }


def validate_resume_checkpoint(
    root: Path, *, config: dict, expected_binding: dict, current_origin: dict
) -> dict:
    from learning.smolvla.checkpoints import _origin

    _origin(current_origin)
    policy = config["checkpointing"]
    limits = validate_policy(policy, parameters=config["parameters"])
    approved = policy["resume"]
    require(approved is not None, "Missing explicit resumed checkpoint approval")
    value = read_json(safe_path(root, "checkpoint.json"), limits.max_json_bytes)
    full_state = config["parameters"]["resume_mode"] == "full_state"
    # A weights-only restart may deliberately change its new optimizer/horizon/runtime.
    # The exact source manifest still binds the old configuration and original job.
    mutable = (
        set()
        if full_state
        else {
            "training_config_sha256",
            "training_code_sha256",
            "runtime_sha256",
        }
    )
    for key in expected_binding.keys() - mutable:
        require(
            value["binding"].get(key) == expected_binding[key],
            "Resume data/model/profile/owner binding changed",
        )
    validate_checkpoint(
        root,
        expected_sha256=approved["checkpoint_sha256"],
        expected_binding=value["binding"],
        require_full_state=full_state,
        limits=limits,
    )
    require(
        value["step"] == approved["step"]
        and value["origin"]["azure_job_id"] == approved["source_azure_job_id"]
        and value["origin"]["azure_job_id"] != current_origin["azure_job_id"]
        and value["origin"]["specification_sha256"] != current_origin["specification_sha256"]
        and value["origin"]["test_only"] == current_origin["test_only"],
        "Resume requires a distinct newly authorized job bound to the exact checkpoint origin",
    )
    if current_origin.get("azure_job_type") == "command":
        require(
            value["origin"].get("azure_job_type")
            == approved.get("source_azure_job_type")
            == "command"
            and "azure_pipeline_job_id" not in value["origin"]
            and "source_azure_pipeline_job_id" not in approved,
            "Command checkpoint continuation cannot synthesize a pipeline hierarchy",
        )
    else:
        require(
            "azure_job_type" not in value["origin"]
            and "source_azure_job_type" not in approved
            and value["origin"]["azure_pipeline_job_id"] == approved["source_azure_pipeline_job_id"]
            and value["origin"]["azure_pipeline_job_id"] != current_origin["azure_pipeline_job_id"],
            "Checkpoint pipeline lineage changed",
        )
    require(
        value["origin"]["azure_job_id"].rsplit("/jobs/", 1)[0]
        == current_origin["azure_job_id"].rsplit("/jobs/", 1)[0],
        "Checkpoint restart must remain in the explicitly approved workspace",
    )
    if full_state:
        from learning.smolvla.checkpoint_state import continuation_metadata

        continuation_metadata(root, value["step"])
        native_config = read_json(root / "pretrained_model" / "train_config.json")
        from learning.gr00t.artifacts import _no_dynamic_configuration

        _no_dynamic_configuration(native_config)
        expected = {
            "steps": config["parameters"]["max_steps"],
            "batch_size": config["parameters"]["batch_size"],
            "seed": config["parameters"]["seed"],
            "save_freq": config["parameters"]["checkpoint_steps"],
            "num_workers": 0,
            "eval_freq": 0,
            "save_checkpoint": True,
        }
        require(
            all(native_config.get(name) == value for name, value in expected.items()),
            "Full-state native training horizon/seed/data/optimizer configuration changed",
        )
        require(
            native_config.get("policy", {}).get("type") == "smolvla"
            and native_config["policy"].get("use_amp") is False
            and native_config["policy"].get("push_to_hub") is False
            and native_config.get("wandb", {}).get("enable") is False
            and native_config.get("env") is None
            and native_config.get("peft") is None
            and native_config.get("use_rabc") is False
            and native_config.get("dataset", {}).get("streaming") is False
            and native_config["dataset"].get("image_transforms", {}).get("enable") is False,
            "Full-state supports only the pinned single-device unaugmented native Smol loop",
        )
        if not value["origin"]["test_only"]:
            from learning.smolvla.adaptation import validate_franka_config

            policy_config = read_json(root / "pretrained_model" / "config.json")
            _no_dynamic_configuration(policy_config)
            validate_franka_config(policy_config)
        allowed_processors = {
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
            "unnormalizer_processor",
            "tokenizer_processor",
            "smolvla_new_line_processor",
        }
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor = read_json(root / "pretrained_model" / name)
            require(isinstance(processor.get("steps"), list), "Missing native processor state")
            for entry in processor["steps"]:
                require(
                    "class" not in entry and entry.get("registry_name") in allowed_processors,
                    "Full-state checkpoint contains unreviewed processor code",
                )
                if entry.get("state_file"):
                    safe_path(root / "pretrained_model", entry["state_file"])
    return value


class NativeCheckpointPublisher:
    def __init__(
        self, *, original_save, context: dict, container, check_deadline, state_provider=None
    ):
        self.original_save = original_save
        self.context, self.container, self.check_deadline = context, container, check_deadline
        self.limits = CheckpointLimits(**context["limits"])
        self.limits.validate()
        self.state_provider = state_provider
        self.published = 0
        self.receipts = []

    def __call__(self, *args, **kwargs) -> None:
        if self.state_provider is None:
            self._save(*args, **kwargs)
            return
        from learning.smolvla.checkpoint_state import capture_rng, restore_rng

        rng = capture_rng()
        try:
            self._save(*args, rng=rng, **kwargs)
        finally:
            # Filesystem/Blob client bookkeeping must not perturb training RNG streams.
            restore_rng(rng)

    def _save(self, *args, rng=None, **kwargs) -> None:
        self.check_deadline()
        require(
            self.published < self.limits.max_checkpoints,
            "Complete checkpoint count budget exhausted",
        )
        bound = inspect.signature(self.original_save).bind(*args, **kwargs)
        step, root = bound.arguments["step"], Path(bound.arguments["checkpoint_dir"])
        require(not root.exists(), "Never rewrite an existing native checkpoint directory")
        if "policy" in bound.arguments:
            import torch

            tensors = list(bound.arguments["policy"].state_dict().values())
            optimizer = bound.arguments.get("optimizer")
            if optimizer is not None:
                state = optimizer.state_dict().get("state", {})
                tensors.extend(
                    value
                    for item in state.values()
                    for value in item.values()
                    if isinstance(value, torch.Tensor)
                )
            for value in tensors:
                require(
                    isinstance(value, torch.Tensor),
                    "Native model checkpoint contains non-tensor state",
                )
                if value.is_floating_point():
                    require(
                        bool(torch.isfinite(value).all()),
                        "Nonfinite model/optimizer checkpoint state",
                    )
        self.original_save(*args, **kwargs)
        self.check_deadline()
        required_native = FULL_STATE_FILES - {
            "training_state/continuation.json",
            "training_state/continuation.safetensors",
        }
        require(
            all((root / name).is_file() for name in required_native),
            "Native checkpoint save returned without complete optimizer/scheduler/RNG state",
        )
        capability = "weights_only"
        if self.state_provider is not None:
            self.state_provider.save(root, step, rng)
            state = read_json(root / "training_state" / "continuation.json")
            require(
                state.get("schema") == CONTINUATION_SCHEMA and state.get("step") == step,
                "State provider did not finish a bound continuation",
            )
            capability = "full_state"
        checksum = seal_checkpoint(
            root,
            step=step,
            binding=self.context["binding"],
            origin=self.context["origin"],
            state_kind=capability,
            prior_optimizer_steps=self.context.get("prior_optimizer_steps", 0),
            resume_from_checkpoint_sha256=self.context["resume_from_checkpoint_sha256"],
            limits=self.limits,
        )
        receipt = publish_checkpoint(
            root,
            self.container,
            prefix=self.context["blob_prefix"],
            expected_sha256=checksum,
            expected_binding=self.context["binding"],
            check_deadline=self.check_deadline,
            limits=self.limits,
        )
        self.receipts.append(receipt)
        self.published += 1
        print("PHYSICALAI_CHECKPOINT " + canonical(receipt).decode(), flush=True)


def run(context: dict) -> None:
    from learning.offline import enforce_offline

    enforce_offline()
    keys(
        context,
        {
            "schema",
            "binding",
            "origin",
            "limits",
            "storage_account_name",
            "blob_container",
            "managed_identity_client_id",
            "blob_prefix",
            "resume_from_checkpoint_sha256",
            "prior_optimizer_steps",
            "training_parameters",
            "resume_checkpoint_root",
            "environment_image",
        },
        "checkpoint job context",
    )
    require(context["schema"] == CONTEXT_SCHEMA, "Wrong checkpoint runner context")
    require(
        context["origin"]["test_only"] is False,
        "Production runner does not synthesize test/cloud identity",
    )
    require(
        os.environ.get("AZUREML_RUN_ID") == context["origin"]["azure_job_id"].rsplit("/", 1)[-1],
        "Checkpoint runner is not the actual approved AML component",
    )
    deadline = JobDeadline(context["origin"]["job_deadline_utc"])
    deadline.check()
    runtime = native_runtime(environment_image=context["environment_image"])
    require(
        runtime["device"] == "cuda" and runtime["device_count"] == 1,
        "Expected one approved CUDA device",
    )
    require(
        digest(canonical(runtime)) == context["binding"]["runtime_sha256"],
        "Checkpoint runtime changed",
    )
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient

    with (
        ManagedIdentityCredential(client_id=context["managed_identity_client_id"]) as credential,
        BlobServiceClient(
            f"https://{context['storage_account_name']}.blob.core.windows.net",
            credential=credential,
            retry_total=0,
            connection_timeout=10,
            read_timeout=30,
        ) as service,
    ):
        train_native(
            context, service.get_container_client(context["blob_container"]), deadline.check
        )


def train_native(context: dict, container, check_deadline) -> None:
    """The same pinned loop/hook wiring used by production and explicit CPU fixture checks."""
    from lerobot.scripts import lerobot_train as native

    from learning.smolvla.checkpoint_state import ContinuationState

    parameters = context["training_parameters"]
    full_resume = parameters["resume_mode"] == "full_state"
    resume_root = Path(context["resume_checkpoint_root"]) if full_resume else None
    resume_step = 0
    if full_resume:
        saved = validate_checkpoint(
            resume_root,
            expected_sha256=context["resume_from_checkpoint_sha256"],
            expected_binding=context["binding"],
            require_full_state=True,
            limits=CheckpointLimits(**context["limits"]),
        )
        resume_step = saved["step"]
    continuation = ContinuationState(
        seed=parameters["seed"], resume_root=resume_root, resume_step=resume_step
    )
    original_load, original_accelerator = native.load_training_state, native.Accelerator

    def load_state(path, optimizer, scheduler):
        require(
            full_resume and Path(path).resolve() == resume_root.resolve(),
            "Unapproved full-state load path",
        )
        step, optimizer, scheduler = original_load(path, optimizer, scheduler)
        require(step == resume_step, "Native optimizer step differs from the complete checkpoint")
        return step, optimizer, scheduler

    def accelerator(*args, **kwargs):
        require("mixed_precision" not in kwargs, "Unexpected implicit AMP override")
        result = original_accelerator(*args, mixed_precision="no", **kwargs)
        require(
            result.num_processes == 1
            and result.mixed_precision == "no"
            and result.gradient_accumulation_steps == 1
            and result.device.type == ("cpu" if context["origin"]["test_only"] else "cuda"),
            "Checkpoint continuation requires one process, one update per batch and no AMP",
        )
        return result

    publisher = NativeCheckpointPublisher(
        original_save=native.save_checkpoint,
        context=context,
        container=container,
        check_deadline=check_deadline,
        state_provider=continuation,
    )
    # Call the pinned native entry directly; do not load arbitrary third-party plugins.
    with (
        patch.object(native, "save_checkpoint", publisher),
        patch.object(native, "cycle", continuation.cycle),
        patch.object(native, "load_training_state", load_state),
        patch.object(native, "Accelerator", accelerator),
    ):
        native.train()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-context", type=Path, required=True)
    parser.add_argument("--checkpoint-context-sha256", required=True)
    args, native_args = parser.parse_known_args()
    require(
        file_digest(args.checkpoint_context) == sha256(args.checkpoint_context_sha256),
        "Checkpoint execution context changed",
    )
    context = read_json(args.checkpoint_context)
    sys.argv = [sys.argv[0], *native_args]
    run(context)


if __name__ == "__main__":
    main()
