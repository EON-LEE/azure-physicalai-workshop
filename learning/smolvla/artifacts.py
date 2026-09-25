from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from learning.common import (
    canonical,
    digest,
    file_digest,
    finite,
    integer,
    inventory,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
    verify_inventory,
)
from learning.contract import ControlProfile, DemonstrationSource, Scope
from learning.gr00t.artifacts import _no_dynamic_configuration, task_contract
from learning.smolvla import ACTION_HORIZON, POLICY_TYPE, UPSTREAM
from learning.smolvla.adaptation import IMAGE_SIZE, validate_franka_config

if TYPE_CHECKING:
    from learning.paused.contract import PausedControlProfile

MODEL_SCHEMA = "physicalai.smolvla-checkpoint/v1"


def processor_digest(files: dict) -> str:
    processors = {
        name: checksum
        for name, checksum in files.items()
        if name.startswith(("policy_preprocessor", "policy_postprocessor"))
    }
    require(bool(processors), "Missing actual Smol normalizers/tokenizer processors")
    return digest(canonical(processors))


def model_contract(
    *,
    checkpoint: Path,
    scope: Scope,
    profile: ControlProfile | PausedControlProfile,
    task: DemonstrationSource,
    backbone_manifest_sha256: str,
    role: str,
    training: dict | None,
) -> dict:
    files = inventory(checkpoint)
    return {
        "schema": MODEL_SCHEMA,
        "policy_type": POLICY_TYPE,
        "upstream": UPSTREAM,
        "scope": asdict(scope),
        "control_profile": asdict(profile),
        "task": task_contract(task),
        "backbone_manifest_sha256": sha256(backbone_manifest_sha256),
        "checkpoint_files": files,
        "weights_sha256": files["model.safetensors"],
        "processor_sha256": processor_digest(files),
        "role": role,
        "training": training,
        "action_horizon": ACTION_HORIZON,
        "n_action_steps": 1,
        "image_size": IMAGE_SIZE,
    }


def validate_backbone(root: Path, *, scope: Scope, expected_sha256: str) -> Path:
    require(
        file_digest(safe_path(root, "backbone.json")) == sha256(expected_sha256),
        "Backbone manifest checksum mismatch",
    )
    value = keys(
        read_json(root / "backbone.json"),
        {"schema", "scope", "upstream", "files"},
        "SmolVLM backbone",
    )
    require(
        value["schema"] == "physicalai.smolvla-backbone/v1"
        and value["scope"] == asdict(scope)
        and value["upstream"] == UPSTREAM,
        "Unapproved backbone identity/scope/license",
    )
    verify_inventory(root, value["files"], exclude={"backbone.json"})
    from learning.smolvla.prepare import verify_vendor_files

    verify_vendor_files(root / "assets", kind="backbone")
    _no_dynamic_configuration(read_json(root / "assets" / "config.json"))
    _no_dynamic_configuration(read_json(root / "assets" / "processor_config.json"))
    return (root / "assets").resolve()


def validate_model(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool = True,
) -> dict:
    return _validate_model(
        root,
        expected_scope=expected_scope,
        expected_model_sha256=expected_model_sha256,
        for_inference=for_inference,
    )


def _validate_model(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool,
    model_schema: str = MODEL_SCHEMA,
    profile_type: type[ControlProfile] | type[PausedControlProfile] = ControlProfile,
    extra_fields: frozenset[str] = frozenset(),
) -> dict:
    expected_scope.validate()
    require(
        file_digest(safe_path(root, "model.json")) == sha256(expected_model_sha256),
        "Model manifest checksum mismatch",
    )
    model = keys(
        read_json(root / "model.json"),
        {
            "schema",
            "policy_type",
            "upstream",
            "scope",
            "control_profile",
            "task",
            "role",
            "training",
            "backbone_manifest_sha256",
            "checkpoint_files",
            "weights_sha256",
            "processor_sha256",
            "action_horizon",
            "n_action_steps",
            "image_size",
        }
        | extra_fields,
        "Smol model manifest",
    )
    require(
        model["schema"] == model_schema
        and model["policy_type"] == POLICY_TYPE
        and model["upstream"] == UPSTREAM,
        "Wrong explicit Smol family/model/backbone/license pins",
    )
    require(model["scope"] == asdict(expected_scope), "Smol model tenant/owner scope mismatch")
    profile_type(
        **keys(model["control_profile"], set(profile_type.__dataclass_fields__), "servo profile")
    ).validate()
    DemonstrationSource(
        kind="reference_controller",
        **keys(model["task"], {"task_id", "instruction", "goal_id"}, "task"),
    ).validate()
    require(
        model["action_horizon"] == 50
        and model["n_action_steps"] == 1
        and model["image_size"] == IMAGE_SIZE,
        "Wrong physical inference cadence/shape",
    )
    require(model["role"] in ("pretrained", "candidate"), "Unknown Smol artifact role")
    if for_inference:
        require(
            model["role"] == "candidate", "Six-DOF vendor base is train-only, not a Franka policy"
        )
    sha256(model["backbone_manifest_sha256"])
    files = model["checkpoint_files"]
    require(isinstance(files, dict), "Missing checkpoint file inventory")
    for name in files:
        require(
            "/" not in name
            and (
                name
                in {
                    "config.json",
                    "model.safetensors",
                    "train_config.json",
                    "README.md",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                }
                or re.fullmatch(r"policy_(pre|post)processor_step_\d+_[a-z_]+\.safetensors", name)
            ),
            "Unapproved executable/pickle/checkpoint content",
        )
    verify_inventory(root / "checkpoint", files)
    require(
        files.get("model.safetensors") == sha256(model["weights_sha256"])
        and processor_digest(files) == sha256(model["processor_sha256"]),
        "Model/processor hash mismatch",
    )
    config = read_json(root / "checkpoint" / "config.json")
    _no_dynamic_configuration(config)
    if model["role"] == "pretrained":
        from learning.smolvla.prepare import verify_vendor_files

        verify_vendor_files(root / "checkpoint", kind="model")
        require(
            model["training"] is None and config["type"] == POLICY_TYPE, "False pretrained lineage"
        )
        return model
    validate_franka_config(config)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = read_json(root / "checkpoint" / name)
        require(isinstance(processor.get("steps"), list), "Missing saved processor steps")
        for step in processor["steps"]:
            require(
                "class" not in step
                and step.get("registry_name")
                in {
                    "rename_observations_processor",
                    "to_batch_processor",
                    "device_processor",
                    "normalizer_processor",
                    "unnormalizer_processor",
                    "tokenizer_processor",
                    "smolvla_new_line_processor",
                },
                "Unreviewed Smol processor code",
            )
            if step.get("state_file"):
                safe_path(root / "checkpoint", step["state_file"])
    training = model["training"]
    require(
        isinstance(training, dict) and training.get("test_only") is False,
        "Not a real trained model",
    )
    for name in (
        "parent_model_sha256",
        "parent_weights_sha256",
        "raw_manifest_sha256",
        "conversion_sha256",
        "config_sha256",
        "code_snapshot_sha256",
        "specification_sha256",
        "updated_parameter_sample_before",
        "updated_parameter_sample_after",
    ):
        sha256(training.get(name), name)
    require(
        training["parent_weights_sha256"] != model["weights_sha256"]
        and training["updated_parameter_sample_before"]
        != training["updated_parameter_sample_after"],
        "No actual weight/parameter update",
    )
    steps = integer(training.get("optimizer_steps"), "actual optimizer steps", 1)
    integer(training.get("cumulative_optimizer_steps"), "cumulative optimizer steps", steps)
    integer(training.get("checkpoint_step"), "checkpoint marker", steps, steps)
    require(
        training.get("resume_mode") in ("new", "weights_only"), "Unapproved optimizer/pickle resume"
    )
    require(
        isinstance(training.get("episodes"), list) and bool(training["episodes"]),
        "Missing training episode lineage",
    )
    for name in ("azure_job_id", "azure_pipeline_job_id"):
        require(
            isinstance(training.get(name), str)
            and re.fullmatch(
                r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
                r"Microsoft.MachineLearningServices/workspaces/[A-Za-z0-9_.-]+/jobs/[A-Za-z0-9_.-]+",
                training[name],
            ),
            "Missing actual Azure pipeline/component job identity",
        )
    require(
        training["azure_job_id"].rsplit("/jobs/", 1)[0]
        == training["azure_pipeline_job_id"].rsplit("/jobs/", 1)[0],
        "Cross-workspace job lineage",
    )
    require(training.get("gpu", {}).get("cuda") is True, "Missing actual CUDA training provenance")
    if training.get("loss") is not None:
        finite(training["loss"], "measured training loss")
    return model
