"""Explicit v3 command admission, separate from the immutable legacy servo validator bundle."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from learning.common import (
    file_digest,
    finite,
    integer,
    keys,
    read_json,
    require,
    safe_path,
    sha256,
    token,
    verify_inventory,
)
from learning.contract import DemonstrationSource, Scope
from learning.gr00t.artifacts import _no_dynamic_configuration
from learning.paused.artifacts import MODE_FIELDS
from learning.paused.contract import (
    CONVERSION_SCHEMA,
    EXECUTION_TIMING,
    RAW_SCHEMA,
    PausedControlProfile,
)
from learning.smolvla import ACTION_HORIZON, POLICY_TYPE, UPSTREAM
from learning.smolvla.adaptation import IMAGE_SIZE, validate_franka_config
from learning.smolvla.artifacts import model_contract as base_contract
from learning.smolvla.artifacts import processor_digest

MODEL_SCHEMA = "physicalai.smolvla-checkpoint/v3"
TRAINING_EXECUTION = "azureml_command"
ADMISSION_KIND = "azureml_command_v3"


def model_contract(
    *,
    checkpoint: Path,
    scope: Scope,
    profile: PausedControlProfile,
    task: DemonstrationSource,
    backbone_manifest_sha256: str,
    role: str,
    training: dict | None,
    criteria_sha256: str,
    frozen_plan_sha256: str,
) -> dict:
    scope.validate()
    profile.validate()
    task.validate()
    require(
        role == "candidate" and isinstance(training, dict),
        "Command v3 is a genuinely trained candidate, not a vendor/prepared parent",
    )
    value = base_contract(
        checkpoint=checkpoint,
        scope=scope,
        profile=profile,
        task=task,
        backbone_manifest_sha256=backbone_manifest_sha256,
        role=role,
        training=training,
    )
    value.update(
        schema=MODEL_SCHEMA,
        training_execution=TRAINING_EXECUTION,
        execution_timing=EXECUTION_TIMING,
        real_time_admission=False,
        timestamp_basis="simulation_time",
        criteria_sha256=sha256(criteria_sha256),
        frozen_plan_sha256=sha256(frozen_plan_sha256),
    )
    return value


def _checkpoint(root: Path, model: dict) -> None:
    files = model["checkpoint_files"]
    require(isinstance(files, dict), "Missing actual candidate checkpoint inventory")
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
        "Command weights/processors differ from the signed inventory",
    )
    config = read_json(root / "checkpoint" / "config.json")
    _no_dynamic_configuration(config)
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


def _training(model: dict) -> None:
    training = model["training"]
    require(
        isinstance(training, dict) and training.get("test_only") is False,
        "Command candidate must retain actual training provenance",
    )
    require(
        training.get("azure_job_type") == "command"
        and not {"azure_pipeline_job_id", "azure_component_job_id", "resume_from_pipeline_job_id"}
        & training.keys(),
        "Command training cannot invent parent/component identities",
    )
    require(
        training.get("raw_schema") == RAW_SCHEMA
        and training.get("conversion_schema") == CONVERSION_SCHEMA
        and all(training.get(name) == model[name] for name in MODE_FIELDS),
        "Command model is not bound to the actual paused data/criteria/conditions",
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
    steps = integer(training.get("optimizer_steps"), "actual optimizer updates", 1)
    cumulative = integer(training.get("cumulative_optimizer_steps"), "cumulative updates", steps)
    mode = training.get("resume_mode")
    require(mode in ("new", "weights_only", "full_state"), "Unapproved optimizer/pickle resume")
    restored = (
        integer(training.get("resume_from_checkpoint_step"), "source step", 1)
        if mode == "full_state"
        else 0
    )
    integer(
        training.get("checkpoint_step"),
        "actual checkpoint marker",
        steps + restored,
        steps + restored,
    )
    require(
        isinstance(training.get("episodes"), list) and bool(training["episodes"]),
        "Missing training episode lineage",
    )
    episode_ids = set()
    from learning.paused.dataset import TRAIN_SEEDS

    for episode in training["episodes"]:
        keys(episode, {"episode_id", "environment_id", "revision", "seed"}, "training episode")
        token(episode["episode_id"], "training episode ID")
        token(episode["environment_id"], "training environment ID")
        sha256(episode["revision"], "training environment revision")
        integer(episode["seed"], "TRAIN seed")
        require(
            episode["episode_id"] not in episode_ids and episode["seed"] in TRAIN_SEEDS,
            "Duplicate or unapproved/validation/held-out training episode",
        )
        episode_ids.add(episode["episode_id"])
    job_id = training.get("azure_job_id")
    require(
        isinstance(job_id, str)
        and re.fullmatch(
            r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
            r"Microsoft.MachineLearningServices/workspaces/[A-Za-z0-9_.-]+/jobs/[A-Za-z0-9_.-]+",
            job_id,
        ),
        "Missing actual standalone AML root job identity",
    )
    if restored:
        sha256(training.get("resume_from_checkpoint_sha256"), "complete source checkpoint")
        previous = integer(
            training.get("resume_from_cumulative_optimizer_steps"),
            "source cumulative updates",
            restored,
        )
        require(
            training.get("optimizer_state_restored") is True
            and training.get("bitwise_continuation_claimed") is False
            and cumulative == previous + steps,
            "Full-state continuation must bind actual state/new updates, not bitwise claims",
        )
        workspace = job_id.rsplit("/jobs/", 1)[0] + "/jobs/"
        source = training.get("resume_from_job_id")
        require(
            training.get("resume_from_job_type") == "command"
            and isinstance(source, str)
            and source.startswith(workspace)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", source[len(workspace) :])
            and source != job_id,
            "Full-state source must be a distinct actual command in the same approved workspace",
        )
    require(training.get("gpu", {}).get("cuda") is True, "Missing actual CUDA training provenance")
    require(training["gpu"].get("device_count") == 1, "Command must use one actual physical GPU")
    if training.get("loss") is not None:
        finite(training["loss"], "measured training loss")


def validate_model(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool = True,
) -> dict:
    """Fully validate original v3 bytes; never create a compatibility view for the v2 validator."""
    expected_scope.validate()
    require(type(for_inference) is bool, "Explicit candidate admission purpose is required")
    require(
        file_digest(safe_path(root, "model.json")) == sha256(expected_model_sha256),
        "Command model manifest checksum mismatch",
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
            "training_execution",
        }
        | MODE_FIELDS,
        "command model manifest",
    )
    require(
        model["schema"] == MODEL_SCHEMA
        and model["training_execution"] == TRAINING_EXECUTION
        and model["role"] == "candidate"
        and model["policy_type"] == POLICY_TYPE
        and model["upstream"] == UPSTREAM,
        "Only explicitly admitted command v3 trained SmolVLA candidates are accepted",
    )
    require(model["scope"] == asdict(expected_scope), "Command model tenant/owner mismatch")
    profile = PausedControlProfile(
        **keys(model["control_profile"], set(PausedControlProfile.__dataclass_fields__), "profile"),
    )
    profile.validate()
    require(
        model["execution_timing"] == EXECUTION_TIMING
        and model["real_time_admission"] is False
        and model["timestamp_basis"] == "simulation_time",
        "Command admission never changes paused or real-time control semantics",
    )
    for name in ("criteria_sha256", "frozen_plan_sha256", "backbone_manifest_sha256"):
        sha256(model[name], name)
    DemonstrationSource(
        kind="reference_controller",
        **keys(model["task"], {"task_id", "instruction", "goal_id"}, "task"),
    ).validate()
    require(
        model["action_horizon"] == ACTION_HORIZON
        and model["n_action_steps"] == 1
        and model["image_size"] == IMAGE_SIZE,
        "Wrong actual physical inference cadence/shape",
    )
    _checkpoint(root, model)
    _training(model)
    return model


def validate_parent(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool = False,
) -> dict:
    require(for_inference is False, "Parent selection is not model inference admission")
    require(
        file_digest(safe_path(root, "model.json")) == sha256(expected_model_sha256),
        "Parent manifest checksum mismatch",
    )
    if read_json(root / "model.json").get("schema") == MODEL_SCHEMA:
        return validate_model(
            root,
            expected_scope=expected_scope,
            expected_model_sha256=expected_model_sha256,
            for_inference=False,
        )
    from learning.paused.artifacts import validate_model as legacy_model

    parent = legacy_model(
        root,
        expected_scope=expected_scope,
        expected_model_sha256=expected_model_sha256,
        for_inference=False,
    )
    require(
        parent["role"] == "pretrained",
        "Command training accepts the approved v2 prepared parent or a real v3 command candidate",
    )
    return parent


def validate_models(plan: dict, models: dict[str, Path], *, scope: Scope) -> dict:
    from learning.paused.evaluation import expected_model, roles, validate_plan

    cases = validate_plan(plan)
    require(plan["scope"] == asdict(scope), "Evaluation plan owner/tenant mismatch")
    keys(models, set(roles(plan)) - {"reference"}, "explicit command evaluation models")
    loaded = {}
    for role, path in models.items():
        model = validate_model(
            path,
            expected_scope=scope,
            expected_model_sha256=expected_model(plan, role),
        )
        require(
            model["control_profile"] == plan["control_profile"]
            and model["criteria_sha256"] == plan["criteria_sha256"]
            and model["frozen_plan_sha256"] == plan["frozen_plan_sha256"],
            "Evaluated model differs from the profile or frozen training/evaluation conditions",
        )
        for episode in model["training"]["episodes"]:
            require(
                all(
                    episode["episode_id"] != case["episode_id"] and episode["seed"] != case["seed"]
                    for case in cases.values()
                ),
                "Training/final-held-out leakage",
            )
        loaded[role] = model
    if "before" in loaded:
        require(
            loaded["before"]["task"] == loaded["after"]["task"]
            and (
                loaded["after"]["training"]["parent_model_sha256"] == plan["policy_before_sha256"]
                or plan["policy_before_sha256"]
                in loaded["after"]["training"].get("ancestor_model_sha256s", [])
            ),
            "Paired task or actual incremental model lineage differs",
        )
    return loaded
