from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
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
from learning.gr00t import ACTION_HORIZON, MODEL_ID, MODEL_REVISION, POLICY_TYPE, SOURCE_COMMIT

MODEL_SCHEMA = "physicalai.gr00t-checkpoint/v1"
UPSTREAM = {"source_commit": SOURCE_COMMIT, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}
PRETRAINED_WEIGHTS = {
    "model-00001-of-00003.safetensors": (
        "4d5e8e2a81a4b25084965ae87b8257a87ebe2f3b6656f83bbb15ba13675940b6"
    ),
    "model-00002-of-00003.safetensors": (
        "03ba0f11339d5ed24920582781c7352e308eb4eb78067c128bd8516b146656c8"
    ),
    "model-00003-of-00003.safetensors": (
        "1f572eb204d7afe3ddbfb890ca56eac1a9bafbdce51ed6fd3ba314dc4298d565"
    ),
}


def weight_digest(files: dict[str, str]) -> str:
    weights = {name: value for name, value in files.items() if name.endswith(".safetensors")}
    require(bool(weights), "Checkpoint has no real safetensors weights")
    return digest(canonical(weights))


def task_contract(task: DemonstrationSource) -> dict:
    task.validate()
    return {"task_id": task.task_id, "instruction": task.instruction, "goal_id": task.goal_id}


def model_contract(
    *,
    scope: Scope,
    profile: ControlProfile,
    task: DemonstrationSource,
    checkpoint: Path,
    role: str,
    training: dict | None,
) -> dict:
    scope.validate()
    profile.validate()
    files = inventory(checkpoint)
    return {
        "schema": MODEL_SCHEMA,
        "policy_type": POLICY_TYPE,
        "upstream": UPSTREAM,
        "scope": asdict(scope),
        "control_profile": asdict(profile),
        "task": task_contract(task),
        "role": role,
        "training": training,
        "checkpoint_files": files,
        "weights_sha256": weight_digest(files),
        "processor_sha256": files.get("experiment_cfg/metadata.json"),
        "action_horizon": ACTION_HORIZON,
        "n_action_steps": 1,
    }


def _no_dynamic_configuration(value: object) -> None:
    if isinstance(value, dict):
        require(
            not {"auto_map", "_auto_class", "trust_remote_code", "custom_pipelines"}.intersection(
                value
            ),
            "Checkpoint cannot select executable/remote processor code",
        )
        for item in value.values():
            _no_dynamic_configuration(item)
    elif isinstance(value, list):
        for item in value:
            _no_dynamic_configuration(item)


def validate_model(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    for_inference: bool = True,
) -> dict:
    expected_scope.validate()
    path = safe_path(root, "model.json")
    require(file_digest(path) == sha256(expected_model_sha256), "Model checksum mismatch")
    model = keys(
        read_json(path),
        {
            "schema",
            "policy_type",
            "upstream",
            "scope",
            "control_profile",
            "task",
            "role",
            "training",
            "checkpoint_files",
            "weights_sha256",
            "processor_sha256",
            "action_horizon",
            "n_action_steps",
        },
        "GR00T model manifest",
    )
    require(
        model["schema"] == MODEL_SCHEMA
        and model["policy_type"] == POLICY_TYPE
        and model["upstream"] == UPSTREAM,
        "Unapproved GR00T N1.5 model/source revision",
    )
    require(model["scope"] == asdict(expected_scope), "Model tenant/owner scope mismatch")
    ControlProfile(
        **keys(
            model["control_profile"],
            set(ControlProfile.__dataclass_fields__),
            "model control profile",
        )
    ).validate()
    task = keys(model["task"], {"task_id", "instruction", "goal_id"}, "approved model task")
    DemonstrationSource(kind="human_teleop", **task).validate()
    require(model["role"] in ("pretrained", "candidate"), "Unknown GR00T artifact role")
    require(
        model["action_horizon"] == ACTION_HORIZON and model["n_action_steps"] == 1,
        "Unapproved GR00T horizon",
    )
    if for_inference:
        require(
            model["role"] == "candidate" and model["training"] is not None,
            "Only actually trained Franka checkpoints can execute as P0/P1",
        )
    files = model["checkpoint_files"]
    require(isinstance(files, dict) and "config.json" in files, "Missing model configuration")
    for name in files:
        require(
            name in ("config.json", "model.safetensors.index.json", "experiment_cfg/metadata.json")
            or re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", name),
            "Checkpoint contains unapproved executable/pickle/processor files",
        )
    verify_inventory(root / "checkpoint", files)
    require(weight_digest(files) == sha256(model["weights_sha256"]), "Weight provenance mismatch")
    config = read_json(root / "checkpoint" / "config.json")
    _no_dynamic_configuration(config)
    require(
        config.get("model_type") == POLICY_TYPE
        and config.get("architectures") == ["GR00T_N1_5"]
        and config.get("action_horizon") == ACTION_HORIZON
        and config.get("action_dim") == 32
        and config.get("action_head_cfg", {}).get("action_horizon") == ACTION_HORIZON
        and config.get("action_head_cfg", {}).get("action_dim") == 32,
        "Model architecture differs from pinned N1.5 padded32/physical9/horizon16",
    )
    if "model.safetensors.index.json" in files:
        index = read_json(root / "checkpoint" / "model.safetensors.index.json")
        require(
            isinstance(index.get("weight_map"), dict)
            and bool(index["weight_map"])
            and set(index["weight_map"].values())
            == {name for name in files if name.endswith(".safetensors")},
            "Shard index references missing or unapproved files",
        )
    if model["role"] == "pretrained":
        require(model["training"] is None, "Base model has false training evidence")
        require(
            {name: value for name, value in files.items() if name.endswith(".safetensors")}
            == PRETRAINED_WEIGHTS,
            "Base weights differ from the pinned NVIDIA model revision",
        )
    else:
        training = keys(
            model["training"],
            {
                "parent_model_sha256",
                "parent_weights_sha256",
                "raw_manifest_sha256",
                "export_sha256",
                "config_sha256",
                "code_snapshot_sha256",
                "azure_job_id",
                "optimizer_steps",
                "cumulative_optimizer_steps",
                "checkpoint_step",
                "resume_from_job_id",
                "resume_mode",
                "test_only",
                "episodes",
                "gpu",
            },
            "actual GR00T training provenance",
        )
        for name in (
            "parent_model_sha256",
            "parent_weights_sha256",
            "raw_manifest_sha256",
            "export_sha256",
            "config_sha256",
            "code_snapshot_sha256",
        ):
            sha256(training[name], name)
        steps = integer(training["optimizer_steps"], "optimizer steps", 1)
        integer(training["checkpoint_step"], "checkpoint step", steps, steps)
        integer(training["cumulative_optimizer_steps"], "cumulative optimizer steps", steps)
        require(
            training["test_only"] is False
            and training["parent_weights_sha256"] != model["weights_sha256"],
            "Real changed weights are required; no synthetic/no-op training",
        )
        require(
            isinstance(training["azure_job_id"], str)
            and re.fullmatch(
                r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.-]+/providers/"
                r"Microsoft.MachineLearningServices/workspaces/[A-Za-z0-9_.-]+/jobs/[A-Za-z0-9_.-]+",
                training["azure_job_id"],
            ),
            "Explicit real Azure ML job resource identity is required",
        )
        require(
            training["resume_mode"] in ("new", "weights_only"),
            "Unsupported pickle/optimizer resume",
        )
        require(
            isinstance(training["episodes"], list)
            and bool(training["episodes"])
            and isinstance(training["gpu"], dict)
            and training["gpu"].get("cuda") is True,
            "Missing actual dataset/GPU provenance",
        )
        before = sha256(
            training["gpu"].get("updated_parameter_sample_before"), "pre-update parameters"
        )
        after = sha256(
            training["gpu"].get("updated_parameter_sample_after"), "post-update parameters"
        )
        require(before != after, "No measured trainable-parameter update")
        integer(training["gpu"].get("trainable_tensors"), "trainable tensor count", 1)
        metadata_path = "experiment_cfg/metadata.json"
        require(
            files.get(metadata_path) == sha256(model["processor_sha256"]),
            "Missing or unbound normalization metadata",
        )
        metadata = read_json(root / "checkpoint" / metadata_path)
        require(
            "new_embodiment" in metadata, "Checkpoint was not trained for the Franka embodiment"
        )
        modalities = metadata["new_embodiment"].get("modalities", {})
        for group in ("state", "action"):
            require(
                set(modalities.get(group, {})) == {"arm", "fingers"}
                and modalities[group]["arm"].get("shape") == [7]
                and modalities[group]["fingers"].get("shape") == [2],
                "Franka normalization metadata has the wrong physical dimensions",
            )
    return model
