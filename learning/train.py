from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from learning import LEROBOT_VERSION
from learning.common import (
    digest,
    file_digest,
    integer,
    inventory,
    keys,
    read_json,
    require,
    sha256,
    token,
    verify_inventory,
    write_json,
)
from learning.contract import (
    CAMERAS,
    EPISODE_KEYS,
    JOINT_NAMES,
    JOINT_UNITS,
    ControlProfile,
    DemonstrationSource,
    EpisodeSpec,
    Provenance,
    Scope,
    timing,
)
from learning.convert import CONVERSION_SCHEMA, LOCAL_REPO_ID, TEACHING_CONVERSION_SCHEMA
from learning.offline import OFFLINE_ENV, require_lerobot

MODEL_SCHEMA = "physicalai.act-checkpoint/v1"
CONVERSION_KEYS = {
    "schema",
    "lerobot_version",
    "repo_id",
    "scope",
    "raw_manifest_sha256",
    "dataset_id",
    "split",
    "fps",
    "physics_hz",
    "joint_names",
    "joint_units",
    "image_size",
    "image_transform",
    "test_only",
    "episodes",
    "files",
}


@dataclass(frozen=True)
class TrainOptions:
    steps: int = 20000
    batch_size: int = 8
    seed: int = 12345
    chunk_size: int = 10
    n_action_steps: int = 1
    timeout_seconds: int = 3600
    device: str = "cuda"
    smoke_test: bool = False

    def validate(self) -> None:
        integer(self.steps, "training steps", 1, 1000000)
        integer(self.batch_size, "batch size", 1, 64)
        integer(self.seed, "training seed", high=2**32 - 1)
        integer(self.chunk_size, "action chunk", 1, 100)
        integer(self.n_action_steps, "action steps", 1, min(10, self.chunk_size))
        integer(self.timeout_seconds, "timeout", 1, 86400)
        require(self.device in ("cpu", "cuda"), "Unsupported training device")
        require(type(self.smoke_test) is bool, "Invalid smoke flag")
        require(self.device == "cuda" or self.smoke_test, "CPU training is test-only")
        if self.smoke_test:
            require(self.steps <= 5 and self.batch_size <= 2, "Smoke run exceeds test-only budget")


def validate_conversion(root: Path, expected_scope: Scope) -> dict:
    expected_scope.validate()
    value = read_json(root / "conversion.json")
    teaching = value.get("schema") == TEACHING_CONVERSION_SCHEMA
    keys(value, CONVERSION_KEYS | ({"control_profile"} if teaching else set()), "conversion")
    require(
        value["schema"] in (CONVERSION_SCHEMA, TEACHING_CONVERSION_SCHEMA),
        "Wrong conversion schema",
    )
    if teaching:
        ControlProfile(**value["control_profile"]).validate()
    return _validate_conversion_contents(root, expected_scope, value, teaching=teaching)


def _validate_conversion_contents(
    root: Path,
    expected_scope: Scope,
    value: dict,
    *,
    teaching: bool,
    extra_episode_keys: frozenset[str] = frozenset(),
) -> dict:
    expected_scope.validate()
    require(value["lerobot_version"] == LEROBOT_VERSION, "Wrong converter version")
    require(value["scope"] == asdict(expected_scope), "Tenant/owner scope mismatch")
    require(value["repo_id"] == LOCAL_REPO_ID, "Remote dataset identifiers are forbidden")
    require(value["split"] == "train", "Only the train split can enter optimization")
    require(value["joint_names"] == list(JOINT_NAMES), "Wrong joint ordering")
    require(value["joint_units"] == list(JOINT_UNITS), "Wrong joint units")
    timing(value["fps"], value["physics_hz"])
    integer(value["image_size"], "image size", 32, 512)
    require(value["image_transform"] == "rgb-bilinear-square/v1", "Unknown image transform")
    require(type(value["test_only"]) is bool, "Missing fixture provenance")
    sha256(value["raw_manifest_sha256"])
    require(isinstance(value["episodes"], list) and bool(value["episodes"]), "Empty training split")
    ids, seeds = set(), set()
    has_fixtures = False
    for index, episode in enumerate(value["episodes"]):
        keys(
            episode,
            EPISODE_KEYS
            | {"episode_index", "terminated", "truncated"}
            | ({"demonstration"} if teaching else set())
            | extra_episode_keys,
            "converted episode",
        )
        EpisodeSpec(**{name: episode[name] for name in EpisodeSpec.__dataclass_fields__}).validate()
        if "demonstration" in episode:
            DemonstrationSource(**episode["demonstration"]).validate()
        provenance = Provenance(
            **keys(episode["provenance"], set(Provenance.__dataclass_fields__), "provenance")
        )
        provenance.validate()
        has_fixtures |= provenance.source_kind == "test_fixture"
        require(
            episode["split"] == "train" and episode["episode_index"] == index,
            "Held-out episode in training conversion",
        )
        require(episode["episode_id"] not in ids, "Duplicate training episode")
        ids.add(token(episode["episode_id"], "training episode ID"))
        seeds.add(integer(episode["seed"], "training scene seed", high=2**32 - 1))
        sha256(episode["revision"], "environment revision")
    require(value["test_only"] == has_fixtures, "Conversion fixture provenance was relabeled")
    verify_inventory(root, value["files"], exclude={"conversion.json"})
    return value


def training_command(dataset: Path, run_output: Path, options: TrainOptions) -> list[str]:
    options.validate()
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--policy.type=act",
        f"--dataset.repo_id={LOCAL_REPO_ID}",
        f"--dataset.root={dataset.resolve()}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=false",
        "--dataset.image_transforms.enable=false",
        f"--output_dir={run_output.resolve()}",
        f"--policy.device={options.device}",
        "--policy.pretrained_backbone_weights=null",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--wandb.mode=disabled",
        "--num_workers=0",
        "--eval_freq=0",
        "--save_checkpoint=true",
        f"--steps={options.steps}",
        f"--batch_size={options.batch_size}",
        f"--seed={options.seed}",
        f"--save_freq={options.steps}",
        f"--log_freq={min(100, options.steps)}",
        f"--policy.chunk_size={options.chunk_size}",
        f"--policy.n_action_steps={options.n_action_steps}",
    ]
    if options.smoke_test:
        command.extend(
            [
                "--policy.dim_model=32",
                "--policy.n_heads=4",
                "--policy.dim_feedforward=64",
                "--policy.n_encoder_layers=1",
                "--policy.n_decoder_layers=1",
                "--policy.n_vae_encoder_layers=1",
                "--policy.latent_dim=8",
            ]
        )
    return command


def train_policy(
    dataset: Path,
    output: Path,
    *,
    expected_scope: Scope,
    expected_conversion_sha256: str,
    model_name: str,
    model_version: str,
    code_snapshot_sha256: str,
    options: TrainOptions,
) -> dict:
    options.validate()
    sha256(code_snapshot_sha256, "code snapshot")
    token(model_name, "model name")
    token(model_version, "model version")
    require(
        file_digest(dataset / "conversion.json") == sha256(expected_conversion_sha256),
        "Conversion provenance checksum mismatch",
    )
    conversion = validate_conversion(dataset, expected_scope)
    require(
        options.smoke_test or not conversion["test_only"], "Fixture data cannot train a candidate"
    )
    require(not output.exists() or not any(output.iterdir()), "Model output is not empty")
    require_lerobot()
    import torch

    if not options.smoke_test:
        require(
            options.device == "cuda"
            and torch.cuda.is_available()
            and bool(os.getenv("AZUREML_RUN_ID")),
            "Production training requires an Azure ML GPU job",
        )
    if options.device == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but no GPU is available")
    output.mkdir(parents=True, exist_ok=True)
    run_output = output / "training"
    command = training_command(dataset, run_output, options)
    env = {**os.environ, **OFFLINE_ENV, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    with (output / "training.log").open("xb") as log:
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=options.timeout_seconds,
        )
    checkpoint = run_output / "checkpoints" / f"{options.steps:06d}" / "pretrained_model"
    # LeRobot widens checkpoint directory names for runs exceeding six digits.
    if options.steps >= 1000000:
        checkpoint = run_output / "checkpoints" / str(options.steps) / "pretrained_model"
    require(checkpoint.is_dir(), "LeRobot did not produce the expected trained checkpoint")
    candidate = output / "checkpoint"
    shutil.copytree(checkpoint, candidate)
    manifest = {
        "schema": MODEL_SCHEMA,
        "policy_type": "act",
        "lerobot_version": LEROBOT_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "scope": asdict(expected_scope),
        "raw_manifest_sha256": conversion["raw_manifest_sha256"],
        "conversion_sha256": expected_conversion_sha256,
        "code_snapshot_sha256": code_snapshot_sha256,
        "training": {
            **asdict(options),
            "azureml_run_id": os.getenv("AZUREML_RUN_ID"),
            "gpu_model": torch.cuda.get_device_name(0) if options.device == "cuda" else None,
            "test_only": options.smoke_test or conversion["test_only"],
            "episodes": [
                {
                    "episode_id": ep["episode_id"],
                    "environment_id": ep["environment_id"],
                    "revision": ep["revision"],
                    "seed": ep["seed"],
                }
                for ep in conversion["episodes"]
            ],
        },
        "observation": {
            "joint_names": list(JOINT_NAMES),
            "joint_units": list(JOINT_UNITS),
            "cameras": list(CAMERAS),
            "image_size": conversion["image_size"],
            "image_transform": conversion["image_transform"],
            "fps": conversion["fps"],
            "physics_hz": conversion["physics_hz"],
        },
        "checkpoint_files": inventory(candidate),
        "training_log_sha256": digest((output / "training.log").read_bytes()),
    }
    write_json(output / "model.json", manifest)
    return manifest
