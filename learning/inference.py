from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from learning import LEROBOT_VERSION
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
    utc,
    vector,
    verify_inventory,
)
from learning.contract import (
    CAMERAS,
    JOINT_NAMES,
    JOINT_UNITS,
    CameraSample,
    Scope,
    bounded_joints,
    png_dimensions,
    timing,
)
from learning.offline import require_lerobot
from learning.train import MODEL_SCHEMA, TrainOptions

MODEL_KEYS = {
    "schema",
    "policy_type",
    "lerobot_version",
    "model_name",
    "model_version",
    "scope",
    "raw_manifest_sha256",
    "conversion_sha256",
    "code_snapshot_sha256",
    "training",
    "observation",
    "checkpoint_files",
    "training_log_sha256",
}


def validate_model_artifact(
    root: Path,
    *,
    expected_scope: Scope,
    expected_model_sha256: str,
    allow_test_model: bool = False,
) -> dict:
    expected_scope.validate()
    path = safe_path(root, "model.json")
    require(
        file_digest(path) == sha256(expected_model_sha256), "Model provenance checksum mismatch"
    )
    model = keys(read_json(path), MODEL_KEYS, "model provenance")
    require(model["schema"] == MODEL_SCHEMA and model["policy_type"] == "act", "Unsupported policy")
    require(model["lerobot_version"] == LEROBOT_VERSION, "Unsupported policy version")
    require(model["scope"] == asdict(expected_scope), "Tenant/owner model scope mismatch")
    token(model["model_name"], "model name")
    token(model["model_version"], "model version")
    for name in (
        "raw_manifest_sha256",
        "conversion_sha256",
        "code_snapshot_sha256",
        "training_log_sha256",
    ):
        sha256(model[name], name)
    observation = keys(
        model["observation"],
        {
            "joint_names",
            "joint_units",
            "cameras",
            "image_size",
            "image_transform",
            "fps",
            "physics_hz",
        },
        "model observation",
    )
    require(observation["joint_names"] == list(JOINT_NAMES), "Wrong model joint order")
    require(observation["joint_units"] == list(JOINT_UNITS), "Wrong model units")
    require(observation["cameras"] == list(CAMERAS), "Wrong model camera set")
    require(observation["image_transform"] == "rgb-bilinear-square/v1", "Wrong image transform")
    integer(observation["image_size"], "model image size", 32, 512)
    timing(observation["fps"], observation["physics_hz"])
    train = keys(
        model["training"],
        set(TrainOptions.__dataclass_fields__)
        | {"azureml_run_id", "gpu_model", "test_only", "episodes"},
        "training provenance",
    )
    TrainOptions(**{name: train[name] for name in TrainOptions.__dataclass_fields__}).validate()
    require(type(train["test_only"]) is bool, "Missing model test-only flag")
    if not allow_test_model:
        require(
            not train["test_only"]
            and not train["smoke_test"]
            and train["device"] == "cuda"
            and isinstance(train["azureml_run_id"], str)
            and bool(train["azureml_run_id"])
            and isinstance(train["gpu_model"], str)
            and bool(train["gpu_model"]),
            "Test/CPU/unattributed model is not a production checkpoint",
        )
    require(
        isinstance(train["episodes"], list) and bool(train["episodes"]), "Missing training episodes"
    )
    ids = set()
    for episode in train["episodes"]:
        keys(episode, {"episode_id", "environment_id", "revision", "seed"}, "training episode")
        token(episode["episode_id"], "training episode")
        token(episode["environment_id"], "training environment")
        sha256(episode["revision"])
        integer(episode["seed"], "training scene seed", high=2**32 - 1)
        require(episode["episode_id"] not in ids, "Duplicate training episode provenance")
        ids.add(episode["episode_id"])
    required_files = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    require(
        isinstance(model["checkpoint_files"], dict)
        and required_files.issubset(model["checkpoint_files"]),
        "Missing policy weights/config/normalization processors",
    )
    for name in model["checkpoint_files"]:
        require(
            "/" not in name
            and (name in required_files | {"train_config.json"} or name.endswith(".safetensors")),
            "Unexpected executable/pickle checkpoint content",
        )
    verify_inventory(root / "checkpoint", model["checkpoint_files"])
    return model


@dataclass(frozen=True)
class PolicyObservation:
    scope: Scope
    environment_id: str
    revision: str
    episode_id: str
    captured_at_utc: str
    monotonic_ns: int
    physics_step: int
    joint_positions: tuple[float, ...]
    images: dict[str, CameraSample]


@dataclass(frozen=True)
class ControlContext:
    scope: Scope
    environment_id: str
    revision: str
    episode_id: str
    epoch: str
    command_id: str
    destination_id: str
    approved: bool
    active: bool
    deadline_monotonic_ns: int

    def binding(self) -> tuple:
        return (
            self.scope,
            self.environment_id,
            self.revision,
            self.episode_id,
            self.epoch,
            self.command_id,
            self.destination_id,
        )


@dataclass(frozen=True)
class SafetyLimits:
    max_joint_velocity: tuple[float, ...] = (0.5,) * 7 + (0.04, 0.04)
    max_observation_age_ms: float = 200.0
    max_inference_latency_ms: float = 80.0
    max_chunk_steps: int = 1

    def validate(self, fps: int) -> None:
        velocities = vector(self.max_joint_velocity, 9, "joint velocity limits")
        for limit, ceiling in zip(velocities, (2.0,) * 7 + (0.1, 0.1), strict=True):
            require(0 < limit <= ceiling, "Velocity limit exceeds reference controller ceiling")
        require(
            0 < finite(self.max_observation_age_ms, "observation age") <= 1000,
            "Invalid observation freshness budget",
        )
        require(
            0 < finite(self.max_inference_latency_ms, "latency budget") < 1000 / fps,
            "Inference budget must fit inside one control interval",
        )
        integer(self.max_chunk_steps, "max chunk steps", 1, min(10, fps))


@dataclass(frozen=True)
class JointCommand:
    targets: tuple[float, ...]
    physics_step: int
    hold_steps: int
    expires_at_monotonic_ns: int
    model_sha256: str
    inference_latency_ms: float


class ChunkPolicy(Protocol):
    fps: int
    physics_hz: int
    chunk_size: int
    n_action_steps: int
    model_sha256: str
    scope: Scope

    def predict_chunk(self, observation: PolicyObservation) -> Sequence[Sequence[float]]: ...

    def reset(self) -> None: ...


class LocalACTPolicy:
    """Only local, checksum-pinned ACT safetensors with saved normalization statistics."""

    def __init__(
        self,
        root: Path,
        *,
        expected_scope: Scope,
        expected_model_sha256: str,
        device: str = "cuda",
        allow_test_model: bool = False,
    ) -> None:
        self.metadata = validate_model_artifact(
            root,
            expected_scope=expected_scope,
            expected_model_sha256=expected_model_sha256,
            allow_test_model=allow_test_model,
        )
        require(
            device == "cuda" or (device == "cpu" and allow_test_model), "CPU inference is test-only"
        )
        checkpoint = root / "checkpoint"
        config_json = read_json(checkpoint / "config.json")
        require(config_json.get("type") == "act", "Checkpoint config is not ACT")
        require(
            config_json.get("pretrained_backbone_weights") is None
            and config_json.get("pretrained_path") is None
            and not config_json.get("push_to_hub")
            and not config_json.get("use_peft", False),
            "Checkpoint requests remote/pretrained or publishing behavior",
        )
        require(
            config_json.get("temporal_ensemble_coeff") is None, "Ensemble timing is not supported"
        )
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor = read_json(checkpoint / filename)
            require(isinstance(processor.get("steps"), list), "Missing saved processor steps")
            for step in processor["steps"]:
                require("class" not in step, "Dynamically imported processor classes are forbidden")
                require(
                    step.get("registry_name")
                    in {
                        "rename_observations_processor",
                        "to_batch_processor",
                        "device_processor",
                        "normalizer_processor",
                        "unnormalizer_processor",
                    },
                    "Unreviewed checkpoint processor",
                )
                if step.get("state_file"):
                    safe_path(checkpoint, step["state_file"])
        require_lerobot()
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        require(device != "cuda" or torch.cuda.is_available(), "CUDA inference needs a real GPU")
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        require(isinstance(config, ACTConfig), "Loaded checkpoint is not ACT")
        config.device = device
        size = self.metadata["observation"]["image_size"]
        expected_inputs = {"observation.state", *(f"observation.images.{name}" for name in CAMERAS)}
        require(
            set(config.input_features) == expected_inputs, "Forbidden/missing policy observation"
        )
        require(
            tuple(config.input_features["observation.state"].shape) == (9,), "Wrong state shape"
        )
        for camera in CAMERAS:
            require(
                tuple(config.input_features[f"observation.images.{camera}"].shape)
                == (3, size, size),
                "Wrong camera tensor shape",
            )
        require(
            set(config.output_features) == {"action"}
            and tuple(config.output_features["action"].shape) == (9,),
            "Wrong actuator output dimensions",
        )
        require(
            config.chunk_size == self.metadata["training"]["chunk_size"]
            and config.n_action_steps == self.metadata["training"]["n_action_steps"],
            "Checkpoint action timing differs from provenance",
        )
        self.policy = ACTPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=True
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.fps = self.metadata["observation"]["fps"]
        self.physics_hz = self.metadata["observation"]["physics_hz"]
        self.chunk_size, self.n_action_steps = config.chunk_size, config.n_action_steps
        self.model_sha256, self.scope = expected_model_sha256, expected_scope
        self.image_size, self.device = size, device

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict_chunk(self, observation: PolicyObservation) -> tuple[tuple[float, ...], ...]:
        import numpy as np
        import torch
        from PIL import Image

        inputs = {
            "observation.state": torch.tensor(observation.joint_positions, dtype=torch.float32)
        }
        for camera in CAMERAS:
            with Image.open(BytesIO(observation.images[camera].png)) as image:
                pixels = np.array(
                    image.convert("RGB").resize(
                        (self.image_size, self.image_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                inputs[f"observation.images.{camera}"] = (
                    torch.from_numpy(pixels).permute(2, 0, 1) / 255
                )
        with torch.inference_mode():
            batch = self.preprocessor(inputs)
            actions = self.postprocessor(self.policy.predict_action_chunk(batch))
        require(
            tuple(actions.shape) == (1, self.chunk_size, 9), "ACT returned wrong action chunk shape"
        )
        return tuple(tuple(action) for action in actions.detach().cpu()[0].tolist())


class GuardedPolicyAdapter:
    """Returns bounded setpoints, never actuates a robot or grants simulator approval."""

    def __init__(
        self,
        policy: ChunkPolicy,
        *,
        limits: SafetyLimits | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.policy, self.limits, self.clock_ns = policy, limits or SafetyLimits(), clock_ns
        self.interval_steps = timing(policy.fps, policy.physics_hz)
        self.interval_ns = round(1_000_000_000 / policy.fps)
        self.limits.validate(policy.fps)
        integer(policy.chunk_size, "checkpoint chunk size", 1, 100)
        integer(policy.n_action_steps, "checkpoint action steps", 1, policy.chunk_size)
        require(policy.n_action_steps <= self.limits.max_chunk_steps, "Unapproved chunk horizon")
        sha256(policy.model_sha256, "checkpoint digest")
        policy.scope.validate()
        self.queue: deque[tuple[float, ...]] = deque()
        self.context: ControlContext | None = None
        self.previous: PolicyObservation | None = None
        self.previous_target: tuple[float, ...] | None = None
        self.chunk_expires_ns = 0
        self.faulted = True

    def _check_context(self, context: ControlContext, now: int) -> None:
        require(
            context.approved is True and context.active is True, "Controller approval is not active"
        )
        require(context.scope == self.policy.scope, "Tenant/owner policy scope mismatch")
        token(context.environment_id, "environment ID")
        sha256(context.revision, "environment revision")
        for value in (
            context.episode_id,
            context.epoch,
            context.command_id,
            context.destination_id,
        ):
            token(value, "control binding")
        integer(context.deadline_monotonic_ns, "command deadline", 1)
        require(now < context.deadline_monotonic_ns, "Command deadline expired")

    def reset(self, context: ControlContext) -> None:
        self.faulted = True
        self.queue.clear()
        self.context, self.previous, self.previous_target = None, None, None
        self._check_context(context, self.clock_ns())
        self.policy.reset()
        self.context = context
        self.faulted = False

    def stop(self) -> None:
        self.faulted = True
        self.queue.clear()
        self.context = None
        self.policy.reset()

    def step(self, observation: PolicyObservation, context: ControlContext) -> JointCommand:
        require(not self.faulted and self.context is not None, "Policy needs an approved reset")
        self.faulted = True
        start = self.clock_ns()
        self._check_context(context, start)
        require(context.binding() == self.context.binding(), "Scene/command binding changed")
        require(
            (
                observation.scope,
                observation.environment_id,
                observation.revision,
                observation.episode_id,
            )
            == (context.scope, context.environment_id, context.revision, context.episode_id),
            "Observation does not belong to the approved command",
        )
        joints = bounded_joints(observation.joint_positions, "observed joints")
        utc(observation.captured_at_utc)
        integer(observation.monotonic_ns, "observation timestamp", 1)
        integer(observation.physics_step, "observation physics step")
        require(
            0 <= start - observation.monotonic_ns <= self.limits.max_observation_age_ms * 1e6,
            "Stale/future policy observation",
        )
        require(set(observation.images) == set(CAMERAS), "Both policy cameras are required")
        for camera, sample in observation.images.items():
            png_dimensions(sample.png)
            integer(sample.rendering_frame, "render frame")
            integer(sample.physics_step, "camera physics step")
            integer(sample.monotonic_ns, "camera timestamp", 1)
            require(
                0 <= observation.physics_step - sample.physics_step < self.interval_steps
                and 0 <= start - sample.monotonic_ns <= self.limits.max_observation_age_ms * 1e6
                and sample.monotonic_ns <= observation.monotonic_ns,
                "Stale/future policy camera",
            )
            if self.previous is not None:
                old = self.previous.images[camera]
                require(
                    sample.rendering_frame > old.rendering_frame
                    and sample.monotonic_ns > old.monotonic_ns
                    and sample.physics_step > old.physics_step,
                    "Repeated policy camera frame",
                )
        if self.previous is not None:
            require(
                observation.physics_step - self.previous.physics_step == self.interval_steps
                and observation.monotonic_ns > self.previous.monotonic_ns
                and utc(observation.captured_at_utc) > utc(self.previous.captured_at_utc),
                "Skipped/repeated policy control tick; reset required",
            )
        if self.queue:
            require(start < self.chunk_expires_ns, "Queued action chunk expired")
        else:
            chunk = self.policy.predict_chunk(observation)
            require(len(chunk) == self.policy.chunk_size, "Wrong predicted chunk length")
            validated = [bounded_joints(action, "predicted action") for action in chunk]
            self.queue.extend(validated[: self.policy.n_action_steps])
            self.chunk_expires_ns = start + self.interval_ns * self.policy.n_action_steps
        target = self.queue.popleft()
        for reference in (joints, self.previous_target):
            if reference is not None:
                for actual, requested, limit in zip(
                    reference, target, self.limits.max_joint_velocity, strict=True
                ):
                    require(
                        abs(requested - actual) <= limit / self.policy.fps + 1e-8,
                        "Policy target exceeds per-tick joint slew/tracking limit",
                    )
        end = self.clock_ns()
        latency_ms = (end - start) / 1e6
        require(
            0 <= latency_ms <= self.limits.max_inference_latency_ms,
            "Policy inference exceeded the control latency budget",
        )
        self._check_context(context, end)
        require(
            end - observation.monotonic_ns <= self.limits.max_observation_age_ms * 1e6,
            "Observation expired during inference",
        )
        self.previous, self.previous_target = observation, target
        self.faulted = False
        return JointCommand(
            target,
            observation.physics_step,
            self.interval_steps,
            min(context.deadline_monotonic_ns, start + self.interval_ns),
            self.policy.model_sha256,
            latency_ms,
        )
