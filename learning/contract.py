from __future__ import annotations

import re
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from learning import CONTRACT_VERSION
from learning.common import (
    ContractError,
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
    utc,
    vector,
)

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1",
    "panda_finger_joint2",
)
JOINT_UNITS = ("rad",) * 7 + ("m",) * 2
DEFAULT_JOINT_VELOCITY_LIMITS = (0.5,) * 7 + (0.04, 0.04)
JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0)
JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04)
CAMERAS = ("inspection", "overview")
SPLITS = ("train", "validation", "test")
MAX_PNG_BYTES = 32 * 1024 * 1024
TEACHING_CONTRACT_VERSION = "physicalai.demonstrations/v2"
FRAME_KEYS = {
    "frame_index",
    "captured_at_utc",
    "monotonic_ns",
    "physics_step",
    "joint_positions",
    "commanded_joint_targets",
    "images",
    "terminated",
    "truncated",
}
EPISODE_KEYS = {
    "episode_id",
    "environment_id",
    "revision",
    "seed",
    "split",
    "provenance",
    "path",
    "sha256",
    "frame_count",
}


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    owner_id: str

    def validate(self) -> None:
        try:
            require(str(UUID(self.tenant_id)) == self.tenant_id, "Invalid tenant UUID")
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError("Invalid tenant UUID") from exc
        sha256(self.owner_id, "opaque owner key")


@dataclass(frozen=True)
class Provenance:
    source_kind: str
    simulator_version: str
    simulator_image_digest: str
    robot_asset_sha256: str
    scene_builder_id: str
    scene_builder_sha256: str
    code_revision: str
    capture_host: str
    gpu_model: str
    render_backend: str = "rtx"

    def validate(self, *, require_live: bool = False) -> None:
        require(self.source_kind in ("isaac_sim", "test_fixture"), "Unknown capture source")
        require(
            self.simulator_version in ("5.1.0", "6.0.0"),
            "Only pinned Isaac Sim 5.1.0 or 6.0.0 captures are supported",
        )
        require(
            isinstance(self.simulator_image_digest, str)
            and self.simulator_image_digest.startswith("sha256:"),
            "Isaac image must be digest-pinned",
        )
        sha256(self.simulator_image_digest[7:], "Isaac image digest")
        sha256(self.robot_asset_sha256, "robot asset checksum")
        sha256(self.scene_builder_sha256, "reviewed scene builder checksum")
        token(self.scene_builder_id, "scene builder ID")
        require(
            isinstance(self.code_revision, str)
            and re.fullmatch(r"[0-9a-f]{40}", self.code_revision),
            "Expected a full source commit",
        )
        require(self.render_backend == "rtx", "Only real RTX camera captures are supported")
        if self.source_kind == "isaac_sim":
            require(self.capture_host == "azure_gpu", "Production capture must run on Azure GPU")
            require(
                isinstance(self.gpu_model, str) and 0 < len(self.gpu_model) <= 128,
                "Actual GPU identity is required",
            )
        else:
            require(
                self.capture_host == "test_cpu" and self.gpu_model == "none",
                "Fixture provenance must be explicitly test-only",
            )
        if require_live:
            require(self.source_kind == "isaac_sim", "Test fixtures are not live learning evidence")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    environment_id: str
    revision: str
    seed: int
    split: str

    def validate(self) -> None:
        token(self.episode_id, "episode ID")
        token(self.environment_id, "environment ID")
        sha256(self.revision, "environment revision")
        integer(self.seed, "scene seed", high=2**32 - 1)
        require(self.split in SPLITS, "Invalid episode split")


@dataclass(frozen=True)
class CameraSample:
    png: bytes
    rendering_frame: int
    physics_step: int
    monotonic_ns: int


@dataclass(frozen=True)
class ControlProfile:
    servo_profile_sha256: str
    profile_id: str = "franka-position-hold-10hz-v1"
    control_hz: int = 10
    physics_hz: int = 60
    hold_steps: int = 6
    velocity_target_mode: str = "zero"
    gravity_compensation: str = "physx_measured_arm_only"

    def validate(self) -> None:
        sha256(self.servo_profile_sha256, "reviewed servo profile")
        require(
            self.profile_id == "franka-position-hold-10hz-v1"
            and self.velocity_target_mode == "zero"
            and self.gravity_compensation == "physx_measured_arm_only",
            "Unsupported teaching servo semantics",
        )
        for value, expected in ((self.control_hz, 10), (self.physics_hz, 60), (self.hold_steps, 6)):
            require(type(value) is int and value == expected, "Teaching requires 10Hz/60Hz hold6")

    @property
    def sha256(self) -> str:
        self.validate()
        return digest(canonical(asdict(self)))


@dataclass(frozen=True)
class DemonstrationSource:
    kind: str
    task_id: str
    instruction: str
    goal_id: str
    source_policy_sha256: str | None = None

    def validate(self) -> None:
        require(
            self.kind in ("human_teleop", "reference_controller", "learned"),
            "Unknown demonstration source",
        )
        token(self.task_id, "approved task")
        token(self.goal_id, "approved goal")
        require(
            isinstance(self.instruction, str)
            and 1 <= len(self.instruction.strip()) <= 512
            and all(ord(character) >= 32 for character in self.instruction),
            "Expected a bounded approved task instruction",
        )
        if self.kind == "learned":
            sha256(self.source_policy_sha256, "demonstrator checkpoint")
        else:
            require(self.source_policy_sha256 is None, "Non-policy demonstration has a checkpoint")


@dataclass(frozen=True)
class AppliedControl:
    physics_step: int
    monotonic_ns: int
    commanded_joint_targets: tuple[float, ...]
    commanded_joint_velocities: tuple[float, ...]
    gravity_efforts: tuple[float, ...]


@dataclass(frozen=True)
class FrameSample:
    captured_at_utc: str
    monotonic_ns: int
    physics_step: int
    joint_positions: tuple[float, ...]
    commanded_joint_targets: tuple[float, ...]
    images: dict[str, CameraSample]
    terminated: bool = False
    truncated: bool = False
    applied_controls: tuple[AppliedControl, ...] | None = None


@dataclass(frozen=True)
class ValidatedEpisode:
    metadata: dict
    frames: tuple[dict, ...]


@dataclass(frozen=True)
class ValidatedDataset:
    root: Path
    manifest: dict
    manifest_sha256: str
    episodes: tuple[ValidatedEpisode, ...]

    def split(self, name: str) -> tuple[ValidatedEpisode, ...]:
        require(name in SPLITS, "Invalid split")
        selected = tuple(ep for ep in self.episodes if ep.metadata["split"] == name)
        require(bool(selected), f"Empty {name} split")
        return selected


def timing(fps: object, physics_hz: object) -> int:
    integer(fps, "fps", 1, 60)
    require(physics_hz == 60 and type(physics_hz) is int, "Reference physics rate is 60 Hz")
    require(physics_hz % fps == 0, "Control rate must divide the physics rate")
    return physics_hz // fps


def bounded_joints(values: object, label: str) -> tuple[float, ...]:
    joints = vector(values, 9, label)
    for name, value, low, high in zip(JOINT_NAMES, joints, JOINT_LOWER, JOINT_UPPER, strict=True):
        require(low - 1e-6 <= value <= high + 1e-6, f"{label}: {name} outside reference limits")
    return joints


def validate_joint_tracking(
    targets: object,
    measured: object,
    previous: object | None,
    *,
    fps: int,
    velocity_limits: tuple[float, ...] = DEFAULT_JOINT_VELOCITY_LIMITS,
) -> None:
    requested = bounded_joints(targets, "commanded targets")
    limits = vector(velocity_limits, 9, "joint velocity limits")
    for label, reference in (("tracking", measured), ("slew", previous)):
        if reference is not None:
            for target, actual, limit in zip(
                requested,
                bounded_joints(reference, label),
                limits,
                strict=True,
            ):
                require(
                    abs(target - actual) <= limit / fps + 1e-8,
                    f"Target exceeds the declared {fps}Hz joint {label} limit",
                )


def png_dimensions(data: bytes) -> tuple[int, int]:
    require(isinstance(data, bytes) and len(data) <= MAX_PNG_BYTES, "Invalid/oversized PNG")
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "Camera capture is not a PNG")
    cursor, width, height = 8, 0, 0
    compressed = bytearray()
    ended = False
    while cursor < len(data):
        require(cursor + 12 <= len(data), "Truncated PNG")
        length = struct.unpack_from(">I", data, cursor)[0]
        kind = data[cursor + 4 : cursor + 8]
        require(cursor + length + 12 <= len(data), "Truncated PNG chunk")
        payload = data[cursor + 8 : cursor + 8 + length]
        checksum = struct.unpack_from(">I", data, cursor + length + 8)[0]
        require(zlib.crc32(kind + payload) & 0xFFFFFFFF == checksum, "PNG CRC mismatch")
        if cursor == 8:
            require(kind == b"IHDR" and length == 13, "Missing PNG header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            require(
                1 <= width <= 4096
                and 1 <= height <= 4096
                and (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0),
                "Expected bounded, noninterlaced 8-bit RGB PNG",
            )
        elif kind == b"IHDR":
            raise ContractError("Duplicate PNG header")
        if kind == b"IDAT":
            compressed.extend(payload)
        cursor += length + 12
        if kind == b"IEND":
            require(length == 0 and cursor == len(data), "PNG has trailing data")
            ended = True
            break
    require(ended and bool(compressed), "Incomplete PNG")
    expected = height * (1 + width * 3)
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise ContractError("Corrupt PNG pixels") from exc
    require(
        len(pixels) == expected and decoder.eof and not decoder.unused_data,
        "PNG pixel size mismatch",
    )
    require(all(pixels[row * (1 + width * 3)] <= 4 for row in range(height)), "Invalid PNG filter")
    return width, height


def validate_frame(
    frame: dict,
    previous: dict | None,
    *,
    index: int,
    control_interval_steps: int,
    control_profile: ControlProfile | None = None,
) -> None:
    keys(frame, FRAME_KEYS | ({"applied_controls"} if control_profile else set()), "frame")
    require(integer(frame["frame_index"], "frame index") == index, "Nonsequential frame index")
    captured = utc(frame["captured_at_utc"])
    integer(frame["monotonic_ns"], "monotonic timestamp", 1)
    integer(frame["physics_step"], "physics step")
    bounded_joints(frame["joint_positions"], "measured joints")
    bounded_joints(frame["commanded_joint_targets"], "issued joint targets")
    if control_profile is not None:
        control_profile.validate()
        validate_joint_tracking(
            frame["commanded_joint_targets"],
            frame["joint_positions"],
            previous["commanded_joint_targets"] if previous is not None else None,
            fps=control_profile.control_hz,
        )
        controls = frame["applied_controls"]
        require(
            isinstance(controls, list) and len(controls) == control_profile.hold_steps,
            "Incomplete actual hold interval",
        )
        timestamp = frame["monotonic_ns"]
        for offset, control in enumerate(controls, start=1):
            keys(control, set(AppliedControl.__dataclass_fields__), "applied control")
            require(
                integer(control["physics_step"], "applied physics step")
                == frame["physics_step"] + offset,
                "Missing/duplicate actual physics tick",
            )
            require(
                integer(control["monotonic_ns"], "applied timestamp", 1) > timestamp,
                "Control must follow its observation with strictly increasing timestamps",
            )
            timestamp = control["monotonic_ns"]
            require(
                bounded_joints(control["commanded_joint_targets"], "held targets")
                == tuple(frame["commanded_joint_targets"]),
                "Intervening changed target; downsampling is forbidden",
            )
            require(
                vector(control["commanded_joint_velocities"], 9, "issued velocities") == (0.0,) * 9,
                "Teaching profile requires actual zero velocity targets",
            )
            gravity = vector(control["gravity_efforts"], 9, "measured gravity efforts")
            require(gravity[7:] == (0.0, 0.0), "Gravity compensation must be arm-only")
        if previous is not None:
            require(
                frame["monotonic_ns"] >= previous["applied_controls"][-1]["monotonic_ns"],
                "Next observation precedes completion of the previous hold",
            )
    require(type(frame["terminated"]) is bool and type(frame["truncated"]) is bool, "Invalid done")
    require(not (frame["terminated"] and frame["truncated"]), "Ambiguous episode end")
    if previous is not None:
        require(not (previous["terminated"] or previous["truncated"]), "Data after episode end")
        require(frame["monotonic_ns"] > previous["monotonic_ns"], "Nonmonotonic frame timestamp")
        require(captured > utc(previous["captured_at_utc"]), "Nonmonotonic UTC timestamp")
        require(
            frame["physics_step"] - previous["physics_step"] == control_interval_steps,
            "Capture gap/control cadence mismatch; do not invent/resample actions",
        )
    images = keys(frame["images"], set(CAMERAS), "images")
    for camera, image in images.items():
        keys(
            image,
            {
                "path",
                "sha256",
                "width",
                "height",
                "rendering_frame",
                "physics_step",
                "monotonic_ns",
            },
            f"{camera} image",
        )
        sha256(image["sha256"])
        integer(image["width"], "image width", 1, 4096)
        integer(image["height"], "image height", 1, 4096)
        integer(image["rendering_frame"], "render frame")
        integer(image["physics_step"], "camera physics step")
        integer(image["monotonic_ns"], "camera monotonic timestamp", 1)
        require(
            0 <= frame["physics_step"] - image["physics_step"] < control_interval_steps,
            "Stale/future camera physics frame",
        )
        require(
            0 <= frame["monotonic_ns"] - image["monotonic_ns"] <= 1_000_000_000,
            "Stale/future camera timestamp",
        )
        if previous is not None:
            old = previous["images"][camera]
            require(
                image["rendering_frame"] > old["rendering_frame"]
                and image["monotonic_ns"] > old["monotonic_ns"]
                and image["physics_step"] > old["physics_step"],
                "Nonmonotonic or repeated camera frame",
            )


def validate_dataset(
    root: Path,
    *,
    expected_scope: Scope,
    require_live: bool = False,
    expected_manifest_sha256: str | None = None,
) -> ValidatedDataset:
    expected_scope.validate()
    path = safe_path(root, "manifest.json")
    checksum = file_digest(path)
    if expected_manifest_sha256 is not None:
        require(checksum == sha256(expected_manifest_sha256), "Manifest checksum mismatch")
    manifest = read_json(path)
    is_teaching = manifest.get("schema") == TEACHING_CONTRACT_VERSION
    keys(
        manifest,
        {
            "schema",
            "dataset_id",
            "scope",
            "fps",
            "physics_hz",
            "joint_names",
            "joint_units",
            "episodes",
        }
        | ({"control_profile"} if is_teaching else set()),
        "manifest",
    )
    require(
        manifest["schema"] in (CONTRACT_VERSION, TEACHING_CONTRACT_VERSION),
        "Unsupported demonstration schema",
    )
    token(manifest["dataset_id"], "dataset ID")
    require(manifest["scope"] == asdict(expected_scope), "Tenant/owner scope mismatch")
    interval = timing(manifest["fps"], manifest["physics_hz"])
    profile = None
    if is_teaching:
        profile = ControlProfile(
            **keys(
                manifest["control_profile"],
                set(ControlProfile.__dataclass_fields__),
                "control profile",
            )
        )
        profile.validate()
        require(
            manifest["fps"] == profile.control_hz and manifest["physics_hz"] == profile.physics_hz,
            "Capture rate differs from the teaching control profile",
        )
    require(manifest["joint_names"] == list(JOINT_NAMES), "Wrong joint ordering")
    require(manifest["joint_units"] == list(JOINT_UNITS), "Wrong joint units")
    require(
        isinstance(manifest["episodes"], list) and 1 <= len(manifest["episodes"]) <= 10000,
        "Empty or oversized dataset",
    )
    episode_ids, paths, seed_splits = set(), {"manifest.json"}, {}
    verified = []
    image_size = None
    for metadata in manifest["episodes"]:
        keys(metadata, EPISODE_KEYS | ({"demonstration"} if is_teaching else set()), "episode")
        if is_teaching:
            DemonstrationSource(
                **keys(
                    metadata["demonstration"],
                    set(DemonstrationSource.__dataclass_fields__),
                    "demonstration source",
                )
            ).validate()
        spec = EpisodeSpec(**{name: metadata[name] for name in EpisodeSpec.__dataclass_fields__})
        spec.validate()
        provenance = keys(
            metadata["provenance"], set(Provenance.__dataclass_fields__), "provenance"
        )
        Provenance(**provenance).validate(require_live=require_live)
        require(spec.episode_id not in episode_ids, "Duplicate/cross-split episode leakage")
        episode_ids.add(spec.episode_id)
        require(
            spec.seed not in seed_splits or seed_splits[spec.seed] == spec.split,
            "Cross-split seed leakage",
        )
        seed_splits[spec.seed] = spec.split
        require(metadata["path"] not in paths, "Reused episode path")
        require(
            metadata["path"] == f"episodes/{spec.episode_id}/frames.jsonl",
            "Unexpected episode path",
        )
        episode_path = safe_path(root, metadata["path"])
        paths.add(metadata["path"])
        require(
            file_digest(episode_path) == sha256(metadata["sha256"]), "Episode checksum mismatch"
        )
        count = integer(metadata["frame_count"], "frame count", 2, 100000)
        require(episode_path.stat().st_size <= count * 16384, "Oversized frame stream")
        frames = []
        with episode_path.open("rb") as stream:
            for index, line in enumerate(stream):
                require(index < count and len(line) <= 16384, "Frame count/size limit exceeded")
                frame = parse_json(line)
                validate_frame(
                    frame,
                    frames[-1] if frames else None,
                    index=index,
                    control_interval_steps=interval,
                    control_profile=profile,
                )
                for camera, image in frame["images"].items():
                    expected_path = f"episodes/{spec.episode_id}/{camera}/{index:08d}.png"
                    require(image["path"] == expected_path, "Unexpected/cross-episode image path")
                    image_path = safe_path(root, image["path"])
                    require(image["path"] not in paths, "Reused image path")
                    paths.add(image["path"])
                    require(image_path.stat().st_size <= MAX_PNG_BYTES, "Oversized PNG")
                    data = image_path.read_bytes()
                    require(digest(data) == image["sha256"], "Image checksum mismatch")
                    dimensions = png_dimensions(data)
                    require(dimensions == (image["width"], image["height"]), "Image shape mismatch")
                    image_size = image_size or dimensions
                    require(dimensions == image_size, "Camera shapes differ across dataset")
                frames.append(frame)
        require(len(frames) == count, "Missing frames")
        require(frames[-1]["terminated"] or frames[-1]["truncated"], "Unfinished episode")
        verified.append(ValidatedEpisode(metadata, tuple(frames)))
    actual = set()
    for artifact in root.rglob("*"):
        require(not artifact.is_symlink(), "Symlink in dataset")
        if artifact.is_file():
            actual.add(artifact.relative_to(root).as_posix())
    require(actual == paths, "Unexpected dataset artifacts; keep evaluator labels separate")
    return ValidatedDataset(root, manifest, checksum, tuple(verified))
