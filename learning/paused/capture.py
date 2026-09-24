from __future__ import annotations

import shutil
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    write_json,
)
from learning.contract import (
    CAMERAS,
    JOINT_NAMES,
    JOINT_UNITS,
    AppliedControl,
    DemonstrationSource,
    EpisodeSpec,
    Provenance,
    Scope,
    ValidatedDataset,
    ValidatedEpisode,
    verify_episode_images,
)
from learning.paused.contract import (
    EXECUTION_TIMING,
    RAW_SCHEMA,
    FrozenCameraSample,
    FrozenPolicyObservation,
    InitialFrozenPublication,
    PausedControlProfile,
    PausedFrameSample,
)

MAX_FRAME_BYTES = 32768
INTEGRATION_SEEDS = frozenset((900002, 900004))


@dataclass(frozen=True)
class PausedEpisodeBudget:
    started_ns: int
    wall_deadline_ns: int
    initial_physics_step: int
    simulation_step_deadline: int

    def validate(self, profile: PausedControlProfile) -> None:
        profile.validate()
        integer(self.started_ns, "original episode wall start", 1)
        integer(self.wall_deadline_ns, "original episode wall deadline", 1)
        integer(self.initial_physics_step, "initial actual physics step")
        integer(self.simulation_step_deadline, "absolute simulation step deadline")
        steps = self.simulation_step_deadline - self.initial_physics_step
        require(
            0 < self.wall_deadline_ns - self.started_ns <= profile.max_episode_wall_ms * 1_000_000
            and 0 < steps <= profile.max_simulation_steps
            and steps % profile.hold_steps == 0,
            "Original paused episode wall/simulation budget is invalid",
        )


def _frame_binding(
    value: PausedFrameSample,
    profile: PausedControlProfile,
    budget: PausedEpisodeBudget,
    episode: EpisodeSpec,
    scope: Scope,
    source: DemonstrationSource,
    index: int,
) -> None:
    observation = value.observation
    require(
        (
            observation.scope,
            observation.environment_id,
            observation.revision,
            observation.episode_id,
        )
        == (scope, episode.environment_id, episode.revision, episode.episode_id),
        "Captured frame escaped its owner/environment/episode scope",
    )
    require(
        observation.control_tick == index
        and observation.physics_step == budget.initial_physics_step + index * profile.hold_steps
        and observation.physics_step + profile.hold_steps <= budget.simulation_step_deadline
        and observation.observation_started_ns >= budget.started_ns
        and value.interval_deadline_ns <= budget.wall_deadline_ns,
        "Capture exceeded the original episode wall/simulation budget or actual cadence",
    )
    require(
        (source.kind == "learned") == (value.policy_started_ns is not None),
        "Only actual learned execution carries measured policy timings",
    )


def _frame_record(sample: PausedFrameSample, episode_id: str, index: int) -> dict:
    result = {
        **sample.observation.metadata(),
        "frame_index": index,
        "observation_sha256": sample.observation.sha256,
        **{
            name: getattr(sample, name)
            for name in PausedFrameSample.__dataclass_fields__
            if name not in ("observation", "applied_controls")
        },
        "applied_controls": [asdict(control) for control in sample.applied_controls],
    }
    for name, image in result["images"].items():
        image["path"] = f"episodes/{episode_id}/{name}/{index:08d}.png"
    return result


def _decode_frame(frame: dict, payloads: dict[str, bytes], index: int) -> PausedFrameSample:
    observation_fields = set(FrozenPolicyObservation.__dataclass_fields__)
    control_fields = set(PausedFrameSample.__dataclass_fields__) - {"observation"}
    keys(
        frame,
        observation_fields | control_fields | {"frame_index", "observation_sha256"},
        "v3 frame",
    )
    require(integer(frame["frame_index"], "frame index") == index, "Nonsequential frame index")
    cameras = {}
    for name, image in keys(frame["images"], set(CAMERAS), "frozen cameras").items():
        camera_fields = set(FrozenCameraSample.__dataclass_fields__) - {"png"}
        keys(image, camera_fields | {"path", "sha256", "width", "height"}, "v3 camera")
        cameras[name] = FrozenCameraSample(
            png=payloads[name], **{key: image[key] for key in camera_fields}
        )
    observation = FrozenPolicyObservation(
        **{
            key: frame[key]
            for key in observation_fields - {"scope", "images", "initial_publication"}
        },
        scope=Scope(**keys(frame["scope"], {"tenant_id", "owner_id"}, "frame scope")),
        images=cameras,
        initial_publication=(
            InitialFrozenPublication(
                **keys(
                    frame["initial_publication"],
                    set(InitialFrozenPublication.__dataclass_fields__),
                    "initial publication proof",
                )
            )
            if frame["initial_publication"] is not None
            else None
        ),
    )
    require(
        observation.sha256 == sha256(frame["observation_sha256"]),
        "Frozen observation checksum mismatch",
    )
    require(isinstance(frame["applied_controls"], list), "Expected actual held controls")
    controls = tuple(
        AppliedControl(**keys(value, set(AppliedControl.__dataclass_fields__), "actual control"))
        for value in frame["applied_controls"]
    )
    return PausedFrameSample(
        observation=observation,
        applied_controls=controls,
        **{key: frame[key] for key in control_fields - {"applied_controls"}},
    )


def _validate(
    root: Path,
    manifest: dict,
    *,
    expected_scope: Scope,
    require_live: bool,
    require_demonstrations: bool,
    manifest_present: bool,
) -> ValidatedDataset:
    expected_scope.validate()
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
            "timestamp_basis",
            "execution_timing",
            "real_time_admission",
            "control_profile",
            "criteria_sha256",
            "frozen_plan_sha256",
            "purpose",
            "episodes",
        },
        "paused dataset manifest",
    )
    require(
        manifest["schema"] == RAW_SCHEMA
        and manifest["execution_timing"] == EXECUTION_TIMING
        and manifest["real_time_admission"] is False
        and manifest["timestamp_basis"] == "simulation_time",
        "Raw v3 requires explicit paused simulation; no old capture relabel",
    )
    token(manifest["dataset_id"], "dataset ID")
    require(manifest["scope"] == asdict(expected_scope), "Dataset owner/tenant scope mismatch")
    profile = PausedControlProfile(
        **keys(
            manifest["control_profile"], set(PausedControlProfile.__dataclass_fields__), "profile"
        )
    )
    profile.validate()
    require(
        type(manifest["fps"]) is int
        and manifest["fps"] == profile.control_sim_hz
        and type(manifest["physics_hz"]) is int
        and manifest["physics_hz"] == profile.physics_hz
        and manifest["joint_names"] == list(JOINT_NAMES)
        and manifest["joint_units"] == list(JOINT_UNITS),
        "Wrong actual joint mapping or simulation-time rate",
    )
    sha256(manifest["criteria_sha256"], "frozen criteria")
    require(
        manifest["purpose"] in ("integration", "demonstration", "evaluation"),
        "Explicit capture purpose is required",
    )
    if manifest["purpose"] != "integration":
        sha256(manifest["frozen_plan_sha256"], "new frozen paused conditions plan")
    elif manifest["frozen_plan_sha256"] is not None:
        sha256(manifest["frozen_plan_sha256"], "frozen conditions plan")
    if require_demonstrations:
        require(manifest["purpose"] == "demonstration", "Only approved demonstrations can train")
    require(
        isinstance(manifest["episodes"], list) and 1 <= len(manifest["episodes"]) <= 10000,
        "Empty or oversized paused dataset",
    )
    paths = {"manifest.json"} if manifest_present else set()
    episode_ids, seed_splits, verified = set(), {}, []
    image_size = None
    for metadata in manifest["episodes"]:
        keys(
            metadata,
            set(EpisodeSpec.__dataclass_fields__)
            | {"provenance", "demonstration", "budget", "path", "sha256", "frame_count"},
            "paused episode",
        )
        episode = EpisodeSpec(**{key: metadata[key] for key in EpisodeSpec.__dataclass_fields__})
        episode.validate()
        source = DemonstrationSource(
            **keys(
                metadata["demonstration"], set(DemonstrationSource.__dataclass_fields__), "source"
            )
        )
        source.validate()
        Provenance(
            **keys(metadata["provenance"], set(Provenance.__dataclass_fields__), "provenance")
        ).validate(require_live=require_live)
        budget = PausedEpisodeBudget(
            **keys(
                metadata["budget"], set(PausedEpisodeBudget.__dataclass_fields__), "episode budget"
            )
        )
        budget.validate(profile)
        require(episode.episode_id not in episode_ids, "Duplicate/cross-split episode leakage")
        require(
            episode.seed not in seed_splits or seed_splits[episode.seed] == episode.split,
            "Cross-split seed leakage",
        )
        episode_ids.add(episode.episode_id)
        seed_splits[episode.seed] = episode.split
        if manifest["purpose"] == "integration":
            require(episode.split == "test", "Integration captures are explicitly TEST only")
        else:
            require(
                episode.seed not in INTEGRATION_SEEDS, "Integration seed cannot be learning data"
            )
        expected = f"episodes/{episode.episode_id}/frames.jsonl"
        require(metadata["path"] == expected and expected not in paths, "Unexpected episode path")
        path = safe_path(root, expected)
        paths.add(expected)
        require(file_digest(path) == sha256(metadata["sha256"]), "Episode checksum mismatch")
        count = integer(
            metadata["frame_count"], "frame count", 2, profile.max_simulation_steps // 6
        )
        require(path.stat().st_size <= count * MAX_FRAME_BYTES, "Oversized v3 frame stream")
        frames, previous, freezes = [], None, set()
        with path.open("rb") as stream:
            for index, line in enumerate(stream):
                require(index < count and len(line) <= MAX_FRAME_BYTES, "Frame count/size exceeded")
                frame = parse_json(line)
                images = keys(frame.get("images"), set(CAMERAS), "frozen images")
                image_size, payloads = verify_episode_images(
                    root,
                    images,
                    episode_id=episode.episode_id,
                    index=index,
                    paths=paths,
                    image_size=image_size,
                )
                sample = _decode_frame(frame, payloads, index)
                sample.validate(profile, previous=previous)
                _frame_binding(sample, profile, budget, episode, expected_scope, source, index)
                require(sample.observation.freeze_id not in freezes, "Reused freeze identity")
                freezes.add(sample.observation.freeze_id)
                frames.append(frame)
                previous = sample
        require(len(frames) == count, "Missing v3 frames")
        require(previous.terminated or previous.truncated, "Unfinished actual capture")
        if require_demonstrations:
            require(previous.terminated and not previous.truncated, "Truncated data cannot train")
        verified.append(ValidatedEpisode(metadata, tuple(frames)))
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlink in paused dataset")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == paths, "Unexpected data artifacts; evaluator labels must remain separate")
    checksum = (
        file_digest(root / "manifest.json")
        if manifest_present
        else digest(canonical(manifest) + b"\n")
    )
    return ValidatedDataset(root, manifest, checksum, tuple(verified))


def validate_dataset(
    root: Path,
    *,
    expected_scope: Scope,
    expected_manifest_sha256: str | None = None,
    require_live: bool = False,
    require_demonstrations: bool = False,
) -> ValidatedDataset:
    path = safe_path(root, "manifest.json")
    if expected_manifest_sha256 is not None:
        require(file_digest(path) == sha256(expected_manifest_sha256), "Manifest checksum mismatch")
    return _validate(
        root,
        read_json(path),
        expected_scope=expected_scope,
        require_live=require_live,
        require_demonstrations=require_demonstrations,
        manifest_present=True,
    )


class PausedEpisodeWriter:
    """Main-thread, manifest-last raw v3 capture; a failed append cannot be retried into success."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_id: str,
        scope: Scope,
        episode: EpisodeSpec,
        provenance: Provenance,
        profile: PausedControlProfile,
        demonstration: DemonstrationSource,
        budget: PausedEpisodeBudget,
        purpose: str,
        criteria_sha256: str,
        frozen_plan_sha256: str | None,
        max_frames: int = 300,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        scope.validate()
        episode.validate()
        provenance.validate()
        demonstration.validate()
        profile.validate()
        budget.validate(profile)
        token(dataset_id, "dataset ID")
        sha256(criteria_sha256, "frozen criteria")
        require(
            purpose in ("integration", "demonstration", "evaluation"), "Unknown capture purpose"
        )
        if purpose == "integration":
            require(episode.split == "test", "Integration capture must be TEST only")
            if frozen_plan_sha256 is not None:
                sha256(frozen_plan_sha256)
        else:
            sha256(frozen_plan_sha256, "new frozen conditions plan")
            require(episode.seed not in INTEGRATION_SEEDS, "Integration seed is not training data")
        integer(max_frames, "frame limit", 2, 300)
        integer(max_bytes, "byte limit", 1024, 20 * 1024**3)
        require(not root.exists() and not root.is_symlink(), "Never overwrite an existing capture")
        root.mkdir(parents=True)
        self.root, self.profile, self.budget = root, profile, budget
        self.scope, self.episode, self.demonstration = scope, episode, demonstration
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.thread_id = threading.get_ident()
        self.count, self.bytes_written, self.previous = 0, 0, None
        self.faulted, self.finalized = False, False
        self.freezes: set[str] = set()
        self.metadata = {
            **asdict(episode),
            "provenance": asdict(provenance),
            "demonstration": asdict(demonstration),
            "budget": asdict(budget),
            "path": f"episodes/{episode.episode_id}/frames.jsonl",
        }
        self.manifest = {
            "schema": RAW_SCHEMA,
            "dataset_id": dataset_id,
            "scope": asdict(scope),
            "fps": profile.control_sim_hz,
            "physics_hz": profile.physics_hz,
            "joint_names": list(JOINT_NAMES),
            "joint_units": list(JOINT_UNITS),
            "timestamp_basis": "simulation_time",
            "execution_timing": EXECUTION_TIMING,
            "real_time_admission": False,
            "control_profile": asdict(profile),
            "criteria_sha256": criteria_sha256,
            "frozen_plan_sha256": frozen_plan_sha256,
            "purpose": purpose,
            "episodes": [self.metadata],
        }
        self.frames_path = safe_path(root, self.metadata["path"], must_exist=False)
        self.frames_path.parent.mkdir(parents=True)
        self.frames_path.touch(exist_ok=False)

    def _writable(self) -> None:
        require(
            threading.get_ident() == self.thread_id, "Capture must stay on the main owner thread"
        )
        require(not self.faulted and not self.finalized, "Capture is faulted or already finalized")

    def append(self, sample: PausedFrameSample) -> None:
        self._writable()
        self.faulted = True
        require(self.count < self.max_frames, "Capture frame limit reached")
        sample.validate(self.profile, previous=self.previous)
        _frame_binding(
            sample,
            self.profile,
            self.budget,
            self.episode,
            self.scope,
            self.demonstration,
            self.count,
        )
        require(sample.observation.freeze_id not in self.freezes, "Reused freeze identity")
        frame = _frame_record(sample, self.episode.episode_id, self.count)
        line = canonical(frame) + b"\n"
        require(len(line) <= MAX_FRAME_BYTES, "Oversized capture frame")
        size = len(line) + sum(len(image.png) for image in sample.observation.images.values())
        require(self.bytes_written + size <= self.max_bytes, "Capture byte limit reached")
        for name, image in sample.observation.images.items():
            path = safe_path(self.root, frame["images"][name]["path"], must_exist=False)
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(image.png)
        with self.frames_path.open("ab") as stream:
            stream.write(line)
        self.count += 1
        self.bytes_written += size
        self.previous = sample
        self.freezes.add(sample.observation.freeze_id)
        self.faulted = False

    def finalize(self) -> Path:
        self._writable()
        self.faulted = True
        require(self.count >= 2, "Empty/too short paused capture")
        require(
            self.previous.terminated or self.previous.truncated, "Missing actual terminal frame"
        )
        self.metadata.update(frame_count=self.count, sha256=file_digest(self.frames_path))
        require(
            self.bytes_written + len(canonical(self.manifest)) + 1 <= self.max_bytes,
            "Capture manifest exceeds byte budget",
        )
        _validate(
            self.root,
            self.manifest,
            expected_scope=self.scope,
            require_live=False,
            require_demonstrations=False,
            manifest_present=False,
        )
        write_json(self.root / "manifest.json", self.manifest)
        self.finalized, self.faulted = True, False
        return self.root / "manifest.json"


def assemble_dataset(
    episode_roots: Sequence[Path],
    destination: Path,
    *,
    dataset_id: str,
    expected_scope: Scope,
    require_live: bool = True,
) -> Path:
    require(bool(episode_roots), "Cannot assemble an empty paused dataset")
    token(dataset_id, "dataset ID")
    sources = [
        validate_dataset(root, expected_scope=expected_scope, require_live=require_live)
        for root in episode_roots
    ]
    manifest = dict(sources[0].manifest, dataset_id=dataset_id, episodes=[])
    ids, seeds = set(), {}
    common = set(manifest) - {"dataset_id", "episodes"}
    for source in sources:
        require(
            all(source.manifest[key] == manifest[key] for key in common),
            "Cannot mix raw versions, profiles, criteria, frozen plans or capture purposes",
        )
        for episode in source.episodes:
            metadata = episode.metadata
            require(metadata["episode_id"] not in ids, "Duplicate/cross-split episode leakage")
            require(
                metadata["seed"] not in seeds or seeds[metadata["seed"]] == metadata["split"],
                "Cross-split seed leakage",
            )
            ids.add(metadata["episode_id"])
            seeds[metadata["seed"]] = metadata["split"]
            manifest["episodes"].append(metadata)
    require(not destination.exists(), "Never overwrite an assembled dataset")
    destination.mkdir(parents=True)
    for source in sources:
        for episode in source.episodes:
            relative = f"episodes/{episode.metadata['episode_id']}"
            shutil.copytree(source.root / relative, destination / relative, symlinks=False)
    _validate(
        destination,
        manifest,
        expected_scope=expected_scope,
        require_live=require_live,
        require_demonstrations=False,
        manifest_present=False,
    )
    write_json(destination / "manifest.json", manifest)
    return destination / "manifest.json"
