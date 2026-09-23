from __future__ import annotations

import shutil
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from learning import CONTRACT_VERSION
from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    require,
    safe_path,
    token,
    write_json,
)
from learning.contract import (
    CAMERAS,
    JOINT_NAMES,
    JOINT_UNITS,
    TEACHING_CONTRACT_VERSION,
    ControlProfile,
    DemonstrationSource,
    EpisodeSpec,
    FrameSample,
    Provenance,
    Scope,
    bounded_joints,
    png_dimensions,
    timing,
    validate_dataset,
    validate_frame,
)


def resolve_joint_targets(
    previous: Sequence[float] | None,
    positions: Sequence[float | None],
    indices: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """Resolve an issued sparse ArticulationAction, never substituting measured joint state."""
    targets = (
        list(bounded_joints(previous, "previous issued targets"))
        if previous is not None
        else [None] * 9
    )
    selected = list(range(9)) if indices is None else list(indices)
    require(len(selected) == len(positions), "Sparse action length mismatch")
    require(len(set(selected)) == len(selected), "Duplicate joint indices")
    for index, position in zip(selected, positions, strict=True):
        integer(index, "joint index", 0, 8)
        if position is not None:
            targets[index] = position
    require(
        all(value is not None for value in targets), "Full issued target state is not yet known"
    )
    return bounded_joints(targets, "issued targets")


class EpisodeWriter:
    """Bounded, single-thread capture; only finalize publishes an uploadable manifest."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_id: str,
        scope: Scope,
        episode: EpisodeSpec,
        provenance: Provenance,
        fps: int = 10,
        physics_hz: int = 60,
        max_frames: int = 3600,
        max_bytes: int = 512 * 1024 * 1024,
        control_profile: ControlProfile | None = None,
        demonstration: DemonstrationSource | None = None,
    ) -> None:
        scope.validate()
        episode.validate()
        provenance.validate()
        token(dataset_id, "dataset ID")
        self.interval = timing(fps, physics_hz)
        require(
            (control_profile is None) == (demonstration is None),
            "Control profile and demonstration source must be supplied together",
        )
        if control_profile is not None:
            control_profile.validate()
            demonstration.validate()
            require(
                fps == control_profile.control_hz and physics_hz == control_profile.physics_hz,
                "Capture must use the actual 10Hz teaching hold profile, not 60Hz decimation",
            )
        self.control_profile = control_profile
        integer(max_frames, "max frames", 2, 100000)
        integer(max_bytes, "max bytes", 1024, 20 * 1024**3)
        require(not root.exists() and not root.is_symlink(), "Capture folder already exists")
        root.mkdir(parents=True)
        self.root, self.scope, self.episode = root, scope, episode
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.thread_id = threading.get_ident()
        self.count, self.bytes_written = 0, 0
        self.previous = None
        self.finalized = False
        self.metadata = {
            **asdict(episode),
            "provenance": asdict(provenance),
            "path": f"episodes/{episode.episode_id}/frames.jsonl",
        }
        self.frames_path = safe_path(root, self.metadata["path"], must_exist=False)
        self.frames_path.parent.mkdir(parents=True)
        self.frames_path.touch(exist_ok=False)
        self.manifest = {
            "schema": TEACHING_CONTRACT_VERSION if control_profile else CONTRACT_VERSION,
            "dataset_id": dataset_id,
            "scope": asdict(scope),
            "fps": fps,
            "physics_hz": physics_hz,
            "joint_names": list(JOINT_NAMES),
            "joint_units": list(JOINT_UNITS),
            "episodes": [self.metadata],
        }
        if control_profile is not None:
            self.manifest["control_profile"] = asdict(control_profile)
            self.metadata["demonstration"] = asdict(demonstration)

    def _writable(self) -> None:
        require(threading.get_ident() == self.thread_id, "Capture must stay on its creating thread")
        require(not self.finalized, "Capture has already been finalized")

    def append(self, sample: FrameSample) -> None:
        self._writable()
        require(self.count < self.max_frames, "Episode frame limit reached; stop capture")
        require(set(sample.images) == set(CAMERAS), "Both real camera frames are required")
        images, payloads = {}, {}
        for camera in CAMERAS:
            captured = sample.images[camera]
            width, height = png_dimensions(captured.png)
            path = f"episodes/{self.episode.episode_id}/{camera}/{self.count:08d}.png"
            images[camera] = {
                "path": path,
                "sha256": digest(captured.png),
                "width": width,
                "height": height,
                "rendering_frame": captured.rendering_frame,
                "physics_step": captured.physics_step,
                "monotonic_ns": captured.monotonic_ns,
            }
            payloads[path] = captured.png
        frame = {
            "frame_index": self.count,
            "captured_at_utc": sample.captured_at_utc,
            "monotonic_ns": sample.monotonic_ns,
            "physics_step": sample.physics_step,
            "joint_positions": list(sample.joint_positions),
            "commanded_joint_targets": list(sample.commanded_joint_targets),
            "images": images,
            "terminated": sample.terminated,
            "truncated": sample.truncated,
        }
        if self.control_profile is not None:
            require(sample.applied_controls is not None, "Missing actual held-control evidence")
            frame["applied_controls"] = [asdict(control) for control in sample.applied_controls]
        else:
            require(sample.applied_controls is None, "v1 cannot carry v2 control evidence")
        validate_frame(
            frame,
            self.previous,
            index=self.count,
            control_interval_steps=self.interval,
            control_profile=self.control_profile,
        )
        dimensions = {(image["width"], image["height"]) for image in images.values()}
        if self.previous is not None:
            dimensions.update(
                (image["width"], image["height"]) for image in self.previous["images"].values()
            )
        require(len(dimensions) == 1, "Capture camera dimensions changed")
        line = canonical(frame) + b"\n"
        size = len(line) + sum(len(payload) for payload in payloads.values())
        require(
            self.bytes_written + size <= self.max_bytes, "Episode byte limit reached; stop capture"
        )
        for rel, payload in payloads.items():
            path = safe_path(self.root, rel, must_exist=False)
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
        with self.frames_path.open("ab") as stream:
            stream.write(line)
        self.count += 1
        self.bytes_written += size
        self.previous = frame

    def finalize(self) -> Path:
        self._writable()
        require(self.count >= 2, "Empty/too short demonstration")
        require(
            self.previous["terminated"] or self.previous["truncated"],
            "Record a real terminal/truncated frame before finalizing",
        )
        self.metadata.update(sha256=file_digest(self.frames_path), frame_count=self.count)
        require(
            self.bytes_written + len(canonical(self.manifest)) + 1 <= self.max_bytes,
            "Manifest exceeds the capture byte budget",
        )
        write_json(self.root / "manifest.json", self.manifest)
        validate_dataset(self.root, expected_scope=self.scope)
        self.finalized = True
        return self.root / "manifest.json"


def assemble_dataset(
    episode_roots: Sequence[Path],
    destination: Path,
    *,
    dataset_id: str,
    expected_scope: Scope,
    require_live: bool = True,
) -> Path:
    require(bool(episode_roots), "Cannot assemble an empty dataset")
    sources = [
        validate_dataset(root, expected_scope=expected_scope, require_live=require_live)
        for root in episode_roots
    ]
    manifest = dict(sources[0].manifest, dataset_id=dataset_id, episodes=[])
    seen, seed_splits = set(), {}
    for source in sources:
        require(
            source.manifest["schema"] == manifest["schema"]
            and source.manifest.get("control_profile") == manifest.get("control_profile"),
            "Cannot mix capture versions or servo profiles",
        )
        require(
            (source.manifest["fps"], source.manifest["physics_hz"])
            == (manifest["fps"], manifest["physics_hz"]),
            "Capture rates differ",
        )
        for episode in source.episodes:
            spec = episode.metadata
            require(spec["episode_id"] not in seen, "Duplicate/cross-split episode leakage")
            require(
                spec["seed"] not in seed_splits or seed_splits[spec["seed"]] == spec["split"],
                "Cross-split seed leakage",
            )
            seen.add(spec["episode_id"])
            seed_splits[spec["seed"]] = spec["split"]
            manifest["episodes"].append(spec)
    require(not destination.exists(), "Output dataset already exists")
    destination.mkdir(parents=True)
    for source in sources:
        for episode in source.episodes:
            rel = f"episodes/{episode.metadata['episode_id']}"
            shutil.copytree(source.root / rel, destination / rel, symlinks=False)
    write_json(destination / "manifest.json", manifest)
    validate_dataset(destination, expected_scope=expected_scope, require_live=require_live)
    return destination / "manifest.json"
