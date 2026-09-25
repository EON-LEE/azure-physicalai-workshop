"""Actual pinned LeRobot CPU conversion of explicit fixtures; no policy weights or cloud calls."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, png
from learning.common import file_digest, require, write_json
from learning.contract import AppliedControl, DemonstrationSource, EpisodeSpec
from learning.paused import (
    CONTROL_PROFILE_V2_ID,
    FrozenCameraSample,
    FrozenPolicyObservation,
    PausedControlProfile,
    PausedFrameSample,
)
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter
from learning.paused.dataset import convert_dataset, validate_conversion


def create_fixture(
    root: Path, *, profile: PausedControlProfile | None = None, full_episode: bool = False
) -> None:
    profile = profile or PausedControlProfile("f" * 64)
    count = profile.max_frames if full_episode else 3
    wall_interval_ms = 500 if full_episode else 2000
    writer = PausedEpisodeWriter(
        root,
        dataset_id="cpu-paused-fixture",
        scope=SCOPE,
        episode=EpisodeSpec("cpu-episode", "cpu-environment", "b" * 64, 10001, "train"),
        provenance=PROVENANCE,
        profile=profile,
        demonstration=DemonstrationSource(
            kind="reference_controller",
            task_id="fixture-only",
            instruction="Test the converter without claiming a physical task.",
            goal_id="rejected",
        ),
        budget=PausedEpisodeBudget(1_000_000_000, 601_000_000_000, 0, profile.max_simulation_steps),
        purpose="demonstration",
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64,
    )
    for index in range(count):
        start = 1_000_000_000 + index * wall_interval_ms * 1_000_000
        utc_start = datetime(2026, 9, 20, tzinfo=UTC) + timedelta(
            milliseconds=wall_interval_ms * index
        )
        stamp = (utc_start + timedelta(milliseconds=100)).isoformat().replace("+00:00", "Z")
        joints = (index * 0.001, *JOINTS[1:])
        observation = FrozenPolicyObservation(
            scope=SCOPE,
            environment_id="cpu-environment",
            revision="b" * 64,
            episode_id="cpu-episode",
            epoch="fixture-epoch",
            captured_at_utc=stamp,
            monotonic_ns=start + 100_000_000,
            physics_step=index * 6,
            joint_positions=joints,
            images={
                name: FrozenCameraSample(
                    png(color=80 + index % 176),
                    index,
                    index * 6,
                    start + 100_000_000,
                    stamp,
                    index,
                    10,
                )
                for name in ("inspection", "overview")
            },
            freeze_id=f"fixture-freeze-{index}",
            state_revision=index * 6,
            control_tick=index,
            control_profile_sha256=profile.sha256,
            observation_started_ns=start,
            joint_sample_ns=start + 50_000_000,
            simulation_time_numerator=index,
            simulation_time_denominator=10,
        )
        value = PausedFrameSample(
            observation=observation,
            commanded_joint_targets=joints,
            applied_controls=tuple(
                AppliedControl(
                    index * 6 + offset,
                    start + 100_000_000 + offset * 20_000_000,
                    joints,
                    (0.0,) * 9,
                    (0.1,) * 7 + (0.0, 0.0),
                )
                for offset in range(1, 7)
            ),
            interval_deadline_ns=start + 5_000_000_000,
            hold_started_ns=start + 100_000_000,
            hold_deadline_ns=start + 2_100_000_000,
        )
        writer.append(replace(value, terminated=index == count - 1))
    writer.finalize()


def run(output: Path, *, profile_version: int = 1, full_episode: bool = False) -> dict:
    require(profile_version in (1, 2), "Choose the explicit supported paused profile version")
    profile = (
        PausedControlProfile("f" * 64)
        if profile_version == 1
        else PausedControlProfile(
            "f" * 64, profile_id=CONTROL_PROFILE_V2_ID, max_simulation_steps=3600
        )
    )
    require(not output.exists(), "Choose a new CPU-only check directory")
    output.mkdir(parents=True)
    source, converted = output / "raw", output / "dataset"
    create_fixture(source, profile=profile, full_episode=full_episode)
    result = convert_dataset(
        source,
        converted,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(source / "manifest.json"),
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
        allow_test_fixture=True,
    )
    validate_conversion(converted, SCOPE)
    import pyarrow.parquet as parquet

    tables = list((converted / "data").rglob("*.parquet"))
    require(bool(tables), "Actual LeRobot did not write Parquet")
    timestamps = []
    for path in tables:
        timestamps.extend(parquet.read_table(path, columns=["timestamp"])["timestamp"].to_pylist())
    count = profile.max_frames if full_episode else 3
    expected = [
        struct.unpack("<f", struct.pack("<f", index / profile.control_sim_hz))[0]
        for index in range(count)
    ]
    require(
        timestamps == expected,
        "Actual LeRobot timestamps differ from the measured fixture simulation clock",
    )
    report = {
        "check": "pinned-lerobot-0.4.4-paused-v3-cpu-conversion",
        "test_only": True,
        "observation_source": "test_fixture",
        "control_profile_id": profile.profile_id,
        "max_simulation_steps": profile.max_simulation_steps,
        "converted_frames": len(timestamps),
        "simulation_timestamps": timestamps,
        "original_wall_interval_seconds": 0.5 if full_episode else 2.0,
        "conversion_sha256": file_digest(converted / "conversion.json"),
        "source_timing_sha256": result["source_timing_sha256"],
        "policy_weights_loaded": False,
        "optimizer_steps": 0,
        "cloud_calls": 0,
        "real_time_admission": False,
        "learning_quality_verified": False,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--full-episode", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output, profile_version=args.profile_version, full_episode=args.full_episode),
            indent=2,
        )
    )
