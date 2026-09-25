from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from learning import convert
from learning.common import (
    canonical,
    digest,
    file_digest,
    finite,
    integer,
    inventory,
    keys,
    parse_json,
    read_json,
    require,
    sha256,
    write_json,
)
from learning.contract import Scope
from learning.paused.capture import MAX_FRAME_BYTES, PausedEpisodeBudget, validate_dataset
from learning.paused.contract import CONVERSION_SCHEMA, EXECUTION_TIMING, PausedControlProfile
from learning.smolvla.adaptation import IMAGE_SIZE
from learning.train import CONVERSION_KEYS, _validate_conversion_contents

TIMING_FILE = "source-timing.jsonl"
TRAIN_SEEDS = frozenset((*range(10001, 10021), *range(11001, 11021)))


def convert_dataset(
    source: Path,
    output: Path,
    *,
    expected_scope: Scope,
    expected_manifest_sha256: str,
    expected_criteria_sha256: str,
    expected_frozen_plan_sha256: str,
    allow_test_fixture: bool = False,
) -> dict:
    validated = validate_dataset(
        source,
        expected_scope=expected_scope,
        expected_manifest_sha256=expected_manifest_sha256,
        require_live=not allow_test_fixture,
        require_demonstrations=True,
    )
    require(
        validated.manifest["criteria_sha256"] == sha256(expected_criteria_sha256)
        and validated.manifest["frozen_plan_sha256"] == sha256(expected_frozen_plan_sha256),
        "Raw data differs from the approved frozen criteria/conditions",
    )
    require(
        all(episode.metadata["seed"] in TRAIN_SEEDS for episode in validated.split("train")),
        "Validation/final-held-out seeds cannot be relabelled into the frozen train cohort",
    )
    result = convert._convert_validated(
        validated,
        output,
        expected_scope=expected_scope,
        image_size=IMAGE_SIZE,
        simulation_timestamps=True,
    )
    with (output / TIMING_FILE).open("xb") as stream:
        for episode_index, episode in enumerate(validated.split("train")):
            initial = episode.frames[0]
            origin = Fraction(
                initial["simulation_time_numerator"], initial["simulation_time_denominator"]
            )
            for raw in episode.frames:
                native = Fraction(
                    raw["simulation_time_numerator"], raw["simulation_time_denominator"]
                )
                record = {
                    **raw,
                    "episode_index": episode_index,
                    "simulation_timestamp": float(native - origin),
                    "source_frame_sha256": digest(canonical(raw)),
                }
                stream.write(canonical(record) + b"\n")
    result.update(
        schema=CONVERSION_SCHEMA,
        timestamp_basis="simulation_time",
        execution_timing=EXECUTION_TIMING,
        real_time_admission=False,
        criteria_sha256=expected_criteria_sha256,
        frozen_plan_sha256=expected_frozen_plan_sha256,
        source_timing_sha256=file_digest(output / TIMING_FILE),
        files=inventory(output),
    )
    write_json(output / "conversion.json", result)
    validate_conversion(output, expected_scope)
    return result


def validate_conversion(root: Path, expected_scope: Scope) -> dict:
    value = keys(
        read_json(root / "conversion.json"),
        CONVERSION_KEYS
        | {
            "control_profile",
            "timestamp_basis",
            "execution_timing",
            "real_time_admission",
            "criteria_sha256",
            "frozen_plan_sha256",
            "source_timing_sha256",
        },
        "paused conversion",
    )
    require(
        value["schema"] == CONVERSION_SCHEMA
        and value["timestamp_basis"] == "simulation_time"
        and value["execution_timing"] == EXECUTION_TIMING
        and value["real_time_admission"] is False,
        "Wrong paused conversion schema/clock; no real-time relabel",
    )
    profile = PausedControlProfile(
        **keys(value["control_profile"], set(PausedControlProfile.__dataclass_fields__), "profile")
    )
    profile.validate()
    require(
        value["fps"] == profile.control_sim_hz and value["physics_hz"] == profile.physics_hz,
        "Wrong simulation-time conversion cadence",
    )
    for name in ("criteria_sha256", "frozen_plan_sha256", "source_timing_sha256"):
        sha256(value[name], name)
    _validate_conversion_contents(
        root, expected_scope, value, teaching=True, extra_episode_keys=frozenset({"budget"})
    )
    require(
        TIMING_FILE in value["files"]
        and file_digest(root / TIMING_FILE) == value["source_timing_sha256"],
        "Original wall/simulation provenance sidecar changed",
    )
    counts = []
    for episode in value["episodes"]:
        require(episode["seed"] in TRAIN_SEEDS, "Unapproved or held-out training seed")
        PausedEpisodeBudget(
            **keys(
                episode["budget"], set(PausedEpisodeBudget.__dataclass_fields__), "episode budget"
            )
        ).validate(profile)
        require(
            episode["terminated"] is True and episode["truncated"] is False,
            "Incomplete/truncated episodes cannot train a paused model",
        )
        counts.append(integer(episode["frame_count"], "source frame count", 2, 300))
    require(
        (root / TIMING_FILE).stat().st_size <= sum(counts) * (MAX_FRAME_BYTES + 512),
        "Oversized timing sidecar",
    )
    expected_episode, expected_frame, origins = 0, 0, {}
    with (root / TIMING_FILE).open("rb") as stream:
        for line in stream:
            require(expected_episode < len(counts), "Extra timing records")
            require(len(line) <= MAX_FRAME_BYTES + 512, "Oversized timing record")
            record = parse_json(line)
            require(
                integer(record["episode_index"], "source episode index") == expected_episode
                and integer(record["frame_index"], "source frame index") == expected_frame
                and record["episode_id"] == value["episodes"][expected_episode]["episode_id"]
                and record["scope"] == asdict(expected_scope),
                "Timing provenance episode/frame/scope was rebound",
            )
            raw = {
                key: item
                for key, item in record.items()
                if key not in ("episode_index", "simulation_timestamp", "source_frame_sha256")
            }
            require(
                digest(canonical(raw)) == sha256(record["source_frame_sha256"]),
                "Original source frame checksum mismatch",
            )
            native = Fraction(
                integer(record["simulation_time_numerator"], "native time numerator"),
                integer(record["simulation_time_denominator"], "native time denominator", 1),
            )
            origin = origins.setdefault(expected_episode, native)
            require(
                abs(native - origin - Fraction(expected_frame, profile.control_sim_hz))
                <= Fraction(1, 1_000_000_000)
                and abs(
                    finite(record["simulation_timestamp"], "simulation timestamp")
                    - float(native - origin)
                )
                <= 1e-9,
                "LeRobot timestamp does not derive from the actual simulation clock",
            )
            expected_frame += 1
            if expected_frame == counts[expected_episode]:
                expected_episode += 1
                expected_frame = 0
    require(
        expected_episode == len(counts) and expected_frame == 0, "Missing source timing records"
    )
    return value
