import json
from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from learning.checks.fixtures import PROVENANCE, SCOPE, overwrite
from learning.common import ContractError, canonical, file_digest, read_json, utc
from learning.contract import EpisodeSpec
from tests.learning.test_paused_contract import profile, sample


def shifted_sample(index, *, terminal=False):
    original = sample()
    source = original.observation
    wall = index * 2_000_000_000

    def stamp(text):
        return (utc(text) + timedelta(seconds=2 * index)).isoformat().replace("+00:00", "Z")

    observation = replace(
        source,
        monotonic_ns=source.monotonic_ns + wall,
        captured_at_utc=stamp(source.captured_at_utc),
        physics_step=source.physics_step + 6 * index,
        control_tick=index,
        state_revision=source.state_revision + 6 * index,
        freeze_id=f"freeze-{index}",
        observation_started_ns=source.observation_started_ns + wall,
        joint_sample_ns=source.joint_sample_ns + wall,
        simulation_time_numerator=10 + index,
        simulation_time_denominator=10,
        images={
            name: replace(
                image,
                monotonic_ns=image.monotonic_ns + wall,
                captured_at_utc=stamp(image.captured_at_utc),
                physics_step=image.physics_step + 6 * index,
                rendering_frame=image.rendering_frame + index,
                simulation_time_numerator=10 + index,
                simulation_time_denominator=10,
            )
            for name, image in source.images.items()
        },
    )
    return replace(
        original,
        observation=observation,
        applied_controls=tuple(
            replace(
                control,
                physics_step=control.physics_step + 6 * index,
                monotonic_ns=control.monotonic_ns + wall,
            )
            for control in original.applied_controls
        ),
        interval_deadline_ns=original.interval_deadline_ns + wall,
        hold_started_ns=original.hold_started_ns + wall,
        hold_deadline_ns=original.hold_deadline_ns + wall,
        policy_started_ns=None,
        policy_finished_ns=None,
        terminated=terminal,
    )


def writer(
    root, *, episode_id="paused-episode", split="train", seed=10001, purpose="demonstration"
):
    from learning.contract import DemonstrationSource
    from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter

    return PausedEpisodeWriter(
        root,
        dataset_id="paused-data",
        scope=SCOPE,
        episode=EpisodeSpec(episode_id, "paused-case", "b" * 64, seed, split),
        provenance=PROVENANCE,
        profile=profile(),
        demonstration=DemonstrationSource(
            kind="reference_controller",
            task_id="manufacturing-part-placement-v1",
            instruction="Move the part to the approved tray.",
            goal_id="rejected",
        ),
        purpose=purpose,
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64 if purpose != "integration" else None,
        budget=PausedEpisodeBudget(
            started_ns=1_000_000_000,
            wall_deadline_ns=601_000_000_000,
            initial_physics_step=60,
            simulation_step_deadline=1860,
        ),
    )


def complete(root):
    result = writer(root)
    result.append(shifted_sample(0))
    result.append(shifted_sample(1, terminal=True))
    result.finalize()
    return root


def test_raw_v3_persists_original_wall_and_actual_simulation_clocks(tmp_path):
    from learning.paused.capture import validate_dataset

    root = complete(tmp_path / "raw")
    value = validate_dataset(root, expected_scope=SCOPE)
    assert value.manifest["schema"] == "physicalai.demonstrations/v3"
    assert value.manifest["timestamp_basis"] == "simulation_time"
    assert value.manifest["real_time_admission"] is False
    assert value.manifest["criteria_sha256"] == "d" * 64
    first, second = value.episodes[0].frames
    assert second["monotonic_ns"] - first["monotonic_ns"] == 2_000_000_000
    assert second["physics_step"] - first["physics_step"] == 6
    assert second["simulation_time_numerator"] == 11
    assert second["simulation_time_denominator"] == 10
    assert second["images"]["overview"]["monotonic_ns"] == 3_900_000_000
    assert first["policy_started_ns"] is None
    assert len(first["applied_controls"]) == 6


@pytest.mark.parametrize(
    "change",
    [
        "checksum",
        "unsafe-path",
        "ground-truth",
        "clock-label",
        "old-version",
        "hold-deadline",
        "fake-freeze",
        "missing-tick",
        "missing-image-field",
        "budget",
        "future-camera",
    ],
)
def test_v3_rejects_tampered_capture_and_relabelled_clock_semantics(tmp_path, change):
    from learning.paused.capture import validate_dataset

    root = complete(tmp_path / "raw")
    manifest = read_json(root / "manifest.json")
    path = root / manifest["episodes"][0]["path"]
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "checksum":
        manifest["episodes"][0]["sha256"] = "0" * 64
    elif change == "clock-label":
        manifest["timestamp_basis"] = "wall_time"
    elif change == "old-version":
        manifest["schema"] = "physicalai.demonstrations/v2"
    elif change == "budget":
        manifest["episodes"][0]["budget"]["simulation_step_deadline"] = 1866
    else:
        if change == "unsafe-path":
            frames[0]["images"]["inspection"]["path"] = "../private.png"
        elif change == "ground-truth":
            frames[0]["success"] = True
        elif change == "hold-deadline":
            frames[0]["hold_deadline_ns"] += 1
        elif change == "fake-freeze":
            frames[1]["freeze_id"] = frames[0]["freeze_id"]
        elif change == "missing-tick":
            frames[0]["applied_controls"].pop()
        elif change == "missing-image-field":
            frames[0]["images"]["overview"].pop("sha256")
        else:
            frames[0]["images"]["overview"]["monotonic_ns"] = frames[0]["monotonic_ns"] + 1
        path.write_bytes(b"".join(canonical(frame) + b"\n" for frame in frames))
        manifest["episodes"][0]["sha256"] = file_digest(path)
    overwrite(root / "manifest.json", manifest)
    with pytest.raises(ContractError):
        validate_dataset(root, expected_scope=SCOPE)


def test_partial_capture_cannot_publish_or_retry_as_a_complete_dataset(tmp_path):
    value = writer(tmp_path / "raw")
    with pytest.raises(ContractError, match="six-tick"):
        value.append(replace(shifted_sample(0), applied_controls=()))
    with pytest.raises(ContractError, match="faulted"):
        value.append(shifted_sample(0))
    assert not (tmp_path / "raw" / "manifest.json").exists()


def test_integration_capture_is_explicit_and_cannot_be_training_data(tmp_path):
    from learning.paused.capture import validate_dataset

    value = writer(tmp_path / "integration", purpose="integration", seed=900004, split="test")
    value.append(shifted_sample(0))
    value.append(shifted_sample(1, terminal=True))
    value.finalize()
    manifest = validate_dataset(value.root, expected_scope=SCOPE).manifest
    assert manifest["purpose"] == "integration"
    assert manifest["frozen_plan_sha256"] is None
    with pytest.raises(ContractError, match="demonstration"):
        validate_dataset(value.root, expected_scope=SCOPE, require_demonstrations=True)


def test_fixture_capture_cannot_be_registered_as_live_or_leak_owner(tmp_path):
    from learning.contract import Scope
    from learning.paused.capture import validate_dataset

    root = complete(tmp_path / "raw")
    with pytest.raises(ContractError, match="live"):
        validate_dataset(root, expected_scope=SCOPE, require_live=True)
    with pytest.raises(ContractError, match="scope"):
        validate_dataset(root, expected_scope=Scope(SCOPE.tenant_id, "0" * 64))


def test_v2_validator_never_accepts_new_paused_mode_by_default(tmp_path):
    from learning.contract import validate_dataset

    with pytest.raises(ContractError):
        validate_dataset(complete(tmp_path / "raw"), expected_scope=SCOPE)


def test_assembly_keeps_seed_and_episode_splits_closed(tmp_path):
    from learning.paused.capture import assemble_dataset

    roots = []
    for name, split in (("train-episode", "train"), ("test-episode", "test")):
        root = tmp_path / name
        value = writer(root, episode_id=name, split=split, seed=10001)
        for index in range(2):
            current = shifted_sample(index, terminal=index == 1)
            value.append(
                replace(current, observation=replace(current.observation, episode_id=name))
            )
        value.finalize()
        roots.append(root)
    with pytest.raises(ContractError, match="seed"):
        assemble_dataset(
            roots,
            tmp_path / "assembled",
            dataset_id="all",
            expected_scope=SCOPE,
            require_live=False,
        )


def test_original_frozen_budget_is_persisted_not_reconstructed_from_last_frame(tmp_path):
    from learning.paused.capture import validate_dataset

    root = complete(tmp_path / "raw")
    expected = asdict(writer(tmp_path / "other").budget)
    actual = validate_dataset(root, expected_scope=SCOPE).episodes[0].metadata["budget"]
    assert actual == expected


def test_capture_stays_on_its_creating_simulator_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    value = writer(tmp_path / "raw")
    with ThreadPoolExecutor(max_workers=1) as executor:
        attempt = executor.submit(value.append, shifted_sample(0))
        with pytest.raises(ContractError, match="thread"):
            attempt.result()
    assert value.count == 0
    assert not (value.root / "manifest.json").exists()


def test_terminal_truncation_is_preserved_but_never_selected_as_demonstration(tmp_path):
    from learning.paused.capture import validate_dataset

    value = writer(tmp_path / "raw")
    value.append(shifted_sample(0))
    value.append(replace(shifted_sample(1), truncated=True))
    value.finalize()
    assert validate_dataset(value.root, expected_scope=SCOPE).episodes[0].frames[-1]["truncated"]
    with pytest.raises(ContractError, match="Truncated"):
        validate_dataset(value.root, expected_scope=SCOPE, require_demonstrations=True)


def test_integration_seed_cannot_be_renamed_into_demonstration(tmp_path):
    with pytest.raises(ContractError, match="Integration seed"):
        writer(tmp_path / "raw", seed=900002, purpose="demonstration")


def test_complete_demonstrations_remain_gated_by_purpose_and_finalized_bytes(tmp_path):
    from learning.paused.capture import validate_dataset

    root = complete(tmp_path / "raw")
    expected = file_digest(root / "manifest.json")
    result = validate_dataset(
        root, expected_scope=SCOPE, expected_manifest_sha256=expected, require_demonstrations=True
    )
    assert result.manifest_sha256 == expected
    with pytest.raises(ContractError, match="Manifest checksum"):
        validate_dataset(root, expected_scope=SCOPE, expected_manifest_sha256="0" * 64)
