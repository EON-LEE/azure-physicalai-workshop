from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, overwrite, png
from learning.common import ContractError, canonical, digest, read_json
from learning.contract import AppliedControl, DemonstrationSource, EpisodeSpec
from learning.paused import (
    FrozenCameraSample,
    FrozenPolicyObservation,
    PausedControlProfile,
    PausedFrameSample,
)
from learning.paused.capture import (
    PausedEpisodeBudget,
    PausedEpisodeWriter,
    assemble_dataset,
    validate_dataset,
)

V2_PROFILE_ID = "franka-position-hold-10hz-paused-v2"


def selected_profile(version=2):
    if version == 1:
        return PausedControlProfile("f" * 64)
    return PausedControlProfile("f" * 64, profile_id=V2_PROFILE_ID, max_simulation_steps=3600)


def capture_writer(root, *, version=2, **kwargs):
    profile = selected_profile(version)
    return PausedEpisodeWriter(
        root,
        dataset_id=f"profile-v{version}-fixture",
        scope=SCOPE,
        episode=EpisodeSpec("budget-fixture", "paused-case", "b" * 64, 10001, "train"),
        provenance=PROVENANCE,
        profile=profile,
        demonstration=DemonstrationSource(
            kind="reference_controller",
            task_id="fixture-only",
            instruction="Test the versioned capture budget without a physical quality claim.",
            goal_id="rejected",
        ),
        budget=PausedEpisodeBudget(
            started_ns=1_000_000_000,
            wall_deadline_ns=601_000_000_000,
            initial_physics_step=0,
            simulation_step_deadline=profile.max_simulation_steps,
        ),
        purpose="demonstration",
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64,
        **kwargs,
    )


def capture_frame(index, *, version=2, terminal=False):
    profile = selected_profile(version)
    start = 1_000_000_000 + index * 500_000_000
    captured = datetime(2026, 9, 25, tzinfo=UTC) + timedelta(milliseconds=index * 500 + 120)
    stamp = captured.isoformat().replace("+00:00", "Z")
    observation = FrozenPolicyObservation(
        scope=SCOPE,
        environment_id="paused-case",
        revision="b" * 64,
        episode_id="budget-fixture",
        epoch="fixture-epoch",
        captured_at_utc=stamp,
        monotonic_ns=start + 120_000_000,
        physics_step=index * 6,
        joint_positions=JOINTS,
        images={
            name: FrozenCameraSample(
                png=png(color=80 + index % 10),
                rendering_frame=index,
                physics_step=index * 6,
                monotonic_ns=start + 120_000_000,
                captured_at_utc=stamp,
                simulation_time_numerator=index,
                simulation_time_denominator=10,
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
    return PausedFrameSample(
        observation=observation,
        commanded_joint_targets=JOINTS,
        applied_controls=tuple(
            AppliedControl(
                physics_step=index * 6 + offset,
                monotonic_ns=start + 150_000_000 + offset * 20_000_000,
                commanded_joint_targets=JOINTS,
                commanded_joint_velocities=(0.0,) * 9,
                gravity_efforts=(0.1,) * 7 + (0.0, 0.0),
            )
            for offset in range(1, 7)
        ),
        interval_deadline_ns=min(start + 5_000_000_000, 601_000_000_000),
        hold_started_ns=start + 150_000_000,
        hold_deadline_ns=min(start + 2_150_000_000, 601_000_000_000),
        terminated=terminal,
    )


def test_v2_profile_is_an_explicit_closed_pair_without_changing_legacy_defaults():
    from learning.paused import CONTROL_PROFILE_ID, CONTROL_PROFILE_V2_ID

    legacy = selected_profile(1)
    current = selected_profile(2)
    legacy.validate()
    current.validate()
    assert CONTROL_PROFILE_ID == "franka-position-hold-10hz-paused-v1"
    assert CONTROL_PROFILE_V2_ID == V2_PROFILE_ID
    assert legacy.max_simulation_steps == 1800 and legacy.max_frames == 300
    assert current.max_simulation_steps == 3600 and current.max_frames == 600
    legacy_fields = asdict(legacy)
    expected_legacy_fields = {
        "servo_profile_sha256": "f" * 64,
        "profile_id": CONTROL_PROFILE_ID,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "physics_hz": 60,
        "control_sim_hz": 10,
        "hold_steps": 6,
        "velocity_target_mode": "zero",
        "gravity_compensation": "physx_measured_arm_only",
        "max_observation_wall_ms": 2000,
        "max_policy_wall_ms": 2000,
        "max_hold_wall_ms": 2000,
        "max_interval_wall_ms": 5000,
        "max_episode_wall_ms": 600000,
        "max_simulation_steps": 1800,
        "max_heartbeat_wall_ms": 2000,
        "max_cartesian_speed_m_s": 0.2,
        "max_axis_goal_error_m": 0.04,
    }
    assert legacy_fields == expected_legacy_fields
    assert asdict(current) == {
        **legacy_fields,
        "profile_id": V2_PROFILE_ID,
        "max_simulation_steps": 3600,
    }
    assert current.sha256 != legacy.sha256 == digest(canonical(expected_legacy_fields))
    assert "max_frames" not in legacy_fields


@pytest.mark.parametrize(
    ("profile_id", "steps"),
    [
        ("franka-position-hold-10hz-paused-v1", 3600),
        (V2_PROFILE_ID, 1800),
        (V2_PROFILE_ID, 3606),
        (V2_PROFILE_ID, 3600.0),
        ("franka-position-hold-10hz-paused-v3", 3600),
    ],
)
def test_unknown_or_mixed_profile_budget_is_rejected(profile_id, steps):
    with pytest.raises(ContractError):
        PausedControlProfile("f" * 64, profile_id=profile_id, max_simulation_steps=steps).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_observation_wall_ms", 2001),
        ("max_policy_wall_ms", 2001),
        ("max_hold_wall_ms", 2001),
        ("max_interval_wall_ms", 5001),
        ("max_episode_wall_ms", 600001),
        ("max_heartbeat_wall_ms", 2001),
        ("max_cartesian_speed_m_s", 0.21),
        ("max_axis_goal_error_m", 0.041),
        ("hold_steps", 7),
    ],
)
def test_v2_does_not_change_any_other_frozen_limit(field, value):
    with pytest.raises(ContractError):
        replace(selected_profile(), **{field: value}).validate()


@pytest.mark.parametrize(("version", "count"), [(1, 300), (2, 600)])
def test_complete_capture_retains_every_interval_and_exact_physics_tick(tmp_path, version, count):
    writer = capture_writer(tmp_path / "raw", version=version)
    assert writer.max_frames == count
    assert writer.max_bytes == 512 * 1024 * 1024
    for index in range(count):
        writer.append(capture_frame(index, version=version, terminal=index == count - 1))
    writer.finalize()
    dataset = validate_dataset(writer.root, expected_scope=SCOPE, require_demonstrations=True)
    frames = dataset.episodes[0].frames
    assert len(frames) == dataset.episodes[0].metadata["frame_count"] == count
    assert sum(len(frame["applied_controls"]) for frame in frames) == count * 6
    assert frames[-1]["applied_controls"][-1]["physics_step"] == count * 6
    assert len(tuple(writer.root.rglob("*.png"))) == count * 2
    assert dataset.manifest["schema"] == "physicalai.demonstrations/v3"
    assert dataset.manifest["real_time_admission"] is False
    assert frames[-1]["monotonic_ns"] - frames[0]["monotonic_ns"] == (count - 1) * 500_000_000
    assert frames[-1]["simulation_time_numerator"] == count - 1
    assert frames[-1]["simulation_time_denominator"] == 10
    assert not (writer.root / "task-states.jsonl").exists()
    with pytest.raises(ContractError, match="live"):
        validate_dataset(writer.root, expected_scope=SCOPE, require_live=True)


@pytest.mark.parametrize(("version", "limit"), [(1, 301), (2, 601)])
def test_explicit_frame_limit_cannot_exceed_selected_profile(tmp_path, version, limit):
    with pytest.raises(ContractError, match="frame limit"):
        capture_writer(tmp_path / "raw", version=version, max_frames=limit)
    assert not (tmp_path / "raw").exists()


def test_v2_writer_retains_explicit_narrower_frame_and_byte_budgets(tmp_path):
    writer = capture_writer(tmp_path / "short", max_frames=2)
    writer.append(capture_frame(0))
    writer.append(capture_frame(1))
    with pytest.raises(ContractError, match="frame limit"):
        writer.append(capture_frame(2))
    assert not (writer.root / "manifest.json").exists()
    bounded = capture_writer(tmp_path / "small", max_bytes=1024)
    with pytest.raises(ContractError, match="byte limit"):
        bounded.append(capture_frame(0))
    assert not (bounded.root / "manifest.json").exists()


def test_v1_capture_cannot_be_promoted_to_v2_by_relabelling_the_manifest(tmp_path):
    writer = capture_writer(tmp_path / "raw", version=1)
    writer.append(capture_frame(0, version=1))
    writer.append(capture_frame(1, version=1, terminal=True))
    writer.finalize()
    manifest = read_json(writer.root / "manifest.json")
    manifest["control_profile"] = asdict(selected_profile(2))
    overwrite(writer.root / "manifest.json", manifest)
    with pytest.raises(ContractError, match="profile"):
        validate_dataset(writer.root, expected_scope=SCOPE)


def test_v1_and_v2_datasets_cannot_be_assembled_together(tmp_path):
    roots = []
    for version in (1, 2):
        writer = capture_writer(tmp_path / f"raw-v{version}", version=version)
        writer.append(capture_frame(0, version=version))
        writer.append(capture_frame(1, version=version, terminal=True))
        writer.finalize()
        roots.append(writer.root)
    with pytest.raises(ContractError, match="mix"):
        assemble_dataset(
            roots,
            tmp_path / "mixed",
            dataset_id="not-admitted",
            expected_scope=SCOPE,
            require_live=False,
        )
    assert not (tmp_path / "mixed").exists()


@pytest.mark.parametrize(("version", "last_tick"), [(1, 299), (2, 599)])
def test_context_admits_only_complete_holds_inside_selected_profile(version, last_tick):
    from tests.learning.test_paused_contract import context

    profile = selected_profile(version)
    observation = capture_frame(last_tick, version=version).observation
    authority = replace(
        context(),
        episode_id=observation.episode_id,
        epoch=observation.epoch,
        freeze_id=observation.freeze_id,
        observation_sha256=observation.sha256,
        control_profile_sha256=profile.sha256,
        state_revision=observation.state_revision,
        episode_initial_physics_step=0,
        simulation_step_deadline=profile.max_simulation_steps,
        interval_started_ns=observation.observation_started_ns,
        interval_deadline_ns=observation.observation_started_ns + 5_000_000_000,
        operation_started_ns=observation.ready_ns,
        operation_deadline_ns=observation.ready_ns + 2_000_000_000,
    )
    authority.validate(profile, observation, now_ns=observation.ready_ns)
    outside = replace(
        observation,
        control_tick=last_tick + 1,
        physics_step=profile.max_simulation_steps,
        images={
            name: replace(image, physics_step=profile.max_simulation_steps)
            for name, image in observation.images.items()
        },
    )
    with pytest.raises(ContractError, match="Simulation step budget"):
        replace(authority, observation_sha256=outside.sha256).validate(
            profile, outside, now_ns=outside.ready_ns
        )
