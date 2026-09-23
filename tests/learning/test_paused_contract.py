from dataclasses import replace

import pytest

from learning.checks.fixtures import JOINTS, SCOPE, png
from learning.common import ContractError
from learning.contract import ControlProfile


def profile():
    from learning.paused import PausedControlProfile

    return PausedControlProfile(servo_profile_sha256="f" * 64)


def observation():
    from learning.paused import FrozenCameraSample, FrozenPolicyObservation

    return FrozenPolicyObservation(
        scope=SCOPE,
        environment_id="paused-case",
        revision="b" * 64,
        episode_id="paused-episode",
        epoch="epoch-new",
        captured_at_utc="2026-09-24T00:00:01Z",
        monotonic_ns=2_000_000_000,
        physics_step=60,
        joint_positions=JOINTS,
        images={
            camera: FrozenCameraSample(
                png=png(),
                rendering_frame=100,
                physics_step=60,
                monotonic_ns=1_900_000_000,
                captured_at_utc="2026-09-24T00:00:00.900000Z",
                simulation_time_numerator=1,
                simulation_time_denominator=1,
            )
            for camera in ("inspection", "overview")
        },
        freeze_id="freeze-new",
        state_revision=60,
        control_tick=0,
        control_profile_sha256=profile().sha256,
        observation_started_ns=1_000_000_000,
        joint_sample_ns=1_100_000_000,
        simulation_time_numerator=1,
        simulation_time_denominator=1,
    )


def context():
    from learning.paused import PausedControlContext

    value = observation()
    return PausedControlContext(
        scope=SCOPE,
        environment_id=value.environment_id,
        revision=value.revision,
        episode_id=value.episode_id,
        epoch=value.epoch,
        command_id="paused-command",
        destination_id="rejected",
        approved=True,
        active=True,
        model_sha256="c" * 64,
        control_profile_sha256=profile().sha256,
        freeze_id=value.freeze_id,
        observation_sha256=value.sha256,
        state_revision=value.state_revision,
        episode_started_ns=1_000_000_000,
        episode_initial_physics_step=60,
        simulation_step_deadline=1860,
        wall_deadline_ns=601_000_000_000,
        interval_started_ns=1_000_000_000,
        interval_deadline_ns=6_000_000_000,
        operation_started_ns=2_000_000_000,
        operation_deadline_ns=4_000_000_000,
    )


def test_paused_profile_has_separate_fixed_clock_limits_and_protocols():
    from learning.paused import protocol_schemas

    value = profile()
    value.validate()
    assert value.execution_timing == "paused_simulation"
    assert value.real_time_admission is False
    assert (value.physics_hz, value.control_sim_hz, value.hold_steps) == (60, 10, 6)
    assert (value.max_observation_wall_ms, value.max_policy_wall_ms, value.max_hold_wall_ms) == (
        2000,
        2000,
        2000,
    )
    assert (value.max_interval_wall_ms, value.max_episode_wall_ms, value.max_simulation_steps) == (
        5000,
        600000,
        1800,
    )
    assert value.sha256 != ControlProfile(servo_profile_sha256="f" * 64).sha256
    schemas = protocol_schemas()
    assert schemas["raw"] == "physicalai.demonstrations/v3"
    assert schemas["model"] == "physicalai.smolvla-checkpoint/v2"
    assert schemas["request"] == "physicalai.smolvla-request/v2"
    assert schemas["paired_results"] == "physicalai.smolvla-paired-results/v3"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("profile_id", "franka-position-hold-10hz-v1"),
        ("execution_timing", "real_time"),
        ("real_time_admission", True),
        ("control_sim_hz", 60),
        ("max_policy_wall_ms", 5000),
        ("max_hold_wall_ms", 2001),
        ("max_interval_wall_ms", 5001),
        ("max_episode_wall_ms", 600001),
        ("max_simulation_steps", 1806),
        ("max_heartbeat_wall_ms", 2001),
    ],
)
def test_frozen_paused_caps_cannot_be_relaxed(field, bad):
    with pytest.raises(ContractError):
        replace(profile(), **{field: bad}).validate()


def test_frozen_observation_keeps_true_wall_times_instead_of_reminting_for_prediction():
    value = observation()
    before = value.sha256
    value.validate(profile(), now_ns=3_500_000_000)
    assert value.monotonic_ns == 2_000_000_000
    assert value.images["inspection"].monotonic_ns == 1_900_000_000
    assert value.sha256 == before
    assert value.simulation_time == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing-camera",
        "future-camera",
        "wrong-physics",
        "wrong-native-time",
        "slow-observation",
        "future-joints",
        "nonfinite",
        "old-profile",
    ],
)
def test_frozen_observation_rejects_invalid_or_relabelled_capture(change):
    value = observation()
    cameras = dict(value.images)
    if change == "missing-camera":
        cameras.pop("overview")
    elif change == "future-camera":
        cameras["overview"] = replace(cameras["overview"], monotonic_ns=value.monotonic_ns + 1)
    elif change == "wrong-physics":
        cameras["overview"] = replace(cameras["overview"], physics_step=59)
    elif change == "wrong-native-time":
        cameras["overview"] = replace(cameras["overview"], simulation_time_numerator=2)
    elif change == "slow-observation":
        value = replace(value, monotonic_ns=3_000_000_001)
    elif change == "future-joints":
        value = replace(value, joint_sample_ns=value.monotonic_ns + 1)
    elif change == "nonfinite":
        value = replace(value, joint_positions=(float("nan"), *JOINTS[1:]))
    else:
        value = replace(value, control_profile_sha256=ControlProfile("f" * 64).sha256)
    value = replace(value, images=cameras)
    with pytest.raises(ContractError):
        value.validate(profile(), now_ns=4_000_000_000)


def test_context_closes_authority_and_simulation_bounds_without_new_wall_lease():
    value = context()
    value.validate(profile(), observation(), now_ns=2_000_000_000)
    assert value.execution_timing == "paused_simulation"
    assert value.real_time_admission is False
    with pytest.raises(ContractError, match="deadline"):
        value.validate(profile(), observation(), now_ns=value.operation_deadline_ns)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("approved", False),
        ("active", False),
        ("freeze_id", "another-freeze"),
        ("state_revision", 61),
        ("observation_sha256", "d" * 64),
        ("epoch", "another-epoch"),
        ("operation_deadline_ns", 4_000_000_001),
        ("interval_deadline_ns", 6_000_000_001),
        ("wall_deadline_ns", 601_000_000_001),
        ("simulation_step_deadline", 1861),
        ("simulation_step_deadline", 1866),
        ("episode_initial_physics_step", 54),
    ],
)
def test_context_cannot_extend_or_rebind_the_frozen_operation(field, bad):
    with pytest.raises(ContractError):
        replace(context(), **{field: bad}).validate(profile(), observation(), now_ns=2_000_000_000)


def test_observation_digest_contains_all_real_input_and_freeze_metadata_but_no_labels():
    value = observation()
    assert value.sha256 != replace(value, freeze_id="different").sha256
    assert value.sha256 != replace(value, joint_positions=(0.01, *JOINTS[1:])).sha256
    assert (
        value.sha256
        != replace(
            value,
            images={
                **value.images,
                "overview": replace(value.images["overview"], png=png(color=90)),
            },
        ).sha256
    )
    payload = value.metadata()
    assert not {"seed", "object_pose", "success", "defect"} & set(payload)
    assert "png" not in payload["images"]["overview"]


def sample():
    from learning.contract import AppliedControl
    from learning.paused import PausedFrameSample

    return PausedFrameSample(
        observation=observation(),
        commanded_joint_targets=JOINTS,
        applied_controls=tuple(
            AppliedControl(
                physics_step=60 + index,
                monotonic_ns=2_300_000_000 + index * 20_000_000,
                commanded_joint_targets=JOINTS,
                commanded_joint_velocities=(0.0,) * 9,
                gravity_efforts=(0.1,) * 7 + (0.0, 0.0),
            )
            for index in range(1, 7)
        ),
        interval_deadline_ns=6_000_000_000,
        hold_started_ns=2_300_000_000,
        hold_deadline_ns=4_300_000_000,
        policy_started_ns=2_000_000_000,
        policy_finished_ns=2_250_000_000,
    )


def test_capture_contract_requires_real_complete_hold_and_distinct_wall_phases():
    value = sample()
    value.validate(profile())
    assert len(value.applied_controls) == 6
    assert value.policy_finished_ns - value.policy_started_ns == 250_000_000


@pytest.mark.parametrize(
    "change",
    ["partial", "skip-tick", "finger-gravity", "velocity", "target", "hold-late", "policy-late"],
)
def test_paused_capture_rejects_partial_or_fabricated_control_intervals(change):
    value = sample()
    controls = list(value.applied_controls)
    if change == "partial":
        controls.pop()
    elif change == "skip-tick":
        controls[2] = replace(controls[2], physics_step=99)
    elif change == "finger-gravity":
        controls[2] = replace(controls[2], gravity_efforts=(0.1,) * 9)
    elif change == "velocity":
        controls[2] = replace(controls[2], commanded_joint_velocities=(0.1,) * 9)
    elif change == "target":
        controls[2] = replace(controls[2], commanded_joint_targets=(0.001, *JOINTS[1:]))
    elif change == "hold-late":
        value = replace(value, hold_deadline_ns=4_300_000_001)
    else:
        value = replace(value, policy_finished_ns=4_000_000_001)
    with pytest.raises(ContractError):
        replace(value, applied_controls=tuple(controls)).validate(profile())
