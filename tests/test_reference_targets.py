"""CPU reference planning tests; the unavailable third GPU proposal is not reconstructed."""

from math import dist

import pytest

from learning.contract import DEFAULT_JOINT_VELOCITY_LIMITS, validate_joint_tracking
from simulation.control import validate_position_target
from simulation.reference_targets import plan_reference_targets, reference_tracking_violations

INITIAL = (
    0.012000000104308128,
    -0.5700000524520874,
    7.042015408298408e-12,
    -2.809999942779541,
    -5.396061028561938e-11,
    3.0369999408721924,
    0.7409999966621399,
    0.03999999910593033,
    0.03999999910593033,
)
FIRST_ISSUED = (
    0.011224433779716492,
    -0.5711446404457092,
    0.0008566574542783201,
    -2.802903890609741,
    1.7120983102358878e-05,
    3.015533924102783,
    0.742156445980072,
    0.04,
    0.04,
)
MEASURED_FRAME_1 = (
    0.011694682762026787,
    -0.5704466700553894,
    0.0003366578894201666,
    -2.8072128295898438,
    6.73422164254589e-06,
    3.0285627841949463,
    0.741454541683197,
    0.039999980479478836,
    0.039999980479478836,
)
SECOND_ISSUED = (
    0.010234907269477844,
    -0.5720019340515137,
    0.0019592351745814085,
    -2.7932844161987305,
    0.00011370430001989007,
    2.9875681400299072,
    0.7436054348945618,
    0.04,
    0.04,
)


def fk(joints):
    return (0.35 + 0.1 * joints[0], 0.25 + 0.1 * joints[2], 0.35)


def plan(measured, requested, previous=None, *, forward=fk):
    return plan_reference_targets(
        measured, requested, previous, forward_kinematics=forward, max_cartesian_speed_m_s=0.1
    )


def test_an_already_bounded_observed_reference_target_is_unchanged():
    actual = plan(INITIAL, FIRST_ISSUED)
    assert actual.targets == FIRST_ISSUED
    assert actual.path_fraction == 1
    validate_joint_tracking(actual.targets, INITIAL, None, fps=10)
    second = plan(MEASURED_FRAME_1, SECOND_ISSUED, FIRST_ISSUED)
    assert second.targets == SECOND_ISSUED
    assert MEASURED_FRAME_1[5] - SECOND_ISSUED[5] == pytest.approx(0.04099464416503906)


def test_synthetic_lag_after_observed_valid_targets_gets_a_bounded_reference_trajectory():
    # This scenario uses the last observed q and issued targets. Its next RMP
    # endpoint and assumed continued lag are synthetic, not the missing third observation.
    measured = MEASURED_FRAME_1
    requested = list(SECOND_ISSUED)
    requested[0] += 0.02
    requested[5] -= 0.08
    measured, requested = tuple(measured), tuple(requested)
    with pytest.raises(ValueError, match="tracking"):
        validate_position_target(measured, requested, SECOND_ISSUED)
    result = plan(measured, requested, SECOND_ISSUED)
    assert 0 < result.path_fraction < 1
    assert result.targets != requested
    validate_joint_tracking(result.targets, measured, SECOND_ISSUED, fps=10)
    fractions = [
        (new - start) / (end - start)
        for start, end, new in zip(measured[:7], requested[:7], result.targets[:7], strict=True)
        if abs(end - start) > 1e-10
    ]
    assert fractions == pytest.approx([result.path_fraction] * len(fractions))
    assert measured[5] > result.targets[5] > requested[5]
    assert result.targets[7:] == requested[7:]
    assert (
        max(abs(a - b) for a, b in zip(measured[:7], result.targets[:7], strict=True))
        <= 0.045 + 1e-12
    )


def test_measured_tracking_and_previous_target_slew_are_intersected_not_chosen_independently():
    measured = INITIAL
    previous = tuple(value - 0.04 if index == 0 else value for index, value in enumerate(INITIAL))
    requested = tuple(value + 0.04 if index == 0 else value for index, value in enumerate(INITIAL))
    result = plan(measured, requested, previous)
    assert result.targets[0] == pytest.approx(measured[0] + 0.005)
    validate_joint_tracking(result.targets, measured, previous, fps=10)
    assert result.limiting_joint_indices == (0,)


def test_headroom_limiter_is_identified_even_when_the_raw_goal_passes_the_hard_guard():
    requested = (INITIAL[0] + 0.047,) + INITIAL[1:]
    validate_position_target(INITIAL, requested, INITIAL)
    result = plan(INITIAL, requested, INITIAL)
    assert result.limiting_joint_indices == (0,)
    assert result.targets[0] - INITIAL[0] == pytest.approx(0.045)


def test_empty_tracking_slew_intersection_fails_explicitly_instead_of_returning_a_hold():
    previous = tuple(value - 0.2 if index == 0 else value for index, value in enumerate(INITIAL))
    with pytest.raises(ValueError, match="feasible"):
        plan(INITIAL, FIRST_ISSUED, previous)


def test_opposite_proposal_without_safe_forward_progress_is_rejected():
    previous = tuple(value - 0.045 if index == 0 else value for index, value in enumerate(INITIAL))
    requested = tuple(value + 0.1 if index == 0 else value for index, value in enumerate(INITIAL))
    with pytest.raises(ValueError, match="progress"):
        plan(INITIAL, requested, previous)


def test_legitimate_zero_arm_goal_hold_and_bounded_contact_pressure_are_preserved():
    measured = INITIAL[:7] + (0.02, 0.02)
    requested = measured[:7] + (0.0183, 0.0183)
    result = plan(measured, requested, measured)
    assert result.targets == requested
    retained = plan(measured, requested, requested)
    assert retained.targets[7:] == requested[7:]
    stationary = plan(measured, measured, measured)
    assert stationary.targets == measured


def test_limiting_the_arm_path_does_not_release_or_rescale_the_finger_pressure_target():
    measured = INITIAL[:7] + (0.02, 0.02)
    requested = (measured[0] + 0.1,) + measured[1:7] + (0.0183, 0.0183)
    result = plan(measured, requested, measured)
    assert 0 < result.path_fraction < 1
    assert result.targets[7:] == requested[7:]
    validate_joint_tracking(result.targets, measured, measured, fps=10)


def test_a_neutral_finger_hold_keeps_the_existing_contact_target():
    measured = INITIAL[:7] + (0.02, 0.02)
    previous = measured[:7] + (0.0183, 0.0183)
    requested = (measured[0] + 0.1,) + measured[1:7] + previous[7:]
    assert plan(measured, requested, previous).targets[7:] == previous[7:]


def test_cartesian_search_is_finite_and_never_crosses_the_previous_command_lower_bound():
    calls = []

    def sensitive_fk(joints):
        calls.append(joints)
        return (0.35 + 10 * (joints[0] - INITIAL[0]), 0.25, 0.35)

    previous = (INITIAL[0] + 0.06,) + INITIAL[1:]
    requested = (INITIAL[0] + 0.1,) + INITIAL[1:]
    with pytest.raises(ValueError, match="Cartesian"):
        plan(INITIAL, requested, previous, forward=sensitive_fk)
    assert len(calls) <= 9


def test_forward_kinematics_limits_the_uniform_path_without_changing_joint_or_tcp_guards():
    calls = []

    def sensitive_fk(joints):
        calls.append(joints)
        return (0.35 + joints[0], 0.25, 0.35)

    requested = tuple(value + 0.05 if index == 0 else value for index, value in enumerate(INITIAL))
    result = plan(INITIAL, requested, INITIAL, forward=sensitive_fk)
    assert len(calls) <= 9  # One origin plus at most eight candidate evaluations.
    assert result.predicted_tcp_step_m <= 0.01
    assert dist(sensitive_fk(INITIAL), sensitive_fk(result.targets)) <= 0.01
    validate_joint_tracking(result.targets, INITIAL, INITIAL, fps=10)


@pytest.mark.parametrize(
    "requested",
    [
        (float("nan"),) + INITIAL[1:],
        (3.0,) + INITIAL[1:],
    ],
)
def test_invalid_rmp_outputs_cannot_be_planned_into_success(requested):
    with pytest.raises(ValueError):
        plan(INITIAL, requested)


def test_diagnostics_identify_tracking_and_slew_joint_values_and_unchanged_units():
    requested = INITIAL[:5] + (INITIAL[5] - 0.08,) + INITIAL[6:7] + (0.03, 0.04)
    violations = reference_tracking_violations(INITIAL, requested, INITIAL)
    arm = next(
        item
        for item in violations
        if item["joint_name"] == "panda_joint6" and item["constraint"] == "tracking"
    )
    assert arm["measured"] == INITIAL[5]
    assert arm["requested"] == requested[5]
    assert arm["limit_per_interval"] == DEFAULT_JOINT_VELOCITY_LIMITS[5] / 10
    assert arm["unit"] == "rad"
    finger = next(item for item in violations if item["joint_name"] == "panda_finger_joint1")
    assert finger["unit"] == "m" and finger["limit_per_interval"] == 0.004
