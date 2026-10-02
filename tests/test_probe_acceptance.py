"""CPU JSON gate tests, not real GPU or model-quality evidence."""

from copy import deepcopy

import pytest

from simulation.probe_acceptance import validate_probe_report


def report(mode="teaching"):
    value = {
        "schema": "physicalai.gpu-control-probe/v1",
        "mode": mode,
        "demonstrator_kind": "reference_controller",
        "environment_id": "learning-g0-teaching",
        "revision": "a" * 64,
        "probe_completed": True,
        "model_weights_loaded": False,
        "learning_quality_proven": False,
        "production_ready": False,
        "physical_task_success": False,
        "physical_status": "cancelled",
        "max_heartbeat_gap_ms": 1100,
        "initial_state": {
            "observed_initial_pose_m": [0.35, 0.25, 0.2],
            "scene_builder_sha256": "b" * 64,
        },
        "control_profile_sha256": "c" * 64,
        "control_intervals": [
            {
                "observation_step": 24 + index * 6,
                "completed_step": 30 + index * 6,
                "control_cycle_ms": 50,
                "inference_latency_ms": 0,
            }
            for index in range(100)
        ],
        "capture": {
            "status": "ready",
            "receipt": {"status": "uploaded", "frame_count": 100, "manifest_sha256": "d" * 64},
        },
        "fixture_actuation": None,
    }
    if mode == "policy-fixture":
        value["fixture_actuation"] = {
            "applied_fixture_sha256": "e" * 64,
            "reference_route_calls": 0,
            "policy_predict_calls": 100,
            "applied_action_count": 600,
        }
    return value


def validate(value, mode="teaching"):
    return validate_probe_report(
        value,
        mode=mode,
        environment_id="learning-g0-teaching",
        revision="a" * 64,
    )


def test_host_accepts_only_the_actual_complete_bound_mechanics_receipt():
    value = validate(report())
    assert value["accepted"] is True
    assert value["control_intervals"] == 100
    assert value["learning_quality_proven"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("probe_completed", False),
        ("schema", "physicalai.operator-attempt-failure/v1"),
        ("revision", "f" * 64),
        ("physical_status", "failed"),
        ("max_heartbeat_gap_ms", 2001),
        ("model_weights_loaded", True),
        ("control_intervals", []),
    ],
)
def test_zero_exit_or_report_presence_cannot_replace_failed_evidence(field, value):
    altered = report() | {field: value}
    with pytest.raises(ValueError):
        validate(altered)


@pytest.mark.parametrize("failure", ["capture", "frames", "physics", "cadence", "latency"])
def test_host_rejects_incomplete_capture_or_widened_guards(failure):
    altered = deepcopy(report())
    if failure == "capture":
        altered["capture"]["status"] = "invalid"
    elif failure == "frames":
        altered["capture"]["receipt"]["frame_count"] = 99
    elif failure == "physics":
        altered["control_intervals"][50]["completed_step"] += 1
    elif failure == "cadence":
        altered["control_intervals"][50]["control_cycle_ms"] = 101
    else:
        altered["control_intervals"][50]["inference_latency_ms"] = 81
    with pytest.raises(ValueError):
        validate(altered)


def test_fixture_needs_actual_model_port_applications_and_zero_reference_route_calls():
    valid = report("policy-fixture")
    assert validate(valid, "policy-fixture")["applied_action_count"] == 600
    for field, value in (
        ("reference_route_calls", 1),
        ("applied_action_count", 599),
        ("policy_predict_calls", 0),
    ):
        altered = deepcopy(valid)
        altered["fixture_actuation"][field] = value
        with pytest.raises(ValueError):
            validate(altered, "policy-fixture")
