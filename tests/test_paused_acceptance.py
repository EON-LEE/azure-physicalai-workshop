"""CPU report gate tests, never an actual demonstration or model-quality proof."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest
from test_paused_control import state
from test_paused_dispatch import paused_core as paused_core
from test_paused_gripper_servo import AUTHORED, ORIGINAL

from simulation.paused_acceptance import validate_attempt
from simulation.paused_gripper_servo import CALIBRATION_ID


def report(paused_core):
    core, record, request = paused_core
    frozen = {
        **asdict(replace(state(), epoch=core.epoch, physics_step=60, world_time=1.0)),
        "epoch": str(core.epoch),
    }
    return record, {
        "schema": "physicalai.paused-reference-attempt/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "mode": "reference-task",
        "physical_status": "succeeded",
        "environment_id": record.environment_id,
        "revision": record.revision,
        "task": request.task.model_dump(),
        "source_kind": "reference_controller",
        "source_revision": "a" * 40,
        "command_id": str(request.command_id),
        "simulator_image_digest": "sha256:" + "b" * 64,
        "control_profile_sha256": core.paused_profile.sha256,
        "criteria_sha256": "c" * 64,
        "frozen_plan_sha256": "d" * 64,
        "physical_task_success": True,
        "model_weights_loaded": False,
        "learning_quality_proven": False,
        "grasp_verified": True,
        "peak_tcp_speed_m_s": 0.19,
        "final_position": [0.22, -0.38, 0.2],
        "initial_state": {
            "observed_initial_pose_m": list(core.spec.part_position),
            "scene_builder_sha256": core.spec.scene_builder_sha256,
        },
        "gripper_servo": {
            "calibration_id": CALIBRATION_ID,
            "status": "restored",
            "contact_forces_measured": False,
            "owner": {
                "owner": core.owner,
                "command_id": str(request.command_id),
                "environment_id": record.environment_id,
                "revision": record.revision,
                "epoch": str(core.epoch),
            },
            "authored": AUTHORED,
            "before": asdict(ORIGINAL),
            "after": asdict(replace(ORIGINAL, stiffness=ORIGINAL.stiffness[:7] + (2000.0, 0.0))),
            "restored": asdict(ORIGINAL),
            "frozen_state": {
                "before": frozen,
                "after": deepcopy(frozen),
            },
        },
        "capture": {"status": "ready", "receipt": {"status": "uploaded", "frame_count": 2}},
        "metrics": {
            "simulation_steps": 12,
            "simulation_elapsed_seconds": 0.2,
            "wall_elapsed_ms": 1000,
            "intervals": [
                {
                    "observation_physics_step": 60 + tick * 6,
                    "completed_physics_step": 66 + tick * 6,
                    "observation_wall_ms": 100,
                    "policy_wall_ms": 100,
                    "hold_wall_ms": 100,
                    "interval_wall_ms": 300,
                }
                for tick in range(2)
            ],
        },
    }


def test_paused_reference_success_requires_its_own_complete_physical_and_capture_evidence(
    paused_core,
):
    environment, value = report(paused_core)
    result = validate_attempt(value, environment=environment, mode="reference-task")
    assert result["accepted"] is True and result["real_time_admission"] is False


@pytest.mark.parametrize(
    "field,changed",
    [
        ("physical_status", "failed"),
        ("grasp_verified", False),
        ("real_time_admission", True),
        ("peak_tcp_speed_m_s", 0.201),
        ("final_position", [0.5, 0.5, 0.5]),
        ("schema", "physicalai.gpu-control-probe/v1"),
        ("model_weights_loaded", True),
    ],
)
def test_wrong_mode_failed_motion_or_widened_physical_limits_cannot_pass(
    paused_core, field, changed
):
    environment, value = report(paused_core)
    with pytest.raises(ValueError):
        validate_attempt(value | {field: changed}, environment=environment, mode="reference-task")


def test_new_mode_still_rejects_an_over_budget_or_omitted_actual_interval(paused_core):
    environment, value = report(paused_core)
    altered = deepcopy(value)
    altered["metrics"]["intervals"][1]["interval_wall_ms"] = 5001
    with pytest.raises(ValueError):
        validate_attempt(altered, environment=environment, mode="reference-task")
    altered = deepcopy(value)
    altered["metrics"]["simulation_steps"] = 13
    with pytest.raises(ValueError):
        validate_attempt(altered, environment=environment, mode="reference-task")


@pytest.mark.parametrize(
    "tamper", ["missing", "ignored", "force", "arm", "restore", "owner", "physics"]
)
def test_paused_gate_requires_actual_owned_gain_readback_not_just_a_success_flag(
    paused_core, tamper
):
    environment, value = report(paused_core)
    servo = deepcopy(value["gripper_servo"])
    value["gripper_servo"] = servo
    if tamper == "missing":
        value.pop("gripper_servo")
    elif tamper == "ignored":
        servo["after"] = servo["before"]
    elif tamper == "force":
        servo["after"]["max_effort"] = ORIGINAL.max_effort[:7] + (8.0, 0.0)
    elif tamper == "arm":
        servo["after"]["stiffness"] = (10.0,) + ORIGINAL.stiffness[1:7] + (2000.0, 0.0)
    elif tamper == "restore":
        servo["restore_error"] = "Foreign drive change"
    elif tamper == "owner":
        servo["owner"]["command_id"] = "different"
    else:
        servo["frozen_state"]["after"]["physics_step"] += 1
    with pytest.raises(ValueError, match="servo|gain|calibration|stiffness"):
        validate_attempt(value, environment=environment, mode="reference-task")
