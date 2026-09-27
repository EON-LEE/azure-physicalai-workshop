"""Synthetic CPU measurement fixtures check report integrity, not live physics or model quality."""

import base64
from dataclasses import asdict, replace
from datetime import timedelta

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_batch_simulation_task import gpu
from test_learned_managed_lifecycle import authority as authority
from test_simulation_batch import spec as spec

from learning.checks.fixtures import JOINTS, png
from learning.common import canonical, digest
from learning.contract import ValidatedDataset, ValidatedEpisode
from learning.paused import FrozenCameraSample
from learning.paused.rollout import derive_trial
from learning.paused.task import TaskState
from simulation import batch_learned, learned_probe


@pytest.fixture
def evidence(authority, tmp_path):
    spec, documents, scene, profile = authority
    _, _, _, grant = batch_learned.validate_inputs(spec, documents)
    now = grant.issued_at + timedelta(seconds=1)
    initial = scene.part_position
    goal = scene.station(grant.authorization.task.goal_id).position
    positions = [(initial, (initial[0], initial[1], initial[2] + 0.003), 0.02)]

    def move(part, tcp, finger, count):
        a, b, old_finger = positions[-1]
        for i in range(1, count + 1):
            t = i / count
            positions.append(
                (
                    tuple(x + (y - x) * t for x, y in zip(a, part, strict=True)),
                    tuple(x + (y - x) * t for x, y in zip(b, tcp, strict=True)),
                    old_finger + (finger - old_finger) * t,
                )
            )

    lift = (*initial[:2], initial[2] + 0.06)
    move(lift, (*lift[:2], lift[2] + 0.003), 0.02, 60)
    above = (*goal[:2], goal[2] + 0.06)
    move(above, (*above[:2], above[2] + 0.003), 0.02, 240)
    move(goal, (*goal[:2], goal[2] + 0.003), 0.02, 60)
    move(goal, (*goal[:2], goal[2] + 0.003), 0.04, 24)
    move(goal, (*goal[:2], goal[2] + 0.08), 0.04, 60)
    move(*positions[-1], 30)
    states = [
        TaskState(
            captured_at_utc=(now + timedelta(seconds=i / 60)).isoformat().replace("+00:00", "Z"),
            monotonic_ns=1_000_000_000 + i * 20_000_000,
            physics_step=60 + i,
            object_position_m=part,
            object_linear_velocity_m_s=(0.0, 0.0, 0.0),
            tcp_position_m=tcp,
            joint_positions=(*JOINTS[:7], finger, finger),
            command_id=str(spec.attempt_id),
            epoch="77777777-7777-4777-8777-777777777777",
            policy_predict_calls=(i + 5) // 6,
            applied_action_count=i,
            reference_route_calls=0,
            applied_model_sha256=spec.model.manifest.sha256 if i else None,
        )
        for i, (part, tcp, finger) in enumerate(positions)
    ]
    frames = []
    for i in range(0, len(states) - 1, 6):
        first, last = states[i], states[i + 6]
        frames.append(
            {
                "physics_step": first.physics_step,
                "joint_positions": list(first.joint_positions),
                "epoch": first.epoch,
                "observation_started_ns": first.monotonic_ns,
                "monotonic_ns": first.monotonic_ns + 1_000_000,
                "initial_publication": None,
                "hold_started_ns": first.monotonic_ns + 2_000_000,
                "hold_deadline_ns": last.monotonic_ns + 1_000_000,
                "policy_started_ns": first.monotonic_ns + 1_000_000,
                "policy_finished_ns": first.monotonic_ns + 2_000_000,
                "applied_controls": [
                    {
                        "physics_step": state.physics_step,
                        "monotonic_ns": state.monotonic_ns,
                    }
                    for state in states[i + 1 : i + 7]
                ],
                "simulation_time_numerator": first.physics_step,
                "simulation_time_denominator": 60,
                "terminated": i + 7 == len(states),
                "truncated": i + 7 != len(states),
            }
        )
    case = learned_probe.physical_case(spec, scene, grant)
    metadata = {
        "episode_id": str(spec.attempt_id),
        "environment_id": case["environment_id"],
        "revision": case["revision"],
        "seed": case["seed"],
        "split": scene.demonstration_split,
        "frame_count": len(frames),
        "demonstration": {"kind": "learned", "source_policy_sha256": spec.model.manifest.sha256},
        "provenance": {
            "scene_builder_sha256": scene.scene_builder_sha256,
            "code_revision": spec.source_revision,
            "simulator_image_digest": spec.platform.container_image.split("@", 1)[1],
            "robot_asset_sha256": spec.asset_bundle.sha256,
        },
        "budget": {
            "started_ns": 1_000_000_000,
            "wall_deadline_ns": 601_000_000_000,
            "initial_physics_step": 60,
            "simulation_step_deadline": 3660,
        },
    }
    manifest = {
        "schema": "physicalai.demonstrations/v3",
        "purpose": "evaluation",
        "scope": {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id},
        "control_profile": asdict(profile),
        "criteria_sha256": spec.criteria_canonical_sha256,
        "frozen_plan_sha256": spec.conditions_canonical_sha256,
        "episodes": [metadata],
    }
    raw = ValidatedDataset(
        tmp_path,
        manifest,
        digest(canonical(manifest)),
        (ValidatedEpisode(metadata, tuple(frames)),),
    )
    last = states[-1]
    images = {
        name: FrozenCameraSample(
            png(320, 320),
            last.physics_step,
            last.physics_step,
            last.monotonic_ns + 1_000_000,
            (now + timedelta(seconds=len(states) / 60)).isoformat().replace("+00:00", "Z"),
            last.physics_step,
            60,
        )
        for name in ("overview", "inspection")
    }
    trial = derive_trial(
        raw,
        states,
        case=case,
        role=spec.role,
        model_sha256=spec.model.manifest.sha256,
        profile=profile,
        final_images=images,
        heartbeat_ns=tuple(state.monotonic_ns for state in states),
        destination_id=grant.authorization.task.goal_id,
        failure_reason=None,
    )
    from test_paused_control import state as frozen_state
    from test_paused_gripper_servo import AUTHORED, ORIGINAL

    from simulation.paused_gripper_servo import CALIBRATION_ID

    frozen = {**asdict(frozen_state()), "epoch": last.epoch, "physics_step": 60}
    report = {
        "schema": learned_probe.SCHEMA,
        "controller": "learned",
        "policy_type": "smolvla",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "source_revision": spec.source_revision,
        "simulator_image_digest": spec.platform.container_image.split("@", 1)[1],
        "model_sha256": spec.model.manifest.sha256,
        "control_profile_sha256": profile.sha256,
        "criteria_sha256": spec.criteria_canonical_sha256,
        "frozen_plan_sha256": spec.conditions_canonical_sha256,
        "operator_grant_sha256": spec.grant.sha256,
        "command_id": str(spec.attempt_id),
        "environment_id": case["environment_id"],
        "revision": case["revision"],
        "physical_status": "succeeded",
        "model_weights_loaded": True,
        "learning_quality_proven": False,
        "metrics": {
            "controller": "learned",
            "applied_model_sha256": spec.model.manifest.sha256,
            "applied_action_count": last.applied_action_count,
            "policy_predict_calls": last.policy_predict_calls,
            "reference_route_calls": 0,
            "simulation_steps": last.applied_action_count,
        },
        "task_states": [asdict(state) for state in states],
        "heartbeat_ns": [state.monotonic_ns for state in states],
        "episode_budget": metadata["budget"],
        "trial": trial,
        "final_images": {name: learned_probe.encode_image(image) for name, image in images.items()},
        "capture": {
            "status": "ready",
            "receipt": {
                "status": "uploaded",
                "manifest_sha256": raw.manifest_sha256,
                "episode_id": str(spec.attempt_id),
                "frame_count": len(frames),
                "manifest_uri": f"{spec.storage_account_url}/{spec.output_container}/"
                f"{spec.owner_id}/{spec.attempt_id}/manifest.json",
            },
        },
        "gripper_servo": {
            "calibration_id": CALIBRATION_ID,
            "status": "restored",
            "contact_forces_measured": False,
            "owner": {
                "owner": spec.owner_id,
                "command_id": str(spec.attempt_id),
                "environment_id": case["environment_id"],
                "revision": case["revision"],
                "epoch": last.epoch,
            },
            "authored": AUTHORED,
            "before": asdict(ORIGINAL),
            "after": asdict(replace(ORIGINAL, stiffness=ORIGINAL.stiffness[:7] + (2000.0, 0.0))),
            "restored": asdict(ORIGINAL),
            "frozen_state": {"before": frozen, "after": frozen},
        },
    }
    descriptor = canonical(
        {
            "schema": "physicalai.paused-model-runtime/v1",
            "python_executable": batch_learned.MODEL_PYTHON,
            "python_sha256": "a" * 64,
            "python_version": "3.11",
            "code_root": "/work",
            "dependency_image": "unit.azurecr.io/ml@sha256:" + "b" * 64,
            "code_image": "unit.azurecr.io/ml@sha256:" + "c" * 64,
            "source_files": {"learning/paused/model.py": "d" * 64},
        }
    )
    value = spec.model_dump(mode="json", by_alias=True)
    value["model_runtime"].update(sha256=digest(descriptor), size_bytes=len(descriptor))
    spec = batch_learned.BatchLearnedSpec.model_validate(value)
    preflight = gpu() | {
        "model_process": {
            "model_sha256": spec.model.manifest.sha256,
            "backbone_sha256": spec.backbone.manifest.sha256,
            "runtime_sha256": spec.model_runtime.sha256,
            "python": {"version": [3, 11]},
            "runtime_descriptor_base64": base64.b64encode(descriptor).decode(),
            "socket_transport": "same-kernel-unix-so_peercred",
        }
    }
    return (
        spec,
        scene,
        profile,
        grant,
        report,
        {
            **{f"inputs/{name}.json": payload for name, payload in documents.items()},
            "inputs/spec.json": canonical(spec.model_dump(mode="json", by_alias=True)),
            "preflight.json": canonical(preflight),
            "raw-manifest.json": canonical(manifest),
            "probe.log": b"CPU fixture, not actual GPU",
            "acceptance.log": b"CPU fixture",
        },
    )


def test_native_single_trial_rescores_measured_grasp_release_and_settle(evidence):
    spec, scene, profile, grant, report, documents = evidence
    result = learned_probe.rescore(report, spec, scene, profile, grant)
    assert result["accepted"] is True and result["reference_route_calls"] == 0
    assert result["task_evidence"]["grasp_verified"] and result["task_evidence"]["settled"]
    assert result["policy_predict_calls"] * 6 == result["applied_action_count"]
    documents.update({"probe.json": canonical(report), "acceptance.json": canonical(result)})
    assert batch_learned.verify_evidence(spec, documents) == report["capture"]["receipt"]


@pytest.mark.parametrize(
    "changed", ["servo", "images", "unbound_timing", "actuator_model", "heartbeat", "budget"]
)
def test_accepted_flag_cannot_waive_measured_servo_camera_or_timing_proof(evidence, changed):
    spec, scene, profile, grant, report, _ = evidence
    if changed == "servo":
        report.pop("gripper_servo")
    elif changed == "images":
        report["final_images"] = {}
    elif changed == "unbound_timing":
        report["final_images"]["overview"]["monotonic_ns"] += 2_000_000_001
    elif changed == "actuator_model":
        report["metrics"]["applied_model_sha256"] = "f" * 64
    elif changed == "heartbeat":
        report["heartbeat_ns"][2] += 10_000_000
    else:
        report["episode_budget"]["started_ns"] += 1
    with pytest.raises(ValueError):
        learned_probe.rescore(report, spec, scene, profile, grant)


def test_actual_private_status_uses_learned_not_reference_validator(evidence):
    from types import SimpleNamespace

    from test_batch_simulation_proof import Blob

    spec, scene, profile, grant, report, documents = evidence
    result = learned_probe.rescore(report, spec, scene, profile, grant)
    documents.update({"probe.json": canonical(report), "acceptance.json": canonical(result)})
    proof = {
        "schema": "physicalai.batch-simulation-result/v1",
        "attempt_id": str(spec.attempt_id),
        "previous_attempt_id": None,
        "spec_sha256": spec.sha256,
        "spec_file_sha256": digest(documents["inputs/spec.json"]),
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "source_revision": spec.source_revision,
        "image": spec.platform.container_image,
        "control_profile_sha256": spec.control_profile_sha256,
        "profile_id": spec.profile_id,
        "accepted": True,
        "outcome": "accepted",
        "native_acceptance": "accepted",
        "learning_quality_proven": False,
        "physical_status": "succeeded",
        "capture_status": "ready",
        "raw_manifest": report["capture"]["receipt"],
        "artifacts": [
            {"path": name, "bytes": len(value), "sha256": digest(value)}
            for name, value in documents.items()
        ],
    }
    blobs = {spec.artifact_prefix + "/" + name: value for name, value in documents.items()}
    blobs[spec.artifact_prefix + "/completion.json"] = canonical(proof)
    raw_name = f"{spec.owner_id}/{spec.attempt_id}/manifest.json"
    blobs[raw_name] = documents["raw-manifest.json"]
    store = object.__new__(batch_learned.LearnedArtifacts)
    store.spec = spec
    store.client = SimpleNamespace(get_blob_client=lambda container, name: Blob(blobs[name]))
    assert store.read_completion()["accepted"] is True
    blobs[raw_name] = b"{}"
    with pytest.raises(ValueError, match="Remote raw manifest"):
        store.read_completion()
