"""Real native codecs over CPU-only metadata/pixels; no Azure/model/physics quality evidence."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
from uuid import UUID

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_paired_evaluation import mapping_value as mapping_value
from test_simulation_batch import spec as spec

from learning.checks.fixtures import png
from learning.common import canonical, digest, file_digest, read_json, utc
from learning.contract import AppliedControl, DemonstrationSource, EpisodeSpec, Provenance, Scope
from learning.paused import FrozenCameraSample, FrozenPolicyObservation, PausedFrameSample
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter, validate_dataset
from learning.paused.rollout import derive_trial
from learning.paused.task import TaskState
from simulation import batch_learned, learned_probe, paired_evaluation
from tests.learning.test_paused_model_artifacts import fixture


def encoded(value):
    return canonical(value) + b"\n"


def inventory(root):
    return {
        path.relative_to(root).as_posix(): {
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def recording(evidence, mapping_value, tmp_path):
    original, scene, profile, grant, report, documents = evidence
    spec_value = original.model_dump(mode="json", by_alias=True)
    spec_value["pairing_plan_sha256"] = "a" * 64
    models, model_roots = {}, {role: tmp_path / "models" / role for role in ("before", "after")}
    for role, root in model_roots.items():
        fixture(root)
        model = read_json(root / "model.json")
        model.update(
            scope={"tenant_id": str(original.platform.tenant_id), "owner_id": original.owner_id},
            control_profile=asdict(profile),
            task=grant.authorization.task.model_dump(),
            criteria_sha256=original.criteria_canonical_sha256,
            frozen_plan_sha256=original.conditions_canonical_sha256,
        )
        model["training"].update(
            criteria_sha256=model["criteria_sha256"],
            frozen_plan_sha256=model["frozen_plan_sha256"],
            episodes=[{"episode_id": "cpu-training-10001", "seed": 10001}],
        )
        if role == "after":
            model["training"]["parent_model_sha256"] = file_digest(
                model_roots["before"] / "model.json"
            )
        (root / "model.json").write_bytes(encoded(model))
        models[role] = model
    spec_value["role"] = "before"
    spec_value["model"]["manifest"].update(
        sha256=file_digest(model_roots["before"] / "model.json"),
        size_bytes=(model_roots["before"] / "model.json").stat().st_size,
    )
    spec_value["model"]["files"] = [
        {
            "path": "checkpoint/" + name,
            "sha256": checksum,
            "size_bytes": (model_roots["before"] / "checkpoint" / name).stat().st_size,
        }
        for name, checksum in models["before"]["checkpoint_files"].items()
    ]
    spec_value["backbone"]["manifest"]["sha256"] = models["before"]["backbone_manifest_sha256"]
    grant = grant.model_copy(
        update={
            "authorization": grant.authorization.model_copy(
                update={"model_sha256": spec_value["model"]["manifest"]["sha256"]}
            )
        }
    )
    documents["inputs/grant.json"] = encoded(grant.model_dump(mode="json", by_alias=True))
    spec_value["grant"].update(
        sha256=digest(documents["inputs/grant.json"]),
        size_bytes=len(documents["inputs/grant.json"]),
    )
    spec = batch_learned.BatchLearnedSpec.model_validate(spec_value)
    documents["inputs/spec.json"] = encoded(spec.model_dump(mode="json", by_alias=True))
    directory = tmp_path / "attempts" / str(spec.attempt_id)
    directory.mkdir(parents=True)
    provenance = Provenance(
        source_kind="isaac_sim",
        simulator_version="6.0.0",
        simulator_image_digest=spec.platform.container_image.split("@", 1)[1],
        robot_asset_sha256=spec.asset_bundle.sha256,
        scene_builder_id="inspection-cell-learning-v1",
        scene_builder_sha256=scene.scene_builder_sha256,
        code_revision=spec.source_revision,
        capture_host="azure_gpu",
        gpu_model="CPU metadata double; not actual hardware",
    )
    budget = PausedEpisodeBudget(**report["episode_budget"])
    writer = PausedEpisodeWriter(
        directory / "capture",
        dataset_id="cpu-codec-only",
        scope=Scope(**models["before"]["scope"]),
        episode=EpisodeSpec(
            str(spec.attempt_id),
            grant.authorization.environment_id,
            grant.authorization.revision,
            scene.seed,
            "test",
        ),
        provenance=provenance,
        profile=profile,
        demonstration=DemonstrationSource(
            "learned",
            **grant.authorization.task.model_dump(),
            source_policy_sha256=spec.model.manifest.sha256,
        ),
        budget=budget,
        purpose="evaluation",
        criteria_sha256=spec.criteria_canonical_sha256,
        frozen_plan_sha256=spec.conditions_canonical_sha256,
    )
    states = [TaskState(**value) for value in report["task_states"][:13]]
    states = [
        replace(value, applied_model_sha256=spec.model.manifest.sha256 if index else None)
        for index, value in enumerate(states)
    ]
    for offset in (0, 6):
        first, last = states[offset], states[offset + 6]
        ready = first.monotonic_ns + 1_000_000
        stamp = (
            (utc(first.captured_at_utc) + timedelta(milliseconds=1))
            .isoformat()
            .replace("+00:00", "Z")
        )
        observation = FrozenPolicyObservation(
            scope=Scope(**models["before"]["scope"]),
            environment_id=grant.authorization.environment_id,
            revision=grant.authorization.revision,
            episode_id=str(spec.attempt_id),
            epoch=first.epoch,
            captured_at_utc=stamp,
            monotonic_ns=ready,
            physics_step=first.physics_step,
            joint_positions=first.joint_positions,
            images={
                name: FrozenCameraSample(
                    png(320, 320),
                    100 + offset,
                    first.physics_step,
                    ready,
                    stamp,
                    first.physics_step,
                    60,
                )
                for name in ("inspection", "overview")
            },
            freeze_id=f"cpu-freeze-{offset}",
            state_revision=offset + 1,
            control_tick=offset // 6,
            control_profile_sha256=profile.sha256,
            observation_started_ns=first.monotonic_ns,
            joint_sample_ns=ready,
            simulation_time_numerator=first.physics_step,
            simulation_time_denominator=60,
        )
        writer.append(
            PausedFrameSample(
                observation=observation,
                commanded_joint_targets=first.joint_positions,
                applied_controls=tuple(
                    AppliedControl(
                        state.physics_step,
                        state.monotonic_ns,
                        first.joint_positions,
                        (0.0,) * 9,
                        (0.0,) * 9,
                    )
                    for state in states[offset + 1 : offset + 7]
                ),
                interval_deadline_ns=first.monotonic_ns + 500_000_000,
                hold_started_ns=ready + 1_000_000,
                hold_deadline_ns=last.monotonic_ns + 10_000_000,
                policy_started_ns=ready,
                policy_finished_ns=ready + 1_000_000,
                terminated=False,
                truncated=offset == 6,
            )
        )
    writer.finalize()
    raw = validate_dataset(
        writer.root, expected_scope=Scope(**models["before"]["scope"]), require_live=True
    )
    last = states[-1]
    images = {
        name: FrozenCameraSample(
            png(320, 320),
            200,
            last.physics_step,
            last.monotonic_ns + 1_000_000,
            (utc(last.captured_at_utc) + timedelta(milliseconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
            last.physics_step,
            60,
        )
        for name in ("overview", "inspection")
    }
    trial = derive_trial(
        raw,
        states,
        case=learned_probe.physical_case(spec, scene, grant),
        role=spec.role,
        model_sha256=spec.model.manifest.sha256,
        profile=profile,
        final_images=images,
        heartbeat_ns=tuple(state.monotonic_ns for state in states),
        destination_id=grant.authorization.task.goal_id,
        failure_reason="CPU fixture cancellation",
    )
    report.update(
        model_sha256=spec.model.manifest.sha256,
        operator_grant_sha256=spec.grant.sha256,
        physical_status="cancelled",
        error={"message": "CPU fixture cancellation"},
        task_states=[asdict(state) for state in states],
        heartbeat_ns=[s.monotonic_ns for s in states],
        final_images={name: learned_probe.encode_image(image) for name, image in images.items()},
        trial=trial,
        metrics={
            "controller": "learned",
            "applied_model_sha256": spec.model.manifest.sha256,
            "applied_action_count": 12,
            "policy_predict_calls": 2,
            "reference_route_calls": 0,
            "simulation_steps": 12,
        },
    )
    report["capture"]["receipt"].update(manifest_sha256=raw.manifest_sha256, frame_count=2)
    preflight = json.loads(documents["preflight.json"])
    preflight["model_process"].update(
        model_sha256=spec.model.manifest.sha256, backbone_sha256=spec.backbone.manifest.sha256
    )
    documents.update(
        {
            "preflight.json": encoded(preflight),
            "probe.json": encoded(report),
            "raw-manifest.json": (writer.root / "manifest.json").read_bytes(),
            "acceptance.json": encoded(learned_probe.rescore(report, spec, scene, profile, grant)),
        }
    )
    common = {
        "schema": "physicalai.batch-simulation-result/v1",
        "attempt_id": str(spec.attempt_id),
        "job_id": spec.job_id,
        "task_id": spec.task_id,
        "previous_attempt_id": None,
        "spec_sha256": spec.sha256,
        "spec_file_sha256": digest(documents["inputs/spec.json"]),
        "source_revision": spec.source_revision,
        "image": spec.platform.container_image,
        "control_profile_sha256": spec.control_profile_sha256,
        "profile_id": spec.profile_id,
        "learning_quality_proven": False,
    }
    completion = {
        **common,
        "accepted": False,
        "outcome": "failed",
        "native_acceptance": "rejected",
        "physical_status": "cancelled",
        "capture_status": "ready",
        "raw_manifest": report["capture"]["receipt"],
        "artifacts": [
            {"path": name, "sha256": digest(data), "bytes": len(data)}
            for name, data in documents.items()
        ],
    }
    claim = {
        **common,
        "accepted": False,
        "outcome": "incomplete",
        "native_acceptance": "missing",
        "claimed_at_utc": (grant.issued_at + timedelta(milliseconds=10)).isoformat(),
        "physical_state_resume": False,
    }
    for name, data in {
        **documents,
        "claim.json": encoded(claim),
        "completion.json": encoded(completion),
    }.items():
        path = directory / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(data)
    mapping_value["evaluation_plan"].update(
        scope=models["before"]["scope"],
        control_profile=asdict(profile),
        control_profile_sha256=profile.sha256,
        criteria_sha256=spec.criteria_canonical_sha256,
        frozen_plan_sha256=spec.conditions_canonical_sha256,
        policy_before_sha256=spec.model.manifest.sha256,
        policy_after_sha256=file_digest(model_roots["after"] / "model.json"),
    )
    mapping_value["evaluation_plan"]["cases"][0].update(
        **{
            key: value
            for key, value in learned_probe.physical_case(spec, scene, grant).items()
            if key != "episode_id"
        }
    )
    mapping_value["runtime"].update(provenance=asdict(provenance), control_profile=asdict(profile))
    mapping_value["evaluation_plan"]["runtime_sha256"] = digest(canonical(mapping_value["runtime"]))
    mapping_value["issued_at_utc"] = (
        (grant.issued_at - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    )
    mapping_value["model_runtime_sha256"] = spec.model_runtime.sha256
    mapping_value["assignments"][0]["physical_attempt_id"] = str(spec.attempt_id)
    mapping = paired_evaluation.PairingPlan.model_validate(mapping_value)
    record = paired_evaluation.AttemptFiles(
        physical_attempt_id=spec.attempt_id, files=inventory(directory)
    )
    snapshot = (grant.expires_at + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    return tmp_path, directory, mapping, record, models, model_roots, spec, snapshot


def test_complete_failed_native_capture_is_verified_and_retained_not_erased(recording):
    _, directory, mapping, record, models, _, spec, snapshot = recording
    original = inventory(directory)
    result = paired_evaluation.verify_attempt(
        directory,
        mapping,
        mapping.assignments[0],
        record,
        models,
        snapshot_at_utc=snapshot,
        mapping_sha256="a" * 64,
    )
    assert result["status"] == "verified" and result["physical_success"] is False
    assert result["physical_trial"]["episode_id"] == str(spec.attempt_id)
    assert result["physical_trial"]["truncated"] is True
    assert result["physical_trial"]["failure_reason"] == "CPU fixture cancellation"
    assert inventory(directory) == original


def test_partial_aggregate_includes_failed_slot_without_fake_missing_trials(recording):
    root, _, mapping, record, _, models, _, snapshot = recording
    evidence = paired_evaluation.PairingEvidence(
        schema="physicalai.managed-paired-evidence/v1",
        mapping_sha256="a" * 64,
        snapshot_at_utc=snapshot,
        attempts=[record],
    )
    result = paired_evaluation.aggregate(
        root,
        mapping,
        evidence,
        mapping_sha256="a" * 64,
        evidence_sha256="b" * 64,
        before_root=models["before"],
        after_root=models["after"],
    )
    assert result["quality_gate_passed"] is False and result["complete"] is False
    assert result["verified_trial_count"] == 1 and len(result["missing"]) == 39
    assert len(result["attempts"]) == 40
    assert result["attempts"][0]["physical_success"] is False
    assert "physical_trial" not in result["attempts"][1]


@pytest.mark.parametrize(
    "changed", ["model", "profile", "criteria", "owner", "role", "physical-id", "retry", "unmapped"]
)
def test_foreign_binding_never_uses_another_physical_slot(recording, changed):
    _, _, mapping, _, models, _, spec, _ = recording
    updates = {
        "model": {
            "model": spec.model.model_copy(
                update={"manifest": spec.model.manifest.model_copy(update={"sha256": "f" * 64})}
            )
        },
        "profile": {"control_profile_sha256": "f" * 64},
        "criteria": {"criteria_canonical_sha256": "f" * 64},
        "owner": {"owner_id": "f" * 64},
        "role": {"role": "after"},
        "physical-id": {"attempt_id": UUID(int=999)},
        "retry": {"previous_attempt_id": UUID(int=999)},
        "unmapped": {"pairing_plan_sha256": None},
    }
    with pytest.raises(ValueError, match="binding"):
        paired_evaluation.bind_attempt(
            mapping,
            mapping.assignments[0],
            spec.model_copy(update=updates[changed]),
            models,
            mapping_sha256="a" * 64,
        )


def test_every_private_raw_pixel_is_hash_bound_not_only_manifest_and_flags(recording):
    _, directory, mapping, record, models, _, _, snapshot = recording
    image = next((directory / "capture").rglob("*.png"))
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte|checksum"):
        paired_evaluation.verify_attempt(
            directory,
            mapping,
            mapping.assignments[0],
            record,
            models,
            snapshot_at_utc=snapshot,
            mapping_sha256="a" * 64,
        )


def test_node_loss_without_completion_remains_incomplete_without_a_failure_trial(recording):
    _, directory, mapping, record, models, _, _, snapshot = recording
    (directory / "completion.json").unlink()
    record = record.model_copy(
        update={
            "files": paired_evaluation.AttemptFiles(
                physical_attempt_id=record.physical_attempt_id, files=inventory(directory)
            ).files
        }
    )
    result = paired_evaluation.verify_attempt(
        directory,
        mapping,
        mapping.assignments[0],
        record,
        models,
        snapshot_at_utc=snapshot,
        mapping_sha256="a" * 64,
    )
    assert result["status"] == "incomplete" and "physical_trial" not in result
    assert result["claimed_at_utc"]


def test_snapshot_cannot_hide_existing_failed_or_extra_attempt_directories(recording):
    root, _, mapping, _, _, models, _, snapshot = recording
    evidence = paired_evaluation.PairingEvidence(
        schema="physicalai.managed-paired-evidence/v1",
        mapping_sha256="a" * 64,
        snapshot_at_utc=snapshot,
        attempts=[],
    )
    with pytest.raises(ValueError, match="omitted|unlisted"):
        paired_evaluation.aggregate(
            root,
            mapping,
            evidence,
            mapping_sha256="a" * 64,
            evidence_sha256="b" * 64,
            before_root=models["before"],
            after_root=models["after"],
        )


@pytest.mark.parametrize("change", ["trace", "timing", "raw-id", "boolean", "late-mapping", "seed"])
def test_rehashed_summaries_cannot_override_native_trace_or_mapping(recording, change):
    _, directory, mapping, record, models, _, _, snapshot = recording
    report = read_json(directory / "probe.json", max_bytes=8 * 1024**2)
    completion = read_json(directory / "completion.json")
    if change == "trace":
        report["task_states"][2]["physics_step"] += 1
    elif change == "timing":
        report["trial"]["latencies_wall_ms"][0] += 0.5
    elif change == "raw-id":
        manifest = read_json(directory / "raw-manifest.json")
        manifest["episodes"][0]["episode_id"] = str(UUID(int=888))
        (directory / "raw-manifest.json").write_bytes(encoded(manifest))
        report["capture"]["receipt"]["manifest_sha256"] = file_digest(
            directory / "raw-manifest.json"
        )
        completion["raw_manifest"] = report["capture"]["receipt"]
    elif change == "boolean":
        completion.update(
            accepted=True,
            outcome="accepted",
            native_acceptance="accepted",
            physical_status="succeeded",
        )
    elif change == "late-mapping":
        mapping = mapping.model_copy(update={"issued_at_utc": snapshot})
    else:
        cases = deepcopy(mapping.evaluation_plan["cases"])
        cases[0]["seed"] = 30002
        mapping = mapping.model_copy(
            update={
                "evaluation_plan": {
                    **mapping.evaluation_plan,
                    "cases": cases,
                }
            }
        )
    (directory / "probe.json").write_bytes(encoded(report))
    for artifact in completion["artifacts"]:
        path = directory / artifact["path"]
        artifact.update(sha256=file_digest(path), bytes=path.stat().st_size)
    (directory / "completion.json").write_bytes(encoded(completion))
    record = paired_evaluation.AttemptFiles(
        physical_attempt_id=record.physical_attempt_id, files=inventory(directory)
    )
    with pytest.raises(ValueError):
        paired_evaluation.verify_attempt(
            directory,
            mapping,
            mapping.assignments[0],
            record,
            models,
            snapshot_at_utc=snapshot,
            mapping_sha256="a" * 64,
        )


def test_failed_physical_trial_cannot_pass_the_existing_success_only_reader(recording):
    _, directory, _, _, _, _, spec, _ = recording
    documents = {
        name: (directory / name).read_bytes()
        for name in (
            "inputs/environment.json",
            "inputs/grant.json",
            "inputs/criteria.json",
            "inputs/conditions.json",
            "probe.json",
            "preflight.json",
            "acceptance.json",
            "raw-manifest.json",
        )
    }
    receipt, result = batch_learned.rescore_evidence(spec, documents)
    assert result["accepted"] is False and receipt["episode_id"] == str(spec.attempt_id)
    with pytest.raises(ValueError, match="not accepted"):
        batch_learned.verify_evidence(spec, documents)


def test_old_report_without_original_heartbeats_is_not_backfilled(recording):
    _, directory, mapping, record, models, _, _, snapshot = recording
    report = read_json(directory / "probe.json")
    report.pop("heartbeat_ns")
    (directory / "probe.json").write_bytes(encoded(report))
    completion = read_json(directory / "completion.json")
    for item in completion["artifacts"]:
        if item["path"] == "probe.json":
            item.update(
                sha256=file_digest(directory / "probe.json"),
                bytes=(directory / "probe.json").stat().st_size,
            )
    (directory / "completion.json").write_bytes(encoded(completion))
    record = paired_evaluation.AttemptFiles(
        physical_attempt_id=record.physical_attempt_id, files=inventory(directory)
    )
    result = paired_evaluation.verify_attempt(
        directory,
        mapping,
        mapping.assignments[0],
        record,
        models,
        snapshot_at_utc=snapshot,
        mapping_sha256="a" * 64,
    )
    assert result["status"] == "incomplete" and "physical_trial" not in result
    assert "heartbeat_ns" not in read_json(directory / "probe.json")


def test_cli_writes_only_a_versioned_incomplete_report_and_preserves_every_input(
    recording, monkeypatch, capsys
):
    root, directory, mapping, record, _, model_roots, spec, snapshot = recording
    path = root / "mapping.json"
    path.write_bytes(encoded(mapping.model_dump(mode="json", by_alias=True)))
    mapping_sha = file_digest(path)
    spec = spec.model_copy(update={"pairing_plan_sha256": mapping_sha})
    (directory / "inputs" / "spec.json").write_bytes(
        encoded(spec.model_dump(mode="json", by_alias=True))
    )
    for name in ("claim.json", "completion.json"):
        value = read_json(directory / name)
        value.update(
            spec_sha256=spec.sha256,
            spec_file_sha256=file_digest(directory / "inputs" / "spec.json"),
        )
        if name == "completion.json":
            for item in value["artifacts"]:
                if item["path"] == "inputs/spec.json":
                    item.update(
                        sha256=value["spec_file_sha256"],
                        bytes=(directory / "inputs" / "spec.json").stat().st_size,
                    )
        (directory / name).write_bytes(encoded(value))
    record = paired_evaluation.AttemptFiles(
        physical_attempt_id=spec.attempt_id, files=inventory(directory)
    )
    evidence = paired_evaluation.PairingEvidence(
        schema="physicalai.managed-paired-evidence/v1",
        mapping_sha256=mapping_sha,
        snapshot_at_utc=snapshot,
        attempts=[record],
    )
    evidence_path = root / "evidence.json"
    evidence_path.write_bytes(encoded(evidence.model_dump(mode="json", by_alias=True)))
    output = root / "mapped-report.json"
    original = inventory(directory)
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired_evaluation",
            "--root",
            str(root),
            "--mapping",
            str(path),
            "--mapping-sha256",
            mapping_sha,
            "--evidence",
            str(evidence_path),
            "--evidence-sha256",
            file_digest(evidence_path),
            "--before-root",
            str(model_roots["before"]),
            "--after-root",
            str(model_roots["after"]),
            "--output",
            str(output),
        ],
    )
    paired_evaluation.main()
    result = read_json(output)
    assert result["schema"] == "physicalai.managed-paired-report/v1"
    assert result["mapping_sha256"] == mapping_sha
    assert result["verified_trial_count"] == 1 and result["quality_gate_passed"] is False
    assert inventory(directory) == original
    assert json.loads(capsys.readouterr().out)["complete"] is False
    with pytest.raises(FileExistsError):
        paired_evaluation.main()


def test_declared_fixture_raw_data_cannot_qualify_a_mapped_trial(recording):
    _, directory, _, _, _, _, spec, _ = recording
    from learning.paused.capture import validate_dataset

    manifest = read_json(directory / "capture" / "manifest.json")
    manifest["episodes"][0]["provenance"].update(
        source_kind="test_fixture", capture_host="test_cpu", gpu_model="none"
    )
    (directory / "capture" / "manifest.json").write_bytes(encoded(manifest))
    with pytest.raises(ValueError, match="live"):
        validate_dataset(
            directory / "capture",
            expected_scope=Scope(str(spec.platform.tenant_id), spec.owner_id),
            require_live=True,
        )
