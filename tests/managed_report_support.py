"""Construct independent CPU codec records. These are not simulated or measured GPU outcomes."""

import base64
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

from apps.api.learning_models import CreateProject, EvaluationRun, LearningProject, PolicyCandidate
from apps.api.learning_ports import JobSpecification
from apps.api.models import SaveEnvironment, utcnow
from apps.api.simulation_reports import ManagedImportReference
from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.managed_reports import (
    ImportContext,
    ManagedEvaluationBinding,
    ManagedImportCompletion,
    prefix,
)
from learning.checks.fixtures import png
from learning.common import canonical, digest, file_digest, read_json, utc
from learning.contract import AppliedControl, DemonstrationSource, EpisodeSpec, Provenance, Scope
from learning.paused import FrozenCameraSample, FrozenPolicyObservation, PausedFrameSample
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter, validate_dataset
from learning.paused.rollout import derive_trial
from learning.paused.task import TaskState
from simulation import batch_learned, learned_probe, paired_evaluation
from simulation.extensions import SceneRegistry
from simulation.paused_configuration import PausedOperatorGrant
from tests.runtime_support import ACTOR, service
from tests.test_paired_artifacts import encoded, inventory
from tests.test_paused_learning_api import paused_project_payload


def bundle(recording, destination, *, model_runtime=None):
    _, template_dir, template_mapping, _, _, template_roots, template_spec, _ = recording
    base_report = read_json(template_dir / "probe.json")
    original_environment = read_json(template_dir / "inputs" / "environment.json")
    task = read_json(template_roots["before"] / "model.json")["task"]
    profile = template_mapping.evaluation_plan["control_profile"]
    start = utcnow() - timedelta(seconds=300)
    factory = service()
    cases, environments, scenes = [], [], []
    for index in range(20):
        document = deepcopy(original_environment["document"])
        document["environment_id"] = f"cpu-managed-{index}"
        document["scene"]["seed"] = 30001 + index
        document["stations"][0]["position_m"][0] += index * 0.002
        environment = factory.save_environment(
            ACTOR, SaveEnvironment(document_json=json.dumps(document))
        )
        scene = SceneRegistry(load_installed=False).build(environment)
        environments.append(environment)
        scenes.append(scene)
        cases.append(
            {
                "environment_id": environment.environment_id,
                "revision": environment.revision,
                "seed": scene.seed,
                "split": "test",
                "scene_builder_sha256": scene.scene_builder_sha256,
                "expected_initial_position_m": list(scene.part_position),
                "goal_position_m": list(scene.station(task["goal_id"]).position),
            }
        )
    criteria = read_json(template_dir / "inputs" / "criteria.json")
    conditions = {
        **criteria,
        "criteria_canonical_sha256": digest(canonical(criteria)),
        "cases": cases,
    }
    roots, models = {}, {}
    for role in ("before", "after"):
        roots[role] = destination / "models" / role
        shutil.copytree(template_roots[role], roots[role])
        model = read_json(roots[role] / "model.json")
        model.update(
            scope={"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key},
            criteria_sha256=digest(canonical(criteria)),
            frozen_plan_sha256=digest(canonical(conditions)),
        )
        model["training"].update(
            criteria_sha256=model["criteria_sha256"],
            frozen_plan_sha256=model["frozen_plan_sha256"],
        )
        if model_runtime is not None:
            model.update(
                schema="physicalai.smolvla-checkpoint/v3", training_execution="azureml_command"
            )
            training = model["training"]
            training["azure_job_id"] = training["azure_pipeline_job_id"] + "-" + role
            training.pop("azure_pipeline_job_id")
            training["azure_job_type"] = "command"
            training["gpu"]["device_count"] = 1
            training["episodes"] = [
                {
                    "episode_id": "cpu-training-10001",
                    "environment_id": "cpu-training",
                    "revision": "a" * 64,
                    "seed": 10001,
                }
            ]
        if role == "after":
            model["training"]["parent_model_sha256"] = file_digest(roots["before"] / "model.json")
        (roots[role] / "model.json").write_bytes(encoded(model))
        models[role] = model
    mapping = template_mapping.model_dump(mode="json", by_alias=True)
    mapping["issued_at_utc"] = (start + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    native = mapping["evaluation_plan"]
    native.update(
        scope=models["before"]["scope"],
        criteria_sha256=models["before"]["criteria_sha256"],
        frozen_plan_sha256=models["before"]["frozen_plan_sha256"],
        policy_before_sha256=file_digest(roots["before"] / "model.json"),
        policy_after_sha256=file_digest(roots["after"] / "model.json"),
    )
    for case, scene, environment in zip(native["cases"], scenes, environments, strict=True):
        case.update(
            seed=scene.seed,
            environment_id=environment.environment_id,
            revision=environment.revision,
            initial_pose_m=list(scene.part_position),
            expected_destination_id=task["goal_id"],
            expected_pose_m=list(scene.station(task["goal_id"]).position),
            scene_builder_sha256=scene.scene_builder_sha256,
        )
    for index, assignment in enumerate(mapping["assignments"]):
        assignment["physical_attempt_id"] = str(UUID(int=1000 + index))
    admission, runtime_bytes = None, None
    if model_runtime is not None:
        runtime = batch_learned.parse_model_runtime(
            {
                **model_runtime,
                "legacy_servo_sha256": profile["servo_profile_sha256"],
                "control_profile_sha256": native["control_profile_sha256"],
                "simulator_image": template_spec.platform.container_image,
                "simulator_source_revision": template_spec.source_revision,
            }
        )
        runtime_bytes = encoded(runtime.model_dump(mode="json", by_alias=True))
        mapping["model_runtime_sha256"] = digest(runtime_bytes)
        admission = batch_learned.command_admission_proof(
            runtime, runtime_sha256=mapping["model_runtime_sha256"]
        )
    mapped = paired_evaluation.PairingPlan.model_validate(mapping)
    root = destination / "files"
    root.mkdir()
    if runtime_bytes is not None:
        (root / "model-runtime.json").write_bytes(runtime_bytes)
    (root / "mapping.json").write_bytes(encoded(mapping))
    mapping_sha = file_digest(root / "mapping.json")
    from learning.paused import PausedControlProfile

    control = PausedControlProfile(**profile)
    records = []
    for index, assignment in enumerate(mapped.assignments):
        case_index = index // 2
        scene, environment = scenes[case_index], environments[case_index]
        model = models[assignment.role]
        value = template_spec.model_dump(mode="json", by_alias=True)
        value.update(
            attempt_id=str(assignment.physical_attempt_id),
            owner_id=ACTOR.owner_key,
            role=assignment.role,
            pairing_plan_sha256=mapping_sha,
            criteria_canonical_sha256=native["criteria_sha256"],
            conditions_canonical_sha256=native["frozen_plan_sha256"],
        )
        value["platform"]["tenant_id"] = str(ACTOR.tenant_id)
        model_prefix = f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/learning/"
        value["model"]["manifest"]["name"] = f"{model_prefix}models/{assignment.role}/model.json"
        value["backbone"]["manifest"]["name"] = f"{model_prefix}backbone/backbone.json"
        value["model_runtime"]["name"] = f"{model_prefix}runtimes/model-runtime.json"
        if runtime_bytes is not None:
            value["model_runtime"].update(
                sha256=digest(runtime_bytes), size_bytes=len(runtime_bytes)
            )
        value["model"]["manifest"].update(
            sha256=file_digest(roots[assignment.role] / "model.json"),
            size_bytes=(roots[assignment.role] / "model.json").stat().st_size,
        )
        value["model"]["files"] = [
            {
                "path": f"checkpoint/{name}",
                "sha256": checksum,
                "size_bytes": (roots[assignment.role] / "checkpoint" / name).stat().st_size,
            }
            for name, checksum in model["checkpoint_files"].items()
        ]
        value["backbone"]["manifest"]["sha256"] = model["backbone_manifest_sha256"]
        grant = read_json(template_dir / "inputs" / "grant.json")
        issued = start + timedelta(seconds=3 + index * 2)
        grant.update(
            tenant_id=str(ACTOR.tenant_id),
            issued_at=issued.isoformat(),
            expires_at=(issued + timedelta(seconds=600)).isoformat(),
        )
        grant["authorization"].update(
            authorization_id=str(uuid4()),
            owner=ACTOR.owner_key,
            environment_id=environment.environment_id,
            revision=environment.revision,
            model_sha256=value["model"]["manifest"]["sha256"],
            criteria_sha256=native["criteria_sha256"],
            frozen_plan_sha256=native["frozen_plan_sha256"],
            wall_expires_at=(issued + timedelta(seconds=590)).isoformat(),
        )
        documents = {
            "inputs/environment.json": encoded(environment.model_dump(mode="json")),
            "inputs/grant.json": encoded(grant),
            "inputs/criteria.json": encoded(criteria),
            "inputs/conditions.json": encoded(conditions),
        }
        for name in ("environment", "grant", "criteria", "conditions"):
            value[name].update(
                sha256=digest(documents[f"inputs/{name}.json"]),
                size_bytes=len(documents[f"inputs/{name}.json"]),
            )
        spec = batch_learned.BatchLearnedSpec.model_validate(value)
        documents["inputs/spec.json"] = encoded(spec.model_dump(mode="json", by_alias=True))
        permission = PausedOperatorGrant.model_validate(grant)
        directory = root / "attempts" / str(spec.attempt_id)
        directory.mkdir(parents=True)
        initial = base_report["task_states"][0]["object_position_m"]
        shift = [scene.part_position[axis] - initial[axis] for axis in range(3)]
        states = [
            replace(
                TaskState(**state),
                command_id=str(spec.attempt_id),
                applied_model_sha256=spec.model.manifest.sha256 if offset else None,
                object_position_m=tuple(
                    state["object_position_m"][axis] + shift[axis] for axis in range(3)
                ),
                tcp_position_m=tuple(
                    state["tcp_position_m"][axis] + shift[axis] for axis in range(3)
                ),
                captured_at_utc=(issued + timedelta(seconds=1 + offset / 60))
                .isoformat()
                .replace("+00:00", "Z"),
            )
            for offset, state in enumerate(base_report["task_states"])
        ]
        writer = PausedEpisodeWriter(
            directory / "capture",
            dataset_id="cpu-codec-only",
            scope=Scope(**native["scope"]),
            episode=EpisodeSpec(
                str(spec.attempt_id),
                environment.environment_id,
                environment.revision,
                scene.seed,
                "test",
            ),
            provenance=Provenance(**mapped.runtime["provenance"]),
            profile=control,
            demonstration=DemonstrationSource(
                "learned", **task, source_policy_sha256=spec.model.manifest.sha256
            ),
            budget=PausedEpisodeBudget(**base_report["episode_budget"]),
            purpose="evaluation",
            criteria_sha256=native["criteria_sha256"],
            frozen_plan_sha256=native["frozen_plan_sha256"],
        )
        for offset in (0, 6):
            first, last = states[offset], states[offset + 6]
            ready = first.monotonic_ns + 1_000_000
            stamp = (
                (utc(first.captured_at_utc) + timedelta(milliseconds=1))
                .isoformat()
                .replace("+00:00", "Z")
            )
            observation = FrozenPolicyObservation(
                scope=Scope(**native["scope"]),
                environment_id=environment.environment_id,
                revision=environment.revision,
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
                control_profile_sha256=control.sha256,
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
            writer.root, expected_scope=Scope(**native["scope"]), require_live=True
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
            case=learned_probe.physical_case(spec, scene, permission),
            role=spec.role,
            model_sha256=spec.model.manifest.sha256,
            profile=control,
            final_images=images,
            heartbeat_ns=tuple(state.monotonic_ns for state in states),
            destination_id=task["goal_id"],
            failure_reason="CPU fixture cancellation",
        )
        report = deepcopy(base_report)
        report.update(
            model_sha256=spec.model.manifest.sha256,
            operator_grant_sha256=spec.grant.sha256,
            criteria_sha256=native["criteria_sha256"],
            frozen_plan_sha256=native["frozen_plan_sha256"],
            command_id=str(spec.attempt_id),
            environment_id=environment.environment_id,
            revision=environment.revision,
            task_states=[asdict(state) for state in states],
            final_images={
                name: learned_probe.encode_image(image) for name, image in images.items()
            },
            trial=trial,
        )
        report["metrics"]["applied_model_sha256"] = spec.model.manifest.sha256
        if admission is not None:
            report["model_admission"] = admission
        report["capture"]["receipt"].update(
            episode_id=str(spec.attempt_id),
            manifest_sha256=raw.manifest_sha256,
            manifest_uri=f"{spec.storage_account_url}/{spec.output_container}/{ACTOR.owner_key}/{spec.attempt_id}/manifest.json",
        )
        report["gripper_servo"]["owner"].update(
            owner=ACTOR.owner_key,
            command_id=str(spec.attempt_id),
            environment_id=environment.environment_id,
            revision=environment.revision,
        )
        preflight = read_json(template_dir / "preflight.json")
        preflight["model_process"]["model_sha256"] = spec.model.manifest.sha256
        if runtime_bytes is not None:
            preflight["model_process"].update(
                runtime_sha256=digest(runtime_bytes),
                admission=admission,
                runtime_descriptor_base64=base64.b64encode(runtime_bytes).decode("ascii"),
            )
        documents.update(
            {
                "preflight.json": encoded(preflight),
                "probe.json": encoded(report),
                "acceptance.json": encoded(
                    learned_probe.rescore(report, spec, scene, control, permission)
                ),
                "raw-manifest.json": (writer.root / "manifest.json").read_bytes(),
            }
        )
        common = read_json(template_dir / "claim.json")
        common.update(
            attempt_id=str(spec.attempt_id),
            job_id=spec.job_id,
            task_id=spec.task_id,
            spec_sha256=spec.sha256,
            spec_file_sha256=digest(documents["inputs/spec.json"]),
            claimed_at_utc=(issued + timedelta(milliseconds=10)).isoformat(),
        )
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
        for name, data in {
            **documents,
            "claim.json": encoded(common),
            "completion.json": encoded(completion),
        }.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        records.append(
            paired_evaluation.AttemptFiles(
                physical_attempt_id=spec.attempt_id, files=inventory(directory)
            )
        )
    evidence = paired_evaluation.PairingEvidence(
        schema="physicalai.managed-paired-evidence/v1",
        mapping_sha256=mapping_sha,
        snapshot_at_utc=(start + timedelta(seconds=100)).isoformat().replace("+00:00", "Z"),
        attempts=records,
    )
    (root / "evidence.json").write_bytes(encoded(evidence.model_dump(mode="json", by_alias=True)))
    output = paired_evaluation.aggregate(
        root,
        mapped,
        evidence,
        mapping_sha256=mapping_sha,
        evidence_sha256=file_digest(root / "evidence.json"),
        before_root=roots["before"],
        after_root=roots["after"],
        **({"model_runtime": runtime_bytes} if runtime_bytes is not None else {}),
    )
    (root / "report.json").write_bytes(encoded(output))
    request = paused_project_payload()
    request.update(
        control_profile_id=control.profile_id,
        control_profile_sha256=control.sha256,
        criteria_sha256=native["criteria_sha256"],
        frozen_plan_sha256=native["frozen_plan_sha256"],
        task_id=task["task_id"],
        instruction=task["instruction"],
        goal_station_id=task["goal_id"],
    )
    request["evaluation_plan"].update(
        control_profile_id=control.profile_id,
        max_simulation_seconds=60,
        seeds=[case["seed"] for case in native["cases"]],
        cases=[
            {key: case[key] for key in ("seed", "environment_id", "revision")}
            for case in native["cases"]
        ],
    )
    project = LearningProject.create(ACTOR, CreateProject.model_validate(request))
    project = project.model_copy(
        update={
            "created_at": start - timedelta(seconds=2),
            "updated_at": start - timedelta(seconds=2),
        }
    )
    candidates = {}
    for role in ("before", "after"):
        model = models[role]
        candidates[role] = PolicyCandidate(
            id=uuid4(),
            owner_key=ACTOR.owner_key,
            actor_id=ACTOR.object_id,
            tenant_id=ACTOR.tenant_id,
            created_at=start - timedelta(seconds=1),
            updated_at=start - timedelta(seconds=1),
            fingerprint=file_digest(roots[role] / "model.json"),
            project_id=project.id,
            dataset_id=uuid4(),
            training_run_id=uuid4(),
            parent_release_id=None,
            policy_type="smolvla",
            model_sha256=file_digest(roots[role] / "model.json"),
            parent_model_sha256=model["training"]["parent_model_sha256"],
            processor_sha256=model["processor_sha256"],
            manifest_sha256=file_digest(roots[role] / "model.json"),
            artifact_id=uuid4(),
            optimizer_steps=model["training"]["optimizer_steps"],
            azure_job_id=model["training"][
                "azure_job_id" if runtime_bytes is not None else "azure_pipeline_job_id"
            ],
            source_commit=model["upstream"]["source_commit"],
            model_revision=model["upstream"]["model_revision"],
            **project.timing_fields(),
        )
    evaluation = EvaluationRun(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=start,
        updated_at=start,
        fingerprint="9" * 64,
        project_id=project.id,
        policy_type="smolvla",
        status="awaiting_import",
        provider="managed_batch",
        backend_job_name=f"learning-{uuid4().hex}",
        deadline=start + timedelta(seconds=600),
        approved_cost_usd="10",
        specification_sha256="8" * 64,
        candidate_id=candidates["after"].id,
        baseline_release_id=None,
        before_candidate_id=candidates["before"].id,
        evaluation_plan_sha256=project.evaluation_plan.sha256,
        **project.timing_fields(),
    )
    spec = JobSpecification(
        owner_key=ACTOR.owner_key,
        project=project,
        run=evaluation,
        candidate=candidates["after"],
        baseline_candidate=candidates["before"],
    )
    binding = ManagedEvaluationBinding(
        schema="physicalai.managed-evaluation-binding/v1",
        provider="managed_batch",
        created_at=start + timedelta(seconds=1),
        study_max_wall_seconds=600,
        study_approved_cost_usd="10",
        mapping_sha256=mapping_sha,
        specification=spec,
    )
    completion = ManagedImportCompletion(
        schema="physicalai.managed-evaluation-import/v1",
        operation_id=uuid4(),
        created_at=start + timedelta(seconds=101),
        binding_sha256=digest(encoded(binding.model_dump(mode="json", by_alias=True))),
        evidence_sha256=file_digest(root / "evidence.json"),
        report_sha256=file_digest(root / "report.json"),
    )
    reference = ManagedImportReference(
        operation_id=completion.operation_id,
        owner_key=ACTOR.owner_key,
        project_id=project.id,
        evaluation_run_id=evaluation.id,
        specification_sha256=evaluation.specification_sha256,
        binding_sha256=completion.binding_sha256,
        completion_sha256=digest(encoded(completion.model_dump(mode="json", by_alias=True))),
    )
    value = ImportContext(
        prefix(project.id, evaluation.id),
        binding,
        completion,
        reference,
        start + timedelta(seconds=2),
    )
    verifier = VerifiedArtifacts(
        None,
        None,
        "https://unused.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    return SimpleNamespace(
        root=root,
        models=roots,
        context=value,
        verifier=verifier,
        mapping=mapped,
        evidence=evidence,
        spec=spec,
    )


def registry_for(bundle):
    from apps.learning_worker.registry import BlobRegistry
    from tests.test_worker_deadlines import ConditionalBlobs

    class Blobs(ConditionalBlobs):
        container_name = "artifacts"

        def __init__(self):
            super().__init__()
            self.modified = {}

        def _upload(self, name, data, **kwargs):
            value = data.read() if hasattr(data, "read") else data
            result = super()._upload(name, value, **kwargs)
            self.modified[name] = utcnow()
            return result

        def download_blob(self, name, **kwargs):
            value = super().download_blob(name, **kwargs)
            data = value.readall()
            value.chunks = lambda: iter((data,))
            value.properties.last_modified = self.modified[name]
            return value

    registry = BlobRegistry.__new__(BlobRegistry)
    registry.container = Blobs()
    registry.client = SimpleNamespace(
        url="https://unitregistry.blob.core.windows.net", close=lambda: None
    )
    registry.budget = None
    base = bundle.context.prefix
    for name, value, modified in (
        (
            "binding.json",
            bundle.context.binding,
            bundle.context.registered_at - timedelta(seconds=1),
        ),
        (
            "completion.json",
            bundle.context.completion,
            bundle.context.completion.created_at + timedelta(seconds=1),
        ),
    ):
        key = registry.key(ACTOR, f"{base}/{name}")
        registry.container.upload_blob(
            name=key, data=encoded(value.model_dump(mode="json", by_alias=True)), overwrite=False
        )
        registry.container.modified[key] = modified
    registry.put(
        ACTOR,
        f"jobs/{bundle.spec.run.backend_job_name}/specification.json",
        bundle.spec.model_dump(mode="json"),
    )
    for path in bundle.root.rglob("*"):
        if path.is_file():
            registry.container.upload_blob(
                name=registry.key(
                    ACTOR, f"{base}/files/{path.relative_to(bundle.root).as_posix()}"
                ),
                data=path.read_bytes(),
                overwrite=False,
            )
    for role, candidate in (
        ("before", bundle.spec.baseline_candidate),
        ("after", bundle.spec.candidate),
    ):
        registry.upload(
            ACTOR,
            candidate.artifact_id,
            bundle.models[role],
            {"manifest_sha256": candidate.model_sha256, "role": "trained_candidate"},
        )
    return registry
