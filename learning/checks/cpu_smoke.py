from __future__ import annotations

import argparse
import inspect
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning.azure import SNAPSHOT_FILES, create_plan, preflight
from learning.capture import EpisodeWriter, assemble_dataset
from learning.checks.fixtures import PROVENANCE, SCOPE, frame
from learning.common import canonical, digest, file_digest, read_json, require, write_json
from learning.contract import EpisodeSpec, bounded_joints, validate_dataset
from learning.convert import LOCAL_REPO_ID, convert_dataset
from learning.evaluation import (
    EpisodeOutcome,
    EvaluationCase,
    EvaluationRecorder,
    build_evaluation_plan,
    evaluate_results,
)
from learning.inference import LocalACTPolicy, PolicyObservation
from learning.offline import require_lerobot
from learning.train import TrainOptions, train_policy


def verify_sdk_preflight(config: dict) -> None:
    from azure.ai.ml.entities import (
        AmlCompute,
        AzureBlobDatastore,
        Data,
        IdentityConfiguration,
        ManagedIdentityConfiguration,
    )

    compute = AmlCompute(
        name=config["compute"],
        size=config["compute_size"],
        min_instances=0,
        max_instances=1,
        enable_node_public_ip=False,
        identity=IdentityConfiguration(
            type="user_assigned",
            user_assigned_identities=[
                ManagedIdentityConfiguration(
                    client_id=config["managed_identity_client_id"],
                    resource_id=config["managed_identity_resource_id"],
                )
            ],
        ),
    )
    datastore = AzureBlobDatastore(
        name=config["datastore"],
        account_name=config["storage_account_name"],
        container_name=config["blob_container"],
    )
    data = {
        asset["name"]: Data(
            name=asset["name"], version=asset["version"], type=asset["type"], path=asset["uri"]
        )
        for asset in config["inputs"].values()
    }
    client = SimpleNamespace(
        workspaces=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                managed_network=SimpleNamespace(isolation_mode="allow_only_approved_outbound")
            )
        ),
        compute=SimpleNamespace(get=lambda name: compute),
        datastores=SimpleNamespace(get=lambda name: datastore),
        data=SimpleNamespace(get=lambda name, version: data[name]),
    )

    def read_azure(arguments):
        if arguments[0] == "account":
            return {
                "id": config["subscription_id"],
                "tenantId": config["tenant_id"],
                "state": "Enabled",
            }
        return {
            "policy": {
                "rules": [
                    {
                        "enabled": True,
                        "definition": {
                            "filters": {
                                "prefixMatch": [
                                    config["blob_container"] + "/" + config["output_prefix"] + "/"
                                ]
                            },
                            "actions": {
                                "baseBlob": {
                                    "delete": {
                                        "daysAfterModificationGreaterThan": config["retention_days"]
                                    }
                                }
                            },
                        },
                    }
                ]
            }
        }

    with patch("learning.azure._az_json", side_effect=read_azure):
        preflight(client, config)


def run(output: Path) -> dict:
    require(not output.exists(), "Choose a new CPU check output directory")
    output.mkdir(parents=True)
    require_lerobot()
    import torch
    from azure.ai.ml import load_job
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(2)
    apis = (
        LeRobotDataset.create,
        LeRobotDataset.add_frame,
        LeRobotDataset.save_episode,
        LeRobotDataset.finalize,
        ACTPolicy.from_pretrained,
        ACTPolicy.predict_action_chunk,
        make_pre_post_processors,
        load_job,
    )
    signatures = {api.__qualname__: str(inspect.signature(api)) for api in apis}
    print(
        "Installed API signatures verified; creating explicitly synthetic CPU fixtures.", flush=True
    )
    roots = []
    for seed, split in enumerate(("train", "validation", "test"), start=1):
        root = output / f"raw-{split}"
        writer = EpisodeWriter(
            root,
            dataset_id="cpu-check-only",
            scope=SCOPE,
            episode=EpisodeSpec(f"cpu-{split}", "customer-line", "f" * 64, seed, split),
            provenance=PROVENANCE,
        )
        for index in range(4):
            sample = frame(index, terminal=index == 3)
            joints = (sample.joint_positions[0] + index * 0.001, *sample.joint_positions[1:])
            writer.append(replace(sample, joint_positions=joints, commanded_joint_targets=joints))
        writer.finalize()
        roots.append(root)
    raw = output / "demonstrations"
    assemble_dataset(
        roots,
        raw,
        dataset_id="cpu-check-only",
        expected_scope=SCOPE,
        require_live=False,
    )
    converted = output / "converted"
    conversion = convert_dataset(
        raw,
        converted,
        expected_scope=SCOPE,
        expected_manifest_sha256=file_digest(raw / "manifest.json"),
        image_size=32,
        allow_test_fixture=True,
    )
    dataset = LeRobotDataset(
        LOCAL_REPO_ID,
        root=converted,
        download_videos=False,
        video_backend="pyav",
    )
    require(dataset.num_episodes == 1 and len(dataset) == 4, "Wrong converted train episode count")
    item = dataset[0]
    require(tuple(item["action"].shape) == (9,), "Wrong converted action shape")
    require(tuple(item["observation.state"].shape) == (9,), "Wrong converted state shape")
    for camera in ("inspection", "overview"):
        require(
            tuple(item[f"observation.images.{camera}"].shape) == (3, 32, 32),
            "Wrong converted image shape",
        )
    print(
        "Real LeRobot v3 conversion/reload passed; running one actual CPU ACT optimizer step.",
        flush=True,
    )
    source_root = Path(__file__).resolve().parents[2]
    source_hash = digest(
        canonical(
            {name: file_digest(source_root.joinpath(*name.split("/"))) for name in SNAPSHOT_FILES}
        )
    )
    trained = output / "trained"
    model = train_policy(
        converted,
        trained,
        expected_scope=SCOPE,
        expected_conversion_sha256=file_digest(converted / "conversion.json"),
        model_name="cpu-check-only",
        model_version="1",
        code_snapshot_sha256=source_hash,
        options=TrainOptions(
            steps=1,
            batch_size=2,
            chunk_size=2,
            n_action_steps=1,
            timeout_seconds=180,
            device="cpu",
            smoke_test=True,
        ),
    )
    model_sha = file_digest(trained / "model.json")
    policy = LocalACTPolicy(
        trained,
        expected_scope=SCOPE,
        expected_model_sha256=model_sha,
        device="cpu",
        allow_test_model=True,
    )
    policy.reset()
    sample = frame(0)
    observation = PolicyObservation(
        SCOPE,
        "customer-line",
        "f" * 64,
        "cpu-test",
        sample.captured_at_utc,
        sample.monotonic_ns,
        sample.physics_step,
        sample.joint_positions,
        sample.images,
    )
    actions = policy.predict_chunk(observation)
    require(len(actions) == 2, "Wrong real checkpoint chunk length")
    for action in actions:
        bounded_joints(action, "real CPU checkpoint prediction")
    print(
        "Actual checkpoint/normalizers reloaded; checking offline AML schemas.",
        flush=True,
    )
    azure_config = read_json(source_root / "learning" / "examples" / "azure-train.json")
    verify_sdk_preflight(azure_config)
    submission_root = output / "azure-offline-plan"
    plan_sha = create_plan(azure_config, submission_root, source_root=source_root)
    loaded = load_job(source=submission_root / "job.json")
    require(set(loaded.jobs) == {"convert", "train"}, "Wrong AML pipeline schema result")
    evaluation_runtime = {
        "azure_resource_id": (
            "/subscriptions/22222222-2222-4222-8222-222222222222/resourceGroups/cpu-test-only"
            "/providers/Microsoft.Compute/virtualMachines/not-a-live-simulator"
        ),
        "run_id": "cpu-fixture-only",
        "provenance": asdict(PROVENANCE),
    }
    case = EvaluationCase(
        "cpu-test", "customer-line", "f" * 64, 3, True, "rejected", (0.3, 0.1, 0.2)
    )
    plan = build_evaluation_plan(
        validate_dataset(raw, expected_scope=SCOPE),
        model,
        [case],
        model_sha256=model_sha,
        baseline_revision="e" * 40,
        expected_runtime=evaluation_runtime,
    )
    evidence = output / "evaluation-fixture"
    recorder = EvaluationRecorder(evidence, plan=plan, runtime=evaluation_runtime)
    for controller in ("baseline", "learned"):
        recorder.record(
            EpisodeOutcome(
                controller,
                case.episode_id,
                case.environment_id,
                case.revision,
                case.seed,
                "2026-09-20T00:00:00Z",
                "2026-09-20T00:00:00.2Z",
                200_000_000,
                12,
                2,
                12,
                1_200_000_000,
                case.expected_destination_position_m,
                case.expected_destination_id,
                True,
                False,
                None,
                (),
                (10.0, 20.0),
            ),
            frame(2).images,
        )
    recorder.finalize()
    verdict = evaluate_results(
        plan,
        evidence,
        model,
        expected_scope=SCOPE,
        expected_model_sha256=model_sha,
        expected_plan_sha256=digest(canonical(plan)),
    )
    require(not verdict["passed"], "CPU fixtures must never pass the live learning gate")
    gate_config = dict(azure_config, kind="gate", parameters={"timeout_seconds": 300})
    reference = azure_config["inputs"]["demonstrations"]
    gate_config["inputs"] = {
        name: {
            **reference,
            "name": f"cpu-check-{name}",
            "uri": reference["uri"] + "/" + name,
            "type": "uri_file" if name == "plan" else "uri_folder",
        }
        for name in ("model", "evidence", "plan")
    }
    gate_root = output / "azure-offline-gate"
    create_plan(gate_config, gate_root, source_root=source_root)
    loaded_gate = load_job(source=gate_root / "job.json")
    require(set(loaded_gate.jobs) == {"validate_evidence"}, "Wrong AML gate schema")
    report = {
        "check": "real-api-cpu-smoke",
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("lerobot", "torch", "torchvision", "datasets", "azure-ai-ml")
        },
        "api_signatures": signatures,
        "converted_episodes": dataset.num_episodes,
        "converted_frames": len(dataset),
        "dataset_format": dataset.meta.info["codebase_version"],
        "actual_optimizer_steps": model["training"]["steps"],
        "real_checkpoint_chunk_shape": [len(actions), len(actions[0])],
        "conversion_test_only": conversion["test_only"],
        "training_test_only": model["training"]["test_only"],
        "model_sha256": model_sha,
        "offline_aml_plan_sha256": plan_sha,
        "offline_aml_schema_loads": ["convert/train", "validate_evidence"],
        "azure_preflight_shape_check": "actual SDK entities; mocked service and CLI reads",
        "gate_rejection_reasons": verdict["reasons"],
        "learning_quality_verified": False,
        "azure_jobs_submitted": 0,
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy CPU check, never a live release gate.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
