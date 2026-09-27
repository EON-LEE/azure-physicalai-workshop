"""Offline bootstrap manifests and native AML plans; never uploads or submits a cloud job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from learning.common import (
    canonical,
    digest,
    file_digest,
    integer,
    keys,
    parse_json,
    read_json,
    relative_path,
    require,
    sha256,
    write_json,
)
from learning.contract import DemonstrationSource, EpisodeSpec, Provenance, Scope
from learning.paused.artifacts import MODEL_SCHEMA
from learning.paused.contract import EXECUTION_TIMING, RAW_SCHEMA, PausedControlProfile
from learning.smolvla import UPSTREAM
from learning.smolvla.artifacts import processor_digest
from learning.smolvla.azure import create_plan, validate_config
from learning.smolvla.prepare import VENDOR_SHA256

BINDING_KEYS = {
    "scope",
    "control_profile",
    "task",
    "execution_timing",
    "real_time_admission",
    "criteria_sha256",
    "frozen_plan_sha256",
}
BOOTSTRAP_SEEDS = frozenset(range(10001, 10021))


def _binding(value: dict) -> tuple[Scope, PausedControlProfile, dict]:
    keys(value, BINDING_KEYS, "paused preparation binding")
    scope = Scope(**keys(value["scope"], {"tenant_id", "owner_id"}, "binding scope"))
    scope.validate()
    profile = PausedControlProfile(
        **keys(value["control_profile"], set(PausedControlProfile.__dataclass_fields__), "profile")
    )
    profile.validate()
    require(
        value["execution_timing"] == EXECUTION_TIMING and value["real_time_admission"] is False,
        "Only explicitly paused bootstrap is supported",
    )
    task = keys(value["task"], {"task_id", "instruction", "goal_id"}, "approved task")
    DemonstrationSource(kind="reference_controller", **task).validate()
    sha256(value["criteria_sha256"], "frozen criteria")
    sha256(value["frozen_plan_sha256"], "model-independent frozen conditions")
    return scope, profile, task


def _read_manifest(path: Path, checksum: str) -> dict:
    require(path.is_file() and not path.is_symlink(), "Expected a nonsymlink local manifest")
    limit = 4 * 1024 * 1024
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    require(len(payload) <= limit, "Supplied manifest exceeds the JSON byte limit")
    require(digest(payload) == sha256(checksum), "Supplied manifest checksum differs")
    return parse_json(payload)


def preparation_binding(
    qualification: Path,
    *,
    expected_qualification_sha256: str,
    scope: Scope,
    expected_profile_sha256: str,
    expected_criteria_sha256: str,
    expected_frozen_plan_sha256: str,
) -> dict:
    """Extract only profile/task identity from the operator's pinned G0 receipt, not TRAIN data."""
    scope.validate()
    evidence = _read_manifest(qualification, expected_qualification_sha256)
    profile = PausedControlProfile(
        **keys(
            evidence["control_profile"], set(PausedControlProfile.__dataclass_fields__), "profile"
        )
    )
    require(
        profile.sha256 == sha256(expected_profile_sha256) == evidence["control_profile_sha256"],
        "Operator-selected profile differs from the supplied qualification",
    )
    require(
        evidence["criteria_sha256"] == sha256(expected_criteria_sha256)
        and evidence["frozen_plan_sha256"] == sha256(expected_frozen_plan_sha256),
        "Qualification uses different frozen criteria/conditions",
    )
    require(
        evidence.get("physical_status") == "succeeded"
        and evidence.get("capture", {}).get("status") == "ready"
        and evidence["capture"].get("receipt", {}).get("status") == "uploaded",
        "The supplied operator qualification does not report a completed reference capture",
    )
    sha256(evidence["capture"]["receipt"]["manifest_sha256"], "qualification capture receipt")
    result = {
        "scope": asdict(scope),
        "control_profile": asdict(profile),
        "task": evidence["task"],
        "execution_timing": EXECUTION_TIMING,
        "real_time_admission": False,
        "criteria_sha256": expected_criteria_sha256,
        "frozen_plan_sha256": expected_frozen_plan_sha256,
    }
    _binding(result)
    return result


def _training_manifest(raw: dict, binding: dict, profile: PausedControlProfile) -> int:
    require(
        raw.get("schema") == RAW_SCHEMA
        and raw.get("purpose") == "demonstration"
        and raw.get("timestamp_basis") == "simulation_time"
        and all(raw.get(name) == binding[name] for name in BINDING_KEYS - {"task"}),
        "TRAIN manifest mode/profile/criteria/owner differs from the approved binding",
    )
    require(
        raw.get("fps") == profile.control_sim_hz and raw.get("physics_hz") == profile.physics_hz,
        "Wrong actual simulation-time cadence",
    )
    episodes = raw.get("episodes")
    require(
        isinstance(episodes, list) and len(episodes) == len(BOOTSTRAP_SEEDS),
        "Bootstrap requires all twenty predeclared TRAIN episodes",
    )
    ids, seeds, count = set(), set(), 0
    for item in episodes:
        require(isinstance(item, dict), "Invalid TRAIN episode metadata")
        spec = EpisodeSpec(**{name: item[name] for name in EpisodeSpec.__dataclass_fields__})
        spec.validate()
        require(
            spec.split == "train" and spec.seed in BOOTSTRAP_SEEDS,
            "Integration/held-out seed is not bootstrap TRAIN",
        )
        require(
            spec.episode_id not in ids and spec.seed not in seeds,
            "Duplicate bootstrap episode or seed",
        )
        ids.add(spec.episode_id)
        seeds.add(spec.seed)
        Provenance(
            **keys(item["provenance"], set(Provenance.__dataclass_fields__), "capture provenance")
        ).validate(require_live=True)
        demonstration = DemonstrationSource(
            **keys(
                item["demonstration"], set(DemonstrationSource.__dataclass_fields__), "demonstrator"
            )
        )
        demonstration.validate()
        require(
            demonstration.kind in ("reference_controller", "human_teleop")
            and {name: item["demonstration"][name] for name in binding["task"]} == binding["task"],
            "First bootstrap needs actual approved teacher data, not a learned-policy substitute",
        )
        require(item["path"] == f"episodes/{spec.episode_id}/frames.jsonl", "Unsafe episode path")
        sha256(item["sha256"], "raw episode stream checksum")
        count += integer(item["frame_count"], "raw frame count", 2, profile.max_frames)
    require(seeds == BOOTSTRAP_SEEDS, "Bootstrap TRAIN cohort is incomplete")
    return count


def _prepared_manifests(parent: dict, backbone: dict, binding: dict, backbone_sha256: str) -> None:
    require(
        parent.get("schema") == MODEL_SCHEMA
        and parent.get("policy_type") == "smolvla"
        and parent.get("upstream") == UPSTREAM
        and parent.get("role") == "pretrained"
        and parent.get("training") is None
        and parent.get("timestamp_basis") == "simulation_time"
        and all(parent.get(name) == value for name, value in binding.items()),
        "A new train-only parent is required; old profile metadata cannot be relabelled",
    )
    require(
        parent.get("action_horizon") == 50
        and parent.get("n_action_steps") == 1
        and parent.get("image_size") == 256,
        "Unexpected prepared parent architecture",
    )
    files = parent.get("checkpoint_files")
    require(isinstance(files, dict), "Missing parent checkpoint inventory")
    for name, checksum in files.items():
        relative_path(name)
        sha256(checksum)
    require(
        files.get("model.safetensors")
        == parent.get("weights_sha256")
        == VENDOR_SHA256["model"]["model.safetensors"]
        and processor_digest(files) == parent.get("processor_sha256"),
        "Prepared parent is not bound to the exact vendor weights/processors",
    )
    require(
        backbone.get("schema") == "physicalai.smolvla-backbone/v1"
        and backbone.get("scope") == binding["scope"]
        and backbone.get("upstream") == UPSTREAM
        and parent.get("backbone_manifest_sha256") == backbone_sha256,
        "Parent/backbone manifest identity mismatch",
    )
    files = backbone.get("files")
    require(isinstance(files, dict) and bool(files), "Missing private backbone inventory")
    for name, checksum in files.items():
        relative_path(name)
        sha256(checksum)
    require(
        files.get("assets/model.safetensors") == VENDOR_SHA256["backbone"]["model.safetensors"],
        "Backbone is not the pinned licensed private model",
    )


def create_package(
    config: dict,
    binding: dict,
    *,
    raw_manifest: Path,
    parent_manifest: Path,
    backbone_manifest: Path,
    output: Path,
    job_name: str,
) -> dict:
    """Validate exact manifest slots and emit native files. Payload/cloud admission is separate."""
    scope, profile, task = _binding(binding)
    validate_config(config)
    require(
        config["kind"] == "train"
        and config.get("execution_timing") == EXECUTION_TIMING
        and config.get("real_time_admission") is False
        and config["owner_id"] == scope.owner_id
        and config["tenant_id"] == scope.tenant_id
        and config["control_profile_sha256"] == profile.sha256
        and config["task_sha256"] == digest(canonical(task))
        and config["criteria_sha256"] == binding["criteria_sha256"]
        and config["frozen_plan_sha256"] == binding["frozen_plan_sha256"],
        "Native training config differs from the selected owner/profile/task/conditions",
    )
    require(
        config["parameters"]["resume_mode"] == "new"
        and config.get("checkpointing") is not None
        and config["checkpointing"]["resume"] is None,
        "First bootstrap requires new training and explicit intermediate checkpoint publication",
    )
    raw = _read_manifest(raw_manifest, config["inputs"]["demonstrations"]["sha256"])
    parent = _read_manifest(parent_manifest, config["inputs"]["parent_model"]["sha256"])
    backbone_sha = config["inputs"]["backbone"]["sha256"]
    backbone = _read_manifest(backbone_manifest, backbone_sha)
    frame_count = _training_manifest(raw, binding, profile)
    _prepared_manifests(parent, backbone, binding, backbone_sha)
    require(not output.exists(), "Never overwrite an operator-reviewed bootstrap package")
    checksum = create_plan(config, output / "plan", deterministic_job_name=job_name)
    job = read_json(output / "plan" / "job.json")
    registrations = output / "registrations"
    registrations.mkdir()
    commands = []
    for name, asset in config["inputs"].items():
        path = registrations / f"{name}.json"
        write_json(
            path,
            {
                "$schema": "https://azuremlschemas.azureedge.net/latest/data.schema.json",
                "name": asset["name"],
                "version": asset["version"],
                "type": asset["type"],
                "path": asset["uri"],
                "tags": {
                    "scope_owner": scope.owner_id,
                    "scope_tenant": scope.tenant_id,
                    "manifest_sha256": asset["sha256"],
                    "control_profile_sha256": profile.sha256,
                },
            },
        )
        commands.append(
            [
                "az",
                "ml",
                "data",
                "create",
                "--file",
                f"registrations/{name}.json",
                "--subscription",
                config["subscription_id"],
                "--resource-group",
                config["resource_group"],
                "--workspace-name",
                config["workspace"],
            ]
        )
    write_json(output / "prepare-binding.json", binding)
    write_json(output / "run-config.json", config)
    receipt = {
        "schema": "physicalai.paused-bootstrap-package/v1",
        "status": "offline_package_requires_payload_preflight_and_operator_approval",
        "plan_sha256": checksum,
        "config_sha256": digest(canonical(config)),
        "job_name": job_name,
        "profile_sha256": profile.sha256,
        "raw_manifest_sha256": config["inputs"]["demonstrations"]["sha256"],
        "parent_model_manifest_sha256": config["inputs"]["parent_model"]["sha256"],
        "backbone_manifest_sha256": backbone_sha,
        "raw_episode_count": len(raw["episodes"]),
        "raw_frame_count": frame_count,
        "optimizer_target_steps": config["parameters"]["max_steps"],
        "checkpoint_interval_steps": config["parameters"]["checkpoint_steps"],
        "checkpoint_publication_count_limit": config["checkpointing"]["limits"]["max_checkpoints"],
        "planned_checkpoint_count": (
            config["parameters"]["max_steps"] + config["parameters"]["checkpoint_steps"] - 1
        )
        // config["parameters"]["checkpoint_steps"],
        "conversion_seconds": job["jobs"]["convert"]["limits"]["timeout"],
        "training_seconds": job["jobs"]["train"]["limits"]["timeout"],
        "job_deadline_utc": config["job_deadline_utc"],
        "compute": config["compute"],
        "compute_size": config["compute_size"],
        "compute_tier": config["compute_tier"],
        "environment_image": config["environment_image"],
        "manifest_metadata_checked": True,
        "payloads_verified": False,
        "cloud_calls": 0,
        "jobs_submitted": 0,
        "learning_quality_verified": False,
        "real_time_admission": False,
    }
    write_json(
        output / "operator-sequence.json",
        {
            "schema": "physicalai.paused-bootstrap-operator-sequence/v1",
            "registration_commands": commands,
            "registration_commands_executed": False,
            "registration_requires": [
                "Explicit parent/operator Azure write authorization.",
                "Stage and verify payload bytes under the three exact reviewed datastore paths.",
                "Prepare a new parent for this profile binding; never rewrite an old manifest.",
            ],
            "submission_requires": [
                "Register the exact config with the existing approved-plan API/worker flow.",
                "New job/time/cost authorization and fresh exact deadline-reconciler enrollment.",
                "Actual private datastore/MI/compute/asset preflight: max one GPU, no auto retry.",
                "Use PolicyJobs.submit through the durable paid-job claim, not az ml job create.",
            ],
            "outputs": job["outputs"],
            "checkpoint_blob_prefix": config["output_prefix"] + "/" + job_name + "/checkpoints",
            "quality_boundary": (
                "Optimizer updates, saved weights and loss are training diagnostics. "
                "Choose checkpoints on validation only. Final held-out cases and physical "
                "quality gates remain separate."
            ),
        },
    )
    write_json(output / "package.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    binding = commands.add_parser(
        "binding", help="Build a new train-only preparation binding from pinned qualification."
    )
    for name in ("qualification", "output"):
        binding.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "qualification-sha256",
        "tenant-id",
        "owner-id",
        "profile-sha256",
        "criteria-sha256",
        "frozen-plan-sha256",
    ):
        binding.add_argument(f"--{name}", required=True)
    plan = commands.add_parser(
        "plan", help="Emit a native job plan and registration files offline."
    )
    for name in (
        "config",
        "binding",
        "raw-manifest",
        "parent-manifest",
        "backbone-manifest",
        "output",
    ):
        plan.add_argument(f"--{name}", type=Path, required=True)
    plan.add_argument("--job-name", required=True)
    args = parser.parse_args()
    if args.command == "binding":
        value = preparation_binding(
            args.qualification,
            expected_qualification_sha256=args.qualification_sha256,
            scope=Scope(args.tenant_id, args.owner_id),
            expected_profile_sha256=args.profile_sha256,
            expected_criteria_sha256=args.criteria_sha256,
            expected_frozen_plan_sha256=args.frozen_plan_sha256,
        )
        write_json(args.output, value)
        print(
            canonical(
                {
                    "binding_sha256": file_digest(args.output),
                    "used_for_training_data": False,
                    "qualification_payload_verified": False,
                    "cloud_calls": 0,
                }
            ).decode()
        )
    else:
        print(
            canonical(
                create_package(
                    read_json(args.config),
                    read_json(args.binding),
                    raw_manifest=args.raw_manifest,
                    parent_manifest=args.parent_manifest,
                    backbone_manifest=args.backbone_manifest,
                    output=args.output,
                    job_name=args.job_name,
                )
            ).decode()
        )


if __name__ == "__main__":
    main()
