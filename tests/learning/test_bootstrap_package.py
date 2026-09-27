from dataclasses import asdict

import pytest

from learning.checks.fixtures import LIVE_PROVENANCE, SCOPE
from learning.common import ContractError, file_digest, write_json
from learning.contract import DemonstrationSource
from learning.paused import PausedControlProfile
from learning.smolvla import UPSTREAM
from learning.smolvla.artifacts import processor_digest
from learning.smolvla.checkpoint_runner import POLICY_SCHEMA
from learning.smolvla.checkpoints import DEFAULT_LIMITS
from learning.smolvla.prepare import VENDOR_SHA256
from tests.learning.test_paused_azure import config as native_config


def manifests(tmp_path):
    """Metadata-only fixtures, never real simulator captures or model payloads."""
    profile = PausedControlProfile(
        "f" * 64,
        profile_id="franka-position-hold-10hz-paused-v2",
        max_simulation_steps=3600,
    )
    task = {
        "task_id": "manufacturing-part-placement-v1",
        "instruction": "Pick up the part and place it in the quarantine tray.",
        "goal_id": "rejected",
    }
    binding = {
        "scope": asdict(SCOPE),
        "control_profile": asdict(profile),
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "task": task,
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
    }
    raw = {
        "schema": "physicalai.demonstrations/v3",
        "dataset_id": "incoming-train",
        "scope": asdict(SCOPE),
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "control_profile": asdict(profile),
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
        "purpose": "demonstration",
        "fps": 10,
        "physics_hz": 60,
        "episodes": [
            {
                "episode_id": f"train-{seed}",
                "environment_id": f"new-mode-{seed}",
                "revision": "c" * 64,
                "seed": seed,
                "split": "train",
                "provenance": asdict(LIVE_PROVENANCE),
                "demonstration": asdict(DemonstrationSource(kind="reference_controller", **task)),
                "frame_count": 424,
                "path": f"episodes/train-{seed}/frames.jsonl",
                "sha256": "9" * 64,
            }
            for seed in range(10001, 10021)
        ],
    }
    backbone = {
        "schema": "physicalai.smolvla-backbone/v1",
        "scope": asdict(SCOPE),
        "upstream": UPSTREAM,
        "files": {"assets/model.safetensors": VENDOR_SHA256["backbone"]["model.safetensors"]},
    }
    backbone_path = tmp_path / "backbone.json"
    write_json(backbone_path, backbone)
    files = {
        "model.safetensors": VENDOR_SHA256["model"]["model.safetensors"],
        "policy_preprocessor.json": "a" * 64,
        "policy_postprocessor.json": "b" * 64,
    }
    parent = {
        "schema": "physicalai.smolvla-checkpoint/v2",
        **binding,
        "timestamp_basis": "simulation_time",
        "upstream": UPSTREAM,
        "policy_type": "smolvla",
        "role": "pretrained",
        "training": None,
        "backbone_manifest_sha256": file_digest(backbone_path),
        "checkpoint_files": files,
        "weights_sha256": files["model.safetensors"],
        "processor_sha256": processor_digest(files),
        "action_horizon": 50,
        "n_action_steps": 1,
        "image_size": 256,
    }
    # The helper checks reviewed manifest metadata only. Model/data payload verification
    # remains mandatory in the existing managed component and is never claimed by this test.
    return binding, raw, parent, backbone


def setup_package(tmp_path):
    binding, raw, parent, backbone = manifests(tmp_path)
    raw_path, parent_path = tmp_path / "manifest.json", tmp_path / "model.json"
    write_json(raw_path, raw)
    write_json(parent_path, parent)
    config = native_config()
    config["control_profile_sha256"] = PausedControlProfile(**binding["control_profile"]).sha256
    from learning.common import canonical, digest

    config["task_sha256"] = digest(canonical(binding["task"]))
    config["checkpointing"] = {
        "schema": POLICY_SCHEMA,
        "limits": asdict(DEFAULT_LIMITS),
        "resume": None,
    }
    for name, path in (
        ("demonstrations", raw_path),
        ("parent_model", parent_path),
        ("backbone", tmp_path / "backbone.json"),
    ):
        config["inputs"][name]["sha256"] = file_digest(path)
        config["inputs"][name]["uri"] += "/" + name
    return binding, config, raw_path, parent_path, tmp_path / "backbone.json"


def test_bootstrap_package_builds_real_native_graph_without_claiming_payload_or_cloud_verification(
    tmp_path,
):
    from learning.paused.bootstrap import create_package

    binding, config, raw, parent, backbone = setup_package(tmp_path)
    output = tmp_path / "offline-package"
    receipt = create_package(
        config,
        binding,
        raw_manifest=raw,
        parent_manifest=parent,
        backbone_manifest=backbone,
        output=output,
        job_name="new-explicit-bootstrap",
    )
    from learning.common import read_json

    job = read_json(output / "plan" / "job.json")
    assert set(job["jobs"]) == {"convert", "train"}
    assert "learning.paused.components" in job["jobs"]["train"]["command"]
    assert receipt["status"] == "offline_package_requires_payload_preflight_and_operator_approval"
    assert receipt["raw_episode_count"] == 20
    assert receipt["raw_frame_count"] == 20 * 424
    assert receipt["payloads_verified"] is False
    assert receipt["cloud_calls"] == receipt["jobs_submitted"] == 0
    assert receipt["learning_quality_verified"] is False
    assert not (output / "plan" / "code" / "learning" / "paused" / "bootstrap.py").exists()
    assert receipt["optimizer_target_steps"] == config["parameters"]["max_steps"]
    assert receipt["checkpoint_publication_count_limit"] == 32
    assert (
        receipt["training_seconds"] + receipt["conversion_seconds"]
        == config["parameters"]["timeout_seconds"]
    )
    for name, asset in config["inputs"].items():
        registered = read_json(output / "registrations" / f"{name}.json")
        assert registered["name"] == asset["name"]
        assert registered["version"] == asset["version"]
        assert registered["path"] == asset["uri"]
        assert registered["type"] == "uri_folder"
        assert registered["tags"]["manifest_sha256"] == asset["sha256"]
    sequence = read_json(output / "operator-sequence.json")
    assert sequence["registration_commands_executed"] is False
    assert len(sequence["registration_commands"]) == 3
    assert all("job" not in command for command in sequence["registration_commands"])
    assert all(
        config["subscription_id"] in command for command in sequence["registration_commands"]
    )


def test_command_bootstrap_package_preserves_one_real_job_and_whole_command_budget(tmp_path):
    from learning.common import read_json
    from learning.paused.bootstrap import create_package
    from learning.paused.command import EXECUTION
    from learning.smolvla.embedded_source import prepare_context

    binding, config, raw, parent, backbone = setup_package(tmp_path)
    context = tmp_path / "image"
    prepare_context(context, direct=True)
    config.update(
        run_id="new-command-bootstrap",
        source_delivery=read_json(context / "source-delivery.json"),
        job_execution=EXECUTION,
        compute_size="Standard_NC24ads_A100_v4",
        compute_tier="LowPriority",
        output_prefix=f"tenants/{config['tenant_id']}/owners/{config['owner_id']}/learning/outputs",
    )
    output = tmp_path / "command-package"
    receipt = create_package(
        config,
        binding,
        raw_manifest=raw,
        parent_manifest=parent,
        backbone_manifest=backbone,
        output=output,
        job_name=config["run_id"],
    )
    job = read_json(output / "plan" / "job.json")
    assert job["type"] == "command" and job["name"] == config["run_id"]
    assert not {"jobs", "code", "inputs", "outputs"}.intersection(job)
    assert receipt["schema"] == "physicalai.paused-bootstrap-package/v2"
    assert receipt["job_execution"] == EXECUTION
    assert receipt["execution_seconds"] == config["parameters"]["timeout_seconds"]
    assert "conversion_seconds" not in receipt and "training_seconds" not in receipt
    assert receipt["raw_episode_count"] == 20 and receipt["raw_frame_count"] == 20 * 424
    assert receipt["payloads_verified"] is False and receipt["jobs_submitted"] == 0
    sequence = read_json(output / "operator-sequence.json")
    assert sequence["schema"] == "physicalai.paused-bootstrap-operator-sequence/v2"
    assert sequence["outputs"] == {}
    assert sequence["private_output_prefix"] == config["output_prefix"] + "/" + config["run_id"]
    assert sequence["job_execution"] == EXECUTION
    assert not any("job" in command for command in sequence["registration_commands"])


@pytest.mark.parametrize(
    "change",
    [
        "integration",
        "missing-seed",
        "extra-seed",
        "wrong-profile",
        "wrong-criteria",
        "fixture",
        "duplicate-episode",
    ],
)
def test_bootstrap_refuses_unqualified_or_incomplete_training_manifest(tmp_path, change):
    from learning.checks.fixtures import PROVENANCE, overwrite
    from learning.common import read_json
    from learning.paused.bootstrap import create_package

    binding, config, raw_path, parent_path, backbone_path = setup_package(tmp_path)
    raw = read_json(raw_path)
    if change == "integration":
        raw["purpose"] = "integration"
        raw["episodes"][0]["seed"] = 900002
    elif change == "missing-seed":
        raw["episodes"].pop()
    elif change == "extra-seed":
        raw["episodes"][0]["seed"] = 11001
    elif change == "wrong-profile":
        raw["control_profile"]["servo_profile_sha256"] = "0" * 64
    elif change == "wrong-criteria":
        raw["criteria_sha256"] = "0" * 64
    elif change == "fixture":
        raw["episodes"][0]["provenance"] = asdict(PROVENANCE)
    else:
        raw["episodes"][1]["episode_id"] = raw["episodes"][0]["episode_id"]
    overwrite(raw_path, raw)
    config["inputs"]["demonstrations"]["sha256"] = file_digest(raw_path)
    with pytest.raises(ContractError):
        create_package(
            config,
            binding,
            raw_manifest=raw_path,
            parent_manifest=parent_path,
            backbone_manifest=backbone_path,
            output=tmp_path / "rejected",
            job_name="bootstrap",
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "change", ["old-profile", "old-conditions", "candidate", "wrong-weights", "unlinked-backbone"]
)
def test_bootstrap_never_relabels_old_prepared_parent_or_candidate(tmp_path, change):
    from learning.checks.fixtures import overwrite
    from learning.common import read_json
    from learning.paused.bootstrap import create_package

    binding, config, raw, parent, backbone = setup_package(tmp_path)
    value = read_json(parent)
    if change == "old-profile":
        value["control_profile"]["servo_profile_sha256"] = "0" * 64
    elif change == "old-conditions":
        value["frozen_plan_sha256"] = "0" * 64
    elif change == "candidate":
        value["role"] = "candidate"
    elif change == "wrong-weights":
        value["weights_sha256"] = "0" * 64
    else:
        value["backbone_manifest_sha256"] = "0" * 64
    overwrite(parent, value)
    config["inputs"]["parent_model"]["sha256"] = file_digest(parent)
    with pytest.raises(ContractError):
        create_package(
            config,
            binding,
            raw_manifest=raw,
            parent_manifest=parent,
            backbone_manifest=backbone,
            output=tmp_path / "rejected",
            job_name="bootstrap",
        )


@pytest.mark.parametrize(
    "change", ["no-checkpoint", "resume", "cross-container-uri", "hash-mismatch"]
)
def test_bootstrap_plan_requires_first_job_checkpointing_and_exact_private_input_locations(
    tmp_path, change
):
    from learning.paused.bootstrap import create_package

    binding, config, raw, parent, backbone = setup_package(tmp_path)
    if change == "no-checkpoint":
        config.pop("checkpointing")
    elif change == "resume":
        config["parameters"]["resume_mode"] = "weights_only"
    elif change == "cross-container-uri":
        config["inputs"]["demonstrations"]["uri"] = config["inputs"]["demonstrations"][
            "uri"
        ].replace(f"/datastores/{config['datastore']}/", "/datastores/learningdemos/")
    else:
        config["inputs"]["backbone"]["sha256"] = "0" * 64
    with pytest.raises(ContractError):
        create_package(
            config,
            binding,
            raw_manifest=raw,
            parent_manifest=parent,
            backbone_manifest=backbone,
            output=tmp_path / "rejected",
            job_name="bootstrap",
        )


def test_binding_can_use_g0_profile_but_never_turns_the_integration_episode_into_training(tmp_path):
    from learning.paused.bootstrap import preparation_binding

    expected, raw, _, _ = manifests(tmp_path)
    profile_sha = PausedControlProfile(**expected["control_profile"]).sha256
    qualification = {
        "control_profile": raw["control_profile"],
        "control_profile_sha256": profile_sha,
        "criteria_sha256": expected["criteria_sha256"],
        "frozen_plan_sha256": expected["frozen_plan_sha256"],
        "task": expected["task"],
        "physical_status": "succeeded",
        "capture": {
            "status": "ready",
            "receipt": {"status": "uploaded", "manifest_sha256": "8" * 64},
        },
        "source_revision": "7" * 40,
        "simulator_image_digest": "sha256:" + "6" * 64,
        "environment_id": "nrt2-integration-test-900002",
    }
    captured = tmp_path / "g0-binding.json"
    write_json(captured, qualification)
    result = preparation_binding(
        captured,
        expected_qualification_sha256=file_digest(captured),
        scope=SCOPE,
        expected_profile_sha256=profile_sha,
        expected_criteria_sha256="d" * 64,
        expected_frozen_plan_sha256="e" * 64,
    )
    assert result == expected
    assert "seed" not in result and "episodes" not in result
    with pytest.raises(ContractError, match="profile"):
        preparation_binding(
            captured,
            expected_qualification_sha256=file_digest(captured),
            scope=SCOPE,
            expected_profile_sha256="0" * 64,
            expected_criteria_sha256="d" * 64,
            expected_frozen_plan_sha256="e" * 64,
        )


def test_changed_qualification_file_is_rejected_without_deriving_a_new_binding(tmp_path):
    from learning.paused.bootstrap import preparation_binding

    record = tmp_path / "qualification.json"
    write_json(record, {"control_profile": {}})
    with pytest.raises(ContractError, match="checksum"):
        preparation_binding(
            record,
            expected_qualification_sha256="0" * 64,
            scope=SCOPE,
            expected_profile_sha256="1" * 64,
            expected_criteria_sha256="2" * 64,
            expected_frozen_plan_sha256="3" * 64,
        )


def test_operator_manifest_read_remains_bounded(tmp_path):
    from learning.paused.bootstrap import preparation_binding

    record = tmp_path / "too-large.json"
    with record.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    with pytest.raises(ContractError, match="byte limit"):
        preparation_binding(
            record,
            expected_qualification_sha256="0" * 64,
            scope=SCOPE,
            expected_profile_sha256="1" * 64,
            expected_criteria_sha256="2" * 64,
            expected_frozen_plan_sha256="3" * 64,
        )
