"""CPU metadata/Unix socket admission tests; not real command training or neural execution."""

import importlib
import os
import socket
from pathlib import Path

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_command_runtime_admission import command_runtime_value as command_runtime_value
from test_isaac_control_actuation import hardware as hardware
from test_paused_dispatch import paused_core as paused_core
from test_paused_learned_runtime import learned as learned
from test_paused_learned_runtime import start, step_to
from test_paused_policy_deployment import installed as installed
from test_simulation_batch import spec as spec
from test_teaching_runtime import teaching as teaching

from learning.common import canonical, file_digest
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.batch_learned import BatchLearnedSpec, parse_model_runtime
from simulation.paused_deployment import InstalledPausedPolicyProvider


def provider_type():
    return importlib.import_module(
        "simulation.command_policy_deployment"
    ).CommandPausedPolicyProvider


@pytest.fixture
def command_installed(installed, learned_spec, command_runtime_value):
    core, request, permission, path, catalogue, socket_path, metadata = installed
    metadata.update(schema="physicalai.smolvla-checkpoint/v3", training_execution="azureml_command")
    metadata["training"].pop("azure_pipeline_job_id")
    metadata["training"].pop("azure_component_job_id", None)
    metadata["training"]["azure_job_type"] = "command"
    metadata["training"]["gpu"]["device_count"] = 1
    metadata["training"]["episodes"] = [
        {
            "episode_id": "cpu-training-10001",
            "environment_id": "cpu-training-case",
            "revision": "a" * 64,
            "seed": 10001,
        }
    ]
    model_root = Path(catalogue["policies"][0]["model_root"])
    (model_root / "model.json").write_bytes(canonical(metadata))
    model_sha = file_digest(model_root / "model.json")
    request = request.model_copy(update={"model_sha256": model_sha})
    permission = permission.model_copy(update={"model_sha256": model_sha})
    grant = catalogue["policies"][0]["grant"]
    grant["authorization"]["model_sha256"] = model_sha
    path.write_bytes(canonical(catalogue))
    value = learned_spec.model_dump(mode="json", by_alias=True)
    prefix = f"tenants/{grant['tenant_id']}/owners/{permission.owner}/learning/"
    value.update(
        owner_id=permission.owner,
        source_revision=grant["source_revision"],
        control_profile_sha256=core.paused_profile.sha256,
        profile_id=core.paused_profile.profile_id,
        criteria_canonical_sha256=permission.criteria_sha256,
        conditions_canonical_sha256=permission.frozen_plan_sha256,
    )
    value["platform"].update(
        container_image="test.azurecr.io/simulator@" + grant["simulator_image_digest"],
        tenant_id=grant["tenant_id"],
    )
    value["model"]["manifest"].update(
        name=prefix + "model/model.json",
        sha256=model_sha,
        size_bytes=(model_root / "model.json").stat().st_size,
    )
    value["backbone"]["manifest"]["name"] = prefix + "backbone/backbone.json"
    spec = BatchLearnedSpec.model_validate(value)
    runtime = parse_model_runtime(
        {
            **command_runtime_value,
            "simulator_image": spec.platform.container_image,
            "simulator_source_revision": spec.source_revision,
            "control_profile_sha256": core.paused_profile.sha256,
            "legacy_servo_sha256": core.paused_profile.servo_profile_sha256,
        }
    )
    return core, request, permission, path, socket_path, metadata, spec, runtime


def test_explicit_command_provider_reuses_original_socket_and_action_guards(command_installed):
    core, request, permission, path, socket_path, metadata, spec, runtime = command_installed
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o660)
        provider = provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )
        assert provider.authorize(permission.owner, request, core.paused_profile) == permission
        assert type(provider).authorize is InstalledPausedPolicyProvider.authorize
        assert type(provider).create is InstalledPausedPolicyProvider.create
        assert type(provider)._check_runtime is InstalledPausedPolicyProvider._check_runtime
        port = provider.create(
            permission.owner, request, core.paused_profile, publication_guard=lambda *args: False
        )
        assert isinstance(port, PausedGuardedPolicyAdapter)
        assert port.policy.model_sha256 == permission.model_sha256
        assert port.policy.expected_peer_uid == os.geteuid()
        assert port.policy.predict_calls == 0
        assert metadata["schema"] == "physicalai.smolvla-checkpoint/v3"
        assert "azure_pipeline_job_id" not in metadata["training"]


def test_legacy_provider_still_rejects_command_v3_without_any_manifest_conversion(
    command_installed,
):
    core, _, _, path, _, _, _, _ = command_installed
    with pytest.raises(ValueError, match="v2|provenance"):
        InstalledPausedPolicyProvider.load(path, file_digest(path), core.paused_profile)


def test_new_command_provider_does_not_admit_a_legacy_v2_candidate(command_installed):
    from learning.common import read_json

    core, _, _, path, _, metadata, spec, runtime = command_installed
    catalogue = read_json(path)
    metadata["schema"] = "physicalai.smolvla-checkpoint/v2"
    metadata.pop("training_execution")
    metadata["training"].pop("azure_job_type")
    metadata["training"]["azure_pipeline_job_id"] = (
        metadata["training"]["azure_job_id"] + "-pipeline"
    )
    model_path = Path(catalogue["policies"][0]["model_root"]) / "model.json"
    model_path.write_bytes(canonical(metadata))
    checksum = file_digest(model_path)
    catalogue["policies"][0]["grant"]["authorization"]["model_sha256"] = checksum
    path.write_bytes(canonical(catalogue))
    spec = spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "manifest": spec.model.manifest.model_copy(update={"sha256": checksum}),
                }
            )
        }
    )
    assert InstalledPausedPolicyProvider.load(path, file_digest(path), core.paused_profile)
    with pytest.raises(ValueError, match="command"):
        provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )


@pytest.mark.parametrize("change", ["descriptor", "profile", "model", "owner", "image"])
def test_command_admission_rejects_foreign_runtime_or_grant(command_installed, change):
    core, _, _, path, _, _, spec, runtime = command_installed
    if change == "descriptor":
        runtime = None
    elif change == "profile":
        runtime = runtime.model_copy(update={"control_profile_sha256": "f" * 64})
    elif change == "model":
        spec = spec.model_copy(
            update={
                "model": spec.model.model_copy(
                    update={
                        "manifest": spec.model.manifest.model_copy(update={"sha256": "f" * 64}),
                    }
                )
            }
        )
    elif change == "owner":
        spec = spec.model_copy(update={"owner_id": "f" * 64})
    else:
        runtime = runtime.model_copy(
            update={"simulator_image": "unit.azurecr.io/ml@sha256:" + "f" * 64}
        )
    with pytest.raises(ValueError):
        provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )


def test_command_provider_socket_targets_reach_the_same_six_guarded_sdk_actions(
    command_installed, learned
):
    import threading
    import time

    from learning.contract import DemonstrationSource, Scope
    from learning.gr00t.ipc import receive_packet, send_packet
    from learning.paused.ipc import make_response, validate_request

    core, request, permission, path, socket_path, _, spec, runtime = command_installed
    assert core is learned.core
    core.clock_ns = time.monotonic_ns
    learned.runtime.paused_policy_worker.clock_ns = time.monotonic_ns
    errors = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o660)
        listener.listen(1)
        listener.settimeout(3)
        provider = provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )
        core.paused_authorizer = core.paused_policy_provider = provider
        learned.provider, learned.request = provider, request
        task = DemonstrationSource(
            "learned", **permission.task.model_dump(), source_policy_sha256=permission.model_sha256
        )

        def serve_one():
            try:
                connection, _ = listener.accept()
                with connection:
                    message = receive_packet(
                        connection, deadline_ns=time.monotonic_ns() + 2_000_000_000
                    )
                    observation, context = validate_request(
                        message,
                        model_sha256=permission.model_sha256,
                        scope=Scope(str(spec.platform.tenant_id), spec.owner_id),
                        profile=core.paused_profile,
                        task=task,
                    )
                    learned.model.entered.set()
                    actions = ((0.001, *observation.joint_positions[1:]),) * 50
                    send_packet(
                        connection,
                        make_response(message, actions, inference_latency_ms=1),
                        deadline_ns=context.operation_deadline_ns,
                    )
            except (ValueError, RuntimeError, OSError) as exc:
                errors.append(exc)

        thread = threading.Thread(target=serve_one)
        thread.start()
        try:
            start(learned)
            step_to(learned, 6)
        finally:
            thread.join(3)
        assert not thread.is_alive() and not errors
    assert len(learned.cell.robot.actions) == 6
    assert all(action.joint_positions[0] == 0.001 for action in learned.cell.robot.actions)
    metrics = learned.cell.paused_driver.metrics()
    assert metrics.policy_predict_calls == 1 and metrics.reference_route_calls == 0
    assert metrics.applied_model_sha256 == permission.model_sha256
    assert learned.model.calls == 0


@pytest.mark.parametrize("field", ["azure_pipeline_job_id", "azure_component_job_id"])
def test_command_provider_rejects_fake_pipeline_lineage_before_policy_creation(
    command_installed, field
):
    core, _, _, path, _, metadata, spec, runtime = command_installed
    from learning.common import read_json

    catalogue = read_json(path)
    metadata["training"][field] = metadata["training"]["azure_job_id"]
    model_path = Path(catalogue["policies"][0]["model_root"]) / "model.json"
    model_path.write_bytes(canonical(metadata))
    checksum = file_digest(model_path)
    catalogue["policies"][0]["grant"]["authorization"]["model_sha256"] = checksum
    path.write_bytes(canonical(catalogue))
    spec = spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "manifest": spec.model.manifest.model_copy(update={"sha256": checksum}),
                }
            )
        }
    )
    with pytest.raises(ValueError):
        provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )


@pytest.mark.parametrize("change", ["expired", "renewed", "task", "manifest", "peer"])
def test_command_path_retains_original_deadline_task_and_manifest_guards(command_installed, change):
    from datetime import timedelta

    from apps.api.models import utcnow
    from learning.common import read_json

    core, request, permission, path, _, _, spec, runtime = command_installed
    catalogue = read_json(path)
    entry = catalogue["policies"][0]
    if change == "expired":
        entry["grant"]["expires_at"] = (utcnow() - timedelta(seconds=1)).isoformat()
    elif change == "renewed":
        entry["grant"]["expires_at"] = (utcnow() + timedelta(seconds=900)).isoformat()
    elif change == "task":
        entry["grant"]["authorization"]["task"]["instruction"] = "Unapproved changed task."
    elif change == "peer":
        entry["expected_peer_uid"] += 1
    else:
        provider = provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )
        (Path(entry["model_root"]) / "model.json").write_bytes(b"changed")
        with pytest.raises(ValueError, match="checksum"):
            provider.create(
                permission.owner,
                request,
                core.paused_profile,
                publication_guard=lambda *args: False,
            )
        return
    path.write_bytes(canonical(catalogue))
    with pytest.raises(ValueError):
        provider_type().load(
            path, file_digest(path), core.paused_profile, spec=spec, runtime=runtime
        )
