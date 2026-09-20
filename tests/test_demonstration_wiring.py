import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from runtime_support import ACTOR, PNG, document, service

from apps.api.models import MotionCommand, SaveEnvironment, utcnow
from contracts.validate_environment import validate_environment
from learning.contract import CameraSample, FrameSample
from simulation.core import SimulationCore
from simulation.demonstrations import Demonstration
from simulation.extensions import SceneRegistry


def active_capture(monkeypatch):
    monkeypatch.setenv("CAPTURE_ENABLED", "true")
    monkeypatch.setenv("SIMULATOR_IMAGE", "test.azurecr.io/simulator@sha256:" + "a" * 64)
    monkeypatch.setenv("FRANKA_ASSET_SHA256", "b" * 64)
    monkeypatch.setenv("SOURCE_REVISION", "c" * 40)
    monkeypatch.setenv("ENTRA_TENANT_ID", str(ACTOR.tenant_id))
    monkeypatch.setenv("STORAGE_ACCOUNT_URL", "https://test.blob.core.windows.net")
    monkeypatch.setenv("AZURE_CLIENT_ID", str(uuid4()))
    monkeypatch.setattr(
        "simulation.demonstrations.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Test GPU Identity\n"),
    )
    doc = document()
    doc["execution"].update(record_demonstration=True, demonstration_split="train")
    backend = service()
    env = backend.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(doc)))
    core = SimulationCore(SceneRegistry(load_installed=False))
    core.activate(ACTOR.owner_key, env)
    core.next_action()
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (-0.5, 0, 0.2), 5)
    observation = core.observe(ACTOR.owner_key, env.environment_id, env.revision, "inspection")
    command = MotionCommand(
        command_id=uuid4(),
        environment_id=env.environment_id,
        revision=env.revision,
        epoch=core.epoch,
        state_revision=core.state_revision,
        observation_id=observation.observation_id,
        object_id=observation.object_id,
        target_station_id="rejected",
        deadline=utcnow() + timedelta(seconds=20),
    )
    core.dispatch(ACTOR.owner_key, command)
    core.next_action()
    return core, command


def test_capture_is_an_explicit_customer_choice_and_requires_a_split():
    doc = document()
    assert not validate_environment(doc)
    doc["execution"]["record_demonstration"] = True
    assert validate_environment(doc)
    doc["execution"]["demonstration_split"] = "train"
    assert not validate_environment(doc)


def test_capture_cannot_start_before_core_approval(monkeypatch, tmp_path):
    core, command = active_capture(monkeypatch)
    with pytest.raises(ValueError, match="approved"):
        Demonstration(core, command.command_id, tmp_path)
    core.begin_motion(command.command_id)
    recorder = Demonstration(core, command.command_id, tmp_path)
    assert recorder.scope.owner_id == ACTOR.owner_key
    assert recorder.writer.manifest["fps"] == 60
    assert recorder.writer.manifest["physics_hz"] == 60
    assert recorder.root == tmp_path / ACTOR.owner_key / str(command.command_id)


def test_capture_fails_without_verified_deployment_provenance(monkeypatch, tmp_path):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    monkeypatch.delenv("SOURCE_REVISION")
    with pytest.raises(ValueError, match="provenance"):
        Demonstration(core, command.command_id, tmp_path)


def test_stale_stop_action_cannot_stop_a_different_active_command(monkeypatch):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    assert not core.should_stop(uuid4())
    core.cancel(ACTOR.owner_key, command.command_id)
    assert core.should_stop(command.command_id)
    core.finish("cancelled", None)
    assert not core.should_stop(command.command_id)


def test_validated_owner_scoped_manifest_is_uploaded_last(monkeypatch, tmp_path):
    core, command = active_capture(monkeypatch)
    core.begin_motion(command.command_id)
    recorder = Demonstration(core, command.command_id, tmp_path)
    positions = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02)
    start = datetime(2026, 9, 20, tzinfo=UTC)
    for index in range(2):
        mono = 1_000_000_000 + index * 16_666_667
        recorder.append(
            FrameSample(
                captured_at_utc=(start + timedelta(seconds=index / 60))
                .isoformat()
                .replace("+00:00", "Z"),
                monotonic_ns=mono,
                physics_step=index,
                joint_positions=positions,
                commanded_joint_targets=positions,
                images={
                    name: CameraSample(PNG, index, index, mono)
                    for name in ("inspection", "overview")
                },
                terminated=index == 1,
            )
        )
    uploaded = []

    class Resource:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_container_client(self, name):
            assert name == "demonstrations"
            return self

        def upload_blob(self, name, stream, **kwargs):
            assert kwargs["overwrite"] is False
            uploaded.append((name, stream.read()))

    monkeypatch.setattr("simulation.demonstrations.ManagedIdentityCredential", Resource)
    monkeypatch.setattr("simulation.demonstrations.BlobServiceClient", Resource)
    result = recorder.finalize_and_upload()
    assert uploaded[-1][0] == f"{ACTOR.owner_key}/{command.command_id}/manifest.json"
    assert all(name.startswith(f"{ACTOR.owner_key}/{command.command_id}/") for name, _ in uploaded)
    manifest = json.loads(uploaded[-1][1])
    assert manifest["scope"]["owner_id"] == ACTOR.owner_key
    assert result.frame_count == 2
    assert result.episode_id == str(command.command_id)
