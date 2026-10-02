"""CPU-only attestation tests; no trained weights, images, Azure clients or GPU execution."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_simulation_batch import spec as spec

from learning.common import canonical, digest, file_digest
from learning.paused import PausedControlProfile
from simulation import batch_learned


@pytest.fixture
def command_runtime_value(learned_spec):
    profile = PausedControlProfile("a" * 64)
    sources = {
        "learning/paused/model.py": "b" * 64,
        "learning/paused/command_model.py": "c" * 64,
        "learning/paused/command_artifacts.py": "d" * 64,
    }
    return {
        "schema": "physicalai.paused-model-runtime/v2",
        "python_executable": batch_learned.MODEL_PYTHON,
        "python_sha256": "e" * 64,
        "python_version": "3.11",
        "code_root": "/work",
        "dependency_image": "unit.azurecr.io/ml@sha256:" + "1" * 64,
        "code_image": "unit.azurecr.io/ml@sha256:" + "2" * 64,
        "source_files": sources,
        "admission_kind": "azureml_command_v3",
        "artifact_schema": "physicalai.smolvla-checkpoint/v3",
        "training_execution": "azureml_command",
        "server_entrypoint": "learning.paused.command_model",
        "provider_entrypoint": "simulation.command_policy_deployment.CommandPausedPolicyProvider",
        "request_schema": "physicalai.smolvla-request/v2",
        "response_schema": "physicalai.smolvla-response/v2",
        "legacy_servo_sha256": profile.servo_profile_sha256,
        "control_profile_sha256": profile.sha256,
        "simulator_image": learned_spec.platform.container_image,
        "simulator_source_revision": learned_spec.source_revision,
        "simulator_source_files": {
            **sources,
            "simulation/batch_learned.py": "f" * 64,
            "simulation/learned_probe.py": "f" * 64,
            "simulation/paired_evaluation.py": "f" * 64,
            "simulation/command_policy_deployment.py": "f" * 64,
        },
    }


def test_command_descriptor_is_an_explicit_closed_version_not_a_legacy_alias(command_runtime_value):
    runtime = batch_learned.parse_model_runtime(command_runtime_value)
    assert type(runtime) is batch_learned.CommandModelRuntime
    assert runtime.schema_version == "physicalai.paused-model-runtime/v2"
    assert runtime.admission_kind == "azureml_command_v3"
    with pytest.raises(ValueError):
        batch_learned.ModelRuntime.model_validate(command_runtime_value)
    encoded = canonical(runtime.model_dump(mode="json", by_alias=True))
    assert digest(encoded) == digest(canonical(command_runtime_value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("admission_kind", "pipeline"),
        ("artifact_schema", "physicalai.smolvla-checkpoint/v2"),
        ("training_execution", "azureml_pipeline"),
        ("server_entrypoint", "learning.paused.model"),
        ("provider_entrypoint", "simulation.paused_deployment.InstalledPausedPolicyProvider"),
        ("request_schema", "physicalai.smolvla-request/v3"),
        ("response_schema", "physicalai.smolvla-response/v1"),
    ],
)
def test_descriptor_rejects_mixed_model_admission_or_arbitrary_entrypoints(
    command_runtime_value, field, value
):
    with pytest.raises(ValueError):
        batch_learned.parse_model_runtime({**command_runtime_value, field: value})


@pytest.mark.parametrize("mutation", ["missing-new", "foreign-path", "different-parser"])
def test_both_source_contexts_must_attest_the_new_admission_code(command_runtime_value, mutation):
    value = deepcopy(command_runtime_value)
    if mutation == "missing-new":
        del value["source_files"]["learning/paused/command_model.py"]
    elif mutation == "foreign-path":
        value["simulator_source_files"]["../learning/loader.py"] = "b" * 64
    else:
        value["simulator_source_files"]["learning/paused/command_artifacts.py"] = "9" * 64
    with pytest.raises(ValueError, match="source|entry|admission|path"):
        batch_learned.parse_model_runtime(value)


def test_command_binding_requires_exact_app_image_profile_and_original_source(
    command_runtime_value, learned_spec
):
    runtime = batch_learned.parse_model_runtime(command_runtime_value)
    profile = PausedControlProfile(runtime.legacy_servo_sha256)
    bound = learned_spec.model_copy(
        update={
            "control_profile_sha256": profile.sha256,
            "profile_id": profile.profile_id,
        }
    )
    batch_learned.validate_command_runtime_binding(runtime, bound, profile)
    for update in (
        {"source_revision": "0" * 40},
        {"control_profile_sha256": "0" * 64},
        {
            "platform": bound.platform.model_copy(
                update={
                    "container_image": "unit.azurecr.io/sim@sha256:" + "0" * 64,
                }
            )
        },
    ):
        with pytest.raises(ValueError, match="binding|runtime|profile"):
            batch_learned.validate_command_runtime_binding(
                runtime, bound.model_copy(update=update), profile
            )
    assert asdict(profile)["servo_profile_sha256"] == runtime.legacy_servo_sha256


def test_legacy_server_command_is_unchanged_and_command_server_is_explicit(command_runtime_value):
    args = {
        "model_root": "/data/model",
        "backbone_root": "/data/backbone",
        "binding": "/data/binding.json",
        "socket_path": "/data/ipc/policy.sock",
        "model_sha256": "a" * 64,
        "uid": 1234,
    }
    legacy = batch_learned.model_command(**args)
    runtime = batch_learned.parse_model_runtime(command_runtime_value)
    command = batch_learned.model_command(**args, runtime=runtime)
    assert legacy[0:3] == [batch_learned.MODEL_PYTHON, "-m", "learning.paused.model"]
    assert command[0:3] == [batch_learned.MODEL_PYTHON, "-m", "learning.paused.command_model"]
    assert command[3:] == legacy[3:]


def test_complete_source_inventory_rejects_extra_bytes_and_symlinks(tmp_path):
    root = tmp_path / "app"
    source = root / "simulation" / "provider.py"
    source.parent.mkdir(parents=True)
    source.write_text("# CPU source fixture\n")
    expected = {"simulation/provider.py": file_digest(source)}
    batch_learned.verify_python_sources(root, ("simulation",), expected)
    extra = source.with_name("extra.py")
    extra.write_text("# Extra unapproved import\n")
    with pytest.raises(ValueError, match="inventory"):
        batch_learned.verify_python_sources(root, ("simulation",), expected)
    extra.unlink()
    link = source.with_name("hidden")
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        batch_learned.verify_python_sources(root, ("simulation",), expected)


def test_runtime_file_hash_is_mandatory_before_v3_selection(
    command_runtime_value, learned_spec, tmp_path
):
    path = tmp_path / "model-runtime.json"
    path.write_bytes(canonical(command_runtime_value))
    with pytest.raises(ValueError, match="checksum|hash"):
        batch_learned.read_model_runtime(path, learned_spec.model_runtime.sha256)
    actual = batch_learned.read_model_runtime(path, file_digest(path))
    assert actual.admission_kind == "azureml_command_v3"


def test_command_source_context_cannot_keep_the_native_legacy_parser_under_a_symlink(
    command_runtime_value, tmp_path
):
    source = tmp_path / "native"
    source.mkdir()
    path = source / "model-runtime.json"
    path.write_bytes(canonical(command_runtime_value))
    link = tmp_path / "alias"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink|sidecar"):
        batch_learned.read_model_runtime(link / "model-runtime.json", file_digest(path))


def test_runtime_v2_requires_the_complete_same_learning_inventory_in_both_contexts(
    command_runtime_value,
):
    value = deepcopy(command_runtime_value)
    value["simulator_source_files"]["learning/paused/unattested_import.py"] = "a" * 64
    with pytest.raises(ValueError, match="source|admission"):
        batch_learned.parse_model_runtime(value)


@pytest.mark.parametrize("where", ["app-extra", "app-changed", "work-extra", "work-changed"])
def test_actual_source_mismatch_stops_before_any_interpreter_or_model_process(
    command_runtime_value, learned_spec, tmp_path, monkeypatch, where
):
    import sys
    import time

    from simulation.paused_configuration import paused_servo_sha256
    from simulation.paused_profiles import paused_profile

    app, work = tmp_path / "app", tmp_path / "work"
    for prefix, files in (
        (app, command_runtime_value["simulator_source_files"]),
        (work, command_runtime_value["source_files"]),
    ):
        for name in files:
            path = prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU source attestation fixture\n")
    native = {name: file_digest(work / name) for name in command_runtime_value["source_files"]}
    simulator = {
        name: file_digest(app / name) for name in command_runtime_value["simulator_source_files"]
    }
    profile = paused_profile(paused_servo_sha256(), learned_spec.profile_id)
    runtime = batch_learned.parse_model_runtime(
        {
            **command_runtime_value,
            "python_sha256": file_digest(Path(sys.executable)),
            "source_files": native,
            "simulator_source_files": simulator,
            "control_profile_sha256": profile.sha256,
            "legacy_servo_sha256": profile.servo_profile_sha256,
        }
    )
    spec = learned_spec.model_copy(update={"control_profile_sha256": profile.sha256})
    root = app if where.startswith("app") else work
    changed = (
        root
        / "learning"
        / "paused"
        / ("extra.py" if where.endswith("extra") else "command_artifacts.py")
    )
    changed.write_text("# Unapproved code must not run\n")
    monkeypatch.setattr(batch_learned, "MODEL_CODE", str(work))
    monkeypatch.setattr(batch_learned, "SIMULATOR_CODE", str(app))
    monkeypatch.setattr(batch_learned, "MODEL_PYTHON", sys.executable)
    calls = []
    monkeypatch.setattr(batch_learned.subprocess, "run", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="inventory"):
        batch_learned.verify_model_runtime(runtime, deadline=time.monotonic() + 30, spec=spec)
    assert calls == []
