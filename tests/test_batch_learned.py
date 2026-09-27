"""Managed learned wrapper CPU tests; no installed policy, cloud account or GPU proof."""

import importlib
import json
from datetime import timedelta

import pytest
from test_simulation_batch import batch_sdk as batch_sdk
from test_simulation_batch import spec as spec

from learning.common import canonical, digest


def adapter():
    return importlib.import_module("simulation.batch_learned")


@pytest.fixture
def learned_spec(spec):
    value = spec.model_dump(mode="json", by_alias=True)
    value["schema"] = "physicalai.batch-learned-evaluation/v1"
    prefix = f"tenants/{spec.platform.tenant_id}/owners/{spec.owner_id}/learning"
    value.update(
        role="candidate",
        model={
            "manifest": {
                "name": prefix + "/candidate/model.json",
                "sha256": "1" * 64,
                "size_bytes": 1024,
            },
            "files": [
                {"path": "checkpoint/model.safetensors", "sha256": "2" * 64, "size_bytes": 1024}
            ],
        },
        backbone={
            "manifest": {
                "name": prefix + "/backbone/backbone.json",
                "sha256": "3" * 64,
                "size_bytes": 1024,
            },
            "files": [{"path": "assets/model.safetensors", "sha256": "4" * 64, "size_bytes": 1024}],
        },
        model_runtime={
            "name": "approved/model-runtime.json",
            "sha256": "5" * 64,
            "size_bytes": 1024,
        },
    )
    return adapter().BatchLearnedSpec.model_validate(value)


def test_learned_task_routes_only_to_the_new_probe_and_keeps_parent_batch_guards(
    learned_spec, batch_sdk
):
    from simulation.batch import CONTAINER_OPTIONS

    job, task = adapter().build_learned_job_task(
        learned_spec,
        learned_spec.storage_account_url + "/artifacts/approved/attempt.json",
        "6" * 64,
    )
    assert task.command_line.startswith("-m simulation.batch_learned run ")
    assert "paused_probe" not in task.command_line
    assert task.container_settings.container_run_options == CONTAINER_OPTIONS
    assert task.constraints.max_task_retry_count == job.constraints.max_task_retry_count == 0
    assert job.constraints.max_wall_clock_time == timedelta(seconds=900)
    assert task.required_slots == 1
    assert job.id == learned_spec.job_id


@pytest.mark.parametrize("change", ["reference", "foreign", "duplicate", "escape", "size"])
def test_learned_spec_cannot_hide_reference_or_foreign_unbounded_model_inputs(learned_spec, change):
    value = learned_spec.model_dump(mode="json", by_alias=True)
    if change == "reference":
        value["role"] = "reference"
    elif change == "foreign":
        value["model"]["manifest"]["name"] = "foreign/model.json"
    elif change == "duplicate":
        value["model"]["files"].append(value["model"]["files"][0])
    elif change == "escape":
        value["model"]["files"][0]["path"] = "../model.py"
    else:
        value["model"]["files"][0]["size_bytes"] = 4 * 1024**3
    with pytest.raises(ValueError):
        adapter().BatchLearnedSpec.model_validate(value)


def test_native_model_environment_is_separate_from_isaac_python_and_tcp(learned_spec):
    env = adapter().model_environment(
        {
            "PYTHONHOME": "/isaac-sim/kit/python",
            "PYTHONPATH": "/isaac-sim/site-packages",
            "LD_LIBRARY_PATH": "/isaac-sim/kit",
            "LD_PRELOAD": "/isaac-sim/libfoo.so",
            "PATH": "/isaac-sim/kit/python/bin:/usr/bin",
            "AZ_BATCH_TASK_ID": "episode",
        }
    )
    assert env["PYTHONPATH"] == "/work"
    assert "PYTHONHOME" not in env and "LD_PRELOAD" not in env and "LD_LIBRARY_PATH" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["AZ_BATCH_TASK_ID"] == "episode"
    command = adapter().model_command(
        model_root="/data/model",
        backbone_root="/data/backbone",
        binding="/data/binding.json",
        socket_path="/data/ipc/policy.sock",
        model_sha256=learned_spec.model.manifest.sha256,
        uid=1234,
    )
    assert command[:3] == ["/opt/smolvla-venv/bin/python", "-m", "learning.paused.model"]
    assert command[-2:] == ["--allowed-client-uid", "1234"]
    assert not any("tcp" in arg or "http" in arg for arg in command)


def test_model_bundle_requires_exact_manifest_inventory_before_any_model_load(learned_spec):
    metadata = {
        "checkpoint_files": {"model.safetensors": "2" * 64},
    }
    adapter().validate_bundle_inventory(learned_spec.model, metadata, model=True)
    with pytest.raises(ValueError, match="inventory"):
        adapter().validate_bundle_inventory(
            learned_spec.model,
            {"checkpoint_files": {"model.safetensors": "f" * 64}},
            model=True,
        )


def test_runtime_descriptor_rejects_isaac_interpreter_and_unpinned_source():
    descriptor = {
        "schema": "physicalai.paused-model-runtime/v1",
        "python_executable": "/opt/smolvla-venv/bin/python",
        "python_sha256": "a" * 64,
        "python_version": "3.11",
        "code_root": "/work",
        "dependency_image": "unit.azurecr.io/ml@sha256:" + "b" * 64,
        "code_image": "unit.azurecr.io/ml@sha256:" + "c" * 64,
        "source_files": {"learning/paused/model.py": "d" * 64},
    }
    adapter().ModelRuntime.model_validate(descriptor)
    for altered in (
        {**descriptor, "python_executable": "/isaac-sim/python.sh"},
        {**descriptor, "python_version": "3.12"},
        {**descriptor, "code_image": "unit.azurecr.io/ml:latest"},
    ):
        with pytest.raises(ValueError):
            adapter().ModelRuntime.model_validate(altered)


def test_model_manifest_change_changes_whole_attempt_and_preserves_raw_spec_hash(learned_spec):
    altered = learned_spec.model_dump(mode="json", by_alias=True)
    altered["model"]["manifest"]["sha256"] = "f" * 64
    changed = adapter().BatchLearnedSpec.model_validate(altered)
    assert changed.sha256 != learned_spec.sha256
    assert changed.job_id == learned_spec.job_id
    encoded = canonical(learned_spec.model_dump(mode="json", by_alias=True))
    assert adapter().BatchLearnedSpec.model_validate(json.loads(encoded)).sha256 == digest(encoded)


def test_optional_pairing_hash_is_preexecution_bound_without_changing_standalone_wire(learned_spec):
    original = learned_spec.model_dump(mode="json", by_alias=True)
    assert "pairing_plan_sha256" not in original
    paired = adapter().BatchLearnedSpec.model_validate(
        {
            **original,
            "pairing_plan_sha256": "e" * 64,
        }
    )
    assert paired.pairing_plan_sha256 == "e" * 64
    assert paired.sha256 != learned_spec.sha256
    assert paired.model_dump(mode="json", by_alias=True)["pairing_plan_sha256"] == "e" * 64
