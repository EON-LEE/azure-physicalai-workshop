"""Strict authority/process lifecycle tests using CPU-only files and owned Unix processes."""

import json
import os
import sys
import time
from datetime import timedelta
from uuid import uuid4

import pytest
from test_batch_learned import learned_spec as learned_spec
from test_batch_simulation_task import Store
from test_paused_environment import paused_document
from test_simulation_batch import spec as spec

from apps.api.models import SaveEnvironment, utcnow
from learning.common import canonical, digest, file_digest
from simulation import batch_learned
from simulation.extensions import SceneRegistry
from simulation.paused_configuration import paused_servo_sha256
from simulation.paused_contracts import ResolvedSimulationAuthorization
from simulation.paused_profiles import paused_profile
from tests.runtime_support import ACTOR, service


@pytest.fixture
def authority(learned_spec, tmp_path, monkeypatch):
    document = paused_document()
    document["scene"]["seed"] = 30001
    document["learning_execution"].update(
        schema="physicalai.paused-simulation/v2",
        profile_id=learned_spec.profile_id,
        max_simulation_seconds=60,
    )
    document["execution"].update(record_demonstration=True, demonstration_split="test")
    environment = service().save_environment(
        ACTOR, SaveEnvironment(document_json=json.dumps(document))
    )
    scene = SceneRegistry(load_installed=False).build(environment)
    profile = paused_profile(paused_servo_sha256(), learned_spec.profile_id)
    task = {
        "task_id": "place-part",
        "instruction": "Place the part in the tray.",
        "goal_id": "rejected",
    }
    criteria = {"execution_timing": "paused_simulation", "real_time_admission": False, "task": task}
    conditions = {
        **criteria,
        "criteria_canonical_sha256": digest(canonical(criteria)),
        "cases": [
            {
                "environment_id": environment.environment_id,
                "revision": environment.revision,
                "seed": scene.seed,
                "split": "test",
                "scene_builder_sha256": scene.scene_builder_sha256,
                "expected_initial_position_m": list(scene.part_position),
                "goal_position_m": list(scene.station(task["goal_id"]).position),
            }
        ],
    }
    now = utcnow()
    permit = ResolvedSimulationAuthorization(
        authorization_id=uuid4(),
        authorization_kind="evaluation_grant",
        owner=learned_spec.owner_id,
        environment_id=environment.environment_id,
        revision=environment.revision,
        controller="learned",
        task=task,
        control_profile_sha256=profile.sha256,
        profile_id=profile.profile_id,
        wall_expires_at=now + timedelta(seconds=590),
        max_episode_wall_seconds=600,
        max_simulation_steps=3600,
        purpose="evaluation",
        criteria_sha256=digest(canonical(criteria)),
        frozen_plan_sha256=digest(canonical(conditions)),
        policy_type="smolvla",
        model_sha256=learned_spec.model.manifest.sha256,
    )
    grant = {
        "schema": "physicalai.paused-operator-grant/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "tenant_id": str(learned_spec.platform.tenant_id),
        "source_revision": learned_spec.source_revision,
        "simulator_image_digest": learned_spec.platform.container_image.split("@", 1)[1],
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=600)).isoformat(),
        "authorization": permit.model_dump(mode="json"),
    }
    documents = {
        "environment": canonical(environment.model_dump(mode="json")),
        "grant": canonical(grant),
        "criteria": canonical(criteria),
        "conditions": canonical(conditions),
    }
    value = learned_spec.model_dump(mode="json", by_alias=True)
    value.update(
        control_profile_sha256=profile.sha256,
        criteria_canonical_sha256=permit.criteria_sha256,
        conditions_canonical_sha256=permit.frozen_plan_sha256,
    )
    for name, payload in documents.items():
        value[name].update(sha256=digest(payload), size_bytes=len(payload))
    spec = batch_learned.BatchLearnedSpec.model_validate(value)
    for name, value in {
        "SOURCE_REVISION": spec.source_revision,
        "SIMULATOR_IMAGE": spec.platform.container_image,
        "ENTRA_TENANT_ID": str(spec.platform.tenant_id),
    }.items():
        monkeypatch.setenv(name, value)
    return spec, documents, scene, profile


def test_original_learned_authority_binds_exact_saved_test_case_model_and_task(authority):
    spec, documents, scene, profile = authority
    _, actual_scene, actual_profile, grant = batch_learned.validate_inputs(spec, documents)
    assert actual_scene == scene and actual_profile == profile
    assert grant.authorization.model_sha256 == spec.model.manifest.sha256
    assert grant.authorization.max_simulation_steps == 3600
    assert grant.expires_at - grant.issued_at == timedelta(seconds=600)


@pytest.mark.parametrize("change", ["expired", "extended", "model", "owner", "task", "reference"])
def test_private_bytes_do_not_authorize_changed_or_expired_learned_grants(authority, change):
    spec, documents, _, _ = authority
    grant = json.loads(documents["grant"])
    if change == "expired":
        grant["issued_at"] = (utcnow() - timedelta(seconds=600)).isoformat()
        grant["expires_at"] = (utcnow() - timedelta(seconds=1)).isoformat()
    elif change == "extended":
        grant["expires_at"] = (utcnow() + timedelta(seconds=901)).isoformat()
    elif change == "reference":
        grant["authorization"]["controller"] = "reference_controller"
    else:
        key = {"model": "model_sha256", "owner": "owner", "task": "task"}[change]
        grant["authorization"][key] = (
            {**grant["authorization"]["task"], "instruction": "A different task"}
            if change == "task"
            else "f" * 64
        )
    documents["grant"] = canonical(grant)
    value = spec.model_dump(mode="json", by_alias=True)
    value["grant"].update(sha256=digest(documents["grant"]), size_bytes=len(documents["grant"]))
    changed = batch_learned.BatchLearnedSpec.model_validate(value)
    with pytest.raises(ValueError):
        batch_learned.validate_inputs(changed, documents)


def test_claim_precedes_all_learned_model_work_and_internal_requeue_cannot_repeat(
    learned_spec, tmp_path, monkeypatch
):
    store, calls = Store(), []
    path = tmp_path / "spec.json"
    path.write_bytes(canonical(learned_spec.model_dump(mode="json", by_alias=True)))

    def interrupted(*args, **kwargs):
        calls.append(True)
        assert store.claimed
        raise KeyboardInterrupt("CPU fixture preemption")

    monkeypatch.setattr(batch_learned, "run_native", interrupted)
    with pytest.raises(KeyboardInterrupt):
        batch_learned.run_managed(learned_spec, path, store, directory=tmp_path / "first")
    result = batch_learned.run_managed(learned_spec, path, store, directory=tmp_path / "requeue")
    assert len(calls) == 1 and result["accepted"] is False
    assert not (tmp_path / "requeue").exists()
    assert store.completion is None


def test_model_socket_startup_timeout_reaps_only_the_owned_process(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_learned, "MODEL_CODE", str(tmp_path))
    ipc = tmp_path / "ipc"
    ipc.mkdir(mode=0o700)
    pid = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        f"import os,time;open({str(pid)!r},'w').write(str(os.getpid()));time.sleep(60)",
    ]
    with (tmp_path / "model.log").open("wb") as log:
        with pytest.raises(ValueError, match="deadline"):
            with batch_learned.model_process(
                command, socket_path=ipc / "policy.sock", deadline=time.monotonic() + 0.5, log=log
            ):
                raise AssertionError("No socket became ready")
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_cpu_managed_imports_cannot_load_neural_or_isaac_packages():
    import subprocess

    code = """
import importlib.abc, sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = fullname.split('.')[0] in {'torch','isaacsim','omni','lerobot'}
        if blocked or fullname == 'azure.batch':
            raise AssertionError('GPU/model/Batch import crossed CPU boundary: '+fullname)
sys.meta_path.insert(0, Boundary())
import simulation.batch_learned, simulation.learned_probe
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_wrong_real_python_version_cannot_pass_the_mixed_image_gate(tmp_path, monkeypatch):
    from pathlib import Path

    if sys.version_info[:2] == (3, 11):
        pytest.skip("This negative gate requires a non-3.11 test interpreter.")
    source = tmp_path / "learning" / "paused" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# CPU-only runtime descriptor fixture\n")
    descriptor = batch_learned.ModelRuntime.model_validate(
        {
            "schema": "physicalai.paused-model-runtime/v1",
            "python_executable": batch_learned.MODEL_PYTHON,
            "python_sha256": file_digest(Path(sys.executable)),
            "python_version": "3.11",
            "code_root": "/work",
            "dependency_image": "unit.azurecr.io/ml@sha256:" + "a" * 64,
            "code_image": "unit.azurecr.io/ml@sha256:" + "b" * 64,
            "source_files": {"learning/paused/model.py": file_digest(source)},
        }
    )
    monkeypatch.setattr(batch_learned, "MODEL_CODE", str(tmp_path))
    monkeypatch.setattr(batch_learned, "MODEL_PYTHON", sys.executable)
    with pytest.raises(ValueError, match="Python 3.11"):
        batch_learned.verify_model_runtime(descriptor, deadline=time.monotonic() + 10)


def test_unlisted_native_python_source_cannot_escape_runtime_fingerprint(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    source = tmp_path / "learning" / "paused" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# CPU source fixture\n")
    (source.parent / "unlisted.py").write_text("# Must not escape image/source binding\n")
    descriptor = batch_learned.ModelRuntime.model_validate(
        {
            "schema": "physicalai.paused-model-runtime/v1",
            "python_executable": batch_learned.MODEL_PYTHON,
            "python_sha256": file_digest(Path(sys.executable)),
            "python_version": "3.11",
            "code_root": "/work",
            "dependency_image": "unit.azurecr.io/ml@sha256:" + "a" * 64,
            "code_image": "unit.azurecr.io/ml@sha256:" + "b" * 64,
            "source_files": {"learning/paused/model.py": file_digest(source)},
        }
    )
    monkeypatch.setattr(batch_learned, "MODEL_CODE", str(tmp_path))
    monkeypatch.setattr(batch_learned, "MODEL_PYTHON", sys.executable)
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    (stdlib / "os.py").write_text("# CPU-only probe fixture\n")
    monkeypatch.setattr(
        batch_learned.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "version": [3, 11],
                    "executable": str(Path(sys.executable).resolve()),
                    "base_prefix": str(tmp_path),
                    "stdlib": str(stdlib),
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="source inventory"):
        batch_learned.verify_model_runtime(descriptor, deadline=time.monotonic() + 10)
