"""One managed learned evaluation, with a separate qualified Python 3.11 model process."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import stat
import subprocess
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from apps.api.models import Model, Revision
from learning.common import (
    canonical,
    digest,
    file_digest,
    finite,
    integer,
    parse_json,
    read_json,
    relative_path,
    require,
    sha256,
    token,
    vector,
)
from simulation.batch import (
    BATCH_TOKEN_SCOPE,
    BatchSimulationSpec,
    BlobInput,
    build_job_task,
    check_regional_capacity,
    inspect_platform,
    read_status,
    submit_requests_once,
)
from simulation.batch_task import PrivateArtifacts

Image = Annotated[str, Field(pattern=r"^[a-z0-9]+\.azurecr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$")]
MODEL_PYTHON = "/opt/smolvla-venv/bin/python"
MODEL_CODE = "/work"
SIMULATOR_CODE = "/app"
APP_SOURCE_ROOTS = ("apps", "contracts", "learning", "simulation")
SOURCE_EXCLUDES = frozenset(
    {"__pycache__", ".cache", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
)
COMMAND_NATIVE_SOURCES = frozenset(
    {
        "learning/paused/command_artifacts.py",
        "learning/paused/command_model.py",
    }
)
COMMAND_APP_SOURCES = COMMAND_NATIVE_SOURCES | {
    "simulation/command_policy_deployment.py",
    "simulation/batch_learned.py",
    "simulation/learned_probe.py",
    "simulation/paired_evaluation.py",
}


class ModelFile(Model):
    path: str
    sha256: Revision
    size_bytes: int = Field(gt=0, le=3 * 1024**3, strict=True)

    @field_validator("path")
    @classmethod
    def safe_path(cls, value):
        relative_path(value)
        return value


class ModelBundle(Model):
    manifest: BlobInput
    files: list[ModelFile] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def bounded_inventory(self):
        require(self.manifest.size_bytes <= 4 * 1024**2, "Model manifest exceeds four MiB.")
        require(
            len({item.path for item in self.files}) == len(self.files),
            "Duplicate model artifact paths.",
        )
        return self


class BatchLearnedSpec(BatchSimulationSpec):
    schema_version: Literal["physicalai.batch-learned-evaluation/v1"] = Field(alias="schema")
    role: Literal["candidate", "before", "after"]
    model: ModelBundle
    backbone: ModelBundle
    model_runtime: BlobInput
    evaluation_split: Literal["test", "validation"] = "test"
    pairing_plan_sha256: Revision | None = None

    @model_serializer(mode="wrap")
    def preserve_standalone_wire(self, handler):
        value = handler(self)
        if self.pairing_plan_sha256 is None:
            value.pop("pairing_plan_sha256", None)
        if self.evaluation_split == "test":
            value.pop("evaluation_split", None)
        return value

    @model_validator(mode="after")
    def owner_scoped_models(self):
        require(
            self.evaluation_split == "test"
            or (self.role == "candidate" and self.pairing_plan_sha256 is None),
            "Validation is an unpaired candidate trial, never final before/after evidence.",
        )
        prefix = f"tenants/{self.platform.tenant_id}/owners/{self.owner_id}/learning/"
        for bundle, name in ((self.model, "model.json"), (self.backbone, "backbone.json")):
            require(
                bundle.manifest.name.startswith(prefix)
                and bundle.manifest.name.endswith("/" + name),
                "Model/backbone inputs must be pinned inside the approved tenant/owner scope.",
            )
        require(
            sum(item.size_bytes for bundle in (self.model, self.backbone) for item in bundle.files)
            <= 8 * 1024**3,
            "Model input staging exceeds the bounded eight-GiB disk budget.",
        )
        require(self.model_runtime.size_bytes <= 65536, "Oversized model-runtime descriptor.")
        return self


def validate_evaluation_cohort(spec: BatchLearnedSpec, scene) -> None:
    seeds = range(20001, 20011) if spec.evaluation_split == "validation" else range(30001, 30021)
    require(
        scene.record_demonstration
        and scene.demonstration_split == spec.evaluation_split
        and scene.seed in seeds
        and (
            spec.evaluation_split == "test"
            or (spec.role == "candidate" and spec.pairing_plan_sha256 is None)
        ),
        "Saved scene does not match its explicit evaluation cohort; no TRAIN/integration relabel.",
    )


class ModelRuntime(Model):
    schema_version: Literal["physicalai.paused-model-runtime/v1"] = Field(alias="schema")
    python_executable: Literal["/opt/smolvla-venv/bin/python"]
    python_sha256: Revision
    python_version: Literal["3.11"]
    code_root: Literal["/work"]
    dependency_image: Image
    code_image: Image
    source_files: dict[str, Revision] = Field(min_length=1, max_length=512)

    @field_validator("source_files")
    @classmethod
    def code_inventory(cls, value):
        for name in value:
            relative_path(name)
            require(name.startswith("learning/") and name.endswith(".py"), "Invalid model source.")
        require("learning/paused/model.py" in value, "Missing actual model-server source.")
        return value


class CommandModelRuntime(ModelRuntime):
    schema_version: Literal["physicalai.paused-model-runtime/v2"] = Field(alias="schema")
    admission_kind: Literal["azureml_command_v3"]
    artifact_schema: Literal["physicalai.smolvla-checkpoint/v3"]
    training_execution: Literal["azureml_command"]
    server_entrypoint: Literal["learning.paused.command_model"]
    provider_entrypoint: Literal["simulation.command_policy_deployment.CommandPausedPolicyProvider"]
    request_schema: Literal["physicalai.smolvla-request/v2"]
    response_schema: Literal["physicalai.smolvla-response/v2"]
    legacy_servo_sha256: Revision
    control_profile_sha256: Revision
    simulator_image: Image
    simulator_source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    simulator_source_files: dict[str, Revision] = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def complete_admission_sources(self):
        for name in self.simulator_source_files:
            path = relative_path(name)
            require(
                path.parts[0] in APP_SOURCE_ROOTS
                and name.endswith(".py")
                and not SOURCE_EXCLUDES.intersection(path.parts),
                "Invalid simulator admission source path.",
            )
        require(
            COMMAND_NATIVE_SOURCES <= set(self.source_files)
            and COMMAND_APP_SOURCES <= set(self.simulator_source_files)
            and self.source_files
            == {
                name: checksum
                for name, checksum in self.simulator_source_files.items()
                if name.startswith("learning/")
            },
            "Both source contexts must attest the same complete native admission code.",
        )
        return self


def parse_model_runtime(value: dict) -> ModelRuntime | CommandModelRuntime:
    schema = value.get("schema")
    if schema == "physicalai.paused-model-runtime/v1":
        return ModelRuntime.model_validate(value)
    require(
        schema == "physicalai.paused-model-runtime/v2", "Unknown model admission runtime version."
    )
    return CommandModelRuntime.model_validate(value)


def read_model_runtime(path: Path, checksum: str) -> ModelRuntime | CommandModelRuntime:
    require(
        path.stat().st_size <= 65536
        and not any(item.is_symlink() for item in (path, *path.parents)),
        "Invalid or Symlink model-runtime sidecar.",
    )
    with path.open("rb") as stream:
        value = stream.read(65537)
    require(len(value) <= 65536, "Model-runtime sidecar exceeds its byte bound.")
    require(digest(value) == sha256(checksum), "Original model-runtime checksum differs.")
    return parse_model_runtime(parse_json(value))


def validate_command_runtime_binding(runtime: CommandModelRuntime, spec, profile) -> None:
    require(
        type(runtime) is CommandModelRuntime,
        "Command v3 requires an explicit v2 runtime descriptor.",
    )
    CommandModelRuntime.model_validate(runtime.model_dump(mode="json", by_alias=True))
    require(
        runtime.simulator_image == spec.platform.container_image
        and runtime.simulator_source_revision == spec.source_revision
        and runtime.control_profile_sha256 == spec.control_profile_sha256 == profile.sha256
        and runtime.legacy_servo_sha256 == profile.servo_profile_sha256
        and spec.profile_id == profile.profile_id,
        "Command admission runtime/image/source/profile binding differs.",
    )


def verify_python_sources(root: Path, prefixes: tuple[str, ...], expected: dict[str, str]) -> None:
    require(
        root.is_dir() and not any(item.is_symlink() for item in (root, *root.parents)),
        "Invalid or Symlink attested source root.",
    )
    actual = {}
    for prefix in prefixes:
        start = root / prefix
        require(not start.is_symlink(), "Symlink in attested source inventory.")
        if not start.exists():
            continue
        require(start.is_dir(), "Expected an attested Python source directory.")
        for directory, names, files in os.walk(start, followlinks=False):
            parent = Path(directory)
            for name in (*names, *files):
                require(not (parent / name).is_symlink(), "Symlink in attested source inventory.")
            names[:] = [name for name in names if name not in SOURCE_EXCLUDES]
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = parent / name
                relative = path.relative_to(root).as_posix()
                require(len(actual) < 2048, "Attested source inventory exceeds its bound.")
                actual[relative] = file_digest(path)
    require(actual == expected, "Complete Python source inventory/checksum differs.")


def command_admission_proof(runtime: CommandModelRuntime, *, runtime_sha256: str) -> dict:
    return {
        "admission_kind": runtime.admission_kind,
        "artifact_schema": runtime.artifact_schema,
        "training_execution": runtime.training_execution,
        "server_entrypoint": runtime.server_entrypoint,
        "provider_entrypoint": runtime.provider_entrypoint,
        "request_schema": runtime.request_schema,
        "response_schema": runtime.response_schema,
        "runtime_sha256": sha256(runtime_sha256),
        "legacy_servo_sha256": runtime.legacy_servo_sha256,
        "control_profile_sha256": runtime.control_profile_sha256,
        "simulator_image": runtime.simulator_image,
        "simulator_source_revision": runtime.simulator_source_revision,
        "simulator_sources_sha256": digest(canonical(runtime.simulator_source_files)),
        "native_sources_sha256": digest(canonical(runtime.source_files)),
    }


def validate_runtime_model(runtime: ModelRuntime, root: Path, *, scope, model_sha256: str) -> dict:
    if type(runtime) is CommandModelRuntime:
        from learning.paused.command_artifacts import validate_model
    else:
        require(type(runtime) is ModelRuntime, "Unknown model admission runtime type.")
        from learning.paused.artifacts import validate_model

    return validate_model(root, expected_scope=scope, expected_model_sha256=model_sha256)


def build_learned_job_task(spec: BatchLearnedSpec, spec_url: str, spec_sha256: str):
    job, task = build_job_task(spec, spec_url, spec_sha256)
    task.command_line = (
        f"-m simulation.batch_learned run --spec attempt.json --spec-sha256 {spec_sha256}"
    )
    return job, task


def validate_bundle_inventory(bundle: ModelBundle, metadata: dict, *, model: bool) -> None:
    inventory = (
        {"checkpoint/" + name: checksum for name, checksum in metadata["checkpoint_files"].items()}
        if model
        else metadata["files"]
    )
    require(
        inventory == {item.path: item.sha256 for item in bundle.files},
        "The private model file inventory differs from the hash-pinned native manifest.",
    )


def model_environment(parent: dict[str, str]) -> dict[str, str]:
    value = {
        key: item
        for key, item in parent.items()
        if not key.startswith("PYTHON")
        and key
        not in {
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CARB_APP_PATH",
            "EXP_PATH",
            "ISAAC_PATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        }
    }
    value.update(
        PYTHONPATH=MODEL_CODE,
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PATH="/opt/smolvla-venv/bin:/usr/local/bin:/usr/bin:/bin",
    )
    return value


def model_command(
    *,
    model_root,
    backbone_root,
    binding,
    socket_path,
    model_sha256,
    uid,
    runtime: ModelRuntime | None = None,
):
    require(
        runtime is None or type(runtime) in (ModelRuntime, CommandModelRuntime),
        "Unknown explicit model-server admission type.",
    )
    return [
        MODEL_PYTHON,
        "-m",
        "learning.paused.command_model"
        if type(runtime) is CommandModelRuntime
        else "learning.paused.model",
        "--model-root",
        str(model_root),
        "--backbone-root",
        str(backbone_root),
        "--model-sha256",
        model_sha256,
        "--binding",
        str(binding),
        "--socket-path",
        str(socket_path),
        "--allowed-client-uid",
        str(uid),
    ]


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    require(value > 0, "The original managed learned deadline expired.")
    return value


def load_inputs(spec: BatchLearnedSpec, paths: dict[str, Path], *, live: bool = True):
    return validate_inputs(
        spec, {name: path.read_bytes() for name, path in paths.items()}, live=live
    )


def validate_inputs(spec: BatchLearnedSpec, documents: dict[str, bytes], *, live: bool = True):
    from apps.api.models import EnvironmentRecord, utcnow
    from simulation.extensions import SceneRegistry
    from simulation.paused_configuration import PausedOperatorGrant, paused_servo_sha256
    from simulation.paused_deployment import InstalledPausedPolicyProvider
    from simulation.paused_profiles import paused_profile

    for name in ("environment", "grant", "criteria", "conditions"):
        require(
            digest(documents[name]) == getattr(spec, name).sha256,
            "An original approved learned input changed.",
        )
    from learning.common import parse_json

    environment = EnvironmentRecord.model_validate(parse_json(documents["environment"]))
    scene = SceneRegistry(load_installed=False).build(environment)
    scene_authority = scene.require_paused_authority()
    profile = paused_profile(paused_servo_sha256(), spec.profile_id)
    grant = PausedOperatorGrant.model_validate(parse_json(documents["grant"]))
    permit = grant.authorization
    require(
        profile.sha256 == spec.control_profile_sha256
        and scene_authority.profile_id == profile.profile_id == permit.profile_id
        and permit.owner == spec.owner_id
        and str(grant.tenant_id) == str(spec.platform.tenant_id)
        and grant.source_revision == spec.source_revision
        and grant.simulator_image_digest == spec.platform.container_image.split("@", 1)[1]
        and permit.environment_id == environment.environment_id
        and permit.revision == environment.revision
        and permit.controller == "learned"
        and permit.policy_type == "smolvla"
        and permit.authorization_kind == "evaluation_grant"
        and permit.purpose == "evaluation"
        and permit.model_sha256 == spec.model.manifest.sha256
        and permit.control_profile_sha256 == profile.sha256
        and permit.criteria_sha256 == spec.criteria_canonical_sha256
        and permit.frozen_plan_sha256 == spec.conditions_canonical_sha256
        and 0 < (grant.expires_at - grant.issued_at).total_seconds() <= 600
        and grant.issued_at < permit.wall_expires_at <= grant.expires_at,
        "Learned authority differs from the exact model, owner, case, profile or original grant.",
    )
    validate_evaluation_cohort(spec, scene)
    if live:
        InstalledPausedPolicyProvider._check_runtime(grant, profile)
        scene_authority.validate_request(
            wall_seconds=min(
                permit.max_episode_wall_seconds, (permit.wall_expires_at - utcnow()).total_seconds()
            ),
            simulation_steps=permit.max_simulation_steps,
        )
    criteria, conditions = parse_json(documents["criteria"]), parse_json(documents["conditions"])
    require(
        digest(canonical(criteria)) == spec.criteria_canonical_sha256
        and digest(canonical(conditions)) == spec.conditions_canonical_sha256
        and conditions.get("criteria_canonical_sha256") == spec.criteria_canonical_sha256,
        "Frozen learned criteria/conditions changed.",
    )
    for document in (criteria, conditions):
        require(
            document.get("execution_timing") == "paused_simulation"
            and document.get("real_time_admission") is False
            and document.get("task") == permit.task.model_dump(),
            "Frozen learned task or execution mode changed.",
        )
    cases = [
        case
        for case in conditions.get("cases", [])
        if case.get("environment_id") == environment.environment_id
        and case.get("revision") == environment.revision
    ]
    require(len(cases) == 1, "The learned case is not uniquely predeclared in frozen conditions.")
    case = cases[0]
    require(
        case["seed"] == scene.seed
        and case["split"] == scene.demonstration_split
        and case["scene_builder_sha256"] == scene.scene_builder_sha256
        and tuple(case["expected_initial_position_m"]) == scene.part_position
        and tuple(case["goal_position_m"]) == scene.station(permit.task.goal_id).position,
        "The measured-scene definition differs from the original frozen learned case.",
    )
    return environment, scene, profile, grant


def verify_model_runtime(
    descriptor: ModelRuntime, *, deadline: float, spec: BatchLearnedSpec | None = None
) -> dict:
    require(os.name == "posix", "The qualified model runtime must share the Linux kernel.")
    require(
        file_digest(Path(MODEL_PYTHON).resolve()) == descriptor.python_sha256,
        "The installed model interpreter differs from the qualified runtime descriptor.",
    )
    if type(descriptor) is CommandModelRuntime:
        from simulation.paused_configuration import paused_servo_sha256
        from simulation.paused_profiles import paused_profile

        require(spec is not None, "Command admission requires its original managed specification.")
        validate_command_runtime_binding(
            descriptor, spec, paused_profile(paused_servo_sha256(), spec.profile_id)
        )
        verify_python_sources(
            Path(SIMULATOR_CODE), APP_SOURCE_ROOTS, descriptor.simulator_source_files
        )
        verify_python_sources(Path(MODEL_CODE), ("learning",), descriptor.source_files)
    files = {
        path.relative_to(MODEL_CODE).as_posix()
        for path in (Path(MODEL_CODE) / "learning").rglob("*.py")
    }
    require(
        files == set(descriptor.source_files),
        "The complete native Python source inventory differs from its image binding.",
    )
    for name, checksum in descriptor.source_files.items():
        path = Path(MODEL_CODE) / name
        require(
            not path.is_symlink() and file_digest(path) == checksum,
            "Installed native model-server source differs from its immutable image binding.",
        )
    result = subprocess.run(
        [
            MODEL_PYTHON,
            "-c",
            (
                "import json,sys,sysconfig,os; "
                "print(json.dumps({'version':list(sys.version_info[:2]),"
                "'executable':os.path.realpath(sys.executable),'base_prefix':sys.base_prefix,"
                "'stdlib':sysconfig.get_path('stdlib')}))"
            ),
        ],
        cwd=MODEL_CODE,
        env=model_environment(dict(os.environ)),
        capture_output=True,
        text=True,
        timeout=min(15, remaining(deadline)),
        check=True,
    )
    proof = json.loads(result.stdout)
    require(
        proof["version"] == [3, 11]
        and Path(proof["executable"]).resolve() == Path(MODEL_PYTHON).resolve()
        and (Path(proof["stdlib"]) / "os.py").is_file()
        and Path(proof["base_prefix"]).is_dir(),
        "The mixed image lacks the real Python 3.11 interpreter/base/stdlib; no Isaac fallback.",
    )
    return proof


@contextmanager
def model_process(command: list[str], *, socket_path: Path, deadline: float, log):
    require(
        socket_path.is_absolute()
        and not socket_path.exists()
        and not socket_path.is_symlink()
        and socket_path.parent.stat().st_mode & 0o077 == 0,
        "A fresh private task-owned Unix socket directory is required.",
    )
    process = subprocess.Popen(
        command,
        cwd=MODEL_CODE,
        env=model_environment(dict(os.environ)),
        stdout=log,
        stderr=subprocess.STDOUT,
        process_group=0,
    )
    try:
        ready_deadline = min(deadline, time.monotonic() + 120)
        while not socket_path.exists():
            require(process.poll() is None, "The actual model server exited before readiness.")
            remaining(ready_deadline)
            time.sleep(min(0.05, remaining(ready_deadline)))
        info = socket_path.lstat()
        require(
            stat.S_ISSOCK(info.st_mode)
            and info.st_uid == os.geteuid()
            and info.st_mode & 0o007 == 0
            and process.poll() is None,
            "The model did not expose the expected protected same-UID Unix socket.",
        )
        yield process
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class ModelBlobInput(BlobInput):
    size_bytes: int = Field(gt=0, le=3 * 1024**3, strict=True)


def download_bundle(store, bundle: ModelBundle, destination: Path, *, model: bool, deadline: float):
    destination.mkdir(mode=0o700)
    manifest_name = "model.json" if model else "backbone.json"
    manifest = destination / manifest_name
    remaining(deadline)
    store.download(bundle.manifest, manifest)
    metadata = read_json(manifest)
    validate_bundle_inventory(bundle, metadata, model=model)
    prefix = bundle.manifest.name.rsplit("/", 1)[0]
    for item in bundle.files:
        remaining(deadline)
        target = destination / item.path
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        store.download(
            ModelBlobInput(
                name=prefix + "/" + item.path, sha256=item.sha256, size_bytes=item.size_bytes
            ),
            target,
        )
        target.chmod(0o400)
    manifest.chmod(0o400)
    remaining(deadline)
    return metadata


def rescore_evidence(spec: BatchLearnedSpec, documents: dict[str, bytes]) -> tuple[dict, dict]:
    from learning.common import parse_json
    from simulation.batch_task import validate_gpu_evidence
    from simulation.learned_probe import rescore

    _, scene, profile, grant = validate_inputs(
        spec,
        {
            name: documents[f"inputs/{name}.json"]
            for name in ("environment", "grant", "criteria", "conditions")
        },
        live=False,
    )
    preflight = parse_json(documents["preflight.json"])
    validate_gpu_evidence(preflight)
    server = preflight["model_process"]
    require(
        server.get("model_sha256") == spec.model.manifest.sha256
        and server.get("backbone_sha256") == spec.backbone.manifest.sha256
        and server.get("runtime_sha256") == spec.model_runtime.sha256
        and server.get("python", {}).get("version") == [3, 11]
        and server.get("socket_transport") == "same-kernel-unix-so_peercred",
        "Learned proof does not bind the real separate model runtime.",
    )
    encoded = base64.b64decode(server["runtime_descriptor_base64"], validate=True)
    require(digest(encoded) == spec.model_runtime.sha256, "Model-runtime descriptor was rebound.")
    runtime = parse_model_runtime(parse_json(encoded))
    report = parse_json(documents["probe.json"])
    if type(runtime) is CommandModelRuntime:
        validate_command_runtime_binding(runtime, spec, profile)
        proof = command_admission_proof(runtime, runtime_sha256=spec.model_runtime.sha256)
        require(
            server.get("admission") == report.get("model_admission") == proof,
            "The actual command provider/server admission differs from its original descriptor.",
        )
    else:
        require(
            "admission" not in server and "model_admission" not in report,
            "Legacy runtime descriptors cannot admit a command-v3 policy.",
        )
    acceptance = rescore(report, spec, scene, profile, grant)
    require(
        parse_json(documents["acceptance.json"]) == acceptance,
        "Native learned acceptance differs from independently recomputed task facts.",
    )
    manifest = parse_json(documents["raw-manifest.json"])
    receipt = report["capture"]["receipt"]
    require(
        digest(documents["raw-manifest.json"]) == receipt["manifest_sha256"]
        and manifest.get("purpose") == "evaluation"
        and manifest.get("scope")
        == {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id}
        and manifest.get("criteria_sha256") == spec.criteria_canonical_sha256
        and manifest.get("frozen_plan_sha256") == spec.conditions_canonical_sha256
        and digest(canonical(manifest["control_profile"])) == profile.sha256
        and len(manifest["episodes"]) == 1,
        "Learned raw manifest differs from its immutable task/profile/owner binding.",
    )
    episode = manifest["episodes"][0]
    require(
        episode["episode_id"] == str(spec.attempt_id)
        and episode["environment_id"] == grant.authorization.environment_id
        and episode["revision"] == grant.authorization.revision
        and episode["seed"] == scene.seed
        and episode["split"] == spec.evaluation_split
        and episode["provenance"]["code_revision"] == spec.source_revision
        and episode["provenance"]["simulator_image_digest"]
        == spec.platform.container_image.split("@", 1)[1]
        and episode["provenance"]["robot_asset_sha256"] == spec.asset_bundle.sha256
        and episode["demonstration"]["kind"] == "learned"
        and episode["demonstration"]["source_policy_sha256"] == spec.model.manifest.sha256
        and episode["frame_count"] * profile.hold_steps == acceptance["applied_action_count"],
        "Learned capture changed its actual model or physical episode.",
    )
    return receipt, acceptance


def verify_evidence(spec: BatchLearnedSpec, documents: dict[str, bytes]) -> dict:
    receipt, acceptance = rescore_evidence(spec, documents)
    require(acceptance["accepted"] is True, "The native learned physical trial was not accepted.")
    return receipt


class LearnedArtifacts(PrivateArtifacts):
    def validate_evidence(self, documents: dict[str, bytes]) -> dict:
        return verify_evidence(self.spec, documents)


def read_model_diagnostic(path: Path, spec: BatchLearnedSpec, grant) -> dict:
    from learning.contract import JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, Scope
    from learning.paused.command_model import (
        DIAGNOSTIC_PREFIX,
        DIAGNOSTIC_SCHEMA,
        MAX_DIAGNOSTIC_BYTES,
        diagnostic_context,
    )
    from learning.paused.contract import PausedControlContext
    from learning.paused.ipc import BINDINGS

    def unavailable(reason):
        return {
            "schema": DIAGNOSTIC_SCHEMA,
            "diagnostic_only": True,
            "kind": "unavailable",
            "reason": reason,
        }

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            start = max(0, stream.tell() - 1024**2)
            stream.seek(start)
            payload = stream.read(1024**2)
        lines = payload.splitlines()
        if start:
            lines = lines[1:]
        prefix = DIAGNOSTIC_PREFIX.encode("ascii")
        records = [line for line in lines if line.startswith(prefix)]
        if len(records) != 1:
            return unavailable("missing_or_ambiguous_bounded_log")
        require(len(records[0]) + 1 <= MAX_DIAGNOSTIC_BYTES, "Diagnostic exceeds its byte bound")
        value = parse_json(records[0][len(prefix) :])
        require(
            value.get("schema") == DIAGNOSTIC_SCHEMA and value.get("diagnostic_only") is True,
            "Unknown model diagnostic",
        )
        if value.get("kind") == "unavailable":
            return unavailable(token(value["reason"], "diagnostic reason"))
        require(
            set(value)
            == {
                "schema",
                "diagnostic_only",
                "kind",
                "message",
                "joint_guard",
                "request",
                "context",
                "inference",
            }
            and value["kind"] == "joint_guard_rejected",
            "Not the closed joint diagnostic",
        )
        request, context, guard, timing = (
            value[name] for name in ("request", "context", "joint_guard", "inference")
        )
        require(
            all(type(item) is dict for item in (request, context, guard, timing)),
            "Diagnostic fields must be objects",
        )
        checked_context = diagnostic_context(
            PausedControlContext(**{**context, "scope": Scope(**context["scope"])})
        )
        require(context == checked_context, "Unknown context fields")
        require(set(request) == BINDINGS, "Incomplete original request binding")
        for key, item in request.items():
            if key == "sequence":
                integer(item, key)
            elif key.endswith("_sha256"):
                sha256(item, key)
            else:
                token(item, key)
        require(
            context["scope"]
            == {"tenant_id": str(spec.platform.tenant_id), "owner_id": spec.owner_id}
            and context["command_id"] == context["episode_id"] == str(spec.attempt_id)
            and context["environment_id"] == grant.authorization.environment_id
            and context["revision"] == grant.authorization.revision
            and context["destination_id"] == grant.authorization.task.goal_id
            and request["task_sha256"] == digest(canonical(grant.authorization.task.model_dump()))
            and request["model_sha256"] == context["model_sha256"] == spec.model.manifest.sha256
            and request["control_profile_sha256"]
            == context["control_profile_sha256"]
            == spec.control_profile_sha256
            and request["context_sha256"] == digest(canonical(context))
            and request["observation_sha256"] == context["observation_sha256"]
            and request["freeze_id"] == context["freeze_id"],
            "Foreign command/model diagnostic",
        )
        require(
            set(guard)
            == {
                "label",
                "joint",
                "joint_index",
                "value",
                "targets",
                "low",
                "high",
                "tolerance",
                "horizon_index",
                "horizon_index_status",
            },
            "Unknown joint diagnostic fields",
        )
        index = integer(guard["joint_index"], "joint index", 0, 8)
        targets = vector(guard["targets"], 9, "original rejected targets")
        low, high = (finite(guard[key], key) for key in ("low", "high"))
        require(
            guard["label"] == "paused Smol action"
            and guard["joint"] == JOINT_NAMES[index]
            and finite(guard["value"], "original rejected value") == targets[index]
            and (low, high) == (JOINT_LOWER[index], JOINT_UPPER[index])
            and finite(guard["tolerance"], "original tolerance") == 1e-6
            and not low - 1e-6 <= targets[index] <= high + 1e-6
            and value["message"]
            == f"paused Smol action: {JOINT_NAMES[index]} outside reference limits",
            "Changed original joint rejection",
        )
        if guard["horizon_index_status"] == "unique_identity":
            integer(guard["horizon_index"], "original horizon index", 0, 49)
        else:
            require(
                guard["horizon_index_status"] == "unknown" and guard["horizon_index"] is None,
                "Guessed horizon index",
            )
        require(set(timing) == {"started_ns", "ended_ns", "latency_ms"}, "Unknown timing fields")
        started = integer(timing["started_ns"], "original inference start")
        ended = integer(timing["ended_ns"], "original inference end", started)
        require(
            finite(timing["latency_ms"], "original inference latency") == (ended - started) / 1e6,
            "Changed original inference latency",
        )
        return value
    except (KeyError, TypeError, ValueError):
        return unavailable("invalid_or_foreign_diagnostic")
    except OSError:
        return unavailable("log_read_unavailable")


def run_native(spec: BatchLearnedSpec, paths, directory: Path, deadline: float, *, store) -> dict:
    from learning.contract import Scope
    from learning.smolvla.artifacts import validate_backbone
    from simulation.batch_task import configure_native_environment, gpu_preflight, run_bounded
    from simulation.learned_probe import DIAGNOSTIC_SCHEMA, diagnostic_error, rescore

    configure_native_environment(spec)
    _, scene, profile, grant = load_inputs(spec, paths)
    preparation_deadline = min(
        deadline, time.monotonic() + (grant.expires_at - datetime.now(UTC)).total_seconds()
    )
    # Large immutable weights use the task's disk, not the qualified two-GiB /data tmpfs.
    staging = Path(os.environ["AZ_BATCH_TASK_WORKING_DIR"]) / "learned-inputs"
    require(staging.is_absolute() and not staging.exists(), "Never reuse partial model staging.")
    staging.mkdir(mode=0o700)
    descriptor_path = staging / "model-runtime.json"
    store.download(spec.model_runtime, descriptor_path)
    descriptor = read_model_runtime(descriptor_path, spec.model_runtime.sha256)
    python_proof = verify_model_runtime(descriptor, deadline=preparation_deadline, spec=spec)
    model_root, backbone_root = staging / "model", staging / "backbone"
    download_bundle(store, spec.model, model_root, model=True, deadline=preparation_deadline)
    download_bundle(store, spec.backbone, backbone_root, model=False, deadline=preparation_deadline)
    scope = Scope(str(spec.platform.tenant_id), spec.owner_id)
    metadata = validate_runtime_model(
        descriptor,
        model_root,
        scope=scope,
        model_sha256=spec.model.manifest.sha256,
    )
    require(
        metadata["backbone_manifest_sha256"] == spec.backbone.manifest.sha256
        and metadata["task"] == grant.authorization.task.model_dump()
        and metadata["control_profile"] == asdict(profile)
        and metadata["criteria_sha256"] == spec.criteria_canonical_sha256
        and metadata["frozen_plan_sha256"] == spec.conditions_canonical_sha256,
        "Actual model/backbone/task provenance differs before neural execution.",
    )
    validate_backbone(backbone_root, scope=scope, expected_sha256=spec.backbone.manifest.sha256)
    binding = staging / "binding.json"
    binding.write_bytes(
        canonical(
            {
                "scope": {"tenant_id": scope.tenant_id, "owner_id": scope.owner_id},
                "control_profile": metadata["control_profile"],
                "execution_timing": "paused_simulation",
                "real_time_admission": False,
                "criteria_sha256": spec.criteria_canonical_sha256,
                "frozen_plan_sha256": spec.conditions_canonical_sha256,
            }
        )
    )
    ipc = directory / "ipc"
    ipc.mkdir(mode=0o700)
    socket_path = ipc / "policy.sock"
    remaining(preparation_deadline)
    preflight = gpu_preflight()
    process = process_error = None
    with (
        (directory / "probe.log").open("xb") as log,
        model_process(
            model_command(
                model_root=model_root,
                backbone_root=backbone_root,
                binding=binding,
                socket_path=socket_path,
                model_sha256=spec.model.manifest.sha256,
                uid=os.geteuid(),
                runtime=descriptor,
            ),
            socket_path=socket_path,
            deadline=preparation_deadline,
            log=log,
        ) as server,
    ):
        load_inputs(spec, paths)
        preflight["model_process"] = {
            "model_sha256": spec.model.manifest.sha256,
            "backbone_sha256": spec.backbone.manifest.sha256,
            "runtime_sha256": spec.model_runtime.sha256,
            "runtime_descriptor_base64": base64.b64encode(descriptor_path.read_bytes()).decode(
                "ascii"
            ),
            "python": python_proof,
            "pid": server.pid,
            "uid": os.geteuid(),
            "socket_transport": "same-kernel-unix-so_peercred",
        }
        if type(descriptor) is CommandModelRuntime:
            preflight["model_process"]["admission"] = command_admission_proof(
                descriptor, runtime_sha256=spec.model_runtime.sha256
            )
        (directory / "preflight.json").write_bytes(canonical(preflight) + b"\n")
        spec_path = directory / "learned-spec.json"
        spec_path.write_bytes(canonical(spec.model_dump(mode="json", by_alias=True)))
        try:
            process = run_bounded(
                [
                    "/isaac-sim/python.sh",
                    "-m",
                    "simulation.learned_probe",
                    "--spec",
                    str(spec_path),
                    "--inputs",
                    str(directory / "inputs"),
                    "--model-root",
                    str(model_root),
                    "--socket-path",
                    str(socket_path),
                    "--model-runtime",
                    str(descriptor_path),
                    "--output",
                    str(directory / "probe.json"),
                ],
                cwd="/app",
                log=log,
                timeout=remaining(deadline),
            )
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            process_error = diagnostic_error("probe_process", exc)
        server_exit = server.poll()
    report_path = directory / "probe.json"
    report, read_error = {}, None
    try:
        if report_path.is_file():
            report = read_json(report_path, max_bytes=8 * 1024**2)
    except (ValueError, OSError) as exc:
        read_error = diagnostic_error("probe_report", exc)
    capture = report.get("capture") or {}
    receipt = capture.get("receipt") or {}
    passed = False
    verdict = {
        "accepted": False,
        "outcome": "failed",
        "native_acceptance": "rejected",
        "controller": "learned",
        "model_sha256": spec.model.manifest.sha256,
        "probe_exit_code": process.returncode if process is not None else None,
        "physical_status": report.get("physical_status"),
        "capture_status": capture.get("status"),
        "raw_manifest": receipt or None,
    }
    complete = False
    acceptance = {"accepted": False, "failure": "No complete native learned capture."}
    evidence_error = None
    process_failed = server_exit is not None or process_error is not None or read_error is not None
    if (
        not process_failed
        and capture.get("status") == "ready"
        and receipt.get("status") == "uploaded"
        and isinstance(report.get("trial"), dict)
        and isinstance(report.get("final_images"), dict)
    ):
        try:
            raw_root = Path("/data/demonstrations") / spec.owner_id / str(spec.attempt_id)
            manifest = raw_root / "manifest.json"
            require(
                file_digest(manifest) == receipt["manifest_sha256"], "Learned raw manifest changed."
            )
            (directory / "raw-manifest.json").write_bytes(manifest.read_bytes())
            acceptance = rescore(report, spec, scene, profile, grant)
            complete = True
            passed = (
                process is not None
                and process.returncode == 0
                and server_exit is None
                and acceptance["accepted"]
            )
        except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
            evidence_error = diagnostic_error("native_evidence", exc)
    publication_error = None
    if not process_failed and evidence_error is None:
        try:
            (directory / "acceptance.json").write_bytes(canonical(acceptance) + b"\n")
            (directory / "acceptance.log").write_text(json.dumps(acceptance) + "\n")
        except (ValueError, OSError) as exc:
            publication_error = diagnostic_error("acceptance_publication", exc)
            passed = False
    verdict.update(
        accepted=passed,
        outcome="accepted"
        if passed
        else "failed"
        if complete and publication_error is None
        else "incomplete",
        native_acceptance="accepted"
        if passed
        else "rejected"
        if complete and publication_error is None
        else "missing",
    )
    if not passed:
        diagnostics = {
            **(report.get("diagnostics") or {}),
            "schema": DIAGNOSTIC_SCHEMA,
            "diagnostic_only": True,
            "command_error": report.get("error"),
            "metrics": report.get("metrics"),
            "model_server_exit_code": server_exit,
            "secondary_errors": list((report.get("diagnostics") or {}).get("secondary_errors", [])),
            "model_diagnostic": read_model_diagnostic(directory / "probe.log", spec, grant),
        }
        diagnostics["secondary_errors"].extend(
            item
            for item in (read_error, process_error, evidence_error, publication_error)
            if item is not None
        )
        model_error = diagnostics["model_diagnostic"]
        command_error = report.get("error") or {}
        verdict["diagnostics"] = diagnostics
        verdict["failure"] = (
            model_error["message"]
            if model_error["kind"] == "joint_guard_rejected"
            else command_error.get("message")
            or report.get("failure")
            or (process_error or read_error or evidence_error or publication_error or {}).get(
                "message"
            )
            or (
                "The real model process exited during learned execution."
                if server_exit is not None
                else acceptance.get(
                    "failure", "The native learned physical trial was not accepted."
                )
            )
        )
    return verdict


def run_managed(spec: BatchLearnedSpec, path: Path, store, *, directory: Path):
    from functools import partial

    from simulation.batch_task import run_episode

    return run_episode(
        spec,
        path,
        store,
        directory=directory,
        runner=partial(run_native, store=store),
        evidence_validator=verify_evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "submit", "status", "run"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-url")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--confirm-submission", action="store_true")
    args = parser.parse_args()
    spec = BatchLearnedSpec.model_validate(read_json(args.spec, max_bytes=1024**2))
    if args.operation in {"plan", "submit"}:
        require(
            args.spec_url is not None,
            "The immutable private learned specification URL is required.",
        )
    if args.operation == "plan":
        job, task = build_learned_job_task(spec, args.spec_url, file_digest(args.spec))
        print(
            json.dumps(
                {
                    "job": job.as_dict(),
                    "task": task.as_dict(),
                    "after_task_created_patch": {"onAllTasksComplete": "terminatejob"},
                },
                indent=2,
            )
        )
        return
    if args.operation == "run":
        require(
            file_digest(args.spec) == args.spec_sha256, "Learned task specification hash changed."
        )
        require(
            os.environ.get("AZ_BATCH_JOB_ID") == spec.job_id
            and os.environ.get("AZ_BATCH_TASK_ID") == spec.task_id
            and os.environ.get("AZ_BATCH_POOL_ID") == spec.platform.pool_id,
            "Actual Batch task differs from the approved learned attempt.",
        )

        def cancelled(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, cancelled)
        from azure.identity import ManagedIdentityCredential

        with ManagedIdentityCredential(
            client_id=str(spec.platform.node_identity_client_id)
        ) as credential:
            with LearnedArtifacts(spec, credential) as store:
                result = run_managed(
                    spec,
                    args.spec,
                    store,
                    directory=Path("/data/managed-simulation") / str(spec.attempt_id),
                )
        print(json.dumps({name: result[name] for name in ("attempt_id", "outcome", "accepted")}))
        if result["accepted"] is not True:
            raise SystemExit(1)
        return
    from azure.batch import BatchClient
    from azure.identity import DefaultAzureCredential

    with DefaultAzureCredential(exclude_interactive_browser_credential=True) as credential:
        if args.operation == "submit":
            require(
                args.confirm_submission, "Explicit learned task submission approval is required."
            )
            check_regional_capacity(spec.platform, credential)
        with BatchClient(
            endpoint=spec.platform.account_url,
            credential=credential,
            credential_scopes=[BATCH_TOKEN_SCOPE],
            retry_total=0,
        ) as client:
            if args.operation == "submit":
                inspect_platform(client, spec.platform)
                result = submit_requests_once(
                    client, *build_learned_job_task(spec, args.spec_url, file_digest(args.spec))
                )
            else:
                with LearnedArtifacts(spec, credential) as store:
                    result = read_status(client, spec, store.read_completion)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
