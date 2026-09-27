"""Offline all-attempt pairing of immutable managed trials; never rewrite physical episode IDs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from apps.api.models import Model, Revision
from learning.common import (
    canonical,
    digest,
    file_digest,
    keys,
    parse_json,
    relative_path,
    require,
    safe_path,
    sha256,
    token,
    utc,
    write_json,
)
from learning.contract import Scope
from learning.paused.capture import validate_dataset
from learning.paused.evaluation import (
    compare_trials,
    expected_model,
    validate_models,
    validate_plan,
)
from learning.paused.rollout import derive_trial, validate_runtime
from learning.paused.task import TaskState
from simulation.batch import PROOF_LIMITS, validate_completion
from simulation.batch_learned import (
    BatchLearnedSpec,
    CommandModelRuntime,
    command_admission_proof,
    parse_model_runtime,
    rescore_evidence,
    validate_inputs,
)
from simulation.learned_probe import decode_image, physical_case

PRIVATE_LIMITS = {
    **PROOF_LIMITS,
    "claim.json": 1024**2,
    "completion.json": 1024**2,
    "capture/manifest.json": 4 * 1024**2,
}


def claim_time(value: str):
    require(isinstance(value, str), "Missing original claim timestamp.")
    # Batch emits +00:00; parse that spelling without changing the original artifact.
    return utc(value[:-6] + "Z" if value.endswith("+00:00") else value)


class Assignment(Model):
    logical_case_id: str
    role: Literal["before", "after"]
    physical_attempt_id: UUID

    @field_validator("logical_case_id")
    @classmethod
    def case_identifier(cls, value):
        return token(value, "logical case")


class PairingPlan(Model):
    schema_version: Literal["physicalai.managed-paired-plan/v1"] = Field(alias="schema")
    issued_at_utc: str
    evaluation_plan: dict
    runtime: dict
    model_runtime_sha256: Revision
    assignments: list[Assignment] = Field(min_length=40, max_length=40)

    @model_validator(mode="after")
    def exact_native_schedule(self):
        utc(self.issued_at_utc)
        validate_plan(self.evaluation_plan)
        require(
            self.evaluation_plan["comparison_kind"] == "paired_policy_eval",
            "Mapping requires two locked before/after models, not a reference bootstrap.",
        )
        validate_runtime(self.runtime, self.evaluation_plan)
        expected = [
            (case["episode_id"], role)
            for index, case in enumerate(self.evaluation_plan["cases"])
            for role in (("before", "after") if index % 2 == 0 else ("after", "before"))
        ]
        require(
            [(item.logical_case_id, item.role) for item in self.assignments] == expected
            and len({item.physical_attempt_id for item in self.assignments}) == 40,
            "Missing, reordered or duplicate frozen logical/physical assignments.",
        )
        return self


class FileProof(Model):
    sha256: Revision
    size_bytes: int = Field(alias="bytes", ge=0, le=64 * 1024**2, strict=True)


class AttemptFiles(Model):
    physical_attempt_id: UUID
    files: dict[str, FileProof] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def bounded_private_inventory(self):
        require(
            "inputs/spec.json" in self.files,
            "Every observed slot needs its original specification.",
        )
        for name, proof in self.files.items():
            relative_path(name)
            require(
                name in PROOF_LIMITS
                or name in {"claim.json", "completion.json", "capture/manifest.json"}
                or name.startswith(f"capture/episodes/{self.physical_attempt_id}/"),
                "Unapproved or foreign physical episode path in private evidence inventory.",
            )
            require(
                proof.size_bytes <= PRIVATE_LIMITS.get(name, 64 * 1024**2),
                "Private proof file exceeds its existing bound.",
            )
        require(
            sum(item.size_bytes for item in self.files.values()) <= 2 * 1024**3,
            "Private attempt evidence exceeds its two-GiB bound.",
        )
        return self


class PairingEvidence(Model):
    schema_version: Literal["physicalai.managed-paired-evidence/v1"] = Field(alias="schema")
    mapping_sha256: Revision
    snapshot_at_utc: str
    attempts: list[AttemptFiles] = Field(max_length=40)

    @model_validator(mode="after")
    def unique_attempts(self):
        utc(self.snapshot_at_utc)
        require(
            len({item.physical_attempt_id for item in self.attempts}) == len(self.attempts),
            "Duplicate physical attempts in the private evidence snapshot.",
        )
        return self


def summarize(
    mapping: PairingPlan,
    verified: list[dict],
    *,
    mapping_sha256: str,
    evidence_sha256: str,
    evidence_verified: bool = False,
) -> dict:
    """Pure scoring projection; only aggregate() establishes actual artifact verification."""
    sha256(mapping_sha256)
    sha256(evidence_sha256)
    planned = {str(item.physical_attempt_id): item for item in mapping.assignments}
    records = {}
    for item in verified:
        physical = item["physical_attempt_id"]
        require(
            physical in planned and physical not in records, "Unknown or duplicate actual attempt."
        )
        assignment = planned[physical]
        require(
            item["logical_case_id"] == assignment.logical_case_id
            and item["role"] == assignment.role,
            "Actual record was reassigned to another frozen case or model role.",
        )
        require(
            item["status"] in {"verified", "incomplete"}, "Unknown attempt verification status."
        )
        if item["status"] == "verified":
            require(
                item["physical_trial"]["episode_id"] == physical
                and item["physical_trial"]["policy"] == assignment.role,
                "The original physical trial identity cannot be rewritten.",
            )
        records[physical] = deepcopy(item)
    attempts = [
        records.get(
            str(item.physical_attempt_id),
            {
                "physical_attempt_id": str(item.physical_attempt_id),
                "logical_case_id": item.logical_case_id,
                "role": item.role,
                "status": "missing",
            },
        )
        for item in mapping.assignments
    ]
    missing = [item["physical_attempt_id"] for item in attempts if item["status"] != "verified"]
    result = {
        "schema": "physicalai.managed-paired-report/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "mapping_sha256": mapping_sha256,
        "evidence_sha256": evidence_sha256,
        "native_plan_sha256": digest(canonical(mapping.evaluation_plan)),
        "scope": mapping.evaluation_plan["scope"],
        "complete": not missing,
        "quality_gate_passed": False,
        "conclusion": "inconclusive",
        "expected_trial_count": 40,
        "verified_trial_count": 40 - len(missing),
        "attempts": attempts,
        "missing": missing,
    }
    if not missing:
        # IDs change only in this explicit numeric projection, never in raw files or stored trials.
        projected = [
            {**item["physical_trial"], "episode_id": item["logical_case_id"]} for item in attempts
        ]
        comparison = compare_trials(
            mapping.evaluation_plan, projected, live_gpu_verified=evidence_verified
        )
        result.update(
            quality_gate_passed=comparison["quality_gate_passed"],
            conclusion=comparison["conclusion"],
            comparison={
                key: value for key, value in comparison.items() if key not in {"schema", "trials"}
            },
            native_scoring_schema=comparison["schema"],
        )
    return result


def verify_files(root: Path, record: AttemptFiles) -> dict[str, bytes]:
    require(
        root.is_dir() and not root.is_symlink(), "Missing immutable attempt evidence directory."
    )
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlink in private attempt evidence.")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        require(len(actual) <= 4096, "Private attempt evidence has too many files.")
    require(actual == set(record.files), "Snapshot omitted or added private attempt artifacts.")
    documents = {}
    for name, proof in record.files.items():
        path = safe_path(root, name)
        require(path.stat().st_size == proof.size_bytes, "Private artifact byte count differs.")
        if name in PROOF_LIMITS or name in {"claim.json", "completion.json"}:
            if name.endswith(".json"):
                payload = path.read_bytes()
                require(digest(payload) == proof.sha256, "Private artifact checksum differs.")
                documents[name] = payload
                continue
        require(file_digest(path) == proof.sha256, "Private payload checksum differs.")
    return documents


def bind_attempt(
    mapping: PairingPlan,
    assignment: Assignment,
    spec: BatchLearnedSpec,
    models: dict,
    *,
    mapping_sha256: str,
) -> dict:
    plan, provenance = mapping.evaluation_plan, mapping.runtime["provenance"]
    case = next(case for case in plan["cases"] if case["episode_id"] == assignment.logical_case_id)
    require(
        spec.attempt_id == assignment.physical_attempt_id
        and spec.previous_attempt_id is None
        and spec.evaluation_split == "test"
        and spec.pairing_plan_sha256 == sha256(mapping_sha256)
        and spec.role == assignment.role
        and spec.model.manifest.sha256 == expected_model(plan, spec.role)
        and spec.owner_id == plan["scope"]["owner_id"]
        and str(spec.platform.tenant_id) == plan["scope"]["tenant_id"]
        and spec.control_profile_sha256 == plan["control_profile_sha256"]
        and spec.profile_id == plan["control_profile"]["profile_id"]
        and spec.criteria_canonical_sha256 == plan["criteria_sha256"]
        and spec.conditions_canonical_sha256 == plan["frozen_plan_sha256"]
        and spec.source_revision == provenance["code_revision"]
        and spec.platform.container_image.split("@", 1)[1] == provenance["simulator_image_digest"]
        and spec.asset_bundle.sha256 == provenance["robot_asset_sha256"]
        and spec.model_runtime.sha256 == mapping.model_runtime_sha256,
        "Attempt differs from its frozen physical ID, model, role, owner or runtime binding.",
    )
    model = models[assignment.role]
    require(
        {item.path: item.sha256 for item in spec.model.files}
        == {"checkpoint/" + name: value for name, value in model["checkpoint_files"].items()}
        and spec.backbone.manifest.sha256 == model["backbone_manifest_sha256"],
        "Attempt model/backbone inventory differs from the actual locked checkpoint.",
    )
    return case


def verify_attempt(
    root: Path,
    mapping: PairingPlan,
    assignment: Assignment,
    record: AttemptFiles,
    models: dict,
    *,
    snapshot_at_utc: str,
    mapping_sha256: str,
) -> dict:
    documents = verify_files(root, record)
    spec = BatchLearnedSpec.model_validate(parse_json(documents["inputs/spec.json"]))
    case = bind_attempt(mapping, assignment, spec, models, mapping_sha256=mapping_sha256)
    result = {
        "physical_attempt_id": str(spec.attempt_id),
        "logical_case_id": assignment.logical_case_id,
        "role": assignment.role,
        "status": "incomplete",
        "reason": "No native terminal proof.",
        "private_artifact_prefix": spec.artifact_prefix,
        "spec_sha256": spec.sha256,
        "files": record.model_dump(mode="json", by_alias=True)["files"],
    }
    claim = None
    if "claim.json" in documents:
        claim = parse_json(documents["claim.json"])
        expected = {
            "schema": "physicalai.batch-simulation-result/v1",
            "attempt_id": str(spec.attempt_id),
            "job_id": spec.job_id,
            "task_id": spec.task_id,
            "spec_sha256": spec.sha256,
            "spec_file_sha256": digest(documents["inputs/spec.json"]),
            "source_revision": spec.source_revision,
            "image": spec.platform.container_image,
            "control_profile_sha256": spec.control_profile_sha256,
            "profile_id": spec.profile_id,
            "previous_attempt_id": None,
            "physical_state_resume": False,
            "learning_quality_proven": False,
            "accepted": False,
            "outcome": "incomplete",
            "native_acceptance": "missing",
        }
        require(
            all(claim.get(name) == value for name, value in expected.items()),
            "The immutable pre-execution claim differs from its assigned attempt.",
        )
        require(
            utc(mapping.issued_at_utc)
            <= claim_time(claim["claimed_at_utc"])
            <= utc(snapshot_at_utc),
            "Assignments were not frozen before the physical attempt claim.",
        )
        result["claimed_at_utc"] = claim["claimed_at_utc"]
    if "completion.json" not in documents:
        return result
    require(claim is not None, "A terminal task cannot lack its original anti-reexecution claim.")
    completion = parse_json(documents["completion.json"])
    artifacts = validate_completion(spec, completion)
    require(
        completion["spec_file_sha256"] == digest(documents["inputs/spec.json"]),
        "Completion references a different private specification.",
    )
    for artifact in artifacts:
        proof = record.files.get(artifact.path)
        require(
            proof is not None
            and proof.sha256 == artifact.sha256
            and proof.size_bytes == artifact.size_bytes,
            "Private snapshot differs from the terminal artifact inventory.",
        )
    require(
        {item.path for item in artifacts} == set(record.files) & set(PROOF_LIMITS),
        "Unreceipted task artifacts were appended after completion.",
    )
    result.update(
        completion_sha256=digest(documents["completion.json"]),
        terminal_outcome=completion["outcome"],
        failure=completion.get("failure"),
    )
    required = {
        "inputs/environment.json",
        "inputs/grant.json",
        "inputs/criteria.json",
        "inputs/conditions.json",
        "probe.json",
        "preflight.json",
        "acceptance.json",
        "raw-manifest.json",
        "capture/manifest.json",
    }
    report = parse_json(documents["probe.json"]) if "probe.json" in documents else {}
    if not required <= set(record.files) or not {
        "trial",
        "heartbeat_ns",
        "episode_budget",
        "task_states",
        "final_images",
    } <= set(report):
        require(
            not completion["accepted"], "Accepted task lacks original native/raw/heartbeat proof."
        )
        result.update(
            reason="Native attempt is incomplete; no failed trial was fabricated.",
            native_failure={
                key: report[key]
                for key in ("phase", "failure_type", "failure", "error", "physical_status")
                if key in report
            },
        )
        return result
    _, scene, profile, grant = validate_inputs(
        spec,
        {
            name: documents[f"inputs/{name}.json"]
            for name in ("environment", "grant", "criteria", "conditions")
        },
        live=False,
    )
    require(
        utc(mapping.issued_at_utc) <= grant.issued_at
        and models[assignment.role]["task"] == grant.authorization.task.model_dump(),
        "The original model/task authority predates or differs from the frozen pairing.",
    )
    actual_case = physical_case(spec, scene, grant)
    require(
        {key: value for key, value in actual_case.items() if key != "episode_id"}
        == {key: value for key, value in case.items() if key != "episode_id"},
        "The actual physical seed, environment, pose or goal differs from the mapped logical case.",
    )
    receipt, acceptance = rescore_evidence(spec, documents)
    require(
        receipt == completion.get("raw_manifest")
        and acceptance["accepted"] == completion["accepted"],
        "Terminal success/failure differs from independently rescored physical evidence.",
    )
    expected_uri = (
        f"{spec.storage_account_url}/{spec.output_container}/"
        f"{spec.owner_id}/{spec.attempt_id}/manifest.json"
    )
    require(receipt["manifest_uri"] == expected_uri, "Raw capture URI has a foreign physical ID.")
    raw = validate_dataset(
        safe_path(root, "capture", must_exist=False),
        expected_scope=Scope(**mapping.evaluation_plan["scope"]),
        require_live=True,
        expected_manifest_sha256=receipt["manifest_sha256"],
    )
    require(
        raw.manifest_sha256 == digest(documents["raw-manifest.json"])
        and len(raw.episodes) == 1
        and raw.episodes[0].metadata["provenance"] == mapping.runtime["provenance"]
        and raw.episodes[0].metadata["budget"] == report["episode_budget"]
        and all(
            image["width"] == image["height"] == 320
            for episode in raw.episodes
            for frame in episode.frames
            for image in frame["images"].values()
        ),
        "Full private raw payload differs from its original runtime, budget or camera proof.",
    )
    states = [
        TaskState(**keys(value, set(TaskState.__dataclass_fields__), "original task state"))
        for value in report["task_states"]
    ]
    images = {name: decode_image(value) for name, value in report["final_images"].items()}
    require(
        claim_time(claim["claimed_at_utc"]) <= utc(states[0].captured_at_utc)
        and max(utc(image.captured_at_utc) for image in images.values()) <= utc(snapshot_at_utc),
        "Native measurements precede the claim or exceed the private evidence snapshot.",
    )
    failure = (report.get("error") or {}).get("message")
    observed = derive_trial(
        raw,
        states,
        case=actual_case,
        role=spec.role,
        model_sha256=spec.model.manifest.sha256,
        profile=profile,
        final_images=images,
        heartbeat_ns=tuple(report["heartbeat_ns"]),
        destination_id=grant.authorization.task.goal_id,
        failure_reason=failure[:512] if failure else None,
    )
    require(
        observed == report["trial"], "Declared trial differs from full raw/task/heartbeat rescore."
    )
    return {
        **result,
        "status": "verified",
        "reason": None,
        "physical_trial": observed,
        "physical_success": acceptance["accepted"],
        "started_at_utc": states[0].captured_at_utc,
        "ended_at_utc": max(
            images.values(), key=lambda image: utc(image.captured_at_utc)
        ).captured_at_utc,
    }


def aggregate(
    root: Path,
    mapping: PairingPlan,
    evidence: PairingEvidence,
    *,
    mapping_sha256: str,
    evidence_sha256: str,
    before_root: Path,
    after_root: Path,
    model_runtime: bytes | None = None,
) -> dict:
    require(
        evidence.mapping_sha256 == sha256(mapping_sha256)
        and utc(mapping.issued_at_utc) <= utc(evidence.snapshot_at_utc),
        "Private evidence references a different frozen mapping or earlier snapshot.",
    )
    assignment_by_id = {item.physical_attempt_id: item for item in mapping.assignments}
    require(
        {item.physical_attempt_id for item in evidence.attempts} <= set(assignment_by_id),
        "A foreign or retried physical attempt cannot replace a frozen slot.",
    )
    attempts_root = safe_path(root, "attempts", must_exist=False)
    present = set()
    if attempts_root.exists():
        require(attempts_root.is_dir(), "Expected a private attempt directory.")
        for path in attempts_root.iterdir():
            require(path.is_dir() and not path.is_symlink(), "Unlisted file/symlink in attempts.")
            present.add(path.name)
            require(len(present) <= 40, "Extra physical attempts cannot be discarded.")
    require(
        present == {str(item.physical_attempt_id) for item in evidence.attempts},
        "Snapshot omitted or introduced unlisted physical attempt directories.",
    )
    admission = None
    model_validator = validate_models
    if model_runtime is not None:
        require(
            len(model_runtime) <= 65536 and digest(model_runtime) == mapping.model_runtime_sha256,
            "Original paired model-runtime descriptor bytes differ.",
        )
        runtime = parse_model_runtime(parse_json(model_runtime))
        if type(runtime) is CommandModelRuntime:
            require(
                runtime.control_profile_sha256 == mapping.evaluation_plan["control_profile_sha256"]
                and runtime.legacy_servo_sha256
                == mapping.evaluation_plan["control_profile"]["servo_profile_sha256"]
                and runtime.simulator_image.split("@", 1)[1]
                == mapping.runtime["provenance"]["simulator_image_digest"]
                and runtime.simulator_source_revision
                == mapping.runtime["provenance"]["code_revision"],
                "Paired command admission differs from the frozen runtime/control/image binding.",
            )
            from learning.paused.command_artifacts import validate_models as command_models

            model_validator = command_models
            admission = command_admission_proof(
                runtime, runtime_sha256=mapping.model_runtime_sha256
            )
    models = model_validator(
        mapping.evaluation_plan,
        {"before": before_root, "after": after_root},
        scope=Scope(**mapping.evaluation_plan["scope"]),
    )
    physical_ids = {str(item.physical_attempt_id) for item in mapping.assignments}
    require(
        all(
            episode["episode_id"] not in physical_ids
            for model in models.values()
            for episode in model["training"]["episodes"]
        ),
        "Physical capture IDs overlap model training data.",
    )
    records = []
    for record in evidence.attempts:
        records.append(
            verify_attempt(
                safe_path(root, f"attempts/{record.physical_attempt_id}", must_exist=False),
                mapping,
                assignment_by_id[record.physical_attempt_id],
                record,
                models,
                snapshot_at_utc=evidence.snapshot_at_utc,
                mapping_sha256=mapping_sha256,
            )
        )
    by_id = {item["physical_attempt_id"]: item for item in records}
    last_end = utc(mapping.issued_at_utc)
    for assignment in mapping.assignments:
        item = by_id.get(str(assignment.physical_attempt_id))
        if item is not None and "claimed_at_utc" in item:
            require(
                claim_time(item["claimed_at_utc"]) >= last_end,
                "Frozen paired attempt order changed.",
            )
            last_end = claim_time(item.get("ended_at_utc", item["claimed_at_utc"]))
    result = summarize(
        mapping,
        records,
        mapping_sha256=mapping_sha256,
        evidence_sha256=evidence_sha256,
        evidence_verified=True,
    )
    if admission is not None:
        result["model_admission"] = admission
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--model-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.mapping.stat().st_size <= 4 * 1024**2, "Oversized frozen mapping.")
    require(args.evidence.stat().st_size <= 32 * 1024**2, "Oversized evidence snapshot.")
    mapping_bytes, evidence_bytes = args.mapping.read_bytes(), args.evidence.read_bytes()
    require(
        digest(mapping_bytes) == sha256(args.mapping_sha256)
        and digest(evidence_bytes) == sha256(args.evidence_sha256),
        "The approved mapping/evidence bytes changed.",
    )
    require(
        not any(
            args.output.resolve().is_relative_to(path.resolve())
            for path in (args.root / "attempts", args.before_root, args.after_root)
        ),
        "The aggregate output must not modify immutable attempt or model directories.",
    )
    mapping = PairingPlan.model_validate(parse_json(mapping_bytes))
    evidence = PairingEvidence.model_validate(parse_json(evidence_bytes))
    runtime_bytes = None
    if args.model_runtime is not None:
        runtime_path = safe_path(args.model_runtime.parent, args.model_runtime.name)
        require(runtime_path.stat().st_size <= 65536, "Oversized model-runtime descriptor.")
        runtime_bytes = runtime_path.read_bytes()
    result = aggregate(
        args.root,
        mapping,
        evidence,
        mapping_sha256=args.mapping_sha256,
        evidence_sha256=args.evidence_sha256,
        before_root=args.before_root,
        after_root=args.after_root,
        model_runtime=runtime_bytes,
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": result["complete"],
                "quality_gate_passed": result["quality_gate_passed"],
                "conclusion": result["conclusion"],
            }
        )
    )


if __name__ == "__main__":
    main()
