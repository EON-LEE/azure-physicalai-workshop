"""Only deployment-pinned timing evidence and release catalogues enable new controls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field

from apps.api.errors import Problem
from apps.api.models import Identifier, Model, Revision
from learning.common import canonical, digest, file_digest, parse_json, require, sha256
from learning.contract import ControlProfile, DemonstrationSource, Scope
from simulation.policy_executor import PolicyPort
from simulation.runtime_contracts import PolicyCommand, PolicyType, TaskDefinition

COMMERCIALLY_APPROVED_POLICY_FAMILIES = frozenset({"smolvla"})


def _require_approved_policy_family(policy_type: PolicyType) -> None:
    require(
        policy_type in COMMERCIALLY_APPROVED_POLICY_FAMILIES,
        "This policy family has no commercial license approval; production loading is disabled",
    )


def servo_profile_sha256() -> str:
    return digest(
        canonical(
            {
                name: file_digest(Path(__file__).with_name(name))
                for name in (
                    "isaac_adapter.py",
                    "camera_observation.py",
                    "physics_scheduling.py",
                    "control.py",
                    "motion.py",
                    "policy_executor.py",
                    "teaching.py",
                )
            }
        )
    )


class TimingEvidence(Model):
    schema_version: Literal["physicalai.control-timing/v1"] = Field(alias="schema")
    evidence_kind: Literal["azure_isaac_gpu"]
    simulator_image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    simulator_version: Literal["6.0.0"]
    robot_asset_sha256: Revision
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    control_profile: ControlProfile
    model_sha256: Revision | None
    policy_type: PolicyType | None
    control_intervals: int = Field(ge=100, le=100000, strict=True)
    camera_pairs: int = Field(ge=100, le=100000, strict=True)
    max_control_cycle_ms: float = Field(gt=0, lt=100)
    max_inference_latency_ms: float = Field(ge=0, le=80)
    max_heartbeat_gap_ms: float = Field(gt=0, le=2000)
    trace_sha256: Revision
    gpu_model: str = Field(min_length=1, max_length=128)


def _read_pinned(path: Path, expected_sha256: str) -> dict:
    sha256(expected_sha256, "deployment configuration checksum")
    require(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size <= 1024 * 1024,
        "Deployment configuration must be a bounded nonsymlink local file",
    )
    data = path.read_bytes()
    require(digest(data) == expected_sha256, "Deployment configuration checksum mismatch")
    return parse_json(data)


def load_control_profile(
    path: Path,
    expected_sha256: str,
    *,
    expected_model_sha256: str | None = None,
    expected_policy_type: PolicyType | None = None,
) -> ControlProfile:
    evidence = TimingEvidence.model_validate(_read_pinned(path, expected_sha256))
    evidence.control_profile.validate()
    image = os.environ.get("SIMULATOR_IMAGE", "").split("@")[-1]
    require(
        evidence.simulator_image_digest == image
        and evidence.robot_asset_sha256 == os.environ.get("FRANKA_ASSET_SHA256")
        and evidence.source_revision == os.environ.get("SOURCE_REVISION")
        and evidence.simulator_version == os.environ.get("ISAAC_SIM_VERSION"),
        "Timing evidence belongs to a different deployed simulator, asset or source revision",
    )
    require(
        evidence.control_profile.servo_profile_sha256 == servo_profile_sha256(),
        "Timing evidence belongs to a different reviewed servo implementation",
    )
    require(evidence.camera_pairs == evidence.control_intervals, "Missing aligned camera intervals")
    if expected_model_sha256 is not None:
        require(
            evidence.model_sha256 == expected_model_sha256
            and (expected_policy_type is None or evidence.policy_type == expected_policy_type),
            "The policy model has no matching measured end-to-end timing evidence",
        )
    return evidence.control_profile


class EnvironmentCase(Model):
    environment_id: Identifier
    revision: Revision


class DeployedRelease(Model):
    policy_release_id: UUID
    policy_type: PolicyType
    scope: Scope
    model_sha256: Revision
    control_profile_sha256: Revision
    task: TaskDefinition
    environment_cases: list[EnvironmentCase] = Field(min_length=1, max_length=1000)
    socket_path: str = Field(min_length=1, max_length=512)
    expected_peer_uid: int = Field(ge=0, strict=True)
    timing_evidence_file: str = Field(min_length=1, max_length=1024)
    timing_evidence_sha256: Revision


class PolicyCatalogue(Model):
    schema_version: Literal["physicalai.policy-catalog/v1"] = Field(alias="schema")
    releases: list[DeployedRelease] = Field(min_length=1, max_length=512)


class DeploymentPolicyProvider:
    def __init__(self, entries: dict[UUID, DeployedRelease]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path, expected_sha256: str, profile: ControlProfile):
        catalogue = PolicyCatalogue.model_validate(_read_pinned(path, expected_sha256))
        entries = {}
        for release in catalogue.releases:
            release.scope.validate()
            require(
                release.policy_release_id not in entries,
                "Duplicate immutable policy release in deployment catalogue",
            )
            require(
                Path(release.socket_path).is_absolute(), "Policy socket must be deployment-local"
            )
            measured = load_control_profile(
                Path(release.timing_evidence_file),
                release.timing_evidence_sha256,
                expected_model_sha256=release.model_sha256,
                expected_policy_type=release.policy_type,
            )
            require(
                measured == profile and release.control_profile_sha256 == profile.sha256,
                "Released model uses a different control profile",
            )
            entries[release.policy_release_id] = release
        return cls(entries)

    def authorize(self, owner: str, request: PolicyCommand, profile: ControlProfile) -> None:
        release = self.entries.get(request.policy_release_id)
        if release is None or release.scope.owner_id != owner:
            raise Problem(404, "policy_missing", "No deployment-approved policy for this owner.")
        if (
            request.model_sha256 != release.model_sha256
            or request.policy_type != release.policy_type
            or request.control_profile_id != profile.profile_id
            or release.control_profile_sha256 != profile.sha256
            or request.task != release.task
            or not any(
                case.environment_id == request.command.environment_id
                and case.revision == request.command.revision
                for case in release.environment_cases
            )
        ):
            raise Problem(
                409,
                "policy_binding_changed",
                "Policy model, task, profile or approved case changed.",
            )

    def create(self, owner: str, request: PolicyCommand, profile: ControlProfile) -> PolicyPort:
        self.authorize(owner, request, profile)
        _require_approved_policy_family(request.policy_type)
        try:
            from learning.smolvla.ipc import RemoteGuardedPolicyAdapter, SocketChunkPolicy
        except ModuleNotFoundError as exc:
            raise RuntimeError("The verified SmolVLA policy adapter is not installed.") from exc

        release = self.entries[request.policy_release_id]
        require(
            hasattr(SocketChunkPolicy, "predict_calls"),
            "The deployment lacks the deadline/peer-checked counted policy IPC adapter",
        )
        policy = SocketChunkPolicy(
            Path(release.socket_path),
            scope=release.scope,
            model_sha256=release.model_sha256,
            profile=profile,
            task=DemonstrationSource(
                "learned",
                request.task.task_id,
                request.task.instruction,
                request.task.goal_id,
                request.model_sha256,
            ),
            expected_peer_uid=release.expected_peer_uid,
        )
        return RemoteGuardedPolicyAdapter(policy)


def load_deployment() -> tuple[ControlProfile | None, DeploymentPolicyProvider | None]:
    profile_file = os.environ.get("CONTROL_TIMING_FILE")
    profile_sha = os.environ.get("CONTROL_TIMING_SHA256")
    catalogue_file = os.environ.get("POLICY_CATALOG_FILE")
    catalogue_sha = os.environ.get("POLICY_CATALOG_SHA256")
    if not any((profile_file, profile_sha, catalogue_file, catalogue_sha)):
        return None, None
    if not profile_file or not profile_sha:
        raise ValueError("Enabling control requires a pinned measured timing file and checksum.")
    profile = load_control_profile(Path(profile_file), profile_sha)
    if not catalogue_file and not catalogue_sha:
        return profile, None
    if not catalogue_file or not catalogue_sha:
        raise ValueError("Enabling learned policies requires a pinned catalogue and checksum.")
    provider = DeploymentPolicyProvider.load(Path(catalogue_file), catalogue_sha, profile)
    for release in provider.entries.values():
        _require_approved_policy_family(release.policy_type)
    return profile, provider
