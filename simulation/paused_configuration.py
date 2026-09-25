"""Pinned operator authority and a distinct complete non-real-time source fingerprint."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field

from apps.api.errors import Problem
from apps.api.models import EnvironmentRecord, Model, utcnow
from learning.common import canonical, digest, file_digest, read_json, require
from learning.paused import (
    CONTROL_PROFILE_ID,
    CONTROL_PROFILE_V2_ID,
    PausedControlProfile,
    protocol_schemas,
)
from simulation.extensions import SceneRegistry
from simulation.paused_contracts import ResolvedSimulationAuthorization, SimulationEpisodeCommand
from simulation.paused_profiles import paused_profile
from simulation.runtime_configuration import _read_pinned


def paused_servo_sha256() -> str:
    base = Path(__file__).parent
    names = (
        "paused_configuration.py",
        "paused_control.py",
        "paused_worker.py",
        "paused_teacher.py",
        "reference_targets.py",
        "paused_gripper_servo.py",
        "paused_runtime.py",
        "paused_learned.py",
        "paused_deployment.py",
        "paused_profiles.py",
        "paused_capture.py",
        "paused_observation.py",
        "paused_contracts.py",
        "isaac_adapter.py",
        "core.py",
        "capture_worker.py",
        "demonstrations.py",
        "camera_observation.py",
        "physics_scheduling.py",
        "control.py",
        "motion.py",
        "run_isaac.py",
        "paused_probe.py",
        "paused_acceptance.py",
    )
    native = base.parent / "learning" / "paused"
    limits = asdict(PausedControlProfile("0" * 64))
    limits.pop("servo_profile_sha256")
    return digest(
        canonical(
            {
                "schema": "physicalai.paused-servo-fingerprint/v1",
                "runtime_files": {name: file_digest(base / name) for name in names},
                "native_contract_files": {
                    name: file_digest(native / name)
                    for name in (
                        "__init__.py",
                        "contract.py",
                        "capture.py",
                        "inference.py",
                        "artifacts.py",
                        "ipc.py",
                        "model.py",
                    )
                },
                "shared_policy_files": {
                    name: file_digest(base.parent / name)
                    for name in (
                        "learning/common.py",
                        "learning/contract.py",
                        "learning/gr00t/ipc.py",
                        "learning/gr00t/artifacts.py",
                        "learning/smolvla/__init__.py",
                        "learning/smolvla/artifacts.py",
                        "learning/smolvla/inference.py",
                        "learning/smolvla/adaptation.py",
                    )
                },
                "protocol_schemas": protocol_schemas(),
                "frozen_profile_limits": limits,
                "supported_profile_limits": {
                    profile_id: {
                        key: value
                        for key, value in asdict(paused_profile("0" * 64, profile_id)).items()
                        if key != "servo_profile_sha256"
                    }
                    for profile_id in (CONTROL_PROFILE_ID, CONTROL_PROFILE_V2_ID)
                },
            }
        )
    )


class PausedOperatorGrant(Model):
    schema_version: Literal["physicalai.paused-operator-grant/v1"] = Field(alias="schema")
    execution_timing: Literal["paused_simulation"]
    real_time_admission: Literal[False]
    tenant_id: UUID
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    simulator_image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    authorization: ResolvedSimulationAuthorization


class OperatorPausedAuthority:
    def __init__(self, grant: PausedOperatorGrant) -> None:
        self.grant = grant

    @classmethod
    def load(
        cls,
        path: Path,
        expected_sha256: str,
        *,
        environment: EnvironmentRecord,
        profile: PausedControlProfile,
        criteria: Path,
        criteria_sha256: str,
        conditions: Path,
        conditions_sha256: str,
    ):
        grant = PausedOperatorGrant.model_validate(_read_pinned(path, expected_sha256))
        now = utcnow()
        require(
            grant.execution_timing == "paused_simulation"
            and grant.real_time_admission is False
            and grant.issued_at <= now < grant.expires_at
            and 0 < (grant.expires_at - grant.issued_at).total_seconds() <= 600,
            "The original paused operator grant is inactive or exceeds its wall budget",
        )
        require(
            grant.source_revision == os.environ.get("SOURCE_REVISION")
            and grant.simulator_image_digest == os.environ.get("SIMULATOR_IMAGE", "").split("@")[-1]
            and str(grant.tenant_id) == os.environ.get("ENTRA_TENANT_ID"),
            "Operator authority does not match the actual runtime source, image or tenant",
        )
        profile.validate()
        spec = SceneRegistry(load_installed=False).build(environment)
        scene_authority = spec.require_paused_authority()
        permit = grant.authorization
        require(
            permit.controller == "reference_controller"
            and permit.authorization_kind == "reference_collection"
            and permit.model_sha256 is None
            and permit.policy_type is None,
            "This operator entry accepts reference only; a real model provider is required",
        )
        require(
            permit.environment_id == environment.environment_id
            and permit.revision == environment.revision
            and permit.control_profile_sha256 == profile.sha256
            and permit.profile_id == profile.profile_id == scene_authority.profile_id
            and permit.wall_expires_at <= grant.expires_at
            and permit.criteria_sha256 == criteria_sha256
            and permit.frozen_plan_sha256 == conditions_sha256,
            "The original operator record does not bind the exact paused case/profile",
        )
        scene_authority.validate_request(
            wall_seconds=min(
                permit.max_episode_wall_seconds, (permit.wall_expires_at - now).total_seconds()
            ),
            simulation_steps=permit.max_simulation_steps,
        )
        criteria_value, plan = read_json(criteria), read_json(conditions)
        require(
            digest(canonical(criteria_value)) == criteria_sha256,
            "Frozen criteria checksum mismatch",
        )
        require(digest(canonical(plan)) == conditions_sha256, "Frozen scene plan checksum mismatch")
        for document in (criteria_value, plan):
            require(
                document.get("execution_timing") == "paused_simulation"
                and document.get("real_time_admission") is False
                and document.get("task") == permit.task.model_dump(),
                "Frozen criteria/plan changed execution mode or approved task",
            )
        require(
            plan.get("criteria_canonical_sha256") == criteria_sha256,
            "Frozen conditions reference different criteria",
        )
        cases = [
            case
            for case in plan.get("cases", [])
            if case.get("environment_id") == environment.environment_id
            and case.get("revision") == environment.revision
        ]
        require(len(cases) == 1, "The saved case is not uniquely authorized in the frozen plan")
        case = cases[0]
        require(
            case["seed"] == spec.seed
            and case["split"] == spec.demonstration_split
            and case["scene_builder_sha256"] == spec.scene_builder_sha256
            and tuple(case["expected_initial_position_m"]) == spec.part_position
            and tuple(case["goal_position_m"]) == spec.station(permit.task.goal_id).position,
            "The actual reviewed scene differs from the original frozen case",
        )
        return cls(grant)

    def authorize(
        self, owner: str, request: SimulationEpisodeCommand, profile: PausedControlProfile
    ):
        permit = self.grant.authorization
        now = utcnow()
        if not self.grant.issued_at <= now < self.grant.expires_at:
            raise Problem(409, "paused_grant_expired", "The original operator grant expired.")
        if (
            owner != permit.owner
            or request.authorization_id != permit.authorization_id
            or request.authorization_kind != permit.authorization_kind
            or request.controller != permit.controller
            or request.environment_id != permit.environment_id
            or request.revision != permit.revision
            or request.task != permit.task
            or profile.sha256 != permit.control_profile_sha256
            or request.profile_id != permit.profile_id
            or profile.profile_id != permit.profile_id
            or request.policy_type != permit.policy_type
            or request.model_sha256 != permit.model_sha256
            or request.wall_expires_at > permit.wall_expires_at
            or request.max_simulation_steps > permit.max_simulation_steps
        ):
            raise Problem(
                403,
                "paused_grant_mismatch",
                "The command does not match the pinned operator grant.",
            )
        return permit
