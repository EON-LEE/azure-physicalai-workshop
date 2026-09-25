"""Explicit installed learned-policy authority, separate from reference operator grants."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field

from apps.api.errors import Problem
from apps.api.models import Model, utcnow
from learning.common import file_digest, require
from learning.contract import DemonstrationSource, Scope
from learning.paused import PausedControlProfile
from learning.paused.artifacts import validate_model
from learning.paused.inference import PausedGuardedPolicyAdapter
from simulation.paused_configuration import (
    OperatorPausedAuthority,
    PausedOperatorGrant,
    paused_servo_sha256,
)
from simulation.runtime_configuration import _read_pinned


class InstalledPausedPolicy(Model):
    grant: PausedOperatorGrant
    model_root: str = Field(min_length=1, max_length=1024)
    socket_path: str = Field(min_length=1, max_length=512)
    expected_peer_uid: int = Field(ge=0, le=2**32 - 1, strict=True)


class PausedPolicyCatalogue(Model):
    schema_version: Literal["physicalai.paused-policy-catalog/v1"] = Field(alias="schema")
    policies: list[InstalledPausedPolicy] = Field(min_length=1, max_length=512)


class InstalledPausedPolicyProvider:
    def __init__(self, entries, profile):
        self.entries = MappingProxyType(dict(entries))
        self.profile = profile

    @staticmethod
    def _check_runtime(grant, profile):
        permit, now = grant.authorization, utcnow()
        require(
            grant.issued_at <= now < grant.expires_at
            and 0 < (grant.expires_at - grant.issued_at).total_seconds() <= 600
            and now < permit.wall_expires_at <= grant.expires_at,
            "The original installed paused grant is expired or exceeds its wall budget",
        )
        require(
            grant.source_revision == os.environ.get("SOURCE_REVISION")
            and grant.simulator_image_digest == os.environ.get("SIMULATOR_IMAGE", "").split("@")[-1]
            and str(grant.tenant_id) == os.environ.get("ENTRA_TENANT_ID")
            and permit.control_profile_sha256 == profile.sha256,
            "The installed paused authority differs from the actual runtime/profile",
        )
        require(
            permit.controller == "learned"
            and permit.policy_type == "smolvla"
            and permit.model_sha256 is not None
            and permit.authorization_kind in {"evaluation_grant", "policy_release"}
            and permit.purpose in {"evaluation", "integration"}
            and (permit.authorization_kind != "evaluation_grant" or permit.purpose == "evaluation"),
            "Learned authority requires an installed candidate evaluation or policy release",
        )

    @classmethod
    def load(cls, path: Path, checksum: str, profile: PausedControlProfile):
        catalogue = PausedPolicyCatalogue.model_validate(_read_pinned(path, checksum))
        profile.validate()
        entries = {}
        for entry in catalogue.policies:
            grant, permit = entry.grant, entry.grant.authorization
            cls._check_runtime(grant, profile)
            require(
                permit.authorization_id not in entries, "Duplicate installed paused authorization"
            )
            scope = Scope(str(grant.tenant_id), permit.owner)
            scope.validate()
            model_root, socket_path = Path(entry.model_root), Path(entry.socket_path)
            require(
                model_root.is_absolute()
                and model_root.is_dir()
                and not model_root.is_symlink()
                and socket_path.is_absolute()
                and not socket_path.is_symlink(),
                "An installed model and a deployment-local protected policy socket are required",
            )
            metadata = validate_model(
                model_root, expected_scope=scope, expected_model_sha256=permit.model_sha256
            )
            require(
                PausedControlProfile(**metadata["control_profile"]) == profile
                and metadata["task"] == permit.task.model_dump()
                and metadata["criteria_sha256"] == permit.criteria_sha256
                and metadata["frozen_plan_sha256"] == permit.frozen_plan_sha256,
                "Candidate profile, task, criteria or frozen conditions differ from authority",
            )
            entries[permit.authorization_id] = entry.model_copy(deep=True)
        return cls(entries, profile)

    def authorize(self, owner, request, profile):
        entry = self.entries.get(request.authorization_id)
        if entry is None or owner != entry.grant.authorization.owner:
            raise Problem(
                404, "paused_policy_missing", "No installed paused policy for this owner."
            )
        self._check_runtime(entry.grant, profile)
        require(profile == self.profile, "The installed paused policy profile changed")
        return OperatorPausedAuthority(entry.grant).authorize(owner, request, profile)

    def create(self, owner, request, profile, *, publication_guard):
        permission = self.authorize(owner, request, profile)
        entry = self.entries[permission.authorization_id]
        require(
            file_digest(Path(entry.model_root) / "model.json") == permission.model_sha256,
            "The installed model manifest checksum changed after admission",
        )
        from learning.paused.ipc import SocketChunkPolicy

        policy = SocketChunkPolicy(
            Path(entry.socket_path),
            scope=Scope(str(entry.grant.tenant_id), owner),
            model_sha256=permission.model_sha256,
            profile=profile,
            task=DemonstrationSource(
                "learned",
                permission.task.task_id,
                permission.task.instruction,
                permission.task.goal_id,
                permission.model_sha256,
            ),
            expected_peer_uid=entry.expected_peer_uid,
        )
        return PausedGuardedPolicyAdapter(
            policy, profile=profile, publication_guard=publication_guard
        )


def load_paused_policy_deployment():
    path = os.environ.get("PAUSED_POLICY_CATALOG_FILE")
    checksum = os.environ.get("PAUSED_POLICY_CATALOG_SHA256")
    if not path and not checksum:
        return None, None
    if not path or not checksum:
        raise ValueError("Paused learned deployment requires a pinned catalogue and checksum.")
    profile = PausedControlProfile(paused_servo_sha256())
    return profile, InstalledPausedPolicyProvider.load(Path(path), checksum, profile)
