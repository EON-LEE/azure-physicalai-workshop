"""Explicit command-v3 admission; legacy providers and the guarded actuator remain unchanged."""

from __future__ import annotations

import os
from pathlib import Path

from learning.common import require
from learning.contract import Scope
from learning.paused import PausedControlProfile
from simulation.batch_learned import (
    BatchLearnedSpec,
    CommandModelRuntime,
    validate_command_runtime_binding,
)
from simulation.paused_deployment import InstalledPausedPolicyProvider, PausedPolicyCatalogue
from simulation.runtime_configuration import _read_pinned


class CommandPausedPolicyProvider(InstalledPausedPolicyProvider):
    admission_kind = "azureml_command_v3"

    @classmethod
    def load(
        cls,
        path: Path,
        checksum: str,
        profile: PausedControlProfile,
        *,
        spec: BatchLearnedSpec,
        runtime: CommandModelRuntime,
    ):
        validate_command_runtime_binding(runtime, spec, profile)
        catalogue = PausedPolicyCatalogue.model_validate(_read_pinned(path, checksum))
        require(
            len(catalogue.policies) == 1, "One managed task requires exactly one command policy."
        )
        profile.validate()
        entry = catalogue.policies[0]
        grant, permit = entry.grant, entry.grant.authorization
        cls._check_runtime(grant, profile)
        require(
            permit.owner == spec.owner_id
            and str(grant.tenant_id) == str(spec.platform.tenant_id)
            and permit.model_sha256 == spec.model.manifest.sha256
            and permit.criteria_sha256 == spec.criteria_canonical_sha256
            and permit.frozen_plan_sha256 == spec.conditions_canonical_sha256
            and entry.expected_peer_uid == os.geteuid(),
            "The command policy grant or protected peer differs from the original managed attempt.",
        )
        model_root, socket_path = Path(entry.model_root), Path(entry.socket_path)
        require(
            model_root.is_absolute()
            and model_root.is_dir()
            and not model_root.is_symlink()
            and socket_path.is_absolute()
            and not socket_path.is_symlink(),
            "A pinned local command model and protected deployment socket are required.",
        )
        from learning.paused.command_artifacts import validate_model

        metadata = validate_model(
            model_root,
            expected_scope=Scope(str(grant.tenant_id), permit.owner),
            expected_model_sha256=permit.model_sha256,
        )
        require(
            metadata["schema"] == runtime.artifact_schema
            and metadata["training_execution"] == runtime.training_execution
            and PausedControlProfile(**metadata["control_profile"]) == profile
            and metadata["task"] == permit.task.model_dump()
            and metadata["criteria_sha256"] == permit.criteria_sha256
            and metadata["frozen_plan_sha256"] == permit.frozen_plan_sha256,
            "The actual command model/profile/task/criteria differs from its attested admission.",
        )
        return cls({permit.authorization_id: entry.model_copy(deep=True)}, profile)
