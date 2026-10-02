"""Owner-thread raw-v3 persistence for an explicitly authorized paused episode."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from learning.contract import DemonstrationSource, EpisodeSpec, FrameSample
from learning.paused import PausedControlProfile, PausedFrameSample
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter, validate_dataset
from simulation.core import SimulationCore
from simulation.demonstrations import (
    DemonstrationRequest,
    deployment_provenance,
    upload_validated_capture,
)
from simulation.paused_contracts import ResolvedSimulationAuthorization
from simulation.runtime_contracts import CaptureReceipt


@dataclass(frozen=True)
class PausedCaptureRequest:
    standard: DemonstrationRequest
    profile: PausedControlProfile
    budget: PausedEpisodeBudget
    authority: ResolvedSimulationAuthorization


def prepare_paused_capture(
    core: SimulationCore, command_id: UUID, budget: PausedEpisodeBudget
) -> PausedCaptureRequest:
    with core.lock:
        key = core.active_command
        if (
            key is None
            or key[1] != command_id
            or core.commands[key].status != "running"
            or key not in core.simulation_commands
            or core.paused_profile is None
            or core.spec is None
            or not core.spec.record_demonstration
        ):
            raise ValueError("Paused capture requires the active, separately approved episode.")
        request = core.simulation_commands[key]
        if (
            budget.started_ns != core.command_started_ns[key]
            or budget.wall_deadline_ns != core.monotonic_deadlines[key]
            or budget.simulation_step_deadline - budget.initial_physics_step
            != request.max_simulation_steps
        ):
            raise ValueError("Capture cannot renew the original episode wall/simulation authority.")
        budget.validate(core.paused_profile)
        source = DemonstrationSource(
            "learned" if request.controller == "learned" else "reference_controller",
            request.task.task_id,
            request.task.instruction,
            request.task.goal_id,
            request.model_sha256,
        )
        environment = core.environment.model_copy(deep=True)
        builder = core.registry.builders[environment.document["scene"]["template_id"]]
        standard = DemonstrationRequest(
            key[0],
            environment,
            core.spec,
            command_id,
            Path(inspect.getfile(type(builder))),
            demonstration=source,
        )
        return PausedCaptureRequest(
            standard,
            core.paused_profile,
            budget,
            core.simulation_authorizations[key].model_copy(deep=True),
        )


class PausedDemonstration:
    def __init__(self, request: PausedCaptureRequest, root: Path | None = None) -> None:
        standard = request.standard
        self.scope, provenance = deployment_provenance(standard)
        self.request = request
        self.episode_id = str(standard.command_id)
        self.root = (root or Path("/data/demonstrations")) / self.scope.owner_id / self.episode_id
        self.writer = PausedEpisodeWriter(
            self.root,
            dataset_id=f"paused-{standard.command_id}",
            scope=self.scope,
            episode=EpisodeSpec(
                self.episode_id,
                standard.environment.environment_id,
                standard.environment.revision,
                standard.spec.seed,
                standard.spec.demonstration_split,
            ),
            provenance=provenance,
            profile=request.profile,
            demonstration=standard.demonstration,
            budget=request.budget,
            purpose=request.authority.purpose,
            criteria_sha256=request.authority.criteria_sha256,
            frozen_plan_sha256=request.authority.frozen_plan_sha256,
        )

    def append(self, sample: FrameSample | PausedFrameSample) -> None:
        if not isinstance(sample, PausedFrameSample):
            raise ValueError("Paused capture cannot relabel a legacy frame as raw v3.")
        self.writer.append(sample)

    def finalize_and_upload(self, on_uploading: Callable[[], None]) -> CaptureReceipt:
        self.writer.finalize()
        validated = validate_dataset(self.root, expected_scope=self.scope, require_live=True)

        def publication_allowed() -> None:
            if time.monotonic_ns() >= self.request.budget.wall_deadline_ns:
                raise RuntimeError(
                    "The original paused episode wall budget expired before publication."
                )
            on_uploading()

        receipt = upload_validated_capture(
            self.root,
            validated,
            self.scope,
            self.episode_id,
            self.writer.count,
            publication_allowed,
        )
        publication_allowed()
        return receipt
