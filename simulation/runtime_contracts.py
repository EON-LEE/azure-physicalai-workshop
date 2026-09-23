"""CPU-safe bridge contracts; importing these never loads an actuator or a model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from apps.api.models import DemonstrationResult, Model

RuntimeMode = Literal["reference", "human_teaching", "learned"]
CapturePhase = Literal["recording", "finalizing", "uploading", "ready", "invalid"]


@dataclass(frozen=True)
class CommandBinding:
    owner: str
    environment_id: str
    revision: str
    epoch: UUID
    command_id: UUID


@dataclass(frozen=True)
class CaptureBinding(CommandBinding):
    capture_id: UUID


@dataclass(frozen=True)
class CaptureReceipt:
    manifest_uri: str
    manifest_sha256: str
    episode_id: str
    frame_count: int


class CaptureStatus(Model):
    capture_id: UUID
    command_id: UUID
    epoch: UUID
    status: CapturePhase
    receipt: DemonstrationResult | None = None
    message: str | None = None


@dataclass(frozen=True)
class CaptureUpdate:
    binding: CaptureBinding
    state: CaptureStatus
