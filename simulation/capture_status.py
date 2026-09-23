"""Owner-private terminal capture journal, separate from the validated raw dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field

from apps.api.models import Model
from learning.common import canonical, parse_json, require, safe_path, sha256
from simulation.runtime_contracts import CaptureBinding, CaptureStatus, CaptureUpdate


class CaptureJournal(Model):
    schema_version: Literal["physicalai.capture-status/v1"] = Field(
        default="physicalai.capture-status/v1", alias="schema"
    )
    binding: CaptureBinding
    state: CaptureStatus


class CaptureStatusStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, owner: str, command_id: UUID) -> Path:
        sha256(owner, "capture owner")
        require(isinstance(command_id, UUID), "Capture command must be a UUID")
        return safe_path(self.root, f"{owner}/{command_id}.json", must_exist=False)

    def _read(self, path: Path) -> CaptureJournal:
        require(path.stat().st_size <= 16384, "Oversized capture journal")
        return CaptureJournal.model_validate(parse_json(path.read_bytes()))

    def put(self, update: CaptureUpdate) -> None:
        binding, state = update.binding, update.state
        require(
            state.command_id == binding.command_id
            and state.capture_id == binding.capture_id
            and state.epoch == binding.epoch,
            "Capture journal binding mismatch",
        )
        require(state.status in {"ready", "invalid"}, "Only terminal capture status is durable")
        path = self._path(binding.owner, binding.command_id)
        entry = CaptureJournal(binding=binding, state=state)
        if path.exists():
            require(
                self._read(path) == entry, "Immutable capture journal binding or result changed"
            )
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(f".{binding.capture_id}.new")
        try:
            with temporary.open("xb") as stream:
                stream.write(canonical(entry.model_dump(mode="json", by_alias=True)) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(
                    self._read(path) == entry,
                    "Immutable capture journal binding or result changed",
                )
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, owner: str, command_id: UUID) -> CaptureStatus | None:
        path = self._path(owner, command_id)
        if not path.exists():
            return None
        entry = self._read(path)
        require(
            entry.binding.owner == owner
            and entry.binding.command_id == command_id
            and entry.state.command_id == command_id
            and entry.state.capture_id == entry.binding.capture_id
            and entry.state.epoch == entry.binding.epoch,
            "Capture journal binding mismatch",
        )
        return entry.state.model_copy(deep=True)
