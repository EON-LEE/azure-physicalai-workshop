"""Curated anonymous viewing, never anonymous identity or control."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

from apps.api.errors import Problem
from apps.api.models import Observation, Principal, utcnow
from apps.api.service import FactoryService, check_fresh
from apps.api.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "examples" / "inspection-cell.json").read_text(encoding="utf-8"))
EVIDENCE = json.loads(
    Path(__file__).with_name("public_demo_evidence.json").read_text(encoding="utf-8")
)


def is_reference_scene(document: dict) -> bool:
    actual = {
        key: value
        for key, value in document.items()
        if key not in {"environment_id", "display_name"}
    }
    expected = {
        key: value
        for key, value in REFERENCE.items()
        if key not in {"environment_id", "display_name"}
    }
    return actual == expected


class PublicDemo:
    def __init__(self, settings: Settings, service: FactoryService) -> None:
        self.settings = settings
        self.service = service
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._snapshot_at = 0.0
        self._frames: dict[str, tuple[float, bytes, Observation]] = {}

    def _publication(self) -> Principal:
        if not self.settings.public_demo_publish_live or not all(
            (
                self.settings.public_demo_owner_id,
                self.settings.public_demo_environment_id,
                self.settings.public_demo_revision,
            )
        ):
            raise Problem(503, "live_not_published", "Live viewing has not been published.")
        actor = Principal(
            tenant_id=self.settings.entra_tenant_id,
            object_id=self.settings.public_demo_owner_id,
        )
        record = self.service.environment(actor, self.settings.public_demo_environment_id).value
        if record.revision != self.settings.public_demo_revision or not is_reference_scene(
            record.document
        ):
            raise Problem(503, "publication_changed", "The approved reference publication changed.")
        return actor

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if self._snapshot is not None and now - self._snapshot_at < 2:
                return deepcopy(self._snapshot)
            simulation = {
                "status": "not_published",
                "live_available": False,
                "message_code": "live_not_published",
                "frame_url": None,
            }
            if self.settings.public_demo_publish_live:
                try:
                    actor = self._publication()
                    status = self.service.runtime(actor)
                    matches = (
                        status.environment_id == self.settings.public_demo_environment_id
                        and status.revision == self.settings.public_demo_revision
                    )
                    if matches and status.status == "ready":
                        simulation = {
                            "status": "ready",
                            "live_available": True,
                            "message_code": "live_ready",
                            "frame_url": "/api/demo/frame",
                        }
                    else:
                        simulation = {
                            "status": "loading"
                            if matches and status.status == "loading"
                            else "unavailable",
                            "live_available": False,
                            "message_code": "live_unavailable",
                            "frame_url": None,
                        }
                except Problem:
                    simulation = {
                        "status": "unavailable",
                        "live_available": False,
                        "message_code": "live_unavailable",
                        "frame_url": None,
                    }
            verified_at = self.service.agent_probe_verified_at
            self._snapshot = {
                "api_version": "public-demo-v1",
                "access": "public_read_only",
                "deployment": "azure",
                "mode": "live" if simulation["live_available"] else "reference",
                "observed_at": utcnow().isoformat(),
                "scene": {
                    "id": "inspection-cell-v1",
                    "name": "Inspection and sorting cell",
                    "length_unit": "m",
                    "stations": deepcopy(REFERENCE["stations"]),
                    "robot": "Franka reference arm",
                    "data_origin": "synthetic_reference_configuration",
                },
                "simulation": simulation,
                "agent": {
                    "provider": "microsoft_foundry",
                    "connectivity": "verified" if verified_at is not None else "configured",
                    "verified_at": verified_at.isoformat() if verified_at is not None else None,
                    "verification_scope": "connectivity_only",
                },
                "learning": deepcopy(EVIDENCE["learning_cpu"]),
                "capabilities": {
                    "anonymous_control": False,
                    "anonymous_editing": False,
                    "public_live_video": bool(simulation["live_available"]),
                },
            }
            self._snapshot_at = time.monotonic()
            return deepcopy(self._snapshot)

    def frame(self, camera: Literal["overview", "inspection"]) -> tuple[bytes, Observation]:
        with self._lock:
            try:
                actor = self._publication()
                cached = self._frames.get(camera)
                if cached is not None and time.monotonic() - cached[0] < 0.5:
                    check_fresh(cached[2], REFERENCE["execution"]["max_observation_age_ms"])
                    return cached[1], cached[2].model_copy(deep=True)
                image, observation = self.service.frame(
                    actor,
                    self.settings.public_demo_environment_id,
                    self.settings.public_demo_revision,
                    camera,
                )
            except Problem as exc:
                self._frames.pop(camera, None)
                raise Problem(
                    503,
                    "public_live_unavailable",
                    "The published live camera is unavailable. No substitute frame was returned.",
                ) from exc
            self._frames[camera] = (time.monotonic(), image, observation.model_copy(deep=True))
            return image, observation
