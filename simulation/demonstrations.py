"""Bound actual simulator samples to an approved, owner-scoped demonstration."""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from apps.api.models import EnvironmentRecord
from learning.capture import EpisodeWriter
from learning.contract import (
    ControlProfile,
    DemonstrationSource,
    EpisodeSpec,
    FrameSample,
    Provenance,
    Scope,
    validate_dataset,
)
from simulation.core import SimulationCore
from simulation.extensions import SceneSpec
from simulation.runtime_contracts import CaptureReceipt


@dataclass(frozen=True)
class DemonstrationRequest:
    owner: str
    environment: EnvironmentRecord
    spec: SceneSpec
    command_id: UUID
    builder_path: Path
    control_profile: ControlProfile | None = None
    demonstration: DemonstrationSource | None = None


class Demonstration:
    @staticmethod
    def prepare(core: SimulationCore, command_id: UUID) -> DemonstrationRequest:
        with core.lock:
            if (
                core.active_command is None
                or core.active_command[1] != command_id
                or core.commands[core.active_command].status != "running"
                or core.spec is None
                or not core.spec.record_demonstration
                or core.environment is None
            ):
                raise ValueError(
                    "Capture requires the active, explicitly approved simulator command."
                )
            owner = core.active_command[0]
            environment = core.environment.model_copy(deep=True)
            spec = core.spec
            builder = core.registry.builders[environment.document["scene"]["template_id"]]
            teaching = core.teaching_by_command.get(core.active_command)
            policy = core.policy_commands.get(core.active_command)
            profile, demonstration = None, None
            if teaching is not None:
                request = core.teaching_sessions[teaching].request
                profile = core.control_profile
                demonstration = DemonstrationSource(
                    request.demonstrator_kind,
                    request.task.task_id,
                    request.task.instruction,
                    request.task.goal_id,
                )
            elif policy is not None:
                profile = core.control_profile
                demonstration = DemonstrationSource(
                    "learned",
                    policy.task.task_id,
                    policy.task.instruction,
                    policy.task.goal_id,
                    policy.model_sha256,
                )
            return DemonstrationRequest(
                owner,
                environment,
                spec,
                command_id,
                Path(inspect.getfile(type(builder))),
                profile,
                demonstration,
            )

    def __init__(
        self,
        core: SimulationCore | DemonstrationRequest,
        command_id: UUID | None = None,
        root: Path | None = None,
    ) -> None:
        if isinstance(core, SimulationCore):
            if command_id is None:
                raise ValueError("An approved capture command is required.")
            request = self.prepare(core, command_id)
        else:
            request = core
        owner, environment, spec = request.owner, request.environment, request.spec
        command_id = request.command_id
        if os.environ.get("CAPTURE_ENABLED") != "true":
            raise ValueError("The deployment has not enabled real demonstration capture.")
        required = (
            "SIMULATOR_IMAGE",
            "FRANKA_ASSET_SHA256",
            "SOURCE_REVISION",
            "ENTRA_TENANT_ID",
            "STORAGE_ACCOUNT_URL",
            "AZURE_CLIENT_ID",
        )
        if any(not os.environ.get(name) for name in required):
            raise ValueError(
                "Trusted deployment provenance and Azure identity are required for capture."
            )
        try:
            gpu = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()
            )
        except subprocess.SubprocessError as exc:
            raise ValueError("Actual GPU identity could not be verified for capture.") from exc
        if len(gpu) != 1:
            raise ValueError("Reference capture requires exactly one identified GPU.")
        image = os.environ["SIMULATOR_IMAGE"]
        if "@sha256:" not in image:
            raise ValueError("Capture requires the actual digest-pinned simulator image.")
        provenance = Provenance(
            source_kind="isaac_sim",
            simulator_version=os.environ.get("ISAAC_SIM_VERSION", "5.1.0"),
            simulator_image_digest=image.split("@", 1)[1],
            robot_asset_sha256=os.environ["FRANKA_ASSET_SHA256"],
            scene_builder_id=environment.document["scene"]["template_id"],
            scene_builder_sha256=hashlib.sha256(request.builder_path.read_bytes()).hexdigest(),
            code_revision=os.environ["SOURCE_REVISION"],
            capture_host="azure_gpu",
            gpu_model=gpu[0],
        )
        self.scope = Scope(os.environ["ENTRA_TENANT_ID"], owner)
        self.episode_id = str(command_id)
        self.root = (root or Path("/data/demonstrations")) / owner / self.episode_id
        self.writer = EpisodeWriter(
            self.root,
            dataset_id=f"capture-{command_id}",
            scope=self.scope,
            episode=EpisodeSpec(
                self.episode_id,
                environment.environment_id,
                environment.revision,
                spec.seed,
                spec.demonstration_split,
            ),
            provenance=provenance,
            fps=request.control_profile.control_hz if request.control_profile is not None else 60,
            physics_hz=60,
            max_frames=3002 if request.control_profile is not None else 18002,
            max_bytes=512 * 1024 * 1024,
            control_profile=request.control_profile,
            demonstration=request.demonstration,
        )

    def append(self, sample: FrameSample) -> None:
        self.writer.append(sample)

    def finalize_and_upload(self, on_uploading: Callable[[], None] | None = None) -> CaptureReceipt:
        manifest = self.writer.finalize()
        validated = validate_dataset(self.root, expected_scope=self.scope, require_live=True)
        if on_uploading is not None:
            on_uploading()
        prefix = f"{self.scope.owner_id}/{self.episode_id}/"
        account = os.environ["STORAGE_ACCOUNT_URL"].rstrip("/")
        container = os.environ.get("DEMONSTRATION_CONTAINER", "demonstrations")
        paths = []
        for episode in validated.episodes:
            paths.append(episode.metadata["path"])
            for frame in episode.frames:
                paths.extend(value["path"] for value in frame["images"].values())
        with (
            ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"]) as credential,
            BlobServiceClient(
                account_url=account,
                credential=credential,
                connection_timeout=5,
                read_timeout=15,
                retry_total=1,
            ) as blobs,
        ):
            target = blobs.get_container_client(container)
            for relative in sorted(set(paths)):
                if on_uploading is not None:
                    on_uploading()
                with (self.root / relative).open("rb") as stream:
                    target.upload_blob(
                        prefix + relative,
                        stream,
                        overwrite=False,
                        content_settings=ContentSettings(
                            content_type="image/png"
                            if relative.endswith(".png")
                            else "application/x-ndjson"
                        ),
                    )
            if on_uploading is not None:
                on_uploading()
            with manifest.open("rb") as stream:
                target.upload_blob(
                    prefix + "manifest.json",
                    stream,
                    overwrite=False,
                    content_settings=ContentSettings(content_type="application/json"),
                )
        return CaptureReceipt(
            manifest_uri=f"{account}/{container}/{prefix}manifest.json",
            manifest_sha256=validated.manifest_sha256,
            episode_id=self.episode_id,
            frame_count=self.writer.count,
        )
