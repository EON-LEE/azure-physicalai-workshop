from __future__ import annotations

import argparse
import os
import socket
import time
from io import BytesIO
from pathlib import Path

from learning.common import ContractError, keys, require
from learning.contract import ControlProfile, DemonstrationSource, Scope, bounded_joints, vector
from learning.gr00t import ACTION_HORIZON
from learning.gr00t.artifacts import validate_model
from learning.gr00t.franka_modality import FrankaDataConfig
from learning.gr00t.ipc import make_response, receive_packet, send_packet, validate_request
from learning.gr00t.source import activate_source
from learning.inference import PolicyObservation


def physical_actions(result: dict) -> tuple[tuple[float, ...], ...]:
    keys(result, {"action.arm", "action.fingers"}, "GR00T physical action")
    arrays = {
        name: value.tolist() if hasattr(value, "tolist") else value
        for name, value in result.items()
    }
    require(
        all(isinstance(value, list) and len(value) == ACTION_HORIZON for value in arrays.values()),
        "GR00T must return the complete sixteen-step physical action chunk",
    )
    return tuple(
        bounded_joints(
            (*vector(arm, 7, "arm radians"), *vector(fingers, 2, "individual finger metres")),
            "GR00T physical joint target",
        )
        for arm, fingers in zip(arrays["action.arm"], arrays["action.fingers"], strict=True)
    )


class LocalGr00tPolicy:
    fps, physics_hz, chunk_size, n_action_steps = 10, 60, ACTION_HORIZON, 1

    def __init__(
        self,
        model_root: Path,
        *,
        source_root: Path,
        scope: Scope,
        model_sha256: str,
        expected_control_profile_sha256: str,
    ) -> None:
        self.metadata = validate_model(
            model_root,
            expected_scope=scope,
            expected_model_sha256=model_sha256,
        )
        self.profile = ControlProfile(**self.metadata["control_profile"])
        require(self.profile.sha256 == expected_control_profile_sha256, "Unapproved servo profile")
        self.task = DemonstrationSource(kind="human_teleop", **self.metadata["task"])
        self.scope, self.model_sha256 = scope, model_sha256
        activate_source(source_root)
        import torch
        from gr00t.model.policy import Gr00tPolicy

        require(torch.cuda.is_available(), "GR00T deployment requires an actual CUDA GPU")
        configuration = FrankaDataConfig()
        self.backend = Gr00tPolicy(
            model_path=str((model_root / "checkpoint").resolve()),
            embodiment_tag="new_embodiment",
            modality_config=configuration.modality_config(),
            modality_transform=configuration.transform(),
            denoising_steps=4,
            device="cuda",
        )

    def reset(self) -> None:
        # N1.5 get_action is stateless; the runtime adapter owns and invalidates action queues.
        return None

    def predict_chunk(self, observation: PolicyObservation):
        import numpy as np
        from PIL import Image

        require(observation.scope == self.scope, "Policy observation owner mismatch")
        inputs = {
            "state.arm": np.asarray([observation.joint_positions[:7]], dtype=np.float32),
            "state.fingers": np.asarray([observation.joint_positions[7:]], dtype=np.float32),
            "annotation.human.task_description": [self.task.instruction],
        }
        for camera in ("inspection", "overview"):
            with Image.open(BytesIO(observation.images[camera].png)) as decoded:
                pixels = np.array(
                    decoded.convert("RGB").resize(
                        (224, 224),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.uint8,
                )
            inputs[f"video.{camera}"] = pixels[None, ...]
        return physical_actions(self.backend.get_action(inputs))


def serve(policy: LocalGr00tPolicy, socket_path: Path) -> None:
    require(os.name == "posix" and socket_path.is_absolute(), "Policy IPC requires Linux AF_UNIX")
    require(
        not socket_path.exists() and not socket_path.is_symlink(),
        "Never replace an existing socket",
    )
    require(socket_path.parent.is_dir(), "Provision a protected policy socket directory first")
    require(socket_path.parent.stat().st_mode & 0o007 == 0, "Socket directory is world-accessible")
    last_sequences: dict[tuple, int] = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o660)
        listener.listen(1)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(0.08)
                request = receive_packet(connection)
                observation, context = validate_request(
                    request,
                    model_sha256=policy.model_sha256,
                    scope=policy.scope,
                    profile=policy.profile,
                    task=policy.task,
                )
                start = time.monotonic_ns()
                require(
                    0 <= start - observation.monotonic_ns <= 200_000_000
                    and start < context.deadline_monotonic_ns,
                    "Stale/future/expired inference request",
                )
                for camera in observation.images.values():
                    require(
                        0 <= observation.physics_step - camera.physics_step < 6
                        and 0 <= start - camera.monotonic_ns <= 200_000_000
                        and camera.monotonic_ns <= observation.monotonic_ns,
                        "Stale/future inference camera",
                    )
                binding = context.binding()
                require(
                    request["sequence"] > last_sequences.get(binding, -1),
                    "Replayed policy request",
                )
                require(
                    binding in last_sequences or len(last_sequences) < 4096,
                    "IPC run capacity reached",
                )
                last_sequences[binding] = request["sequence"]
                actions = policy.predict_chunk(observation)
                end = time.monotonic_ns()
                require(
                    end < context.deadline_monotonic_ns, "Command expired during GR00T inference"
                )
                send_packet(
                    connection,
                    make_response(
                        request,
                        actions,
                        inference_latency_ms=(end - start) / 1e6,
                    ),
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment-only private GR00T process; never an HTTP API."
    )
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--control-profile-sha256", required=True)
    args = parser.parse_args()
    try:
        policy = LocalGr00tPolicy(
            args.model_root,
            source_root=args.source_root,
            scope=Scope(args.tenant_id, args.owner_id),
            model_sha256=args.model_sha256,
            expected_control_profile_sha256=args.control_profile_sha256,
        )
        serve(policy, args.socket_path)
    except (ContractError, OSError) as exc:
        raise SystemExit(f"GR00T policy process stopped: {exc}") from exc


if __name__ == "__main__":
    main()
