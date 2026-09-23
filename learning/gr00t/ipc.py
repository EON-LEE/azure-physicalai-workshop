from __future__ import annotations

import base64
import binascii
import socket
import stat
import struct
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from learning.common import (
    ContractError,
    canonical,
    digest,
    finite,
    integer,
    keys,
    parse_json,
    require,
    sha256,
    token,
    utc,
)
from learning.contract import (
    CAMERAS,
    CameraSample,
    ControlProfile,
    DemonstrationSource,
    Scope,
    bounded_joints,
    png_dimensions,
)
from learning.gr00t import ACTION_HORIZON, POLICY_TYPE
from learning.gr00t.artifacts import UPSTREAM, task_contract
from learning.inference import ControlContext, GuardedPolicyAdapter, PolicyObservation, SafetyLimits

REQUEST_SCHEMA = "physicalai.gr00t-request/v1"
RESPONSE_SCHEMA = "physicalai.gr00t-response/v1"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
BINDINGS = {
    "request_id",
    "sequence",
    "model_sha256",
    "control_profile_sha256",
    "task_sha256",
    "context_sha256",
    "observation_sha256",
}


def make_request(
    observation: PolicyObservation,
    context: ControlContext,
    *,
    model_sha256: str,
    profile: ControlProfile,
    task: DemonstrationSource,
    request_id: str,
    sequence: int,
) -> dict:
    token(request_id, "request ID")
    integer(sequence, "IPC sequence")
    sha256(model_sha256, "checkpoint")
    raw = asdict(observation)
    raw["images"] = {
        name: {
            "png_base64": base64.b64encode(image.png).decode("ascii"),
            "sha256": digest(image.png),
            "rendering_frame": image.rendering_frame,
            "physics_step": image.physics_step,
            "monotonic_ns": image.monotonic_ns,
        }
        for name, image in observation.images.items()
    }
    result = {
        "schema": REQUEST_SCHEMA,
        "policy_type": POLICY_TYPE,
        "request_id": request_id,
        "sequence": sequence,
        "model_sha256": model_sha256,
        "control_profile_sha256": profile.sha256,
        "task_sha256": digest(canonical(task_contract(task))),
        "context_sha256": digest(canonical(asdict(context))),
        "observation_sha256": digest(canonical(raw)),
        "context": asdict(context),
        "observation": raw,
    }
    require(len(canonical(result)) <= MAX_MESSAGE_BYTES, "Policy request exceeds IPC budget")
    return result


def validate_request(
    request: dict,
    *,
    model_sha256: str,
    scope: Scope,
    profile: ControlProfile,
    task: DemonstrationSource,
) -> tuple[PolicyObservation, ControlContext]:
    keys(request, BINDINGS | {"schema", "policy_type", "context", "observation"}, "policy request")
    require(
        request["schema"] == REQUEST_SCHEMA
        and request["policy_type"] == POLICY_TYPE
        and request["model_sha256"] == model_sha256
        and request["control_profile_sha256"] == profile.sha256
        and request["task_sha256"] == digest(canonical(task_contract(task))),
        "Unbound model/task/control-profile request",
    )
    token(request["request_id"], "request ID")
    integer(request["sequence"], "IPC sequence")
    context = keys(request["context"], set(ControlContext.__dataclass_fields__), "control context")
    raw = keys(request["observation"], set(PolicyObservation.__dataclass_fields__), "observation")
    require(context["scope"] == raw["scope"] == asdict(scope), "IPC tenant/owner scope mismatch")
    require(
        context["approved"] is True
        and context["active"] is True
        and context["destination_id"] == task.goal_id,
        "IPC request is not approved for this task goal",
    )
    require(
        request["context_sha256"] == digest(canonical(context))
        and request["observation_sha256"] == digest(canonical(raw)),
        "IPC request content checksum mismatch",
    )
    for name in ("environment_id", "revision", "episode_id"):
        require(raw[name] == context[name], "IPC observation/command binding mismatch")
    sha256(raw["revision"])
    for name in ("environment_id", "episode_id", "epoch", "command_id", "destination_id"):
        token(context[name], name)
    integer(context["deadline_monotonic_ns"], "deadline", 1)
    integer(raw["monotonic_ns"], "observation monotonic timestamp", 1)
    integer(raw["physics_step"], "physics step")
    utc(raw["captured_at_utc"])
    joints = bounded_joints(raw["joint_positions"], "observed joints")
    images = {}
    for camera, image in keys(raw["images"], set(CAMERAS), "policy cameras").items():
        keys(
            image,
            {"png_base64", "sha256", "rendering_frame", "physics_step", "monotonic_ns"},
            "camera",
        )
        require(isinstance(image["png_base64"], str), "Image must be bounded base64")
        require(len(image["png_base64"]) <= MAX_MESSAGE_BYTES // 2, "Camera exceeds IPC budget")
        try:
            png = base64.b64decode(image["png_base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ContractError("Invalid policy PNG encoding") from exc
        require(digest(png) == sha256(image["sha256"]), "Policy camera checksum mismatch")
        png_dimensions(png)
        images[camera] = CameraSample(
            png,
            integer(image["rendering_frame"], "render frame"),
            integer(image["physics_step"], "camera physics step"),
            integer(image["monotonic_ns"], "camera timestamp", 1),
        )
    return (
        PolicyObservation(**{**raw, "scope": scope, "images": images, "joint_positions": joints}),
        ControlContext(**{**context, "scope": scope}),
    )


def make_response(request: dict, actions, *, inference_latency_ms: float) -> dict:
    require(len(actions) == ACTION_HORIZON, "Wrong GR00T chunk length")
    chunk = [list(bounded_joints(action, "GR00T action")) for action in actions]
    latency = finite(inference_latency_ms, "inference latency")
    require(0 <= latency <= 80, "GR00T inference exceeded approved 80ms budget")
    return {
        "schema": RESPONSE_SCHEMA,
        "policy_type": POLICY_TYPE,
        **{name: request[name] for name in BINDINGS},
        "actions": chunk,
        "inference_latency_ms": latency,
    }


def validate_response(response: dict, request: dict) -> tuple[tuple[float, ...], ...]:
    keys(
        response,
        BINDINGS | {"schema", "policy_type", "actions", "inference_latency_ms"},
        "policy response",
    )
    require(
        response["schema"] == RESPONSE_SCHEMA
        and response["policy_type"] == POLICY_TYPE
        and all(response[name] == request[name] for name in BINDINGS),
        "Swapped/replayed policy response",
    )
    return tuple(
        tuple(action)
        for action in make_response(
            request,
            response["actions"],
            inference_latency_ms=response["inference_latency_ms"],
        )["actions"]
    )


def send_packet(connection: socket.socket, value: dict) -> None:
    data = canonical(value)
    require(0 < len(data) <= MAX_MESSAGE_BYTES, "IPC packet exceeds limit")
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive_packet(connection: socket.socket) -> dict:
    def receive(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = connection.recv(size - len(chunks))
            require(bool(part), "Policy IPC peer disconnected")
            chunks.extend(part)
        return bytes(chunks)

    size = struct.unpack("!I", receive(4))[0]
    require(0 < size <= MAX_MESSAGE_BYTES, "IPC packet exceeds limit")
    return parse_json(receive(size))


class SocketChunkPolicy:
    fps, physics_hz, chunk_size, n_action_steps = 10, 60, ACTION_HORIZON, 1

    def __init__(
        self,
        socket_path: Path,
        *,
        scope: Scope,
        model_sha256: str,
        profile: ControlProfile,
        task: DemonstrationSource,
    ) -> None:
        require(
            socket_path.is_absolute()
            and not socket_path.is_symlink()
            and stat.S_ISSOCK(socket_path.stat().st_mode),
            "Deployment must provision an actual nonsymlink Unix policy socket",
        )
        require(
            socket_path.stat().st_mode & 0o007 == 0, "Policy socket must not be world-accessible"
        )
        scope.validate()
        sha256(model_sha256)
        self.socket_path, self.scope, self.model_sha256 = socket_path, scope, model_sha256
        self.profile, self.task = profile, task
        self.metadata = {
            "policy_type": POLICY_TYPE,
            "model_sha256": model_sha256,
            "scope": asdict(scope),
            "control_profile": asdict(profile),
            "task": task_contract(task),
            "upstream": UPSTREAM,
        }
        self.context: ControlContext | None = None
        self.sequence = 0

    def reset(self) -> None:
        self.context = None

    def predict_chunk(self, observation: PolicyObservation):
        require(self.context is not None, "IPC policy has no approved control context")
        request = make_request(
            observation,
            self.context,
            model_sha256=self.model_sha256,
            profile=self.profile,
            task=self.task,
            request_id=str(uuid4()),
            sequence=self.sequence,
        )
        self.sequence += 1
        deadline = min(self.context.deadline_monotonic_ns, time.monotonic_ns() + 80_000_000)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(max(0.001, (deadline - time.monotonic_ns()) / 1e9))
            connection.connect(str(self.socket_path))
            send_packet(connection, request)
            response = receive_packet(connection)
        require(time.monotonic_ns() <= deadline, "Policy IPC control deadline exceeded")
        return validate_response(response, request)


class RemoteGuardedPolicyAdapter(GuardedPolicyAdapter):
    def __init__(self, policy: SocketChunkPolicy):
        super().__init__(
            policy,
            limits=SafetyLimits(
                max_observation_age_ms=200,
                max_inference_latency_ms=80,
                max_chunk_steps=1,
            ),
        )

    def step(self, observation: PolicyObservation, context: ControlContext):
        self.policy.context = context
        return super().step(observation, context)
