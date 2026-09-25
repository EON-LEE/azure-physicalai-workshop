from __future__ import annotations

import base64
import binascii
import os
import socket
import stat
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
    require,
    sha256,
    token,
)
from learning.contract import CAMERAS, DemonstrationSource, Scope, bounded_joints
from learning.gr00t.artifacts import task_contract
from learning.gr00t.ipc import MAX_MESSAGE_BYTES, check_peer, receive_packet, remaining, send_packet
from learning.paused.contract import (
    EXECUTION_TIMING,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    FrozenCameraSample,
    FrozenPolicyObservation,
    InitialFrozenPublication,
    PausedControlContext,
    PausedControlProfile,
)

BINDINGS = frozenset(
    {
        "request_id",
        "sequence",
        "model_sha256",
        "control_profile_sha256",
        "task_sha256",
        "context_sha256",
        "observation_sha256",
        "freeze_id",
    }
)
MODE_KEYS = frozenset({"policy_type", "execution_timing", "real_time_admission"})


def _mode(value: dict) -> None:
    require(
        value["policy_type"] == "smolvla"
        and value["execution_timing"] == EXECUTION_TIMING
        and value["real_time_admission"] is False,
        "Paused IPC cannot claim real-time or another policy family",
    )


def make_request(
    observation: FrozenPolicyObservation,
    context: PausedControlContext,
    *,
    model_sha256: str,
    profile: PausedControlProfile,
    task: DemonstrationSource,
    request_id: str,
    sequence: int,
    now_ns: int | None = None,
) -> dict:
    context.validate(profile, observation, now_ns=time.monotonic_ns() if now_ns is None else now_ns)
    task.validate()
    require(
        context.model_sha256 == sha256(model_sha256) and context.destination_id == task.goal_id,
        "Paused model or approved task goal differs",
    )
    token(request_id, "request ID")
    integer(sequence, "sequence")
    raw = observation.metadata()
    for name, image in raw["images"].items():
        image["png_base64"] = base64.b64encode(observation.images[name].png).decode("ascii")
    value = {
        "schema": REQUEST_SCHEMA,
        "policy_type": "smolvla",
        "execution_timing": EXECUTION_TIMING,
        "real_time_admission": False,
        "request_id": request_id,
        "sequence": sequence,
        "model_sha256": model_sha256,
        "control_profile_sha256": profile.sha256,
        "task_sha256": digest(canonical(task_contract(task))),
        "context_sha256": context.sha256,
        "observation_sha256": observation.sha256,
        "freeze_id": observation.freeze_id,
        "context": asdict(context),
        "observation": raw,
    }
    require(len(canonical(value)) <= MAX_MESSAGE_BYTES, "Paused IPC request exceeds byte budget")
    return value


def validate_request(
    value: dict,
    *,
    model_sha256: str,
    scope: Scope,
    profile: PausedControlProfile,
    task: DemonstrationSource,
    now_ns: int | None = None,
) -> tuple[FrozenPolicyObservation, PausedControlContext]:
    keys(value, set(BINDINGS | MODE_KEYS | {"schema", "context", "observation"}), "paused request")
    _mode(value)
    scope.validate()
    profile.validate()
    task.validate()
    require(
        value["schema"] == REQUEST_SCHEMA
        and value["model_sha256"] == sha256(model_sha256)
        and value["control_profile_sha256"] == profile.sha256
        and value["task_sha256"] == digest(canonical(task_contract(task))),
        "Unbound paused model/profile/task request",
    )
    token(value["request_id"], "request ID")
    integer(value["sequence"], "sequence")
    raw = keys(
        value["observation"],
        set(FrozenPolicyObservation.__dataclass_fields__),
        "frozen observation",
    )
    ctx = keys(value["context"], set(PausedControlContext.__dataclass_fields__), "paused context")
    require(raw["scope"] == ctx["scope"] == asdict(scope), "Paused IPC owner scope differs")
    images = {}
    fields = set(FrozenCameraSample.__dataclass_fields__) - {"png"}
    for name, image in keys(raw["images"], set(CAMERAS), "camera set").items():
        keys(image, fields | {"sha256", "width", "height", "png_base64"}, "frozen camera")
        require(
            isinstance(image["png_base64"], str)
            and len(image["png_base64"]) <= MAX_MESSAGE_BYTES // 2,
            "Camera exceeds the IPC byte limit",
        )
        try:
            payload = base64.b64decode(image["png_base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ContractError("Invalid camera base64") from exc
        sample = FrozenCameraSample(png=payload, **{key: image[key] for key in fields})
        require(
            sample.metadata() == {key: item for key, item in image.items() if key != "png_base64"},
            "Camera bytes/shape/checksum or original metadata changed",
        )
        images[name] = sample
    proof = raw["initial_publication"]
    observation = FrozenPolicyObservation(
        **{
            key: item
            for key, item in raw.items()
            if key not in ("scope", "images", "initial_publication")
        },
        scope=scope,
        images=images,
        initial_publication=(
            InitialFrozenPublication(
                **keys(proof, set(InitialFrozenPublication.__dataclass_fields__), "publication")
            )
            if proof is not None
            else None
        ),
    )
    context = PausedControlContext(**{**ctx, "scope": scope})
    context.validate(profile, observation, now_ns=time.monotonic_ns() if now_ns is None else now_ns)
    require(
        context.model_sha256 == model_sha256
        and context.destination_id == task.goal_id
        and context.sha256 == value["context_sha256"]
        and observation.sha256 == value["observation_sha256"]
        and observation.freeze_id == value["freeze_id"],
        "Paused request content or frozen authority was rebound",
    )
    return observation, context


def make_response(value: dict, actions, *, inference_latency_ms: float) -> dict:
    require(value["schema"] == REQUEST_SCHEMA, "Wrong paused request schema")
    _mode(value)
    require(len(actions) == 50, "Wrong actual Smol horizon")
    targets = [list(bounded_joints(action, "paused Smol action")) for action in actions]
    latency = finite(inference_latency_ms, "actual model wall latency")
    require(0 <= latency <= 2000, "Paused model exceeded its fixed two-second wall budget")
    return {
        "schema": RESPONSE_SCHEMA,
        **{key: value[key] for key in BINDINGS | MODE_KEYS},
        "actions": targets,
        "inference_latency_ms": latency,
    }


def validate_response(value: dict, request: dict) -> tuple[tuple[float, ...], ...]:
    keys(
        value,
        set(BINDINGS | MODE_KEYS | {"schema", "actions", "inference_latency_ms"}),
        "paused response",
    )
    _mode(value)
    require(
        value["schema"] == RESPONSE_SCHEMA
        and all(value[key] == request[key] for key in BINDINGS | MODE_KEYS),
        "Swapped, replayed or real-time response",
    )
    checked = make_response(
        request, value["actions"], inference_latency_ms=value["inference_latency_ms"]
    )
    return tuple(tuple(action) for action in checked["actions"])


class SocketChunkPolicy:
    policy_type = "smolvla"
    execution_timing, real_time_admission = EXECUTION_TIMING, False
    chunk_size, n_action_steps, fps, physics_hz = 50, 1, 10, 60

    def __init__(
        self,
        socket_path: Path,
        *,
        scope: Scope,
        model_sha256: str,
        profile: PausedControlProfile,
        task: DemonstrationSource,
        expected_peer_uid: int | None = None,
    ) -> None:
        require(
            os.name == "posix"
            and socket_path.is_absolute()
            and not socket_path.is_symlink()
            and stat.S_ISSOCK(socket_path.stat().st_mode)
            and socket_path.stat().st_mode & 0o007 == 0,
            "Provision an actual protected Linux policy socket",
        )
        scope.validate()
        profile.validate()
        task.validate()
        sha256(model_sha256)
        self.socket_path, self.scope, self.model_sha256 = socket_path, scope, model_sha256
        self.profile, self.task = profile, task
        self.expected_peer_uid = os.geteuid() if expected_peer_uid is None else expected_peer_uid
        integer(self.expected_peer_uid, "approved peer UID", 0, 2**32 - 1)
        self.sequence = 0

    def reset(self) -> None:
        # Every request is stateless; sequence remains monotonic across approved resets.
        return None

    @property
    def predict_calls(self) -> int:
        return self.sequence

    def predict_chunk(self, observation: FrozenPolicyObservation, context: PausedControlContext):
        deadline = min(
            context.operation_deadline_ns, context.interval_deadline_ns, context.wall_deadline_ns
        )
        remaining(deadline)
        request = make_request(
            observation,
            context,
            model_sha256=self.model_sha256,
            profile=self.profile,
            task=self.task,
            request_id=str(uuid4()),
            sequence=self.sequence,
        )
        self.sequence += 1
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(remaining(deadline))
            connection.connect(str(self.socket_path))
            check_peer(connection, expected_uid=self.expected_peer_uid)
            send_packet(connection, request, deadline_ns=deadline)
            response = receive_packet(connection, deadline_ns=deadline)
        result = validate_response(response, request)
        remaining(deadline)
        return result


def serve(policy, socket_path: Path, *, allowed_client_uid: int | None = None) -> None:
    require(
        policy.policy_type == "smolvla"
        and policy.execution_timing == EXECUTION_TIMING
        and policy.real_time_admission is False,
        "Only an explicitly loaded paused Smol checkpoint may serve this protocol",
    )
    require(
        os.name == "posix"
        and socket_path.is_absolute()
        and not socket_path.exists()
        and not socket_path.is_symlink()
        and socket_path.parent.is_dir()
        and socket_path.parent.stat().st_mode & 0o007 == 0,
        "Provision a protected fresh Linux socket path",
    )
    expected_uid = os.geteuid() if allowed_client_uid is None else allowed_client_uid
    last_sequences: dict[tuple, int] = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o660)
        listener.listen(1)
        while True:
            connection, _ = listener.accept()
            with connection:
                receive_deadline = time.monotonic_ns() + 2_000_000_000
                check_peer(connection, expected_uid=expected_uid)
                request = receive_packet(connection, deadline_ns=receive_deadline)
                observation, context = validate_request(
                    request,
                    model_sha256=policy.model_sha256,
                    scope=policy.scope,
                    profile=policy.profile,
                    task=policy.task,
                )
                deadline = min(receive_deadline, context.operation_deadline_ns)
                remaining(deadline)
                key = (
                    context.environment_id,
                    context.revision,
                    context.episode_id,
                    context.epoch,
                    context.command_id,
                )
                require(
                    request["sequence"] > last_sequences.get(key, -1),
                    "Repeated/out-of-order paused request",
                )
                require(
                    key in last_sequences or len(last_sequences) < 10000,
                    "Session replay table is full",
                )
                last_sequences[key] = request["sequence"]
                policy.reset()
                started = time.monotonic_ns()
                actions = policy.predict_chunk(observation, context)
                ended = time.monotonic_ns()
                remaining(deadline)
                send_packet(
                    connection,
                    make_response(request, actions, inference_latency_ms=(ended - started) / 1e6),
                    deadline_ns=deadline,
                )
