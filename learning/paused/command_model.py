"""Explicitly attested command-v3 admission with unchanged paused prediction/IPC controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from learning.common import (
    ContractError,
    canonical,
    digest,
    finite,
    integer,
    read_json,
    require,
    sha256,
    token,
    vector,
)
from learning.contract import JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, Scope, bounded_joints
from learning.paused.command_artifacts import ADMISSION_KIND, MODEL_SCHEMA, validate_model
from learning.paused.contract import (
    CONTEXT_SCHEMA,
    EXECUTION_TIMING,
    FrozenPolicyObservation,
    PausedControlContext,
    PausedControlProfile,
)
from learning.paused.ipc import BINDINGS, make_response, serve
from learning.paused.model import LocalPausedSmolVLAPolicy

DIAGNOSTIC_SCHEMA = "physicalai.paused-command-diagnostic/v1"
DIAGNOSTIC_PREFIX = "PHYSICALAI_COMMAND_DIAGNOSTIC "
MAX_DIAGNOSTIC_BYTES = 16 * 1024


def _unavailable(reason: str) -> dict:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "diagnostic_only": True,
        "kind": "unavailable",
        "reason": reason,
    }


def diagnostic_context(context: PausedControlContext) -> dict:
    require(
        type(context) is PausedControlContext and type(context.scope) is Scope,
        "Missing original validated context",
    )
    context.scope.validate()
    fields = {"scope": {"tenant_id": context.scope.tenant_id, "owner_id": context.scope.owner_id}}
    for key in (
        "environment_id",
        "episode_id",
        "epoch",
        "command_id",
        "destination_id",
        "freeze_id",
    ):
        fields[key] = token(getattr(context, key), key)
    for key in ("revision", "model_sha256", "control_profile_sha256", "observation_sha256"):
        fields[key] = sha256(getattr(context, key), key)
    for key in (
        "state_revision",
        "episode_started_ns",
        "episode_initial_physics_step",
        "simulation_step_deadline",
        "wall_deadline_ns",
        "interval_started_ns",
        "interval_deadline_ns",
        "operation_started_ns",
        "operation_deadline_ns",
    ):
        fields[key] = integer(getattr(context, key), key)
    require(
        context.schema == CONTEXT_SCHEMA
        and context.execution_timing == EXECUTION_TIMING
        and context.approved is True
        and context.active is True
        and context.real_time_admission is False,
        "Not the original paused authority",
    )
    fields.update(
        schema=context.schema,
        execution_timing=context.execution_timing,
        real_time_admission=False,
        approved=True,
        active=True,
    )
    return fields


def _request_bindings(request, observation, context) -> tuple[dict, dict]:
    require(
        type(request) is dict and type(observation) is FrozenPolicyObservation,
        "Missing original validated request objects",
    )
    bindings = {}
    for key in sorted(BINDINGS):
        item = request[key]
        bindings[key] = (
            integer(item, key)
            if key == "sequence"
            else sha256(item, key)
            if key.endswith("_sha256")
            else token(item, key)
        )
    fields = diagnostic_context(context)
    require(
        bindings["context_sha256"] == context.sha256 == digest(canonical(fields))
        and bindings["observation_sha256"] == observation.sha256 == context.observation_sha256
        and bindings["freeze_id"] == observation.freeze_id == context.freeze_id
        and bindings["model_sha256"] == context.model_sha256
        and bindings["control_profile_sha256"] == context.control_profile_sha256
        and observation.scope == context.scope,
        "Original request bindings changed",
    )
    return bindings, fields


def guard_diagnostic(error: BaseException) -> dict:
    """Copy only allowlisted primitives from the original, exact pinned guard stack."""
    if type(error) is not ContractError:
        return _unavailable("not_joint_guard")
    codes = (serve.__code__, make_response.__code__, bounded_joints.__code__)
    frames = {}
    trace = error.__traceback__
    try:
        for _ in range(64):
            if trace is None:
                break
            for index, code in enumerate(codes):
                if trace.tb_frame.f_code is code:
                    if index in frames:
                        return _unavailable("ambiguous_traceback")
                    frames[index] = trace.tb_frame
            trace = trace.tb_next
        if trace is not None:
            return _unavailable("traceback_limit")
        if list(frames) != [0, 1, 2]:
            return _unavailable("not_joint_guard")
        server, response, guard = (frames[index].f_locals for index in range(3))
        raw, joints, joint = guard.get("values"), guard.get("joints"), guard.get("name")
        require(
            type(raw) in (tuple, list)
            and type(joints) is tuple
            and type(joint) is str
            and joint in JOINT_NAMES
            and guard.get("label") == "paused Smol action",
            "Not a finite nine-joint guard failure",
        )
        targets = vector(joints, 9, "diagnostic targets")
        require(vector(raw, 9, "original targets") == targets, "Original targets changed")
        index = JOINT_NAMES.index(joint)
        value, low, high = (finite(guard.get(key), key) for key in ("value", "low", "high"))
        message = f"paused Smol action: {joint} outside reference limits"
        require(
            error.args == (message,)
            and (low, high) == (JOINT_LOWER[index], JOINT_UPPER[index])
            and value == targets[index]
            and not low - 1e-6 <= value <= high + 1e-6,
            "Not the original joint limit rejection",
        )
        actions, request = response.get("actions"), server.get("request")
        require(
            type(actions) in (tuple, list)
            and len(actions) == 50
            and actions is server.get("actions")
            and request is response.get("value"),
            "Missing original response objects",
        )
        matches = [row for row, action in enumerate(actions) if action is raw]
        bindings, context = _request_bindings(
            request, server.get("observation"), server.get("context")
        )
        started = integer(server.get("started"), "actual inference start")
        ended = integer(server.get("ended"), "actual inference end", started)
        latency = finite(response.get("inference_latency_ms"), "actual inference latency")
        require(latency == (ended - started) / 1e6, "Original inference timing changed")
        return {
            "schema": DIAGNOSTIC_SCHEMA,
            "diagnostic_only": True,
            "kind": "joint_guard_rejected",
            "message": message,
            "joint_guard": {
                "label": guard["label"],
                "joint": joint,
                "joint_index": index,
                "value": value,
                "targets": list(targets),
                "low": low,
                "high": high,
                "tolerance": 1e-6,
                "horizon_index": matches[0] if len(matches) == 1 else None,
                "horizon_index_status": "unique_identity" if len(matches) == 1 else "unknown",
            },
            "request": bindings,
            "context": context,
            "inference": {"started_ns": started, "ended_ns": ended, "latency_ms": latency},
        }
    except (ContractError, KeyError, AttributeError, TypeError, OverflowError):
        return _unavailable("invalid_traceback_evidence")
    finally:
        frames.clear()


def emit_guard_diagnostic(error: BaseException, *, stream: TextIO | None = None) -> bool:
    value = guard_diagnostic(error)
    line = DIAGNOSTIC_PREFIX + canonical(value).decode("ascii") + "\n"
    if len(line.encode("ascii")) > MAX_DIAGNOSTIC_BYTES:
        line = DIAGNOSTIC_PREFIX + canonical(_unavailable("diagnostic_size_limit")).decode() + "\n"
    target = sys.stderr if stream is None else stream
    try:
        require(target.write(line) == len(line), "Incomplete structured diagnostic write")
        target.flush()
    except (OSError, ValueError):
        return False
    return True


class LocalCommandPausedSmolVLAPolicy(LocalPausedSmolVLAPolicy):
    model_admission, artifact_schema = ADMISSION_KIND, MODEL_SCHEMA

    def __init__(
        self,
        root: Path,
        *,
        backbone_root: Path,
        scope: Scope,
        model_sha256: str,
        expected_control_profile_sha256: str,
        expected_criteria_sha256: str,
        expected_frozen_plan_sha256: str,
    ) -> None:
        metadata = validate_model(root, expected_scope=scope, expected_model_sha256=model_sha256)
        require(
            metadata["criteria_sha256"] == expected_criteria_sha256
            and metadata["frozen_plan_sha256"] == expected_frozen_plan_sha256,
            "Command v3 model belongs to different frozen criteria/conditions",
        )
        self._load(
            root,
            backbone_root=backbone_root,
            scope=scope,
            model_sha256=model_sha256,
            expected_control_profile_sha256=expected_control_profile_sha256,
            metadata=metadata,
            profile=PausedControlProfile(**metadata["control_profile"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit command-v3 paused Smol model process.")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--backbone-root", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--allowed-client-uid", type=int)
    args = parser.parse_args()
    binding = read_json(args.binding)
    require(
        binding.get("execution_timing") == EXECUTION_TIMING
        and binding.get("real_time_admission") is False,
        "Explicit paused model-server binding is required",
    )
    try:
        policy = LocalCommandPausedSmolVLAPolicy(
            args.model_root,
            backbone_root=args.backbone_root,
            scope=Scope(**binding["scope"]),
            model_sha256=args.model_sha256,
            expected_control_profile_sha256=PausedControlProfile(
                **binding["control_profile"]
            ).sha256,
            expected_criteria_sha256=binding["criteria_sha256"],
            expected_frozen_plan_sha256=binding["frozen_plan_sha256"],
        )
        serve(policy, args.socket_path, allowed_client_uid=args.allowed_client_uid)
    except (ContractError, OSError) as exc:
        logged = emit_guard_diagnostic(exc)
        suffix = "" if logged else " (structured diagnostic unavailable)"
        raise SystemExit(f"Command-v3 paused Smol process stopped: {exc}{suffix}") from exc


if __name__ == "__main__":
    main()
