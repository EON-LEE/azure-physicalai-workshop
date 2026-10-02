"""Pure evidence sink for the operator-approved runtime loop; never an actuator or grant issuer."""

from __future__ import annotations

import copy
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from learning.common import (
    canonical,
    digest,
    file_digest,
    finite,
    integer,
    inventory,
    keys,
    parse_json,
    read_json,
    require,
    safe_path,
    sha256,
    token,
    utc,
    vector,
    verify_inventory,
)
from learning.contract import (
    CAMERAS,
    CameraSample,
    ControlProfile,
    DemonstrationSource,
    Provenance,
    Scope,
    png_dimensions,
)
from learning.evaluation import validate_runtime
from learning.gr00t.bootstrap import validate_plan as validate_bootstrap_plan
from learning.gr00t.evaluation import score_trial
from learning.gr00t.evaluation import validate_plan as validate_paired_plan
from learning.smolvla import POLICY_TYPE

GRANT_SCHEMA = "physicalai.operator-rollout-grant/v1"
RECORDING_SCHEMA = "physicalai.physical-rollout-recording/v1"
PAIRED_RESULTS = "physicalai.smolvla-paired-results/v2"
BOOTSTRAP_RESULTS = "physicalai.smolvla-bootstrap-results/v2"


@dataclass(frozen=True)
class CaseBinding:
    evaluation_run_id: str
    purpose: str
    scope: Scope
    policy: str
    model_sha256: str | None
    episode_id: str
    attempt: int
    environment_id: str
    revision: str
    seed: int
    scene_builder_sha256: str
    goal_id: str
    command_id: str
    epoch: str
    control_profile_sha256: str
    runtime_sha256: str
    deadline_at_utc: str
    deadline_monotonic_ns: int
    policy_type: str = POLICY_TYPE


@dataclass(frozen=True)
class MeasuredState:
    captured_at_utc: str
    monotonic_ns: int
    physics_step: int
    object_position_m: tuple[float, float, float]


@dataclass(frozen=True)
class ControlTrace:
    command_id: str
    epoch: str
    monotonic_ns: int
    physics_step: int
    policy_predict_calls: int
    applied_action_count: int
    reference_route_calls: int
    applied_model_sha256: str | None
    latencies_ms: tuple[float, ...] = ()
    cycle_durations_ms: tuple[float, ...] = ()
    last_applied_monotonic_ns: int | None = None
    safety_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalOutcome:
    status: str
    destination_id: str
    failure_reason: str | None
    capture_id: str
    capture_manifest_uri: str
    capture_manifest_sha256: str
    capture_frame_count: int


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path: Path, value: object) -> None:
    _write(path, canonical(value) + b"\n")


def _run_contract(plan: dict, runtime: dict, grant: dict) -> list[dict]:
    validate_runtime(runtime)
    Provenance(**runtime["provenance"]).validate(require_live=True)
    keys(
        grant,
        {
            "schema",
            "purpose",
            "evaluation_run_id",
            "scope",
            "operator_principal_sha256",
            "plan_sha256",
            "runtime_sha256",
            "control_profile_sha256",
            "task",
            "issued_at_utc",
            "expires_at_utc",
            "max_episode_seconds",
            "max_total_seconds",
        },
        "operator rollout grant",
    )
    require(grant["schema"] == GRANT_SCHEMA, "Unknown operator grant schema")
    token(grant["evaluation_run_id"], "evaluation run")
    sha256(grant["operator_principal_sha256"], "operator principal")
    scope = Scope(**keys(grant["scope"], {"tenant_id", "owner_id"}, "grant scope"))
    scope.validate()
    require(
        grant["scope"] == plan["scope"]
        and grant["plan_sha256"] == digest(canonical(plan))
        and grant["runtime_sha256"] == plan["runtime_sha256"] == digest(canonical(runtime))
        and grant["control_profile_sha256"] == plan["control_profile_sha256"],
        "Operator grant does not bind the exact plan/runtime/owner/profile",
    )
    issued, expires = utc(grant["issued_at_utc"]), utc(grant["expires_at_utc"])
    total = integer(grant["max_total_seconds"], "total rollout budget", 1, 3600)
    per_episode = integer(grant["max_episode_seconds"], "motion budget", 1, 30)
    require(0 < (expires - issued).total_seconds() <= total, "Invalid grant time window")
    task = DemonstrationSource(
        kind="reference_controller",
        **keys(
            grant["task"],
            {"task_id", "instruction", "goal_id"},
            "approved rollout task",
        ),
    )
    task.validate()
    if grant["purpose"] == "reference_bootstrap":
        validate_bootstrap_plan(plan, schema="physicalai.smolvla-bootstrap-plan/v1")
        roles = ("reference", "candidate")
    else:
        require(grant["purpose"] == "paired_policy_eval", "Not an evaluation-only operator grant")
        validate_paired_plan(plan, schema="physicalai.smolvla-paired-plan/v1")
        roles = ("before", "after")
    require(len(plan["cases"]) >= plan["minimum_pairs"], "Incomplete frozen evaluation cohort")
    require(
        len(plan["cases"]) * len(roles) * per_episode <= total,
        "Schedule exceeds total motion budget",
    )
    schedule = []
    for index, case in enumerate(plan["cases"]):
        require(
            case["expected_destination_id"] == task.goal_id, "Case goal differs from approved task"
        )
        for role in roles if index % 2 == 0 else roles[::-1]:
            schedule.append({"policy": role, "case": copy.deepcopy(case)})
    return schedule


def _expected_model(plan: dict, role: str) -> str | None:
    if role == "reference":
        return None
    return plan["candidate_model_sha256"] if role == "candidate" else plan[f"policy_{role}_sha256"]


def _state(value: MeasuredState) -> None:
    utc(value.captured_at_utc)
    integer(value.monotonic_ns, "actual monotonic timestamp", 1)
    integer(value.physics_step, "actual physics step")
    vector(value.object_position_m, 3, "measured object pose")


def _binding(
    value: CaseBinding, initial: MeasuredState, scheduled: dict, plan: dict, grant: dict
) -> None:
    _state(initial)
    case = scheduled["case"]
    require(
        value.policy_type == POLICY_TYPE
        and value.purpose == grant["purpose"]
        and value.evaluation_run_id == grant["evaluation_run_id"]
        and asdict(value.scope) == grant["scope"]
        and value.policy == scheduled["policy"]
        and value.model_sha256 == _expected_model(plan, value.policy)
        and value.runtime_sha256 == grant["runtime_sha256"]
        and value.control_profile_sha256 == grant["control_profile_sha256"],
        "Attempt authority/model/runtime differs from the frozen operator grant",
    )
    require(
        all(
            getattr(value, name) == case[name]
            for name in (
                "episode_id",
                "attempt",
                "environment_id",
                "revision",
                "seed",
                "scene_builder_sha256",
            )
        ),
        "Attempt does not match the frozen case",
    )
    token(value.command_id, "actual runtime command")
    token(value.epoch, "actual scene epoch")
    require(value.goal_id == case["expected_destination_id"], "Attempt goal changed")
    require(
        all(
            abs(actual - expected) <= 0.001
            for actual, expected in zip(
                initial.object_position_m, case["initial_pose_m"], strict=True
            )
        ),
        "Actual initial pose differs from the frozen case",
    )
    start = utc(initial.captured_at_utc)
    deadline = utc(value.deadline_at_utc)
    require(
        utc(grant["issued_at_utc"]) <= start < deadline <= utc(grant["expires_at_utc"]),
        "Attempt lies outside the operator grant window",
    )
    ns = integer(value.deadline_monotonic_ns, "immutable motion deadline", 1) - initial.monotonic_ns
    require(
        abs(ns - grant["max_episode_seconds"] * 1_000_000_000) <= 1_000_000,
        "Motion deadline differs from the fixed paired per-episode budget",
    )
    require(
        abs((deadline - start).total_seconds() * 1e9 - ns) <= 1_000_000,
        "UTC and monotonic motion deadlines disagree",
    )


def _camera_metadata(state: MeasuredState, frames: dict[str, CameraSample]) -> dict:
    require(set(frames) == set(CAMERAS), "Both actual evaluation cameras are required")
    values = {}
    for name, frame in frames.items():
        width, height = png_dimensions(frame.png)
        integer(frame.rendering_frame, "native render identity")
        integer(frame.physics_step, "camera physics step")
        integer(frame.monotonic_ns, "camera monotonic time", 1)
        require(
            frame.physics_step == state.physics_step
            and 0 <= state.monotonic_ns - frame.monotonic_ns <= 200_000_000,
            "Evaluation camera is not from the actual measured state",
        )
        values[name] = {
            "sha256": digest(frame.png),
            "width": width,
            "height": height,
            "rendering_frame": frame.rendering_frame,
            "physics_step": frame.physics_step,
            "monotonic_ns": frame.monotonic_ns,
        }
    return values


def _trace(
    value: ControlTrace, previous: ControlTrace | None, binding: CaseBinding, initial: MeasuredState
) -> None:
    require(
        value.command_id == binding.command_id and value.epoch == binding.epoch,
        "Stale command/epoch trace",
    )
    timestamp = previous.monotonic_ns if previous else initial.monotonic_ns
    step = previous.physics_step if previous else initial.physics_step
    require(
        integer(value.monotonic_ns, "trace timestamp", 1) > timestamp, "Nonmonotonic control trace"
    )
    require(
        integer(value.physics_step, "trace physics step") >= step, "Physics trace moved backwards"
    )
    for name in ("policy_predict_calls", "applied_action_count", "reference_route_calls"):
        count = integer(getattr(value, name), name, 0, 2000)
        require(
            count >= (getattr(previous, name) if previous else 0),
            "Controller counter moved backwards",
        )
    prior_applied = previous.applied_action_count if previous else 0
    require(
        value.applied_action_count - prior_applied <= value.physics_step - step,
        "Applied action count exceeds actual physics ticks",
    )
    learned = binding.policy != "reference"
    require(
        value.reference_route_calls == 0 if learned else value.policy_predict_calls == 0,
        "Scripted reference substituted for a learned controller",
    )
    if value.applied_action_count:
        require(
            value.applied_model_sha256 == binding.model_sha256
            and (value.policy_predict_calls > 0 if learned else value.reference_route_calls > 0),
            "Applied model/counter evidence differs from actual controller",
        )
        last = integer(value.last_applied_monotonic_ns, "last actual actuation", 1)
        require(
            initial.monotonic_ns <= last <= min(value.monotonic_ns, binding.deadline_monotonic_ns),
            "Actuation occurred after the immutable motion deadline",
        )
        if value.applied_action_count > prior_applied:
            require(
                last
                > (
                    previous.last_applied_monotonic_ns
                    if previous and prior_applied
                    else initial.monotonic_ns
                ),
                "New actuation has no advancing actual timestamp",
            )
    else:
        require(
            value.applied_model_sha256 is None and value.last_applied_monotonic_ns is None,
            "A model cannot be recorded as applied before actual actuation",
        )
    for sequence, label in (
        (value.latencies_ms, "inference/controller latency"),
        (value.cycle_durations_ms, "whole-cycle duration"),
    ):
        require(isinstance(sequence, (tuple, list)), f"Invalid {label} measurements")
        for measurement in sequence:
            require(finite(measurement, label) >= 0, f"Invalid {label}")
    delta_calls = value.policy_predict_calls - (previous.policy_predict_calls if previous else 0)
    require(
        len(value.latencies_ms)
        <= (delta_calls if learned else value.applied_action_count - prior_applied),
        "Latency measurements exceed actual new controller calls",
    )
    require(
        isinstance(value.safety_violations, (tuple, list))
        and len(value.safety_violations) <= 32
        and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value.safety_violations),
        "Invalid safety violation evidence",
    )


def _terminal(
    outcome: TerminalOutcome,
    final: MeasuredState,
    initial: MeasuredState,
    binding: CaseBinding,
    events: list[ControlTrace],
    capture_manifest: bytes,
    images: dict,
    case: dict,
    runtime: dict,
    approved_task: dict,
) -> dict:
    _state(final)
    require(
        outcome.status in ("succeeded", "failed", "cancelled", "timed_out"),
        "No actual terminal outcome",
    )
    require(
        outcome.failure_reason is None
        if outcome.status == "succeeded"
        else isinstance(outcome.failure_reason, str) and 0 < len(outcome.failure_reason) <= 512,
        "Failure outcome requires its actual reason",
    )
    token(outcome.destination_id, "actual selected destination")
    token(outcome.capture_id, "actual capture receipt")
    require(events or outcome.status != "succeeded", "No control trace for a successful attempt")
    last = events[-1] if events else None
    require(
        final.monotonic_ns > initial.monotonic_ns
        and final.monotonic_ns >= (last.monotonic_ns if last else initial.monotonic_ns)
        and final.physics_step >= (last.physics_step if last else initial.physics_step)
        and utc(final.captured_at_utc) > utc(initial.captured_at_utc),
        "Final measurement precedes actual control completion",
    )
    if outcome.status == "succeeded":
        require(
            final.monotonic_ns <= binding.deadline_monotonic_ns
            and utc(final.captured_at_utc) <= utc(binding.deadline_at_utc),
            "Successful outcome exceeded the approved deadline",
        )
    require(
        isinstance(capture_manifest, bytes) and len(capture_manifest) <= 4 * 1024 * 1024,
        "Missing/oversized actual capture manifest",
    )
    require(
        digest(capture_manifest) == sha256(outcome.capture_manifest_sha256),
        "Actual capture manifest checksum mismatch",
    )
    capture = parse_json(capture_manifest)
    require(
        capture.get("scope") == asdict(binding.scope)
        and capture.get("schema")
        in (
            "physicalai.demonstrations/v1",
            "physicalai.demonstrations/v2",
        ),
        "Capture belongs to another scope/schema",
    )
    require(
        isinstance(capture.get("episodes"), list) and len(capture["episodes"]) == 1,
        "Actual command capture must contain exactly its own episode",
    )
    episode = capture["episodes"][0]
    require(
        episode.get("episode_id") == binding.command_id
        and all(
            episode.get(name) == getattr(binding, name)
            for name in ("environment_id", "revision", "seed")
        )
        and episode.get("split") == "test"
        and episode.get("frame_count")
        == integer(outcome.capture_frame_count, "captured frames", 2, 18002),
        "Capture receipt is not bound to this exact held-out runtime command",
    )
    provenance = Provenance(**episode["provenance"])
    provenance.validate(require_live=True)
    for name in (
        "simulator_version",
        "simulator_image_digest",
        "robot_asset_sha256",
        "code_revision",
        "gpu_model",
    ):
        require(
            getattr(provenance, name) == runtime["provenance"][name],
            "Capture runtime identity changed",
        )
    if binding.policy != "reference":
        require(
            capture["schema"] == "physicalai.demonstrations/v2",
            "Learned rollout lacks actual v2 servo capture",
        )
    if capture["schema"] == "physicalai.demonstrations/v2":
        require(
            ControlProfile(**capture["control_profile"]).sha256 == binding.control_profile_sha256,
            "Capture servo profile differs from the frozen evaluation",
        )
        source = DemonstrationSource(**episode["demonstration"])
        source.validate()
        require(
            source.kind == ("reference_controller" if binding.policy == "reference" else "learned")
            and source.source_policy_sha256 == binding.model_sha256
            and source.goal_id == binding.goal_id,
            "Actual capture was not produced by this controller/task",
        )
        require(
            {name: getattr(source, name) for name in ("task_id", "instruction", "goal_id")}
            == approved_task,
            "Captured task instruction differs from the operator evaluation grant",
        )
    location = urlsplit(outcome.capture_manifest_uri)
    require(
        location.scheme == "https"
        and location.hostname is not None
        and location.hostname.endswith(".blob.core.windows.net")
        and not (location.query or location.fragment or location.username or location.password)
        and location.path
        == f"/demonstrations/{binding.scope.owner_id}/{binding.command_id}/manifest.json",
        "Capture receipt points outside the owned immutable command artifact",
    )
    latencies = [number for event in events for number in event.latencies_ms]
    cycles = [number for event in events for number in event.cycle_durations_ms]
    violations = list(dict.fromkeys(item for event in events for item in event.safety_violations))
    if any(duration > 100 for duration in cycles):
        violations.append("whole_control_cycle_exceeded_100ms")
    if binding.policy != "reference" and any(duration > 80 for duration in latencies):
        violations.append("policy_inference_exceeded_80ms")
    if outcome.status == "succeeded" and binding.policy != "reference":
        require(
            last is not None and len(cycles) == last.policy_predict_calls,
            "Successful policy run lacks complete measured end-to-end cycle evidence",
        )
    trial = {
        "episode_id": binding.episode_id,
        "attempt": binding.attempt,
        "seed": binding.seed,
        "policy": binding.policy,
        "model_sha256": binding.model_sha256,
        "environment_id": binding.environment_id,
        "revision": binding.revision,
        "observed_initial_pose_m": list(initial.object_position_m),
        "scene_builder_sha256": binding.scene_builder_sha256,
        "final_pose_m": list(final.object_position_m),
        "destination_id": outcome.destination_id,
        "terminated": outcome.status in ("succeeded", "failed"),
        "truncated": outcome.status in ("cancelled", "timed_out"),
        "failure_reason": outcome.failure_reason,
        "latencies_ms": latencies,
        "duration_ms": (final.monotonic_ns - initial.monotonic_ns) / 1e6,
        "safety_violations": violations,
        "policy_predict_calls": last.policy_predict_calls if last else 0,
        "applied_action_count": last.applied_action_count if last else 0,
        "reference_route_calls": last.reference_route_calls if last else 0,
        "final_images": {name: metadata["sha256"] for name, metadata in images.items()},
    }
    score_trial(trial, case, learned=binding.policy != "reference")
    return trial


class PhysicalRolloutRecorder:
    """One persistence thread. The trusted runtime supplies immutable actual measurements."""

    def __init__(
        self,
        root: Path,
        *,
        plan: dict,
        runtime: dict,
        grant: dict,
        expected_plan_sha256: str,
        expected_grant_sha256: str,
    ) -> None:
        require(
            digest(canonical(plan)) == sha256(expected_plan_sha256), "Frozen plan checksum mismatch"
        )
        require(
            digest(canonical(grant)) == sha256(expected_grant_sha256),
            "Operator grant checksum mismatch",
        )
        self._schedule = _run_contract(plan, runtime, grant)
        require(not root.exists() and not root.is_symlink(), "Never overwrite rollout evidence")
        self.root, self.plan, self.runtime, self.grant = (
            root,
            copy.deepcopy(plan),
            copy.deepcopy(runtime),
            copy.deepcopy(grant),
        )
        self.active = None
        self.completed: list[dict] = []
        self.failed: list[dict] = []
        self.visited = 0
        self.closed = False
        self.faulted = False
        self.thread_id = threading.get_ident()
        self.used_commands: set[str] = set()
        self.last_finished: MeasuredState | None = None
        root.mkdir(parents=True, mode=0o700)
        for name, value in (
            ("plan", self.plan),
            ("runtime", self.runtime),
            ("grant", self.grant),
            ("schedule", {"attempts": self._schedule}),
        ):
            _json(root / f"{name}.json", value)

    @property
    def schedule(self) -> list[dict]:
        return copy.deepcopy(self._schedule)

    def _next(self) -> dict:
        require(
            threading.get_ident() == self.thread_id, "Use the recorder's single persistence thread"
        )
        require(
            not self.closed and self.visited < len(self._schedule), "No remaining frozen attempt"
        )
        return self._schedule[self.visited]

    def start_case(
        self, binding: CaseBinding, initial: MeasuredState, camera_frames: dict[str, CameraSample]
    ) -> None:
        require(
            self.active is None and not self.faulted, "Finish or abort the current attempt first"
        )
        scheduled = self._next()
        self.faulted = True
        _binding(binding, initial, scheduled, self.plan, self.grant)
        if self.last_finished is not None:
            require(
                initial.monotonic_ns > self.last_finished.monotonic_ns
                and utc(initial.captured_at_utc) > utc(self.last_finished.captured_at_utc),
                "A paired attempt started before the preceding physical episode ended",
            )
        require(
            binding.command_id not in self.used_commands, "Runtime command was reused across trials"
        )
        metadata = _camera_metadata(initial, camera_frames)
        relative = f"{binding.policy}/{binding.episode_id}/{binding.attempt}"
        path = safe_path(self.root, relative + "/start.json", must_exist=False).parent
        path.mkdir(parents=True, mode=0o700)
        start = {"binding": asdict(binding), "initial": asdict(initial), "initial_images": metadata}
        for name, frame in camera_frames.items():
            _write(path / f"initial-{name}.png", frame.png)
        _json(path / "start.json", start)
        _write(path / "control.jsonl", b"")
        self.active = {
            "binding": copy.deepcopy(binding),
            "initial": copy.deepcopy(initial),
            "events": [],
            "path": path,
            "relative": relative,
            "chain_sha256": digest(canonical(start)),
        }
        self.used_commands.add(binding.command_id)
        self.faulted = False

    def append_control(self, trace: ControlTrace) -> None:
        self._next()
        require(
            self.active is not None and not self.faulted and not self.closed,
            "No writable active attempt",
        )
        self.faulted = True
        active = self.active
        require(
            len(active["events"]) < 2000,
            "Per-attempt control trace exceeds the bounded motion budget",
        )
        _trace(
            trace,
            active["events"][-1] if active["events"] else None,
            active["binding"],
            active["initial"],
        )
        row = {
            "index": len(active["events"]),
            "previous_sha256": active["chain_sha256"],
            "event": asdict(trace),
        }
        checksum = digest(canonical(row))
        encoded = canonical({**row, "sha256": checksum}) + b"\n"
        trace_path = active["path"] / "control.jsonl"
        require(
            len(encoded) <= 16384 and trace_path.stat().st_size + len(encoded) <= 4 * 1024 * 1024,
            "Control trace exceeds its per-attempt byte budget",
        )
        with trace_path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        active["events"].append(copy.deepcopy(trace))
        active["chain_sha256"] = checksum
        self.faulted = False

    def finish_case(
        self,
        outcome: TerminalOutcome,
        final: MeasuredState,
        camera_frames: dict[str, CameraSample],
        capture_manifest: bytes,
    ) -> None:
        require(
            self.active is not None and not self.faulted and not self.closed,
            "No complete active attempt",
        )
        self.faulted = True
        active = self.active
        images = _camera_metadata(final, camera_frames)
        start = read_json(active["path"] / "start.json")
        for name, image in images.items():
            require(
                image["rendering_frame"] > start["initial_images"][name]["rendering_frame"],
                "Final camera frame did not advance",
            )
        trial = _terminal(
            outcome,
            final,
            active["initial"],
            active["binding"],
            active["events"],
            capture_manifest,
            images,
            self._next()["case"],
            self.runtime,
            self.grant["task"],
        )
        for name, frame in camera_frames.items():
            _write(active["path"] / f"{name}.png", frame.png)
        _write(active["path"] / "capture-manifest.json", capture_manifest)
        terminal = {
            "outcome": asdict(outcome),
            "final": asdict(final),
            "final_images": images,
            "trace_sha256": active["chain_sha256"],
            "trial": trial,
        }
        _json(active["path"] / "terminal.json", terminal)
        self.completed.append({"path": active["relative"], "trial": trial})
        self.last_finished = final
        self.visited += 1
        self.active = None
        self.faulted = False

    def abort_case(self, phase: str, error: str) -> None:
        scheduled = self._next()
        token(phase, "failure phase")
        require(
            isinstance(error, str) and 0 < len(error) <= 1024,
            "Explicit bounded failure is required",
        )
        record = {
            "schedule_index": self.visited,
            "policy": scheduled["policy"],
            "episode_id": scheduled["case"]["episode_id"],
            "attempt": scheduled["case"]["attempt"],
            "phase": phase,
            "error": error,
            "measurements_complete": False,
        }
        _json(self.root / f"failure-{self.visited:04d}.json", record)
        self.failed.append(record)
        self.visited += 1
        self.active = None
        self.faulted = False

    def finalize(self) -> Path:
        require(
            threading.get_ident() == self.thread_id, "Use the recorder's single persistence thread"
        )
        require(not self.closed, "Rollout collection is already sealed")
        complete = (
            self.active is None and not self.failed and len(self.completed) == len(self._schedule)
        )
        _json(
            self.root / "collection.json",
            {
                "complete": complete,
                "expected_attempts": len(self._schedule),
                "visited_attempts": self.visited,
                "completed_attempts": len(self.completed),
                "failed_attempts": len(self.failed),
            },
        )
        self.closed = True
        require(
            complete,
            "Incomplete physical rollout; retained failures cannot be fabricated into results",
        )
        recording = {
            "schema": RECORDING_SCHEMA,
            "plan_sha256": digest(canonical(self.plan)),
            "grant_sha256": digest(canonical(self.grant)),
            "expected_attempts": len(self._schedule),
            "attempt_paths": [item["path"] for item in self.completed],
            "files": inventory(self.root),
        }
        _json(self.root / "rollout-manifest.json", recording)
        result = {
            "schema": BOOTSTRAP_RESULTS
            if self.grant["purpose"] == "reference_bootstrap"
            else PAIRED_RESULTS,
            "scope": self.plan["scope"],
            "plan_sha256": digest(canonical(self.plan)),
            "runtime": self.runtime,
            "trials": [item["trial"] for item in self.completed],
            "recording": {
                "path": "rollout-manifest.json",
                "sha256": file_digest(self.root / "rollout-manifest.json"),
            },
        }
        verify_recording(self.root, result, self.plan)
        _json(self.root / "results.json", result)
        return self.root / "results.json"


def _read_cameras(root: Path, metadata: dict, prefix: str, state: MeasuredState) -> dict:
    frames = {}
    for camera, value in keys(metadata, set(CAMERAS), "recorded cameras").items():
        path = safe_path(root, f"{prefix}{camera}.png")
        require(path.stat().st_size <= 32 * 1024 * 1024, "Oversized recorded camera")
        frames[camera] = CameraSample(
            path.read_bytes(),
            value["rendering_frame"],
            value["physics_step"],
            value["monotonic_ns"],
        )
    require(
        _camera_metadata(state, frames) == metadata, "Recorded camera metadata/checksum mismatch"
    )
    return frames


def verify_recording(root: Path, result: dict, plan: dict) -> None:
    descriptor = keys(result.get("recording"), {"path", "sha256"}, "physical recording descriptor")
    require(descriptor["path"] == "rollout-manifest.json", "Unexpected recording artifact path")
    require(
        file_digest(safe_path(root, descriptor["path"])) == sha256(descriptor["sha256"]),
        "Rollout recording checksum mismatch",
    )
    recording = keys(
        read_json(root / descriptor["path"]),
        {
            "schema",
            "plan_sha256",
            "grant_sha256",
            "expected_attempts",
            "attempt_paths",
            "files",
        },
        "physical recording",
    )
    require(recording["schema"] == RECORDING_SCHEMA, "Unknown physical recording version")
    verify_inventory(root, recording["files"], exclude={"results.json", "rollout-manifest.json"})
    recorded_plan, runtime, grant = (
        read_json(root / f"{name}.json") for name in ("plan", "runtime", "grant")
    )
    require(
        recorded_plan == plan
        and runtime == result["runtime"]
        and recording["plan_sha256"] == result["plan_sha256"] == digest(canonical(plan))
        and recording["grant_sha256"] == digest(canonical(grant)),
        "Recording authority binding changed",
    )
    schedule = _run_contract(plan, runtime, grant)
    require(
        result["scope"] == plan["scope"]
        and result["schema"]
        == (BOOTSTRAP_RESULTS if grant["purpose"] == "reference_bootstrap" else PAIRED_RESULTS),
        "Wrong physical collection scope/purpose",
    )
    require(
        read_json(root / "schedule.json") == {"attempts": schedule}, "Frozen attempt order changed"
    )
    require(
        recording["expected_attempts"]
        == len(schedule)
        == len(result["trials"])
        == len(recording["attempt_paths"]),
        "Missing physical attempts",
    )
    require(
        read_json(root / "collection.json")
        == {
            "complete": True,
            "expected_attempts": len(schedule),
            "visited_attempts": len(schedule),
            "completed_attempts": len(schedule),
            "failed_attempts": 0,
        },
        "Incomplete physical collection cannot be promoted",
    )
    seen_commands = set()
    previous_final = None
    for index, scheduled in enumerate(schedule):
        case, role = scheduled["case"], scheduled["policy"]
        relative = f"{role}/{case['episode_id']}/{case['attempt']}"
        require(recording["attempt_paths"][index] == relative, "Reordered/omitted physical attempt")
        start = read_json(safe_path(root, relative + "/start.json"))
        binding_data = dict(start["binding"])
        binding_data["scope"] = Scope(**binding_data["scope"])
        binding = CaseBinding(**binding_data)
        initial = MeasuredState(**start["initial"])
        _binding(binding, initial, scheduled, plan, grant)
        if previous_final is not None:
            require(
                initial.monotonic_ns > previous_final.monotonic_ns
                and utc(initial.captured_at_utc) > utc(previous_final.captured_at_utc),
                "Physical attempts overlap or replay earlier measurements",
            )
        require(binding.command_id not in seen_commands, "Reused physical runtime command")
        seen_commands.add(binding.command_id)
        path = root / relative
        _read_cameras(path, start["initial_images"], "initial-", initial)
        events, chain = [], digest(canonical(start))
        trace_path = safe_path(root, relative + "/control.jsonl")
        require(trace_path.stat().st_size <= 4 * 1024 * 1024, "Oversized physical trace")
        with trace_path.open("rb") as stream:
            for event_index, line in enumerate(stream):
                row = parse_json(line)
                keys(row, {"index", "previous_sha256", "event", "sha256"}, "control trace row")
                require(
                    event_index < 2000
                    and row["index"] == event_index
                    and row["previous_sha256"] == chain
                    and digest(
                        canonical({key: value for key, value in row.items() if key != "sha256"})
                    )
                    == row["sha256"],
                    "Broken physical control trace hash chain",
                )
                event = ControlTrace(**row["event"])
                _trace(event, events[-1] if events else None, binding, initial)
                events.append(event)
                chain = row["sha256"]
        terminal = read_json(safe_path(root, relative + "/terminal.json"))
        require(
            terminal["trace_sha256"] == chain, "Terminal does not bind the complete control trace"
        )
        final = MeasuredState(**terminal["final"])
        _read_cameras(path, terminal["final_images"], "", final)
        for camera in CAMERAS:
            require(
                terminal["final_images"][camera]["rendering_frame"]
                > start["initial_images"][camera]["rendering_frame"],
                "Repeated final render frame",
            )
        trial = _terminal(
            TerminalOutcome(**terminal["outcome"]),
            final,
            initial,
            binding,
            events,
            safe_path(root, relative + "/capture-manifest.json").read_bytes(),
            terminal["final_images"],
            case,
            runtime,
            grant["task"],
        )
        require(
            trial == terminal["trial"] == result["trials"][index],
            "Summary differs from actual recorded control/capture evidence",
        )
        previous_final = final
