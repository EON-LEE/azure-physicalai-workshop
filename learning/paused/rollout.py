"""Manifest-last paused evaluation recorder and independent verifier; never an actuator."""

from __future__ import annotations

import copy
import shutil
import threading
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from learning.common import (
    ContractError,
    canonical,
    digest,
    file_digest,
    integer,
    inventory,
    keys,
    parse_json,
    read_json,
    relative_path,
    require,
    safe_path,
    sha256,
    token,
    utc,
    verify_inventory,
    write_json,
)
from learning.contract import CAMERAS, Provenance, Scope, ValidatedDataset
from learning.paused.capture import PausedEpisodeBudget, validate_dataset
from learning.paused.contract import (
    BOOTSTRAP_RESULT_SCHEMA,
    EXECUTION_TIMING,
    GRANT_SCHEMA,
    RECORDING_SCHEMA,
    RESULT_SCHEMA,
    FrozenCameraSample,
    PausedControlProfile,
)
from learning.paused.evaluation import _trial, expected_model, roles, validate_plan
from learning.paused.task import PREDICATE_SOURCE, TaskState, evaluate_task_states

RUNTIME_SCHEMA = "physicalai.paused-evaluation-runtime/v1"
MAX_TASK_STATE_BYTES = 8192


def validate_runtime(runtime: dict, plan: dict, *, require_live: bool = True) -> None:
    keys(
        runtime,
        {
            "schema",
            "execution_timing",
            "real_time_admission",
            "provenance",
            "control_profile",
            "source_files",
            "probe_receipt_sha256",
            "test_only",
        },
        "paused runtime",
    )
    require(
        runtime["schema"] == RUNTIME_SCHEMA
        and runtime["execution_timing"] == EXECUTION_TIMING
        and runtime["real_time_admission"] is False
        and runtime["control_profile"] == plan["control_profile"]
        and digest(canonical(runtime)) == plan["runtime_sha256"],
        "Wrong actual paused runtime/profile binding",
    )
    Provenance(
        **keys(runtime["provenance"], set(Provenance.__dataclass_fields__), "runtime provenance")
    ).validate(require_live=require_live)
    require(
        type(runtime["test_only"]) is bool
        and runtime["test_only"] == (runtime["provenance"]["source_kind"] == "test_fixture"),
        "Runtime fixture provenance was relabelled",
    )
    if require_live:
        require(runtime["test_only"] is False, "Fixture runtime is not actual evaluation evidence")
    files = runtime["source_files"]
    require(isinstance(files, dict) and bool(files), "Missing actual runtime source identities")
    for name, checksum in files.items():
        relative_path(name)
        sha256(checksum)
    require(
        files.get("simulation/control.py") == PREDICATE_SOURCE["sha256"],
        "Actual task watchdog source differs; require a new reviewed parity check",
    )
    sha256(runtime["probe_receipt_sha256"], "actual paused runtime probe receipt")


def _grant(plan: dict, runtime: dict, grant: dict) -> list[tuple[str, dict]]:
    keys(
        grant,
        {
            "schema",
            "purpose",
            "execution_timing",
            "real_time_admission",
            "evaluation_run_id",
            "scope",
            "operator_principal_sha256",
            "plan_sha256",
            "runtime_sha256",
            "control_profile_sha256",
            "criteria_sha256",
            "frozen_plan_sha256",
            "issued_at_utc",
            "expires_at_utc",
            "max_episode_wall_seconds",
            "max_episode_physics_steps",
            "max_total_wall_seconds",
        },
        "paused operator grant",
    )
    require(
        grant["schema"] == GRANT_SCHEMA
        and grant["purpose"] == plan["comparison_kind"]
        and grant["execution_timing"] == EXECUTION_TIMING
        and grant["real_time_admission"] is False
        and grant["scope"] == plan["scope"]
        and grant["plan_sha256"] == digest(canonical(plan))
        and grant["runtime_sha256"] == digest(canonical(runtime))
        and all(
            grant[name] == plan[name]
            for name in (
                "control_profile_sha256",
                "criteria_sha256",
                "frozen_plan_sha256",
            )
        ),
        "Operator grant does not bind the exact new-mode plan/runtime/owner/conditions",
    )
    token(grant["evaluation_run_id"], "evaluation run ID")
    sha256(grant["operator_principal_sha256"], "operator principal hash")
    profile = PausedControlProfile(**plan["control_profile"])
    profile.validate()
    integer(
        grant["max_episode_wall_seconds"],
        "episode wall cap",
        1,
        profile.max_episode_wall_ms // 1000,
    )
    integer(
        grant["max_episode_physics_steps"],
        "episode physics cap",
        profile.hold_steps,
        profile.max_simulation_steps,
    )
    require(
        grant["max_episode_physics_steps"] % profile.hold_steps == 0,
        "Episode physics cap must contain full holds",
    )
    total = integer(
        grant["max_total_wall_seconds"], "total explicit operator wall budget", 1, 86400
    )
    require(
        0 < (utc(grant["expires_at_utc"]) - utc(grant["issued_at_utc"])).total_seconds() <= total,
        "Invalid absolute operator grant interval",
    )
    schedule = []
    order = roles(plan)
    for index, case in enumerate(plan["cases"]):
        schedule.extend((role, case) for role in (order if index % 2 == 0 else order[::-1]))
    require(
        len(schedule) * grant["max_episode_wall_seconds"] <= total,
        "The all-attempt frozen schedule exceeds its approved total wall budget",
    )
    return schedule


def derive_trial(
    raw: ValidatedDataset,
    states: list[TaskState],
    *,
    case: dict,
    role: str,
    model_sha256: str | None,
    profile: PausedControlProfile,
    final_images: dict[str, FrozenCameraSample],
    heartbeat_ns: tuple[int, ...],
    destination_id: str,
    failure_reason: str | None,
) -> dict:
    require(len(raw.episodes) == 1, "Each evaluation attempt needs one actual capture")
    require(
        raw.manifest["purpose"] == "evaluation",
        "Evaluation cannot reuse training/integration capture",
    )
    episode = raw.episodes[0]
    require(
        all(
            episode.metadata[key] == case[key]
            for key in ("episode_id", "seed", "environment_id", "revision")
        )
        and episode.metadata["provenance"]["scene_builder_sha256"] == case["scene_builder_sha256"]
        and raw.manifest["control_profile"] == asdict(profile),
        "Trial raw capture does not bind the frozen case/profile/builder",
    )
    learned = role != "reference"
    require(
        episode.metadata["demonstration"]["kind"]
        == ("learned" if learned else "reference_controller")
        and episode.metadata["demonstration"]["source_policy_sha256"] == model_sha256,
        "Demonstration route/model differs from the actual evaluated controller",
    )
    budget = PausedEpisodeBudget(**episode.metadata["budget"])
    require(
        len(states) == len(episode.frames) * profile.hold_steps + 1,
        "Incomplete per-tick task trace",
    )
    first, last = states[0], states[-1]
    require(
        first.physics_step == budget.initial_physics_step
        and first.applied_action_count == 0
        and first.policy_predict_calls == 0
        and first.reference_route_calls == 0,
        "Task trace must begin before evaluated actuation",
    )
    by_step = {state.physics_step: state for state in states}
    for frame_index, frame in enumerate(episode.frames):
        observed_state = by_step.get(frame["physics_step"])
        require(
            observed_state is not None
            and frame["epoch"] == first.epoch
            and tuple(frame["joint_positions"]) == tuple(observed_state.joint_positions),
            "Frozen raw joints/epoch differ from actual task-state readback",
        )
        for control in frame["applied_controls"]:
            state = by_step.get(control["physics_step"])
            require(
                state is not None
                and control["monotonic_ns"] <= state.monotonic_ns <= frame["hold_deadline_ns"]
                and state.applied_model_sha256 == model_sha256
                and (
                    state.reference_route_calls == 0
                    and state.policy_predict_calls == frame_index + 1
                    if learned
                    else state.policy_predict_calls == 0
                    and state.reference_route_calls >= frame_index + 1
                ),
                "Actual task/control tick time or model/reference route binding differs",
            )
    task = evaluate_task_states(
        states, initial=case["initial_pose_m"], goal=case["expected_pose_m"], profile=profile
    )
    require(set(final_images) == set(CAMERAS), "Both final real camera frames are required")
    final_sha, end = {}, last.monotonic_ns
    final_simulation_time = Fraction(
        episode.frames[-1]["simulation_time_numerator"],
        episode.frames[-1]["simulation_time_denominator"],
    ) + Fraction(profile.hold_steps, profile.physics_hz)
    for name, image in final_images.items():
        require(
            isinstance(image, FrozenCameraSample)
            and image.physics_step == last.physics_step
            and 0
            <= image.monotonic_ns - last.monotonic_ns
            <= profile.max_observation_wall_ms * 1_000_000
            and utc(image.captured_at_utc) >= utc(last.captured_at_utc)
            and abs(image.simulation_time - final_simulation_time)
            <= Fraction(1, 2 * profile.physics_hz),
            "Final image is not from the current final physics state",
        )
        metadata = image.metadata()
        final_sha[name] = metadata["sha256"]
        end = max(end, image.monotonic_ns)
    require(
        budget.started_ns <= first.monotonic_ns <= end <= budget.wall_deadline_ns,
        "Trial wall authority expired",
    )
    previous = budget.started_ns
    gaps = []
    for stamp in heartbeat_ns:
        integer(stamp, "actual main heartbeat timestamp", 1)
        require(previous <= stamp <= end, "Invalid/reordered heartbeat trace")
        if stamp != previous:
            gaps.append((stamp - previous) / 1e6)
        previous = stamp
    gaps.append((end - previous) / 1e6)
    require(bool(heartbeat_ns), "Missing main-thread heartbeat evidence")
    frames = episode.frames
    return {
        "episode_id": case["episode_id"],
        "seed": case["seed"],
        "attempt": case["attempt"],
        "policy": role,
        "model_sha256": model_sha256,
        "environment_id": case["environment_id"],
        "revision": case["revision"],
        "scene_builder_sha256": case["scene_builder_sha256"],
        "observed_initial_pose_m": list(first.object_position_m),
        "final_pose_m": list(last.object_position_m),
        "destination_id": destination_id,
        "terminated": frames[-1]["terminated"],
        "truncated": frames[-1]["truncated"],
        "failure_reason": failure_reason,
        "safety_violations": task["safety_violations"],
        "policy_predict_calls": last.policy_predict_calls,
        "applied_action_count": last.applied_action_count,
        "reference_route_calls": last.reference_route_calls,
        "final_images": final_sha,
        "latencies_wall_ms": [
            (frame["policy_finished_ns"] - frame["policy_started_ns"]) / 1e6
            for frame in frames
            if frame["policy_started_ns"] is not None
        ],
        "observation_wall_ms": [
            (
                (
                    frame["observation_completed_ns"]
                    if frame["initial_publication"]
                    else frame["monotonic_ns"]
                )
                - frame["observation_started_ns"]
            )
            / 1e6
            for frame in frames
        ],
        "hold_wall_ms": [
            (frame["applied_controls"][-1]["monotonic_ns"] - frame["hold_started_ns"]) / 1e6
            for frame in frames
        ],
        "interval_wall_ms": [
            (frame["applied_controls"][-1]["monotonic_ns"] - frame["observation_started_ns"]) / 1e6
            for frame in frames
        ],
        "heartbeat_gap_ms": gaps,
        "wall_duration_ms": (end - budget.started_ns) / 1e6,
        "simulation_duration_ms": (last.physics_step - first.physics_step)
        * 1000
        / profile.physics_hz,
        "task_evidence": task,
    }


def _attempt_path(root: Path, role: str, case: dict) -> Path:
    token(role, "controller role")
    token(case["episode_id"], "episode ID")
    return root / role / case["episode_id"] / str(case["attempt"])


class PausedRolloutRecorder:
    def __init__(
        self,
        root: Path,
        *,
        plan: dict,
        runtime: dict,
        grant: dict,
        allow_test_fixture: bool = False,
    ) -> None:
        validate_plan(plan)
        validate_runtime(runtime, plan, require_live=not allow_test_fixture)
        self.schedule = _grant(plan, runtime, grant)
        require(not root.exists(), "Never overwrite evaluation artifacts")
        root.mkdir(parents=True)
        self.root, self.plan, self.runtime, self.grant = (
            root,
            copy.deepcopy(plan),
            copy.deepcopy(runtime),
            copy.deepcopy(grant),
        )
        self.schedule = _grant(self.plan, self.runtime, self.grant)
        self.scope = Scope(**plan["scope"])
        self.profile = PausedControlProfile(**plan["control_profile"])
        self.allow_test_fixture, self.thread_id = allow_test_fixture, threading.get_ident()
        self.trials, self.attempts, self.faulted = [], [], False
        self.finalized = False
        self.incomplete_attempts: list[dict] = []
        self._last_end_ns = 0
        self._last_end_utc = utc(grant["issued_at_utc"])
        self._commands: set[str] = set()
        for name, value in (("plan.json", plan), ("runtime.json", runtime), ("grant.json", grant)):
            write_json(root / name, value)
        write_json(
            root / "schedule.json", [{"policy": role, "case": case} for role, case in self.schedule]
        )

    def record_attempt(
        self,
        *,
        policy: str,
        capture_root: Path,
        states: list[TaskState],
        final_images: dict[str, FrozenCameraSample],
        heartbeat_ns: tuple[int, ...],
        destination_id: str,
        failure_reason: str | None,
    ) -> dict:
        require(
            threading.get_ident() == self.thread_id and not self.faulted,
            "Recording thread changed or faulted",
        )
        require(
            len(self.attempts) < len(self.schedule), "All frozen attempts were already recorded"
        )
        self.faulted = True
        expected_role, case = self.schedule[len(self.attempts)]
        require(
            policy == expected_role, "Attempt order/controller differs from the frozen schedule"
        )
        raw = validate_dataset(
            capture_root, expected_scope=self.scope, require_live=not self.allow_test_fixture
        )
        require(
            raw.manifest["criteria_sha256"] == self.plan["criteria_sha256"]
            and raw.manifest["frozen_plan_sha256"] == self.plan["frozen_plan_sha256"]
            and raw.episodes[0].metadata["provenance"] == self.runtime["provenance"],
            "Raw source differs from the approved actual runtime/conditions",
        )
        trial = derive_trial(
            raw,
            states,
            case=case,
            role=policy,
            model_sha256=expected_model(self.plan, policy),
            profile=self.profile,
            final_images=final_images,
            heartbeat_ns=heartbeat_ns,
            destination_id=destination_id,
            failure_reason=failure_reason,
        )
        _trial(trial, case, learned=policy != "reference", profile=self.profile)
        start, end = (
            utc(states[0].captured_at_utc),
            max(utc(image.captured_at_utc) for image in final_images.values()),
        )
        end_ns = max(image.monotonic_ns for image in final_images.values())
        require(
            utc(self.grant["issued_at_utc"]) <= start <= end <= utc(self.grant["expires_at_utc"])
            and self._last_end_utc <= start
            and self._last_end_ns <= states[0].monotonic_ns
            and states[0].command_id not in self._commands
            and trial["wall_duration_ms"] <= self.grant["max_episode_wall_seconds"] * 1000
            and trial["applied_action_count"] <= self.grant["max_episode_physics_steps"],
            "Actual attempt exceeds original operator grant authority",
        )
        path = _attempt_path(self.root, policy, case)
        path.mkdir(parents=True)
        shutil.copytree(capture_root, path / "capture", symlinks=False)
        with (path / "task-states.jsonl").open("xb") as stream:
            for state in states:
                line = canonical(asdict(state)) + b"\n"
                require(len(line) <= MAX_TASK_STATE_BYTES, "Task trace exceeds row bounds")
                stream.write(line)
        cameras = {}
        for name, image in final_images.items():
            with (path / f"{name}.png").open("xb") as stream:
                stream.write(image.png)
            cameras[name] = image.metadata()
        manifest = {
            "schema": "physicalai.paused-physical-attempt/v1",
            "policy": policy,
            "case": case,
            "capture_manifest_sha256": raw.manifest_sha256,
            "task_states_sha256": file_digest(path / "task-states.jsonl"),
            "heartbeat_ns": list(heartbeat_ns),
            "final_images": cameras,
            "trial": trial,
        }
        write_json(path / "attempt.json", manifest)
        self.attempts.append(
            {
                "path": (path / "attempt.json").relative_to(self.root).as_posix(),
                "sha256": file_digest(path / "attempt.json"),
            }
        )
        self.trials.append(trial)
        self._commands.add(states[0].command_id)
        self._last_end_ns, self._last_end_utc = end_ns, end
        self.faulted = False
        return trial

    def record_failure(
        self,
        *,
        policy: str,
        failure_code: str,
        phase: str,
        observed_at_utc: str,
        monotonic_ns: int,
        command_id: str | None,
    ) -> Path:
        require(threading.get_ident() == self.thread_id, "Recording thread changed")
        require(not self.finalized, "Recording has already been finalized")
        require(len(self.attempts) < len(self.schedule), "No unrecorded attempt remains")
        role, case = self.schedule[len(self.attempts)]
        require(policy == role, "Failure does not match the next frozen attempt")
        token(failure_code, "bounded failure code")
        token(phase, "failed phase")
        integer(monotonic_ns, "observed failure monotonic time", 1)
        observed = utc(observed_at_utc)
        if command_id is not None:
            token(command_id, "actual failed command")
        require(
            observed >= self._last_end_utc and monotonic_ns >= self._last_end_ns,
            "Nonmonotonic failure evidence",
        )
        path = _attempt_path(self.root, role, case)
        path.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "physicalai.paused-incomplete-attempt/v1",
            "policy": role,
            "case": case,
            "failure_code": failure_code,
            "phase": phase,
            "observed_at_utc": observed_at_utc,
            "monotonic_ns": monotonic_ns,
            "actual_command_id": command_id,
            "planned_model_sha256": expected_model(self.plan, role),
            "observed_after_grant_expiry": observed > utc(self.grant["expires_at_utc"]),
            "physical_outcome_verified": False,
        }
        artifact = path / "incomplete-attempt.json"
        write_json(artifact, failure)
        self.attempts.append(
            {
                "path": artifact.relative_to(self.root).as_posix(),
                "sha256": file_digest(artifact),
            }
        )
        self.incomplete_attempts.append(failure)
        self._last_end_ns, self._last_end_utc = monotonic_ns, observed
        self.faulted = False
        return artifact

    def finalize(self) -> Path:
        require(threading.get_ident() == self.thread_id, "Recording thread changed")
        require(not self.finalized, "Recording has already been finalized")
        if self.faulted or self.incomplete_attempts or len(self.attempts) != len(self.schedule):
            write_json(
                self.root / "incomplete-recording.json",
                {
                    "schema": "physicalai.paused-incomplete-recording/v1",
                    "execution_timing": EXECUTION_TIMING,
                    "real_time_admission": False,
                    "scope": asdict(self.scope),
                    "plan_sha256": digest(canonical(self.plan)),
                    "expected_trial_count": len(self.schedule),
                    "recorded_complete_trial_count": len(self.trials),
                    "incomplete_attempts": self.incomplete_attempts,
                    "not_recorded": [
                        {"policy": role, "case": case}
                        for role, case in self.schedule[len(self.attempts) :]
                    ],
                    "artifacts": self.attempts,
                    "files": inventory(self.root),
                    "quality_gate_passed": False,
                    "learning_quality_verified": False,
                },
            )
            self.faulted = True
            self.finalized = True
            raise ContractError("Cannot publish an incomplete/faulted frozen evaluation schedule")
        result = {
            "schema": BOOTSTRAP_RESULT_SCHEMA
            if self.plan["comparison_kind"] == "reference_bootstrap"
            else RESULT_SCHEMA,
            "scope": asdict(self.scope),
            "execution_timing": EXECUTION_TIMING,
            "real_time_admission": False,
            "plan_sha256": digest(canonical(self.plan)),
            "runtime": self.runtime,
            "trials": self.trials,
            "recording": {
                "schema": RECORDING_SCHEMA,
                "grant_sha256": file_digest(self.root / "grant.json"),
                "attempts": self.attempts,
                "files": inventory(self.root),
            },
        }
        _verify(self.root, result, self.plan, self.scope, require_live=not self.allow_test_fixture)
        write_json(self.root / "results.json", result)
        self.faulted, self.finalized = True, True
        return self.root / "results.json"


def _read_task_states(
    path: Path, expected_sha256: str, profile: PausedControlProfile
) -> list[TaskState]:
    profile.validate()
    maximum_count = profile.max_simulation_steps + 1
    require(
        path.stat().st_size <= maximum_count * MAX_TASK_STATE_BYTES,
        "Oversized actual task-state trace",
    )
    require(file_digest(path) == sha256(expected_sha256), "Task-state trace changed")
    states = []
    with path.open("rb") as stream:
        while line := stream.readline(MAX_TASK_STATE_BYTES + 1):
            require(
                len(line) <= MAX_TASK_STATE_BYTES and len(states) < maximum_count,
                "Task trace exceeds episode bounds",
            )
            states.append(
                TaskState(
                    **keys(parse_json(line), set(TaskState.__dataclass_fields__), "task state")
                )
            )
    require(len(states) >= 2, "Missing actual per-tick task trace")
    return states


def _verify(root: Path, result: dict, plan: dict, scope: Scope, *, require_live: bool) -> None:
    keys(
        result,
        {
            "schema",
            "scope",
            "execution_timing",
            "real_time_admission",
            "plan_sha256",
            "runtime",
            "trials",
            "recording",
        },
        "paused physical results",
    )
    require(
        result["schema"]
        == (
            BOOTSTRAP_RESULT_SCHEMA
            if plan["comparison_kind"] == "reference_bootstrap"
            else RESULT_SCHEMA
        )
        and result["execution_timing"] == EXECUTION_TIMING
        and result["real_time_admission"] is False
        and result["scope"] == asdict(scope) == plan["scope"]
        and result["plan_sha256"] == digest(canonical(plan)),
        "Result schema/mode/owner/evaluation plan changed",
    )
    validate_plan(plan)
    validate_runtime(result["runtime"], plan, require_live=require_live)
    recording = keys(
        result["recording"], {"schema", "grant_sha256", "attempts", "files"}, "recording"
    )
    require(recording["schema"] == RECORDING_SCHEMA, "Wrong paused recording schema")
    verify_inventory(root, recording["files"], exclude={"results.json"})
    require(
        read_json(root / "plan.json") == plan
        and read_json(root / "runtime.json") == result["runtime"]
        and file_digest(root / "grant.json") == recording["grant_sha256"],
        "Recording plan/runtime/grant bytes differ",
    )
    grant = read_json(root / "grant.json")
    schedule = _grant(plan, result["runtime"], grant)
    require(
        len(recording["attempts"]) == len(result["trials"]) == len(schedule),
        "Incomplete all-attempt recording",
    )
    profile = PausedControlProfile(**plan["control_profile"])
    last_end_ns, last_end_utc, commands = 0, utc(grant["issued_at_utc"]), set()
    for index, ((role, case), artifact) in enumerate(
        zip(schedule, recording["attempts"], strict=True)
    ):
        keys(artifact, {"path", "sha256"}, "attempt artifact")
        path = _attempt_path(root, role, case)
        require(
            artifact["path"] == (path / "attempt.json").relative_to(root).as_posix()
            and file_digest(safe_path(root, artifact["path"])) == sha256(artifact["sha256"]),
            "Attempt path or checksum differs",
        )
        attempt = keys(
            read_json(path / "attempt.json"),
            {
                "schema",
                "policy",
                "case",
                "capture_manifest_sha256",
                "task_states_sha256",
                "heartbeat_ns",
                "final_images",
                "trial",
            },
            "actual attempt",
        )
        require(
            attempt["schema"] == "physicalai.paused-physical-attempt/v1"
            and attempt["case"] == case
            and attempt["policy"] == role,
            "Attempt controller/frozen case changed",
        )
        raw = validate_dataset(
            path / "capture",
            expected_scope=scope,
            expected_manifest_sha256=attempt["capture_manifest_sha256"],
            require_live=require_live,
        )
        require(
            raw.manifest["criteria_sha256"] == plan["criteria_sha256"]
            and raw.manifest["frozen_plan_sha256"] == plan["frozen_plan_sha256"]
            and raw.episodes[0].metadata["provenance"] == result["runtime"]["provenance"],
            "Recorded raw dataset runtime/criteria/conditions changed",
        )
        states = _read_task_states(
            path / "task-states.jsonl", attempt["task_states_sha256"], profile
        )
        cameras = {}
        fields = set(FrozenCameraSample.__dataclass_fields__) - {"png"}
        for name, meta in keys(attempt["final_images"], set(CAMERAS), "final images").items():
            keys(meta, fields | {"sha256", "width", "height"}, "final camera")
            image_path = safe_path(path, f"{name}.png")
            require(image_path.stat().st_size <= 32 * 1024 * 1024, "Oversized final PNG")
            image = FrozenCameraSample(
                png=image_path.read_bytes(), **{key: meta[key] for key in fields}
            )
            require(image.metadata() == meta, "Final image pixels/metadata changed")
            cameras[name] = image
        observed = derive_trial(
            raw,
            states,
            case=case,
            role=role,
            model_sha256=expected_model(plan, role),
            profile=profile,
            final_images=cameras,
            heartbeat_ns=tuple(attempt["heartbeat_ns"]),
            destination_id=attempt["trial"]["destination_id"],
            failure_reason=attempt["trial"]["failure_reason"],
        )
        require(
            observed == attempt["trial"] == result["trials"][index],
            "Declared trial differs from actual source evidence",
        )
        _trial(observed, case, learned=role != "reference", profile=profile)
        require(
            utc(grant["issued_at_utc"])
            <= utc(states[0].captured_at_utc)
            <= max(utc(image.captured_at_utc) for image in cameras.values())
            <= utc(grant["expires_at_utc"])
            and last_end_utc <= utc(states[0].captured_at_utc)
            and last_end_ns <= states[0].monotonic_ns
            and states[0].command_id not in commands
            and observed["wall_duration_ms"] <= grant["max_episode_wall_seconds"] * 1000
            and observed["applied_action_count"] <= grant["max_episode_physics_steps"],
            "Recorded attempt exceeded the original operator wall/physics authority",
        )
        last_end_ns = max(image.monotonic_ns for image in cameras.values())
        last_end_utc = max(utc(image.captured_at_utc) for image in cameras.values())
        commands.add(states[0].command_id)


def verify_recording(root: Path, plan: dict, scope: Scope, expected_results_sha256: str) -> dict:
    path = safe_path(root, "results.json")
    require(
        file_digest(path) == sha256(expected_results_sha256), "Physical results checksum mismatch"
    )
    result = read_json(path, max_bytes=64 * 1024 * 1024)
    _verify(root, result, plan, scope, require_live=True)
    return result
