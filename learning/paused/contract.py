from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from types import MappingProxyType

from learning.common import canonical, digest, finite, integer, require, sha256, token, utc, vector
from learning.contract import (
    CAMERAS,
    AppliedControl,
    CameraSample,
    Scope,
    bounded_joints,
    png_dimensions,
    validate_joint_tracking,
)

EXECUTION_TIMING = "paused_simulation"
CONTROL_PROFILE_ID = "franka-position-hold-10hz-paused-v1"
CONTROL_PROFILE_V2_ID = "franka-position-hold-10hz-paused-v2"
_PROFILE_STEP_LIMITS = {CONTROL_PROFILE_ID: 1800, CONTROL_PROFILE_V2_ID: 3600}
RAW_SCHEMA = "physicalai.demonstrations/v3"
CONVERSION_SCHEMA = "physicalai.lerobot-conversion/v3"
MODEL_SCHEMA = "physicalai.smolvla-checkpoint/v2"
REQUEST_SCHEMA = "physicalai.smolvla-request/v2"
RESPONSE_SCHEMA = "physicalai.smolvla-response/v2"
CONTEXT_SCHEMA = "physicalai.paused-control-context/v1"
OBSERVATION_SCHEMA = "physicalai.frozen-policy-observation/v1"
COMMAND_SCHEMA = "physicalai.paused-joint-command/v1"
PUBLICATION_SCHEMA = "physicalai.initial-frozen-publication/v1"
PLAN_SCHEMA = "physicalai.smolvla-paired-plan/v2"
RESULT_SCHEMA = "physicalai.smolvla-paired-results/v3"
REPORT_SCHEMA = "physicalai.smolvla-paired-report/v2"
BOOTSTRAP_PLAN_SCHEMA = "physicalai.smolvla-bootstrap-plan/v2"
BOOTSTRAP_RESULT_SCHEMA = "physicalai.smolvla-bootstrap-results/v3"
BOOTSTRAP_REPORT_SCHEMA = "physicalai.smolvla-bootstrap-report/v2"
GRANT_SCHEMA = "physicalai.operator-rollout-grant/v2"
RECORDING_SCHEMA = "physicalai.physical-rollout-recording/v2"


def protocol_schemas() -> dict[str, str]:
    return {
        "raw": RAW_SCHEMA,
        "conversion": CONVERSION_SCHEMA,
        "model": MODEL_SCHEMA,
        "request": REQUEST_SCHEMA,
        "response": RESPONSE_SCHEMA,
        "context": CONTEXT_SCHEMA,
        "observation": OBSERVATION_SCHEMA,
        "command": COMMAND_SCHEMA,
        "initial_publication": PUBLICATION_SCHEMA,
        "paired_plan": PLAN_SCHEMA,
        "paired_results": RESULT_SCHEMA,
        "paired_report": REPORT_SCHEMA,
        "bootstrap_plan": BOOTSTRAP_PLAN_SCHEMA,
        "bootstrap_results": BOOTSTRAP_RESULT_SCHEMA,
        "bootstrap_report": BOOTSTRAP_REPORT_SCHEMA,
        "grant": GRANT_SCHEMA,
        "recording": RECORDING_SCHEMA,
    }


def _mode(execution_timing: str, real_time_admission: bool) -> None:
    require(
        execution_timing == EXECUTION_TIMING and real_time_admission is False,
        "Paused simulation cannot claim or inherit real-time admission",
    )


def _simulation_time(numerator: int, denominator: int) -> Fraction:
    return Fraction(
        integer(numerator, "native simulation time numerator"),
        integer(denominator, "native simulation time denominator", 1),
    )


@dataclass(frozen=True)
class PausedControlProfile:
    servo_profile_sha256: str
    profile_id: str = CONTROL_PROFILE_ID
    execution_timing: str = EXECUTION_TIMING
    real_time_admission: bool = False
    physics_hz: int = 60
    control_sim_hz: int = 10
    hold_steps: int = 6
    velocity_target_mode: str = "zero"
    gravity_compensation: str = "physx_measured_arm_only"
    max_observation_wall_ms: int = 2000
    max_policy_wall_ms: int = 2000
    max_hold_wall_ms: int = 2000
    max_interval_wall_ms: int = 5000
    max_episode_wall_ms: int = 600000
    max_simulation_steps: int = 1800
    max_heartbeat_wall_ms: int = 2000
    max_cartesian_speed_m_s: float = 0.2
    max_axis_goal_error_m: float = 0.04

    def validate(self) -> None:
        sha256(self.servo_profile_sha256, "new paused servo fingerprint")
        _mode(self.execution_timing, self.real_time_admission)
        require(
            isinstance(self.profile_id, str)
            and self.profile_id in _PROFILE_STEP_LIMITS
            and self.velocity_target_mode == "zero"
            and self.gravity_compensation == "physx_measured_arm_only",
            "Wrong paused-simulation profile or held-control semantics",
        )
        expected = {
            "physics_hz": 60,
            "control_sim_hz": 10,
            "hold_steps": 6,
            "max_observation_wall_ms": 2000,
            "max_policy_wall_ms": 2000,
            "max_hold_wall_ms": 2000,
            "max_interval_wall_ms": 5000,
            "max_episode_wall_ms": 600000,
            "max_simulation_steps": _PROFILE_STEP_LIMITS[self.profile_id],
            "max_heartbeat_wall_ms": 2000,
        }
        for name, exact in expected.items():
            integer(getattr(self, name), name, exact, exact)
        require(
            finite(self.max_cartesian_speed_m_s, "physical speed") == 0.2
            and finite(self.max_axis_goal_error_m, "goal tolerance") == 0.04,
            "Paused simulation does not relax physical safety or goal accuracy",
        )

    @property
    def sha256(self) -> str:
        self.validate()
        return digest(canonical(asdict(self)))

    @property
    def max_frames(self) -> int:
        self.validate()
        return self.max_simulation_steps // self.hold_steps


@dataclass(frozen=True)
class FrozenCameraSample(CameraSample):
    captured_at_utc: str
    simulation_time_numerator: int
    simulation_time_denominator: int

    @property
    def simulation_time(self) -> Fraction:
        return _simulation_time(self.simulation_time_numerator, self.simulation_time_denominator)

    def metadata(self) -> dict:
        width, height = png_dimensions(self.png)
        return {
            "sha256": digest(self.png),
            "width": width,
            "height": height,
            "rendering_frame": self.rendering_frame,
            "physics_step": self.physics_step,
            "monotonic_ns": self.monotonic_ns,
            "captured_at_utc": self.captured_at_utc,
            "simulation_time_numerator": self.simulation_time_numerator,
            "simulation_time_denominator": self.simulation_time_denominator,
        }


@dataclass(frozen=True)
class InitialFrozenPublication:
    publication_record_id: str
    publication_record_sha256: str
    capture_sha256: str
    freeze_established_ns: int
    published_at_utc: str
    published_monotonic_ns: int
    age_at_observation_start_ns: int
    schema: str = PUBLICATION_SCHEMA
    origin: str = "original_current_frozen_publication"

    def validate(self, observation: FrozenPolicyObservation) -> None:
        token(self.publication_record_id, "trusted runtime publication record")
        sha256(self.publication_record_sha256, "runtime publication record checksum")
        sha256(self.capture_sha256, "original sensor capture checksum")
        integer(self.freeze_established_ns, "actual prepublication freeze", 1)
        integer(self.published_monotonic_ns, "original publication timestamp", 1)
        integer(self.age_at_observation_start_ns, "original publication age", 0, 2_000_000_000)
        require(
            self.schema == PUBLICATION_SCHEMA
            and self.origin == "original_current_frozen_publication"
            and observation.control_tick == 0,
            "Original-publication proof is allowed only for the first control interval",
        )
        require(
            self.published_monotonic_ns == observation.monotonic_ns
            and self.published_at_utc == observation.captured_at_utc
            and self.capture_sha256 == observation.capture_sha256
            and self.age_at_observation_start_ns
            == observation.observation_started_ns - self.published_monotonic_ns,
            "Original publication was restamped, rebound or outside its original age",
        )
        require(
            self.freeze_established_ns
            <= min(
                observation.joint_sample_ns,
                *(image.monotonic_ns for image in observation.images.values()),
            ),
            "Actual freeze must precede original camera/joint publication",
        )


@dataclass(frozen=True)
class FrozenPolicyObservation:
    scope: Scope
    environment_id: str
    revision: str
    episode_id: str
    epoch: str
    captured_at_utc: str
    monotonic_ns: int
    physics_step: int
    joint_positions: tuple[float, ...]
    images: Mapping[str, FrozenCameraSample]
    freeze_id: str
    state_revision: int
    control_tick: int
    control_profile_sha256: str
    observation_started_ns: int
    joint_sample_ns: int
    simulation_time_numerator: int
    simulation_time_denominator: int
    initial_publication: InitialFrozenPublication | None = None
    observation_completed_ns: int | None = None
    schema: str = OBSERVATION_SCHEMA
    execution_timing: str = EXECUTION_TIMING
    real_time_admission: bool = False

    def __post_init__(self) -> None:
        require(
            self.initial_publication is None
            or isinstance(self.initial_publication, InitialFrozenPublication),
            "Expected typed initial publication evidence",
        )
        require(isinstance(self.images, Mapping), "Frozen camera samples must be a mapping")
        require(
            all(isinstance(item, FrozenCameraSample) for item in self.images.values()),
            "New frozen camera identity/timestamps are required; no v2 relabel",
        )
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))
        require(isinstance(self.joint_positions, (tuple, list)), "Expected actual joint vector")
        object.__setattr__(self, "joint_positions", tuple(self.joint_positions))

    @property
    def simulation_time(self) -> Fraction:
        return _simulation_time(self.simulation_time_numerator, self.simulation_time_denominator)

    def metadata(self) -> dict:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in ("images", "scope", "initial_publication")
            },
            "scope": asdict(self.scope),
            "images": {name: item.metadata() for name, item in self.images.items()},
            "initial_publication": asdict(self.initial_publication)
            if self.initial_publication is not None
            else None,
        }

    @property
    def capture_sha256(self) -> str:
        metadata = self.metadata()
        for name in (
            "episode_id",
            "freeze_id",
            "control_tick",
            "observation_started_ns",
            "observation_completed_ns",
            "initial_publication",
        ):
            metadata.pop(name)
        return digest(canonical(metadata))

    @property
    def ready_ns(self) -> int:
        if self.initial_publication is None:
            return self.monotonic_ns
        return integer(self.observation_completed_ns, "new observation work completion", 1)

    @property
    def sha256(self) -> str:
        return digest(canonical(self.metadata()))

    def validate(self, profile: PausedControlProfile, *, now_ns: int) -> None:
        profile.validate()
        self.scope.validate()
        _mode(self.execution_timing, self.real_time_admission)
        require(self.schema == OBSERVATION_SCHEMA, "Wrong frozen observation schema")
        for value in (self.environment_id, self.episode_id, self.epoch, self.freeze_id):
            token(value, "frozen observation binding")
        sha256(self.revision, "environment revision")
        require(self.control_profile_sha256 == profile.sha256, "Wrong paused observation profile")
        captured = utc(self.captured_at_utc)
        for name in ("state_revision", "physics_step", "control_tick"):
            integer(getattr(self, name), name)
        for name in ("monotonic_ns", "observation_started_ns", "joint_sample_ns"):
            integer(getattr(self, name), name, 1)
        integer(now_ns, "current monotonic timestamp", 1)
        if self.initial_publication is None:
            require(
                self.observation_completed_ns is None,
                "Only the first original-publication proof separates capture and new work times",
            )
            sample_start = self.observation_started_ns
        else:
            self.initial_publication.validate(self)
            sample_start = self.initial_publication.freeze_established_ns
        require(
            sample_start <= self.joint_sample_ns <= self.monotonic_ns <= self.ready_ns <= now_ns
            and self.observation_started_ns <= self.ready_ns
            and self.ready_ns - self.observation_started_ns
            <= profile.max_observation_wall_ms * 1_000_000,
            "Future samples or expired paused observation phase",
        )
        bounded_joints(self.joint_positions, "actual frozen joints")
        require(set(self.images) == set(CAMERAS), "Both actual cameras are required")
        dimensions = set()
        for image in self.images.values():
            dimensions.add(png_dimensions(image.png))
            integer(image.rendering_frame, "actual rendering frame")
            integer(image.monotonic_ns, "original camera timestamp", 1)
            integer(image.physics_step, "camera physics step")
            require(
                image.physics_step == self.physics_step
                and sample_start <= image.monotonic_ns <= self.monotonic_ns
                and utc(image.captured_at_utc) <= captured
                and abs(image.simulation_time - self.simulation_time)
                <= Fraction(1, 2 * profile.physics_hz),
                "Future/stale camera or changed physics during frozen observation",
            )
        require(len(dimensions) == 1, "Camera dimensions differ")


@dataclass(frozen=True)
class PausedControlContext:
    scope: Scope
    environment_id: str
    revision: str
    episode_id: str
    epoch: str
    command_id: str
    destination_id: str
    approved: bool
    active: bool
    model_sha256: str
    control_profile_sha256: str
    freeze_id: str
    observation_sha256: str
    state_revision: int
    episode_started_ns: int
    episode_initial_physics_step: int
    simulation_step_deadline: int
    wall_deadline_ns: int
    interval_started_ns: int
    interval_deadline_ns: int
    operation_started_ns: int
    operation_deadline_ns: int
    schema: str = CONTEXT_SCHEMA
    execution_timing: str = EXECUTION_TIMING
    real_time_admission: bool = False

    @property
    def sha256(self) -> str:
        return digest(canonical(asdict(self)))

    def validate(
        self,
        profile: PausedControlProfile,
        observation: FrozenPolicyObservation,
        *,
        now_ns: int,
    ) -> None:
        observation.validate(profile, now_ns=now_ns)
        _mode(self.execution_timing, self.real_time_admission)
        require(
            self.schema == CONTEXT_SCHEMA and self.approved is True and self.active is True,
            "Paused controller authority is not active",
        )
        for value in (self.command_id, self.destination_id):
            token(value, "controller binding")
        integer(self.state_revision, "actual state revision")
        sha256(self.model_sha256, "actual trained model")
        require(
            (
                self.scope,
                self.environment_id,
                self.revision,
                self.episode_id,
                self.epoch,
                self.control_profile_sha256,
                self.freeze_id,
                self.state_revision,
                self.observation_sha256,
            )
            == (
                observation.scope,
                observation.environment_id,
                observation.revision,
                observation.episode_id,
                observation.epoch,
                profile.sha256,
                observation.freeze_id,
                observation.state_revision,
                observation.sha256,
            ),
            "Frozen observation, scene, owner, model profile or authority binding changed",
        )
        for name in (
            "episode_started_ns",
            "wall_deadline_ns",
            "interval_started_ns",
            "interval_deadline_ns",
            "operation_started_ns",
            "operation_deadline_ns",
        ):
            integer(getattr(self, name), name, 1)
        require(
            self.episode_started_ns
            <= self.interval_started_ns
            == observation.observation_started_ns
            and self.interval_started_ns <= observation.ready_ns <= self.operation_started_ns
            and self.operation_started_ns
            <= now_ns
            < self.operation_deadline_ns
            <= self.interval_deadline_ns
            <= self.wall_deadline_ns,
            "Paused absolute wall deadline expired, renewed or detached from observation",
        )
        require(
            self.wall_deadline_ns - self.episode_started_ns
            <= profile.max_episode_wall_ms * 1_000_000
            and self.interval_deadline_ns - self.interval_started_ns
            <= profile.max_interval_wall_ms * 1_000_000
            and self.operation_deadline_ns - self.operation_started_ns
            <= profile.max_policy_wall_ms * 1_000_000,
            "Paused operation extends the original fixed wall budget",
        )
        integer(self.episode_initial_physics_step, "initial physics step")
        integer(self.simulation_step_deadline, "simulation step deadline")
        steps = self.simulation_step_deadline - self.episode_initial_physics_step
        require(
            0 < steps <= profile.max_simulation_steps
            and steps % profile.hold_steps == 0
            and observation.physics_step
            == self.episode_initial_physics_step + observation.control_tick * profile.hold_steps
            and observation.physics_step + profile.hold_steps <= self.simulation_step_deadline,
            "Simulation step budget/cadence differs from the actual frozen interval",
        )


@dataclass(frozen=True)
class PausedJointCommand:
    targets: tuple[float, ...]
    physics_step: int
    hold_steps: int
    expires_at_monotonic_ns: int
    model_sha256: str
    inference_latency_ms: float
    freeze_id: str
    observation_sha256: str
    state_revision: int
    control_profile_sha256: str
    context_sha256: str
    schema: str = COMMAND_SCHEMA
    execution_timing: str = EXECUTION_TIMING
    real_time_admission: bool = False


@dataclass(frozen=True)
class PausedFrameSample:
    observation: FrozenPolicyObservation
    commanded_joint_targets: tuple[float, ...]
    applied_controls: tuple[AppliedControl, ...]
    interval_deadline_ns: int
    hold_started_ns: int
    hold_deadline_ns: int
    policy_started_ns: int | None = None
    policy_finished_ns: int | None = None
    terminated: bool = False
    truncated: bool = False

    def validate(
        self, profile: PausedControlProfile, *, previous: PausedFrameSample | None = None
    ) -> None:
        observation = self.observation
        observation.validate(profile, now_ns=observation.ready_ns)
        require(len(self.applied_controls) == profile.hold_steps, "Incomplete actual six-tick hold")
        validate_joint_tracking(
            self.commanded_joint_targets,
            observation.joint_positions,
            previous.commanded_joint_targets if previous else None,
            fps=profile.control_sim_hz,
        )
        for name in ("interval_deadline_ns", "hold_started_ns", "hold_deadline_ns"):
            integer(getattr(self, name), name, 1)
        ready = observation.ready_ns
        require(
            (self.policy_started_ns is None) == (self.policy_finished_ns is None),
            "Incomplete actual policy timing evidence",
        )
        if self.policy_started_ns is not None:
            integer(self.policy_started_ns, "policy start", 1)
            integer(self.policy_finished_ns, "policy finish", 1)
            require(
                ready <= self.policy_started_ns <= self.policy_finished_ns
                and self.policy_finished_ns - self.policy_started_ns
                <= profile.max_policy_wall_ms * 1_000_000,
                "Expired/reordered actual policy phase",
            )
            ready = self.policy_finished_ns
        require(
            ready <= self.hold_started_ns < self.hold_deadline_ns <= self.interval_deadline_ns
            and self.hold_deadline_ns - self.hold_started_ns <= profile.max_hold_wall_ms * 1_000_000
            and self.interval_deadline_ns - observation.observation_started_ns
            <= profile.max_interval_wall_ms * 1_000_000,
            "Hold/interval deadline renewed or outside fixed wall budgets",
        )
        timestamp = self.hold_started_ns
        for offset, control in enumerate(self.applied_controls, 1):
            require(
                control.physics_step == observation.physics_step + offset
                and timestamp < control.monotonic_ns <= self.hold_deadline_ns,
                "Missing/duplicate actual physics tick or expired held-tick phase",
            )
            integer(control.physics_step, "applied physics step")
            integer(control.monotonic_ns, "actual held-tick timestamp", 1)
            timestamp = control.monotonic_ns
            require(
                bounded_joints(control.commanded_joint_targets, "actual held targets")
                == bounded_joints(self.commanded_joint_targets, "requested target")
                and vector(control.commanded_joint_velocities, 9, "actual velocities")
                == (0.0,) * 9,
                "Changed target or nonzero velocity inside held control",
            )
            gravity = vector(control.gravity_efforts, 9, "actual gravity compensation")
            require(gravity[7:] == (0.0, 0.0), "Gravity compensation must remain arm-only")
        require(
            type(self.terminated) is bool
            and type(self.truncated) is bool
            and not (self.terminated and self.truncated),
            "Invalid/ambiguous episode termination",
        )
        if previous is not None:
            old = previous.observation
            require(not (previous.terminated or previous.truncated), "Data after episode end")
            require(
                (
                    observation.scope,
                    observation.environment_id,
                    observation.revision,
                    observation.episode_id,
                    observation.epoch,
                )
                == (old.scope, old.environment_id, old.revision, old.episode_id, old.epoch)
                and observation.physics_step == old.physics_step + profile.hold_steps
                and observation.control_tick == old.control_tick + 1
                and observation.state_revision > old.state_revision
                and observation.freeze_id != old.freeze_id
                and observation.observation_started_ns >= previous.applied_controls[-1].monotonic_ns
                and utc(observation.captured_at_utc) > utc(old.captured_at_utc)
                and abs(observation.simulation_time - old.simulation_time - Fraction(1, 10))
                <= Fraction(1, 1_000_000_000),
                "Reused freeze, skipped physics or nonmonotonic wall/simulation capture",
            )
            for name, image in observation.images.items():
                prior = old.images[name]
                require(
                    image.rendering_frame > prior.rendering_frame
                    and image.monotonic_ns > prior.monotonic_ns
                    and utc(image.captured_at_utc) > utc(prior.captured_at_utc),
                    "Repeated/nonmonotonic actual camera frame",
                )
