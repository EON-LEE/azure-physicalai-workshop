"""Imported only after SimulationApp is initialized in the pinned Isaac container."""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from io import BytesIO
from math import dist, isfinite
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (
    RMPFlowController,
)
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from apps.api.models import MotionPhase
from learning.capture import resolve_joint_targets
from learning.common import finite, integer, require, vector
from learning.contract import (
    DEFAULT_JOINT_VELOCITY_LIMITS,
    JOINT_LOWER,
    JOINT_NAMES,
    JOINT_UNITS,
    JOINT_UPPER,
    AppliedControl,
    CameraSample,
    FrameSample,
    Scope,
)
from learning.inference import PolicyObservation
from learning.paused import FrozenCameraSample, FrozenPolicyObservation, InitialFrozenPublication
from simulation.asset_references import validate_usd_bundle
from simulation.camera_observation import (
    camera_evidence,
    observation_barrier,
    paused_observation_barrier,
    render_identity,
)
from simulation.control import (
    HoldCapture,
    TaskWatchdog,
    check_measured_motion,
    validate_position_target,
)
from simulation.core import SimulationCore
from simulation.extensions import SceneSpec, can_reset_in_place
from simulation.motion import (
    GripperRamp,
    InspectionRoute,
    arm_gravity_efforts,
    move_toward,
    rotate_toward,
)
from simulation.paused_control import FrozenPhysicsState
from simulation.paused_gripper_servo import DriveReadback, PausedGripperServo
from simulation.paused_observation import PausedPublication, PausedPublicationCache
from simulation.paused_teacher import is_pick_place_task
from simulation.physics_scheduling import physics_scheduling_readback, require_control_scheduling
from simulation.policy_executor import PolicyExecutor
from simulation.reference_targets import (
    GRASP_CONTACT_PHASES,
    MAX_REFERENCE_TRACE_INTERVALS,
    REFERENCE_LIMIT_FRACTION,
    plan_reference_targets,
    reference_contact_point,
    reference_gripper_targets,
    reference_tcp_target,
    reference_tracking_violations,
    verify_grasp_calibration_asset,
    verify_paused_franka_asset,
)
from simulation.runtime_contracts import PolicyCommand, TeachingStart


class IsaacWorkcell:
    dt = 1 / 60

    def __init__(self) -> None:
        self.world = None
        self.cameras = {}
        self.last_render_frame = {}
        self.empty_frames = {}
        self.controller = None
        self.target = None
        self.phase = 0
        self.steps = 0
        self.last_effector_position = None
        self.recording = None
        self.issued_targets = None
        self.route = None
        self.orientation_target = None
        self.initial_part_position = None
        self.grasp_verified = False
        self.peak_tcp_speed = 0.0
        self.control_mode = "reference"
        self.control_done = True
        self.control_succeeded = False
        self.control_core = None
        self.control_binding = None
        self.control_profile = None
        self.policy_executor = None
        self.hold_capture = None
        self.held_targets = None
        self.held_sequence = None
        self.held_expires_ns = 0
        self.hold_offset = 0
        self.policy_command = None
        self.finish_requested = False
        self.actuation_guard = None
        self.warmup_steps = 0
        self.reset_initial_position = None
        self.render_monotonic_ns = 0
        self.kinematics = None
        self.articulation_kinematics = None
        self.camera_timebases = {}
        self.camera_observation_metadata = None
        self.control_warmup_timings = []
        self.physics_scheduling = {}
        self.scene_epoch: UUID | None = None
        self.paused_driver = None
        self.parked_state: FrozenPhysicsState | None = None
        self.paused_scene_core = None
        self.paused_publications = PausedPublicationCache()
        self.reference_target_diagnostic: dict | None = None
        self._reference_target_trace: list[dict] = []
        self._paused_camera_diagnostic: dict = {}
        self._gripper_asset_evidence: dict = {}
        self._grasp_frame_binding: dict | None = None
        self._paused_gripper_servo: PausedGripperServo | None = None
        self._paused_gripper_state_evidence: dict = {}

    def load(self, spec: SceneSpec) -> None:
        self._validate_asset_bundle()
        self.stop()
        self.paused_driver = None
        self.reference_target_diagnostic = None
        self._reference_target_trace = []
        self._paused_camera_diagnostic = {}
        self._gripper_asset_evidence = {}
        self._grasp_frame_binding = None
        self.parked_state = None
        self.paused_publications = PausedPublicationCache()
        if self.world is not None and can_reset_in_place(self.spec, spec):
            self.spec = spec
            self.world.reset(soft=True)
            self._prepare_episode()
            return
        for camera in self.cameras.values():
            camera.destroy()
        self.cameras.clear()
        self.last_render_frame.clear()
        self.empty_frames.clear()
        self.camera_timebases.clear()
        if self.world is not None:
            self.world.stop()
            self.world.clear()
            World.clear_instance()
        create_new_stage()
        self.spec = spec
        self.world = World(stage_units_in_meters=1, physics_dt=self.dt, rendering_dt=self.dt)
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Floor",
                name="floor",
                position=np.array([0, 0, -0.025]),
                scale=np.array([3, 3, 0.05]),
                color=np.array([0.1, 0.13, 0.18]),
            )
        )
        stage = self.world.stage
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1500)
        station_colors = {
            "source": spec.platform_color,
            "inspection": (0.8, 0.5, 0.12),
            "accepted": (0.12, 0.5, 0.25),
            "rejected": (0.7, 0.12, 0.16),
        }
        for index, station in enumerate(spec.stations):
            if station.role not in station_colors:
                raise ValueError(f"Unsupported station role: {station.role}")
            x, y, z = station.position
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Station{index}",
                    name=f"station-{index}",
                    position=np.array([x, y, z - 0.045]),
                    scale=np.array([0.16, 0.16, 0.04]),
                    color=np.array(station_colors[station.role]),
                )
            )
        self.robot = self.world.scene.add(
            Franka(
                prim_path="/World/Robot",
                name="reference-arm",
                usd_path=os.environ["FRANKA_USD_PATH"],
                gripper_open_position=np.array([0.04, 0.04]),
            )
        )
        self.part = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Part",
                name="part-001",
                position=np.array(spec.part_position),
                scale=np.array([0.05, 0.05, 0.05]),
                mass=0.05,
                color=np.array([0.15, 0.7, 0.45]),
                physics_material=PhysicsMaterial(
                    prim_path="/World/PartContact",
                    static_friction=0.8,
                    dynamic_friction=0.6,
                    restitution=0.0,
                ),
            )
        )
        self._create_defect()
        source = spec.station(spec.source_id).position
        for name, eye, target, up in (
            ("overview", (1.3, 1.15, 1.2), (0.25, 0, 0.28), (0, 0, 1)),
            ("inspection", (source[0], source[1], source[2] + 0.48), source, (0, 1, 0)),
        ):
            camera = Camera(
                prim_path=f"/World/{name}",
                name=name,
                frequency=60 if spec.record_demonstration else 10,
                resolution=(320, 320) if spec.record_demonstration else (960, 540),
            )
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
            quaternion = view.GetInverse().ExtractRotationQuat()
            orientation = np.array([quaternion.GetReal(), *quaternion.GetImaginary()])
            camera.set_world_pose(
                position=np.array(eye), orientation=orientation, camera_axes="usd"
            )
            camera.set_clipping_range(0.01, 100)
            # Camera lengths use stage metres, not millimetres.
            camera.set_focal_length(0.018 if name == "overview" else 0.028)
            camera.set_horizontal_aperture(0.020955)
            camera.set_vertical_aperture(0.020955 * 540 / 960)
            self.cameras[name] = camera
        self.world.reset()
        self.dynamics = Articulation(prim_paths_expr="/World/Robot", name="reference-arm-dynamics")
        self.dynamics.initialize()
        for camera in self.cameras.values():
            camera.initialize()
            if spec.learning_execution is not None:
                camera.add_rgb_to_frame()
        self._prepare_episode()

    def _create_defect(self) -> None:
        stage = self.world.stage
        self.defect = UsdGeom.Cube.Define(stage, "/World/Part/SurfaceDefect")
        self.defect.CreateSizeAttr(1)
        self.defect.CreateDisplayColorAttr([Gf.Vec3f(0.015, 0.015, 0.015)])
        transform = UsdGeom.Xformable(self.defect.GetPrim())
        transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.501))
        transform.AddScaleOp().Set(Gf.Vec3f(0.08, 0.8, 0.01))
        material = UsdShade.Material.Define(stage, "/World/DefectMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/DefectMaterial/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.015, 0.015, 0.015)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        parent_binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Part"))
        UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
            parent_binding.GetDirectBindingRel(), UsdShade.Tokens.weakerThanDescendants
        )
        UsdShade.MaterialBindingAPI.Apply(self.defect.GetPrim()).Bind(material)
        bound, _ = UsdShade.MaterialBindingAPI(self.defect.GetPrim()).ComputeBoundMaterial()
        if bound.GetPath() != material.GetPath():
            raise RuntimeError("The reference surface-defect material is not visibly bound.")

    def _prepare_episode(self) -> None:
        self._restore_paused_gripper_servo()
        self.control_mode = "reference"
        self.control_done = True
        self.actuation_guard = None
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=self.dt)
        self._configure_cameras()
        self._place_part_at_reset()
        self.defect.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if self.spec.defective else UsdGeom.Tokens.invisible
        )
        self.last_render_frame.clear()
        self.empty_frames.clear()
        self.camera_timebases.clear()
        self.robot.gripper.set_joint_positions(self.robot.gripper.joint_opened_positions)
        self.robot.set_joint_velocities(np.zeros(9))
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=self.robot.get_joint_positions(), joint_velocities=np.zeros(9)
            )
        )
        self.world.play()
        for camera in self.cameras.values():
            camera.resume()
        self.steps = 0
        self.target = None
        self.route = None
        self.last_effector_position = None
        for _ in range(18):
            self._compensate_gravity()
            self.world.step(render=True)
            self.render_monotonic_ns = time.monotonic_ns()
            self.steps += 1
        self.reset_initial_position = self.position()
        if (
            self.spec.initial_part_position is not None
            and dist(self.reset_initial_position, self.spec.part_position) > 0.001
        ):
            raise RuntimeError(
                "The learning part did not settle within its pinned 1 mm start pose."
            )

    def _place_part_at_reset(self) -> None:
        if self.controller is not None or (
            self.control_mode != "reference" and not self.control_done
        ):
            raise RuntimeError("Part placement is allowed only during a stopped scene reset.")
        self.part.set_world_pose(position=np.array(self.spec.part_position))
        self.part.set_linear_velocity(np.zeros(3))
        self.part.set_angular_velocity(np.zeros(3))

    def initial_state_evidence(self) -> dict:
        if self.reset_initial_position is None or self.spec.scene_builder_sha256 is None:
            raise RuntimeError(
                "Measured initial pose and reviewed builder evidence are unavailable."
            )
        return {
            "observed_initial_pose_m": self.reset_initial_position,
            "scene_builder_sha256": self.spec.scene_builder_sha256,
        }

    @staticmethod
    def _validate_asset_bundle() -> None:
        validate_usd_bundle(
            Path(os.environ["FRANKA_ASSET_ROOT"]), Path(os.environ["FRANKA_USD_PATH"])
        )

    def start(self, target_id: str, recording=None) -> None:
        self._restore_paused_gripper_servo()
        self.control_mode = "reference"
        self.control_done = True
        self.policy_executor = None
        self.hold_capture = None
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=self.dt)
        self._configure_cameras()
        self.target = target_id
        self.phase = 0
        self.controller = RMPFlowController(
            name="bounded-inspection-route", robot_articulation=self.robot
        )
        joints = self.robot.get_joint_positions()
        position, rotation = self.controller.rmp_flow.get_end_effector_pose(joints[:7])
        self.orientation_target = tuple(float(value) for value in rot_matrix_to_quat(rotation))
        self.route = InspectionRoute(
            tuple(position),
            self.position(),
            self.spec.station(self.spec.inspection_id).position,
            self.spec.station(target_id).position,
            self.spec.requested_speed * 0.65,
            self.dt,
        )
        self.initial_part_position = self.position()
        self.grasp_verified = False
        self.peak_tcp_speed = 0.0
        self.gripper_ramp = GripperRamp(tuple(joints[7:]), self.dt)
        self.last_effector_position = None
        self.recording = recording
        if recording is not None:
            hold = tuple(float(value) for value in self.robot.get_joint_positions())
            self.robot.apply_action(ArticulationAction(joint_positions=np.array(hold)))
            self.issued_targets = resolve_joint_targets(None, hold)
        self.world.play()

    def _configure_cameras(self) -> bool:
        profiled = self.control_mode != "reference"
        resolution = (320, 320) if profiled or self.spec.record_demonstration else (960, 540)
        frequency = -1 if profiled else (60 if self.spec.record_demonstration else 10)
        changed = False
        for camera in self.cameras.values():
            if tuple(camera.get_resolution()) != resolution:
                camera.set_resolution(resolution)
                changed = True
            if camera.get_frequency() != frequency:
                camera.set_frequency(frequency)
                changed = True
        if changed:
            self.last_render_frame.clear()
        return changed

    def _start_control(self, mode, command, core: SimulationCore, recording) -> None:
        self._restore_paused_gripper_servo()
        if core.control_profile is None:
            raise RuntimeError("The aligned control profile has not been enabled.")
        self._check_control_scheduling("command_start")
        self.control_mode, self.control_done = mode, False
        self.control_succeeded = False
        self.control_core = core
        self.control_binding = core.binding(command.command_id)
        self.control_profile = core.control_profile
        self.control_profile.validate()
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=0.0)
        self.target = command.target_station_id
        self.route = None
        self.controller = None
        self.policy_executor = None
        self.recording = recording
        self.hold_capture = (
            HoldCapture(recording, self.control_profile) if recording is not None else None
        )
        self.held_targets, self.held_sequence = None, None
        self.issued_targets = None
        self.hold_offset = 0
        self.finish_requested = False
        self.last_jog_sequence = 0
        kinematics = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.kinematics = LulaKinematicsSolver(**kinematics)
        self.articulation_kinematics = ArticulationKinematicsSolver(
            self.robot, self.kinematics, "right_gripper"
        )
        self.jog_goal = self._measured_tcp()
        self.jog_gripper = "hold"
        self.jog_speed = min(0.05, self.spec.requested_speed)
        _, rotation = self.articulation_kinematics.compute_end_effector_pose()
        self.orientation_target = tuple(float(value) for value in rot_matrix_to_quat(rotation))
        self.task_watchdog = TaskWatchdog(
            self.position(), self.spec.station(command.target_station_id).position
        )
        self.initial_part_position = self.position()
        self.last_effector_position = self._measured_tcp()
        self.grasp_verified = False
        self.peak_tcp_speed = 0.0
        self.warmup_steps = 6 if self._configure_cameras() else 0
        self.control_next_ns = core.clock_ns()
        self.control_timings = []
        self.world.play()

    def _step_control_physics(self, *, render: bool) -> None:
        self._check_control_scheduling("before_physics_tick")
        before_index = int(self.world.current_time_step_index)
        before_time = float(self.world.current_time)
        self.world.step(render=render, update_fabric=True)
        if (
            int(self.world.current_time_step_index) != before_index + 1
            or abs(float(self.world.current_time) - before_time - self.dt) > 1e-6
        ):
            raise RuntimeError("The control tick did not advance exactly one 60 Hz physics step.")

    def _check_control_scheduling(self, phase: str) -> None:
        import carb

        observed = physics_scheduling_readback(
            carb.settings.get_settings(), phase=phase, context=self.world.get_physics_context()
        )
        self.physics_scheduling[phase] = observed
        require_control_scheduling(observed)

    def prime_control_profile(self, on_tick=None) -> None:
        if self.controller is not None or not self.control_done or self.recording is not None:
            raise RuntimeError("Control renderer warm-up must occur before motion admission.")
        self._check_control_scheduling("scene_ready")
        previous_mode = self.control_mode
        self.control_mode = "human_teaching"
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=0.0)
        self._configure_cameras()
        self.control_warmup_timings = []
        targets = tuple(float(value) for value in self.robot.get_joint_positions())
        started = time.monotonic()
        self.world.play()
        try:
            for index in range(60):
                if time.monotonic() - started > 10:
                    raise RuntimeError(
                        "Fixed unarmed renderer warm-up exceeded its ten-second bound."
                    )
                tick = time.monotonic_ns()
                self._compensate_gravity()
                self._issue_command(targets, (0.0,) * 9)
                capture_initial = index == 59 and self.paused_scene_core is not None
                self._step_control_physics(render=not capture_initial)
                self.steps += 1
                self.render_monotonic_ns = time.monotonic_ns()
                if capture_initial:
                    publication = self._paused_publication(self.paused_scene_core)
                    self.paused_publications.record(publication)
                self.control_warmup_timings.append(
                    {
                        "index": index,
                        "physics_step": self.steps,
                        "duration_ms": (self.render_monotonic_ns - tick) / 1_000_000,
                    }
                )
                if on_tick is not None:
                    on_tick()
            self.reset_initial_position = self.position()
            if (
                self.spec.initial_part_position is not None
                and dist(self.reset_initial_position, self.spec.part_position) > 0.001
            ):
                raise RuntimeError("Unarmed warm-up changed the pinned initial part pose.")
            self._check_control_scheduling("warmup_complete")
        finally:
            self.control_mode = previous_mode

    def start_teaching(self, request: TeachingStart, core: SimulationCore, recording=None) -> None:
        self._start_control("human_teaching", request, core, recording)
        if request.demonstrator_kind == "reference_controller":
            self.jog_speed = min(0.1, self.spec.requested_speed)
        self.controller = RMPFlowController(
            name="bounded-cartesian-teaching",
            robot_articulation=self.robot,
            physics_dt=1 / self.control_profile.control_hz,
        )

    def start_learned(
        self,
        request: PolicyCommand,
        core: SimulationCore,
        executor: PolicyExecutor,
        recording=None,
    ) -> None:
        self._start_control("learned", request.command, core, recording)
        self.policy_executor = executor
        executor.start()

    def request_finish(self) -> None:
        if self.control_mode != "human_teaching" or self.control_done:
            raise RuntimeError(
                "Only an active teaching session can request a held boundary finish."
            )
        self.finish_requested = True

    def _measured_tcp(self) -> tuple[float, float, float]:
        if self.kinematics is None or self.articulation_kinematics is None:
            raise RuntimeError("The reviewed right-gripper measurement frame is unavailable.")
        base_position, base_orientation = self.robot.get_world_pose()
        self.kinematics.set_robot_base_pose(base_position, base_orientation)
        position, _ = self.articulation_kinematics.compute_end_effector_pose()
        return tuple(float(value) for value in position)

    def frozen_physics_state(self, epoch: UUID) -> FrozenPhysicsState:
        position, orientation = self.part.get_world_pose()
        return FrozenPhysicsState(
            epoch=epoch,
            physics_step=int(self.world.current_time_step_index),
            world_time=float(self.world.current_time),
            joint_positions=tuple(float(value) for value in self.robot.get_joint_positions()),
            joint_velocities=tuple(float(value) for value in self.robot.get_joint_velocities()),
            object_position=tuple(float(value) for value in position),
            object_orientation=tuple(float(value) for value in orientation),
            object_linear_velocity=tuple(float(value) for value in self.part.get_linear_velocity()),
            object_angular_velocity=tuple(
                float(value) for value in self.part.get_angular_velocity()
            ),
        )

    def render_frozen_scene(self) -> None:
        if self.scene_epoch is None:
            raise RuntimeError("An actual scene epoch is required for a frozen preview.")
        before = self.frozen_physics_state(self.scene_epoch)
        if self.parked_state is not None and before != self.parked_state:
            raise RuntimeError("The parked scene did not remain physically frozen.")
        started_ns = time.monotonic_ns()
        self.world.render()
        if time.monotonic_ns() - started_ns > 2_000_000_000:
            raise RuntimeError("Frozen preview exceeded the hard main-thread heartbeat budget.")
        if self.frozen_physics_state(self.scene_epoch) != before:
            raise RuntimeError("The preview renderer changed a frozen physical scene.")
        self.parked_state = before

    def prepare_paused_reference(self, request, core: SimulationCore) -> None:
        if request.controller != "reference_controller" or core.paused_profile is None:
            raise RuntimeError(
                "A paused learned episode cannot select the scripted reference servo."
            )
        self._prepare_paused_servo(request, core)
        if is_pick_place_task(request.task, self.spec, request.target_station_id):
            try:
                variant_sets = self.world.stage.GetPrimAtPath(self.robot.prim_path).GetVariantSets()
                self._grasp_frame_binding = verify_grasp_calibration_asset(
                    Path(os.environ.get("FRANKA_ASSET_ROOT", "")),
                    Path(os.environ.get("FRANKA_USD_PATH", "")),
                    os.environ.get("FRANKA_ASSET_SHA256", ""),
                    {
                        name: variant_sets.GetVariantSet(name).GetVariantSelection()
                        for name in ("Mesh", "Gripper")
                    },
                )
            except (AttributeError, OSError, RuntimeError) as exc:
                raise ValueError(
                    "The verified reference grasp calibration asset is unavailable."
                ) from exc
        self.controller = RMPFlowController(
            name="paused-reference-expert",
            robot_articulation=self.robot,
            physics_dt=1 / core.paused_profile.control_sim_hz,
        )
        _, rotation = self.articulation_kinematics.compute_end_effector_pose()
        self.orientation_target = tuple(float(value) for value in rot_matrix_to_quat(rotation))
        self._gripper_asset_evidence = self._read_gripper_asset_evidence()
        self.world.play()

    def prepare_paused_learned(self, request, core: SimulationCore) -> None:
        if request.controller != "learned" or request.model_sha256 is None:
            raise RuntimeError("An explicitly authorized paused learned model is required.")
        self._prepare_paused_servo(request, core)
        self.world.play()

    def _prepare_paused_servo(self, request, core: SimulationCore) -> None:
        if core.paused_profile is None:
            raise RuntimeError("A separately approved paused servo profile is required.")
        self.spec.require_paused_authority()
        self._check_control_scheduling("command_start")
        self.control_mode = "paused_simulation"
        self.control_done = False
        self.control_core = core
        self.control_binding = core.binding(request.command_id)
        self.control_profile = core.paused_profile
        self.scene_epoch = core.epoch
        self.target = request.target_station_id
        self.route = None
        self.controller = None
        self._grasp_frame_binding = None
        self._gripper_asset_evidence = {}
        self.parked_state = None
        self.policy_executor = None
        self.issued_targets = None
        self.reference_target_diagnostic = None
        self._reference_target_trace = []
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=0.0)
        self._configure_cameras()
        configuration = interface_config_loader.load_supported_lula_kinematics_solver_config(
            "Franka"
        )
        self.kinematics = LulaKinematicsSolver(**configuration)
        self.articulation_kinematics = ArticulationKinematicsSolver(
            self.robot, self.kinematics, "right_gripper"
        )
        self.last_effector_position = self._measured_tcp()
        self.task_watchdog = TaskWatchdog(self.position(), self.spec.station(self.target).position)
        self.grasp_verified = False
        self.peak_tcp_speed = 0.0
        self.paused_measured_success = False
        self._prepare_paused_gripper_servo()

    def _read_paused_drive_snapshot(self) -> DriveReadback:
        controller = self.robot.get_articulation_controller()
        kp, kd = controller.get_gains()
        return DriveReadback(
            tuple(float(value) for value in kp),
            tuple(float(value) for value in kd),
            tuple(float(value) for value in controller.get_max_efforts()),
        )

    def _read_paused_gripper_asset(self) -> dict:
        from pxr import UsdPhysics

        root = self.robot.prim_path
        variants = self.world.stage.GetPrimAtPath(root).GetVariantSets()
        identity = verify_paused_franka_asset(
            Path(os.environ.get("FRANKA_ASSET_ROOT", "")),
            Path(os.environ.get("FRANKA_USD_PATH", "")),
            os.environ.get("FRANKA_ASSET_SHA256", ""),
            {
                name: variants.GetVariantSet(name).GetVariantSelection()
                for name in ("Mesh", "Gripper")
            },
        )
        driven = self.world.stage.GetPrimAtPath(root + "/panda_hand/panda_finger_joint1")
        passive = self.world.stage.GetPrimAtPath(root + "/panda_hand/panda_finger_joint2")
        require(
            driven and passive and driven.HasAPI(UsdPhysics.DriveAPI, "linear"),
            "The actual authored linear finger drive is unavailable",
        )
        drive = UsdPhysics.DriveAPI(driven, "linear")
        values = {}
        for name, attribute in (
            ("drive_type", drive.GetTypeAttr()),
            ("authored_stiffness", drive.GetStiffnessAttr()),
            ("authored_damping", drive.GetDampingAttr()),
            ("authored_max_force", drive.GetMaxForceAttr()),
        ):
            require(
                attribute and attribute.HasAuthoredValueOpinion(),
                "Expected authored finger drive data",
            )
            values[name] = attribute.Get() if name == "drive_type" else float(attribute.Get())
        schemas = passive.GetMetadata("apiSchemas")
        require(schemas is not None, "The actual passive mimic schema is unavailable")
        schemas = list(schemas.GetAddedOrExplicitItems())
        reference = passive.GetRelationship("physxMimicJoint:rotX:referenceJoint").GetTargets()
        require(reference == [driven.GetPath()], "The actual mimic reference differs")
        return {
            **identity,
            **values,
            "joint_names": tuple(self.robot.dof_names),
            "driven_joint_index": self.robot.get_dof_index("panda_finger_joint1"),
            "driven_joint_type": driven.GetTypeName(),
            "passive_joint_index": self.robot.get_dof_index("panda_finger_joint2"),
            "passive_has_drive": any(name.startswith("PhysicsDriveAPI:") for name in schemas),
            "mimic_axis": "rotX" if "PhysxMimicJointAPI:rotX" in schemas else None,
            "mimic_gearing": float(passive.GetAttribute("physxMimicJoint:rotX:gearing").Get()),
            "mimic_reference": driven.GetName(),
        }

    def _write_paused_finger_stiffness(self, value: float) -> None:
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.timeline import get_timeline_interface

        require(
            not get_timeline_interface().is_stopped()
            and SimulationManager.get_physics_sim_view() is not None
            and self.dynamics.is_physics_handle_valid(),
            "A live non-stopped physics view is required; USD gain fallback is forbidden",
        )
        before = self.frozen_physics_state(self.scene_epoch)
        self.dynamics.set_gains(
            kps=np.array([[value]]),
            indices=np.array([0]),
            joint_indices=np.array([7]),
            save_to_usd=False,
        )
        if self.frozen_physics_state(self.scene_epoch) != before:
            raise RuntimeError("The native gain change altered the frozen physical state.")

    def _prepare_paused_gripper_servo(self) -> None:
        if self._paused_gripper_servo is None:
            self._paused_gripper_servo = PausedGripperServo(
                read=self._read_paused_drive_snapshot,
                write_stiffness=self._write_paused_finger_stiffness,
            )

        def calibrate():
            before = self.frozen_physics_state(self.scene_epoch)
            self._paused_gripper_state_evidence = {
                "before": {**asdict(before), "epoch": str(before.epoch)},
                "first_publication_precedes_calibration": self.paused_publications.publication
                is not None,
            }
            self._paused_gripper_servo.apply(
                self.control_binding, self._read_paused_gripper_asset()
            )
            after = self.frozen_physics_state(self.scene_epoch)
            self._paused_gripper_state_evidence["after"] = {
                **asdict(after),
                "epoch": str(after.epoch),
            }
            if after != before:
                raise RuntimeError("Paused gain calibration changed the frozen physical state.")

        self.control_core.apply_guarded(self.control_binding, calibrate)

    def _restore_paused_gripper_servo(self) -> None:
        if self._paused_gripper_servo is not None:
            self._paused_gripper_servo.restore()

    def paused_gripper_servo_evidence(self) -> dict | None:
        if self._paused_gripper_servo is None:
            return None
        return {
            **self._paused_gripper_servo.evidence(),
            "frozen_state": deepcopy(self._paused_gripper_state_evidence),
        }

    def paused_reference_route_point(self, tcp, phase) -> tuple[float, float, float]:
        if self._grasp_frame_binding is None or phase not in GRASP_CONTACT_PHASES:
            return tcp
        base_position, base_orientation = self.robot.get_world_pose()
        self.kinematics.set_robot_base_pose(base_position, base_orientation)
        _, rotation = self.articulation_kinematics.compute_end_effector_pose()
        orientation = tuple(float(value) for value in rot_matrix_to_quat(rotation))
        return reference_contact_point(tcp, orientation)

    @staticmethod
    def _drive_field_readback(values, name: str) -> dict:
        require(values is not None and len(values) == 9, "Expected nine actual drive readouts")
        readback, issues = [], []
        for index, raw in enumerate(values):
            require(not isinstance(raw, (bool, str, bytes)), "A drive readout is not numeric")
            value = float(raw)
            readback.append(value if isfinite(value) else str(value))
            classification = (
                "nonfinite"
                if not isfinite(value)
                else "nonpositive_limit"
                if name == "max_effort" and value <= 0
                else "negative_gain"
                if name != "max_effort" and value < 0
                else None
            )
            if classification is not None:
                issues.append(
                    {
                        "joint_index": index,
                        "joint_name": JOINT_NAMES[index],
                        "classification": classification,
                    }
                )
        return {
            "status": "invalid" if issues else "available",
            "readback": tuple(readback),
            "issues": issues,
        }

    def _read_gripper_drive_evidence(self) -> dict:
        drives = {"source": "loaded_articulation_controller", "fields": {}}
        try:
            controller = self.robot.get_articulation_controller()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            return {**drives, "status": "unavailable", "error": str(exc)[:512]}
        try:
            kp, kd = controller.get_gains()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            for name in ("stiffness", "damping"):
                drives["fields"][name] = {"status": "unavailable", "error": str(exc)[:512]}
        else:
            for name, values in (("stiffness", kp), ("damping", kd)):
                try:
                    drives["fields"][name] = self._drive_field_readback(values, name)
                except (ValueError, TypeError, OverflowError) as exc:
                    drives["fields"][name] = {"status": "unavailable", "error": str(exc)[:512]}
        try:
            drives["fields"]["max_effort"] = self._drive_field_readback(
                controller.get_max_efforts(), "max_effort"
            )
        except (AttributeError, RuntimeError, ValueError, TypeError, OverflowError) as exc:
            drives["fields"]["max_effort"] = {"status": "unavailable", "error": str(exc)[:512]}
        valid = 0
        for name, field in drives["fields"].items():
            if field["status"] == "available":
                drives[name] = field.pop("readback")
                field.pop("issues")
                valid += 1
        drives["status"] = "available" if valid == 3 else "partial" if valid else "unavailable"
        return drives

    def _read_gripper_asset_evidence(self) -> dict:
        evidence = {
            "contact_forces_measured": False,
            "drives": self._read_gripper_drive_evidence(),
            "links": {},
        }
        for name in ("panda_leftfinger", "panda_rightfinger"):
            try:
                from pxr import Usd, UsdPhysics

                root = self.world.stage.GetPrimAtPath(f"{self.robot.prim_path}/{name}")
                if not root:
                    raise RuntimeError("The loaded finger asset prim is unavailable.")
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                bounds = []
                for index, prim in enumerate(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
                    if index >= 64:
                        raise RuntimeError(
                            "The finger collision hierarchy exceeds the diagnostic cap."
                        )
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        require(len(bounds) < 8, "The finger collision count exceeds its cap")
                        box = cache.ComputeRelativeBound(prim, root).ComputeAlignedRange()
                        lower = vector(tuple(float(v) for v in box.GetMin()), 3, "collision lower")
                        upper = vector(tuple(float(v) for v in box.GetMax()), 3, "collision upper")
                        require(
                            all(a <= b for a, b in zip(lower, upper, strict=True)),
                            "Invalid collision extent",
                        )
                        bounds.append(
                            {"prim_path": str(prim.GetPath()), "lower_m": lower, "upper_m": upper}
                        )
                require(bounds, "No composed finger collision extents are available")
                geometry = {
                    "status": "available",
                    "backend": "usd_asset_geometry",
                    "traversal": "instance_proxies",
                    "frame": "finger_link_local",
                    "collision_extents": bounds,
                }
            except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
                geometry = {"status": "unavailable", "error": str(exc)[:512]}
            evidence["links"][name] = {"collision_extents": geometry}
        return evidence

    def _read_finger_world_poses(self) -> dict:
        poses = {}
        for name in ("panda_leftfinger", "panda_rightfinger"):
            try:
                from isaacsim.core.experimental.utils.stage import get_current_stage
                from isaacsim.core.experimental.utils.xform import get_world_pose

                prim = get_current_stage(backend="fabric").GetPrimAtPath(
                    f"{self.robot.prim_path}/{name}"
                )
                if not prim:
                    raise RuntimeError("The active Fabric finger prim is unavailable.")
                position, orientation = get_world_pose(prim, device="cpu")
                poses[name] = {
                    "status": "available",
                    "backend": "fabric_hierarchy",
                    "physics_step": int(self.world.current_time_step_index),
                    "position_m": vector(
                        tuple(float(v) for v in position.numpy()), 3, "finger world position"
                    ),
                    "orientation_wxyz": vector(
                        tuple(float(v) for v in orientation.numpy()), 4, "finger world orientation"
                    ),
                }
            except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
                poses[name] = {"status": "unavailable", "error": str(exc)[:512]}
        return poses

    def paused_reference_targets(
        self, point: tuple[float, float, float], closed: bool, *, phase: str, control_tick: int
    ) -> tuple[float, ...]:
        if self.control_mode != "paused_simulation" or self.controller is None:
            raise RuntimeError(
                "The explicitly authorized paused reference controller is unavailable."
            )
        diagnostic = self.reference_target_diagnostic = {
            "schema": "physicalai.reference-target-diagnostic/v1",
            "command_id": str(self.control_binding.command_id),
            "epoch": str(self.scene_epoch),
            "phase": phase,
            "control_tick": control_tick,
            "physics_step": int(self.world.current_time_step_index),
            "simulation_time": float(self.world.current_time),
            "physics_dt": self.dt,
            "joint_names": JOINT_NAMES,
            "joint_units": JOINT_UNITS,
            "joint_lower": JOINT_LOWER,
            "joint_upper": JOINT_UPPER,
            "hard_step_limits": tuple(value / 10 for value in DEFAULT_JOINT_VELOCITY_LIMITS),
            "planning_limit_fraction": REFERENCE_LIMIT_FRACTION,
            "planning_step_limits": tuple(
                value / 10 * REFERENCE_LIMIT_FRACTION for value in DEFAULT_JOINT_VELOCITY_LIMITS
            ),
            "previous_issued_targets": self.issued_targets,
            "last_issued_targets": None,
            "actual_hold_steps": 0,
            "failure": None,
            "part_position": self.position(),
        }
        try:
            require(
                isinstance(phase, str) and 0 < len(phase) <= 64, "A reference phase is required"
            )
            require(
                len(self._reference_target_trace) < MAX_REFERENCE_TRACE_INTERVALS,
                "The bounded reference trace cannot discard previous actual intervals",
            )
            integer(control_tick, "reference control tick", 0, MAX_REFERENCE_TRACE_INTERVALS - 1)
            joints = vector(
                tuple(float(value) for value in self.robot.get_joint_positions()),
                9,
                "reference measured joints",
            )
            diagnostic["measured_joint_positions"] = joints
            diagnostic["measured_joint_velocities"] = vector(
                tuple(float(value) for value in self.robot.get_joint_velocities()),
                9,
                "reference measured velocities",
            )
            point = vector(point, 3, "reference Cartesian target")
            contact_phase = self._grasp_frame_binding is not None and phase in GRASP_CONTACT_PHASES
            diagnostic["reference_target_frame"] = (
                "inner_pad_centroid" if contact_phase else "right_gripper"
            )
            diagnostic["reference_target_position"] = point
            diagnostic["measured_tcp_frame"] = "right_gripper"
            diagnostic["measured_tcp"] = self._measured_tcp()
            diagnostic["measured_route_point"] = self.paused_reference_route_point(
                diagnostic["measured_tcp"], phase
            )
            diagnostic["orientation_previous"] = self.orientation_target
            self.orientation_target = rotate_toward(
                self.orientation_target, (0.0, 0.0, 1.0, 0.0), 0.5 / 10
            )
            diagnostic["orientation_target"] = self.orientation_target
            tcp_target = (
                reference_tcp_target(point, self.orientation_target) if contact_phase else point
            )
            diagnostic["cartesian_target"] = tcp_target
            diagnostic["cartesian_target_frame"] = "right_gripper"
            policy = self.controller.get_articulation_motion_policy()
            diagnostic["rmp_dt"] = finite(policy.get_default_physics_dt(), "RMP integration dt")
            diagnostic["rmp_maximum_substep_size"] = finite(
                self.controller.rmp_flow.maximum_substep_size, "RMP maximum substep"
            )
            diagnostic["rmp_ignores_state_updates"] = (
                self.controller.rmp_flow.ignore_robot_state_updates
            )
            require(
                diagnostic["rmp_dt"] == 1 / self.control_profile.control_sim_hz
                and diagnostic["rmp_maximum_substep_size"] > 0
                and diagnostic["rmp_ignores_state_updates"] is False,
                "The reference expert requires actual measured feedback "
                "and the declared 0.1s horizon",
            )
            proposed = self.controller.forward(
                target_end_effector_position=np.array(tcp_target),
                target_end_effector_orientation=np.array(self.orientation_target),
            )
            if proposed.joint_positions is None:
                raise RuntimeError("The reference expert produced no actual position command.")
            diagnostic["raw_rmp_positions_repr"] = repr(proposed.joint_positions[:9])[:512]
            require(
                len(proposed.joint_positions) == 7, "The RMP arm proposal requires seven joints"
            )
            require(
                proposed.joint_velocities is None or len(proposed.joint_velocities) == 7,
                "RMP arm velocity evidence requires seven joints",
            )
            positions = vector(
                tuple(float(value) for value in proposed.joint_positions), 7, "RMP arm positions"
            )
            diagnostic["raw_rmp_joint_positions"] = positions
            diagnostic["raw_rmp_joint_velocities"] = (
                vector(
                    tuple(float(value) for value in proposed.joint_velocities),
                    7,
                    "RMP arm velocities",
                )
                if proposed.joint_velocities is not None
                else None
            )
            selected = (
                list(range(7))
                if proposed.joint_indices is None
                else proposed.joint_indices.tolist()
                if hasattr(proposed.joint_indices, "tolist")
                else list(proposed.joint_indices)
            )
            require(
                sorted(selected) == list(range(7)),
                "The reference expert must propose all seven distinct arm joints",
            )
            targets = list(joints)
            for index, value in zip(selected, positions, strict=True):
                integer(index, "reference joint index", 0, 6)
                targets[index] = value
            diagnostic["raw_rmp_joint_indices"] = selected
            targets[7:] = reference_gripper_targets(joints, self.issued_targets, closed=closed)
            diagnostic["gripper_command"] = "close" if closed else "open"
            diagnostic["gripper_target_origin"] = (
                "previous_issued" if self.issued_targets is not None else "initial_measured"
            )
            diagnostic["requested_targets"] = tuple(targets)
            diagnostic["limit_violations"] = reference_tracking_violations(
                joints, tuple(targets), self.issued_targets
            )

            def forward_kinematics(candidate):
                position, _ = self.controller.rmp_flow.get_end_effector_pose(
                    np.array(candidate[:7])
                )
                return tuple(float(value) for value in position)

            plan = plan_reference_targets(
                joints,
                tuple(targets),
                self.issued_targets,
                forward_kinematics=forward_kinematics,
                max_cartesian_speed_m_s=min(0.1, self.spec.requested_speed),
            )
            diagnostic["plan"] = asdict(plan)
            diagnostic["limiting_joint_names"] = tuple(
                JOINT_NAMES[index] for index in plan.limiting_joint_indices
            )
            return validate_position_target(joints, plan.targets, self.issued_targets)
        except (RuntimeError, ValueError, TypeError) as exc:
            diagnostic["failure"] = str(exc)
            print(
                "PHYSICALAI_REFERENCE_TARGET "
                + json.dumps(diagnostic, sort_keys=True, allow_nan=False),
                flush=True,
            )
            raise

    def reference_target_evidence(self) -> dict:
        return {
            "latest": deepcopy(self.reference_target_diagnostic),
            "intervals": deepcopy(self._reference_target_trace),
            "max_retained_intervals": MAX_REFERENCE_TRACE_INTERVALS,
            "gripper_asset": deepcopy(self._gripper_asset_evidence),
            "grasp_frame_calibration": deepcopy(self._grasp_frame_binding),
        }

    def apply_paused_tick(self, targets: tuple[float, ...]) -> AppliedControl:
        self._check_control_scheduling("before_control_tick")
        if not self.control_core.actuation_allowed(self.control_binding):
            raise RuntimeError("The paused episode authority is no longer active.")
        if self._paused_gripper_servo is None:
            raise RuntimeError("The required paused gripper servo was not calibrated.")
        self._paused_gripper_servo.verify(self.control_binding)
        efforts = self._compensate_gravity()
        self._issue_command(targets, (0.0,) * 9)
        if self.reference_target_diagnostic is not None:
            self.reference_target_diagnostic["last_issued_targets"] = targets
        self._step_control_physics(render=False)
        if self.reference_target_diagnostic is not None:
            self.reference_target_diagnostic["actual_hold_steps"] += 1
        self.steps += 1
        stamp = self.control_core.clock_ns()
        current = self._measured_tcp()
        joints = tuple(float(value) for value in self.robot.get_joint_positions())
        speed = check_measured_motion(
            self.last_effector_position,
            current,
            joints,
            dt=self.dt,
            speed_limit=self.spec.requested_speed,
        )
        self.last_effector_position = current
        self.peak_tcp_speed = max(self.peak_tcp_speed, speed)
        self.paused_measured_success = self.task_watchdog.observe(
            tcp=current, part=self.position(), joints=joints
        )
        self.grasp_verified = self.task_watchdog.grasp_verified
        diagnostic = self.reference_target_diagnostic
        if diagnostic is not None:
            diagnostic["completed_physics_step"] = int(self.world.current_time_step_index)
            diagnostic["last_measured_joint_positions"] = joints
            diagnostic["last_measured_tcp"] = current
            diagnostic["peak_hold_tcp_speed_m_s"] = max(
                diagnostic.get("peak_hold_tcp_speed_m_s", 0.0), speed
            )
            diagnostic["grasp_verified"] = self.grasp_verified
            diagnostic["measured_goal_error_m"] = self.task_watchdog.goal_error_m
            if diagnostic["actual_hold_steps"] == 6:
                diagnostic["last_part_position"] = self.position()
                diagnostic["last_finger_world_poses"] = self._read_finger_world_poses()
                self._reference_target_trace.append(deepcopy(diagnostic))
        return AppliedControl(
            int(self.world.current_time_step_index), stamp, targets, (0.0,) * 9, efforts
        )

    def paused_goal_reached(self) -> bool:
        return self.paused_measured_success

    @staticmethod
    def _encode_published_rgb(frame) -> bytes:
        rgba = frame.get("rgb")
        if (
            rgba is None
            or rgba.ndim != 3
            or rgba.shape[2] not in (3, 4)
            or rgba.dtype != np.uint8
            or not np.any(rgba[:, :, :3])
        ):
            raise ValueError(
                "A real nonempty uint8 RGB publication is required in the camera frame."
            )
        rgb = rgba[:, :, :3].copy()
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="PNG", compress_level=1)
        return output.getvalue()

    def _paused_publication(self, core, *, deadline_ns: int | None = None) -> PausedPublication:
        self._paused_camera_diagnostic = {
            "schema": "physicalai.paused-camera-barrier/v1",
            "epoch": str(core.epoch),
            "failure": None,
        }
        try:
            return self._publish_paused_camera(core, deadline_ns=deadline_ns)
        except (RuntimeError, ValueError, TypeError) as exc:
            self._paused_camera_diagnostic["failure"] = str(exc)
            print(
                "PHYSICALAI_PAUSED_CAMERA "
                + json.dumps(self._paused_camera_diagnostic, sort_keys=True, allow_nan=False),
                flush=True,
            )
            raise

    def paused_camera_evidence(self) -> dict:
        return deepcopy(self._paused_camera_diagnostic)

    def _paused_native_clocks(self) -> dict:
        import carb

        evidence = {
            "status": "available",
            "multitick_enabled": carb.settings.get_settings().get(
                "/rtx/hydra/supportMultiTickRate"
            ),
        }
        if type(evidence["multitick_enabled"]) is not bool:
            evidence.update(
                status="unavailable", multitick_error="The actual multitick setting is unavailable."
            )
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            evidence["simulation_time"] = finite(
                SimulationManager.get_simulation_time(), "actual native simulation time"
            )
            evidence["physics_steps"] = integer(
                SimulationManager.get_num_physics_steps(), "actual native physics count"
            )
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
            evidence.update(status="unavailable", manager_error=str(exc)[:512])
        try:
            from isaacsim.core.experimental.utils.stage import get_current_stage

            stage = get_current_stage(backend="fabric")
            prim = stage.GetPrimAtPath("/ExternalSimulationTime")
            attribute = prim.GetAttribute("omni:time") if prim else None
            if not attribute:
                raise RuntimeError("Fabric /ExternalSimulationTime.omni:time is unavailable.")
            evidence["external_simulation_time"] = finite(attribute.Get(), "actual Fabric time")
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
            evidence.update(status="unavailable", fabric_error=str(exc)[:512])
        return evidence

    def _publish_paused_camera(self, core, *, deadline_ns: int | None = None) -> PausedPublication:
        if (
            core.paused_profile is None
            or core.tenant_id is None
            or core.environment is None
            or core.epoch != self.scene_epoch
            or core.owner is None
        ):
            raise RuntimeError("A bound paused scene owner, epoch and profile are required.")
        freeze_ns = core.clock_ns()
        deadline_ns = min(
            freeze_ns + 2_000_000_000,
            deadline_ns if deadline_ns is not None else freeze_ns + 2_000_000_000,
        )
        before = self.frozen_physics_state(core.epoch)
        joint_ns = core.clock_ns()

        def verify_frozen():
            current = self.frozen_physics_state(core.epoch)
            if current != before:
                self._paused_camera_diagnostic["changed_frozen_state"] = {
                    **asdict(current),
                    "epoch": str(current.epoch),
                }
                raise RuntimeError("Physics changed during the frozen camera publication.")

        self._paused_camera_diagnostic["initial_frozen_state"] = {
            **asdict(before),
            "epoch": str(before.epoch),
        }
        self._paused_camera_diagnostic["freeze_established_ns"] = freeze_ns
        self._paused_camera_diagnostic["joint_sample_ns"] = joint_ns
        previous = {
            name: render_identity(camera.get_current_frame().get("rendering_frame"))
            for name, camera in self.cameras.items()
        }
        rendered_ns = paused_observation_barrier(
            self.world,
            self.cameras,
            dt=self.dt,
            physics_step=before.physics_step,
            clock_ns=core.clock_ns,
            deadline_ns=deadline_ns,
            previous_identities=previous,
            read_native_clocks=self._paused_native_clocks,
            verify_frozen=verify_frozen,
            diagnostics=self._paused_camera_diagnostic,
        )
        rendered_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        images = {}
        for name, camera in self.cameras.items():
            metadata = dict(camera.get_current_frame())
            identity = render_identity(metadata.get("rendering_frame"))
            image = self._encode_published_rgb(metadata)
            if render_identity(camera.get_current_frame().get("rendering_frame")) != identity:
                raise RuntimeError("Camera identity changed while reading its published RGB.")
            images[name] = FrozenCameraSample(
                image,
                identity[0],
                before.physics_step,
                rendered_ns,
                rendered_utc,
                identity[0],
                identity[1],
            )
        verify_frozen()
        published_ns = core.clock_ns()
        if published_ns >= deadline_ns:
            raise RuntimeError(
                "Actual frozen camera publication exceeded its original wall deadline."
            )
        self._paused_camera_diagnostic["published_ns"] = published_ns
        return PausedPublication(
            publication_id=uuid4(),
            scope=Scope(core.tenant_id, core.owner),
            environment_id=core.environment.environment_id,
            revision=core.environment.revision,
            profile_sha256=core.paused_profile.sha256,
            state_revision=core.state_revision,
            frozen_state=before,
            freeze_established_ns=freeze_ns,
            joint_sample_ns=joint_ns,
            published_ns=published_ns,
            captured_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            images=tuple(images.items()),
        )

    def paused_observation(self, request, core, episode, control_tick):
        if control_tick == 0:
            publication = self.paused_publications.take(
                scope=Scope(core.tenant_id, core.owner),
                environment_id=request.environment_id,
                revision=request.revision,
                profile_sha256=core.paused_profile.sha256,
                state_revision=core.state_revision,
                state=self.frozen_physics_state(core.epoch),
                now_ns=episode.interval_started_ns,
            )
        else:
            publication = self._paused_publication(core, deadline_ns=episode.operation_deadline_ns)
        first_image = dict(publication.images)["inspection"]
        observed = FrozenPolicyObservation(
            scope=publication.scope,
            environment_id=publication.environment_id,
            revision=publication.revision,
            episode_id=str(request.command_id),
            epoch=str(publication.frozen_state.epoch),
            captured_at_utc=publication.captured_at_utc,
            monotonic_ns=publication.published_ns,
            physics_step=publication.frozen_state.physics_step,
            joint_positions=publication.frozen_state.joint_positions,
            images=dict(publication.images),
            freeze_id=str(episode.freeze_id),
            state_revision=publication.state_revision,
            control_tick=control_tick,
            control_profile_sha256=publication.profile_sha256,
            observation_started_ns=episode.interval_started_ns,
            joint_sample_ns=publication.joint_sample_ns,
            simulation_time_numerator=first_image.simulation_time_numerator,
            simulation_time_denominator=first_image.simulation_time_denominator,
        )
        if control_tick == 0:
            proof = InitialFrozenPublication(
                publication_record_id=str(publication.publication_id),
                publication_record_sha256=publication.record_sha256,
                capture_sha256=observed.capture_sha256,
                freeze_established_ns=publication.freeze_established_ns,
                published_at_utc=publication.captured_at_utc,
                published_monotonic_ns=publication.published_ns,
                age_at_observation_start_ns=episode.interval_started_ns - publication.published_ns,
            )
            observed = replace(
                observed, initial_publication=proof, observation_completed_ns=core.clock_ns()
            )
        observed.validate(core.paused_profile, now_ns=core.clock_ns())
        return observed

    def _issue_command(self, targets, velocities) -> None:
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.array(targets), joint_velocities=np.array(velocities)
            )
        )
        self.issued_targets = tuple(float(value) for value in targets)

    def _teaching_targets(self, joints):
        core, profile = self.control_core, self.control_profile
        intent = core.teaching_intent(self.control_binding)
        minimum_hold_ns = round(1_000_000_000 / profile.control_hz)
        targets = list(joints)
        if self.issued_targets is not None:
            targets[7:] = self.issued_targets[7:]
        if intent is None or intent.expires_at_monotonic_ns - core.clock_ns() < minimum_hold_ns:
            return tuple(targets), None, core.monotonic_deadlines[core.active_command]
        current = self._measured_tcp()
        if intent.sequence != self.last_jog_sequence:
            self.jog_goal = tuple(a + b for a, b in zip(current, intent.delta_xyz_m, strict=True))
            check_measured_motion(None, self.jog_goal, joints, dt=self.dt, speed_limit=0.05)
            if intent.gripper != "hold":
                self.jog_gripper = intent.gripper
            self.last_jog_sequence = intent.sequence
        target = move_toward(current, self.jog_goal, self.jog_speed / profile.control_hz)
        if any(abs(a - b) > 1e-7 for a, b in zip(current, target, strict=True)):
            requested = self.controller.forward(
                target_end_effector_position=np.array(target),
                target_end_effector_orientation=np.array(self.orientation_target),
            )
            if requested.joint_positions is None:
                raise RuntimeError("The Cartesian jog solver returned no position targets.")
            selected = (
                range(len(requested.joint_positions))
                if requested.joint_indices is None
                else requested.joint_indices
            )
            for index, value in zip(selected, requested.joint_positions, strict=True):
                if index < 7 and value is not None:
                    targets[index] = float(value)
        if self.jog_gripper in {"open", "close"}:
            fingers = (0.04, 0.04) if self.jog_gripper == "open" else (0.0, 0.0)
            targets[7:] = move_toward(tuple(joints[7:]), fingers, 0.025 / profile.control_hz)
        return tuple(targets), intent.sequence, intent.expires_at_monotonic_ns

    def _next_control_interval(self, *, started_ns: int) -> None:
        self.interval_started_ns = started_ns
        self.control_next_ns = self.interval_started_ns + 100_000_000
        self.interval_phases = {
            "observation_render_ms": 0.0,
            "observation_read_ms": 0.0,
            "policy_or_teacher_ms": 0.0,
            "actuator_submission_ms": 0.0,
            "physics_and_publish_ms": 0.0,
            "capture_queue_ms": 0.0,
            "measurement_guard_ms": 0.0,
        }
        barrier_details = {}
        try:
            self.render_monotonic_ns = observation_barrier(
                self.world,
                self.cameras,
                dt=self.dt,
                physics_step=self.steps,
                clock_ns=self.control_core.clock_ns,
                deadline_ns=self.control_next_ns,
                published_ns=self.render_monotonic_ns,
                previous_identities={
                    name: self.last_render_frame[(name, "recording")]
                    for name in self.cameras
                    if (name, "recording") in self.last_render_frame
                },
                diagnostics=barrier_details,
            )
        finally:
            self.camera_observation_metadata = camera_evidence(
                self.world, self.cameras, physics_step=self.steps
            ) | {
                "warmup_steps": self.warmup_steps,
                "hold_offset": self.hold_offset,
                "barrier": barrier_details,
            }
        rendered_ns = self.control_core.clock_ns()
        self.interval_phases["observation_render_ms"] = (
            rendered_ns - self.interval_started_ns
        ) / 1_000_000
        joints = tuple(float(value) for value in self.robot.get_joint_positions())
        observed = self._sample_before_command(joints)
        observed_ns = self.control_core.clock_ns()
        self.interval_phases["observation_read_ms"] = (observed_ns - rendered_ns) / 1_000_000
        if self.control_mode == "learned":
            context = self.control_core.policy_context(self.control_binding)
            observation = PolicyObservation(
                context.scope,
                context.environment_id,
                context.revision,
                context.episode_id,
                observed.captured_at_utc,
                observed.monotonic_ns,
                observed.physics_step,
                observed.joint_positions,
                observed.images,
            )
            self.policy_command = self.policy_executor.predict(observation)
            targets = self.policy_command.targets
            self.held_expires_ns = min(
                self.policy_command.expires_at_monotonic_ns, self.control_next_ns
            )
            self.held_sequence = None
        else:
            targets, self.held_sequence, self.held_expires_ns = self._teaching_targets(joints)
        self.held_targets = validate_position_target(joints, targets, self.issued_targets)
        planned_ns = self.control_core.clock_ns()
        self.interval_phases["policy_or_teacher_ms"] = (planned_ns - observed_ns) / 1_000_000
        self.hold_offset = 0
        if self.hold_capture is not None:
            self.hold_capture.begin(replace(observed, commanded_joint_targets=self.held_targets))
        self.interval_phases["capture_queue_ms"] += (
            self.control_core.clock_ns() - planned_ns
        ) / 1_000_000

    def _advance_control(self) -> bool:
        core = self.control_core
        if not core.actuation_allowed(self.control_binding):
            raise RuntimeError("The control command is no longer active.")
        tick_started_ns = core.clock_ns()
        self._check_control_scheduling("before_control_tick")
        if self.warmup_steps:

            def prepare_cameras():
                self._compensate_gravity()
                self._issue_command(self.robot.get_joint_positions(), (0.0,) * 9)

            core.apply_guarded(self.control_binding, prepare_cameras)
            self._step_control_physics(render=True)
            self.render_monotonic_ns = core.clock_ns()
            self.steps += 1
            self.warmup_steps -= 1
            return False
        if self.held_targets is None or self.hold_offset == self.control_profile.hold_steps:
            if core.clock_ns() < self.control_next_ns:
                return False
            if len(self.control_timings) >= 3000:
                raise RuntimeError("The bounded control-interval retention budget was exhausted.")
            self._next_control_interval(started_ns=tick_started_ns)
        efforts = None

        def submit(targets):
            nonlocal efforts
            if core.clock_ns() >= self.held_expires_ns:
                raise RuntimeError("The issued control authority expired during its hold.")
            if self.held_sequence is not None and not core.teaching_hold_allowed(
                self.control_binding, self.held_sequence
            ):
                raise RuntimeError("Teaching deadman was released during an active hold.")
            efforts = self._compensate_gravity()
            self._issue_command(targets, (0.0,) * 9)

        submitted_ns = core.clock_ns()
        if self.control_mode == "learned":
            self.policy_executor.apply(
                self.policy_command, physics_step=self.steps + 1, actuator=submit
            )
        else:
            core.apply_guarded(self.control_binding, lambda: submit(self.held_targets))
        physics_ns = core.clock_ns()
        self.interval_phases["actuator_submission_ms"] += (physics_ns - submitted_ns) / 1_000_000
        render = self.hold_offset + 1 == self.control_profile.hold_steps
        self._step_control_physics(render=render)
        if render:
            self.render_monotonic_ns = core.clock_ns()
        completed_ns = core.clock_ns()
        self.interval_phases["physics_and_publish_ms"] += (completed_ns - physics_ns) / 1_000_000
        self.steps += 1
        self.hold_offset += 1
        if self.hold_capture is not None:
            self.hold_capture.applied(
                AppliedControl(self.steps, core.clock_ns(), self.held_targets, (0.0,) * 9, efforts)
            )
        measured_ns = core.clock_ns()
        self.interval_phases["capture_queue_ms"] += (measured_ns - completed_ns) / 1_000_000
        current = self._measured_tcp()
        joints = tuple(float(value) for value in self.robot.get_joint_positions())
        speed = check_measured_motion(
            self.last_effector_position,
            current,
            joints,
            dt=self.dt,
            speed_limit=self.spec.requested_speed,
        )
        self.peak_tcp_speed = max(self.peak_tcp_speed, speed)
        self.last_effector_position = current
        complete = self.task_watchdog.observe(tcp=current, part=self.position(), joints=joints)
        self.grasp_verified = self.task_watchdog.grasp_verified
        self.interval_phases["measurement_guard_ms"] += (core.clock_ns() - measured_ns) / 1_000_000
        if self.hold_offset == self.control_profile.hold_steps:
            self.control_timings.append(
                {
                    "observation_step": self.steps - self.control_profile.hold_steps,
                    "completed_step": self.steps,
                    **self.interval_phases,
                    "control_cycle_ms": (core.clock_ns() - self.interval_started_ns) / 1_000_000,
                    "inference_latency_ms": (
                        self.policy_command.inference_latency_ms
                        if self.control_mode == "learned"
                        else 0.0
                    ),
                }
            )
            if core.clock_ns() - self.interval_started_ns > 100_000_000:
                raise RuntimeError(
                    "The measured control interval exceeded the 10 Hz profile budget."
                )
            if self.finish_requested or (self.control_mode == "learned" and complete):
                if not complete:
                    raise RuntimeError(
                        "Measured grasp, release, goal and settling checks did not pass."
                    )
                core.apply_guarded(
                    self.control_binding,
                    lambda: self._issue_command(self.robot.get_joint_positions(), (0.0,) * 9),
                )
                self.control_done, self.control_succeeded = True, True
                if self.policy_executor is not None:
                    self.policy_executor.stop()
                return True
        return False

    def _sample_before_command(
        self, targets: tuple[float, ...], *, terminated: bool = False, truncated: bool = False
    ) -> FrameSample:
        images = {}
        for name, camera in self.cameras.items():
            metadata = camera.get_current_frame()
            rendering_time = metadata.get("rendering_time")
            if (
                rendering_time is None
                or abs(float(rendering_time) - self.world.current_time) > self.dt / 2
            ):
                raise ValueError(
                    "Demonstration camera is not synchronized with the current physics observation."
                )
            image = self.capture(name, consumer="recording")
            if image is None:
                raise ValueError(
                    "A genuinely new camera frame is required for every recorded control step."
                )
            images[name] = CameraSample(
                image,
                render_identity(metadata["rendering_frame"])[0],
                self.steps,
                self.render_monotonic_ns,
            )
        return FrameSample(
            captured_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            monotonic_ns=time.monotonic_ns(),
            physics_step=self.steps,
            joint_positions=tuple(float(value) for value in self.robot.get_joint_positions()),
            commanded_joint_targets=targets,
            images=images,
            terminated=terminated,
            truncated=truncated,
        )

    def finish_recording(self, truncated: bool):
        if self.recording is None:
            return None
        recorder = self.recording
        try:
            if self.paused_driver is not None:
                self.paused_driver.finish_recording(truncated=truncated)
            elif self.hold_capture is not None:
                self.hold_capture.finish(truncated=truncated)
            else:
                sample = self._sample_before_command(
                    self.issued_targets, terminated=not truncated, truncated=truncated
                )
                recorder.append(sample)
                recorder.seal()
        finally:
            self.recording = None

    def position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.part.get_world_pose()[0])

    def motion_phase(self) -> MotionPhase:
        if self.world is None or not self.world.is_playing():
            return "stopped"
        if self.paused_driver is not None:
            if self.paused_driver.done:
                return "complete" if self.paused_driver.succeeded else "stopped"
            return "transporting" if self.grasp_verified else "approaching"
        if self.control_mode != "reference":
            if self.control_done:
                return "complete" if self.control_succeeded else "stopped"
            return "transporting" if self.grasp_verified else "approaching"
        if self.controller is None:
            return "complete" if self.route is not None and self.route.done else "idle"
        phases: dict[str, MotionPhase] = {
            "lift-clear": "approaching",
            "approach-part": "approaching",
            "lower-to-part": "approaching",
            "grasp": "grasping",
            "lift-part": "lifting",
            "to-inspection": "inspection_station",
            "inspect": "inspection_station",
            "lift-inspected-part": "transporting",
            "to-destination": "transporting",
            "lower-to-destination": "transporting",
            "release": "releasing",
            "retreat": "returning",
        }
        return phases[self.route.current.name]

    def _compensate_gravity(self) -> tuple[float, ...]:
        gravity = self.dynamics.get_generalized_gravity_forces()
        if gravity is None:
            raise RuntimeError("PhysX gravity compensation is unavailable for the reference arm.")
        efforts = arm_gravity_efforts(tuple(float(value) for value in gravity[0]))
        self.robot.set_joint_efforts(np.array(efforts))
        return efforts

    def advance(self) -> bool:
        if self.world is None:
            return False
        if self.paused_driver is not None:
            return self.paused_driver.advance()
        if (
            self.spec.learning_execution is not None
            and self.control_done
            and self.controller is None
        ):
            self.render_frozen_scene()
            return False
        if not self.world.is_playing():
            self.world.render()
            return False
        if self.control_mode != "reference" and not self.control_done:
            return self._advance_control()
        self._compensate_gravity()
        if self.controller is not None:
            joints = self.robot.get_joint_positions()
            position, _ = self.controller.rmp_flow.get_end_effector_pose(joints[:7])
            target, closed = self.route.next_target(
                tuple(position), float(sum(joints[7:])), self.position()
            )
            self.orientation_target = rotate_toward(
                self.orientation_target, (0.0, 0.0, 1.0, 0.0), 0.5 * self.dt
            )
            requested = self.controller.forward(
                target_end_effector_position=np.array(target),
                target_end_effector_orientation=np.array(self.orientation_target),
            )
            targets = joints.copy()
            desired_velocity = np.zeros(7)
            indices = requested.joint_indices
            selected = range(len(requested.joint_positions)) if indices is None else indices
            for index, value in zip(selected, requested.joint_positions, strict=True):
                if value is not None:
                    targets[index] = float(value)
            velocities = requested.joint_velocities
            for offset, index in enumerate(selected):
                if index < 7:
                    desired_velocity[index] = (
                        float(velocities[offset])
                        if velocities is not None and velocities[offset] is not None
                        else (targets[index] - joints[index]) / self.dt
                    )
            desired_velocity = np.array(move_toward((0.0,) * 7, tuple(desired_velocity), 0.6))
            targets[:7] = joints[:7] + desired_velocity * self.dt
            predicted, _ = self.controller.rmp_flow.get_end_effector_pose(targets[:7])
            displacement = np.linalg.norm(predicted - position)
            maximum = self.spec.requested_speed * 0.65 * self.dt
            if displacement > maximum:
                targets[:7] = joints[:7] + (targets[:7] - joints[:7]) * (maximum / displacement)
                desired_velocity *= maximum / displacement
            finger_positions, finger_velocities = self.gripper_ramp.next_command(closed)
            targets[7:] = finger_positions
            actions = ArticulationAction(
                joint_positions=targets,
                joint_velocities=np.concatenate((desired_velocity, finger_velocities)),
            )
            if self.recording is not None:
                positions = actions.joint_positions
                if positions is None:
                    raise ValueError("The controller returned no position command to record.")
                indices = actions.joint_indices
                targets = resolve_joint_targets(
                    self.issued_targets,
                    positions.tolist() if isinstance(positions, np.ndarray) else positions,
                    indices.tolist() if isinstance(indices, np.ndarray) else indices,
                )
                sample = self._sample_before_command(targets)

            def submit_reference():
                self._issue_command(actions.joint_positions, actions.joint_velocities)

            if self.actuation_guard is not None:
                self.actuation_guard(submit_reference)
            else:
                submit_reference()
            if self.recording is not None:
                self.issued_targets = targets
                self.recording.append(sample)
        render = self.spec.record_demonstration or self.steps % 6 == 0
        self.world.step(render=render)
        if render:
            self.render_monotonic_ns = time.monotonic_ns()
        self.steps += 1
        if self.controller is not None:
            current, _ = self.controller.rmp_flow.get_end_effector_pose(
                self.robot.get_joint_positions()[:7]
            )
            speed = check_measured_motion(
                self.last_effector_position,
                tuple(float(value) for value in current),
                tuple(float(value) for value in self.robot.get_joint_positions()),
                dt=self.dt,
                speed_limit=self.spec.requested_speed,
            )
            self.peak_tcp_speed = max(self.peak_tcp_speed, speed)
            self.last_effector_position = current.copy()
            if self.route.index >= 5 and not self.grasp_verified:
                if self.position()[2] < self.initial_part_position[2] + 0.05:
                    self.stop()
                    raise RuntimeError(
                        "The gripper reached lift height but the part was not grasped."
                    )
                self.grasp_verified = True
            if self.steps % 120 == 0:
                print(
                    "PHYSICALAI_MOTION "
                    + json.dumps(
                        {
                            "step": self.steps,
                            "phase": self.route.index,
                            "tcp": current.tolist(),
                            "target": self.route.target,
                            "part": self.position(),
                            "fingers": self.robot.get_joint_positions()[7:].tolist(),
                            "peak_tcp_speed_m_s": self.peak_tcp_speed,
                        }
                    ),
                    flush=True,
                )
            if self.route.done:
                goal = self.spec.station(self.target).position
                if any(abs(a - b) > 0.04 for a, b in zip(self.position(), goal, strict=True)):
                    self.stop()
                    raise RuntimeError("The part did not reach the commanded station volume.")
                print(
                    "PHYSICALAI_TRAJECTORY "
                    + json.dumps(
                        {
                            "final_position": self.position(),
                            "grasp_verified": self.grasp_verified,
                            "peak_tcp_speed_m_s": self.peak_tcp_speed,
                            "simulation_seconds": self.route.elapsed,
                            "gravity_compensation": "physx_feed_forward",
                        }
                    ),
                    flush=True,
                )
                self._issue_command(self.robot.get_joint_positions(), (0.0,) * 9)
                self.controller = None
                return True
        return False

    def capture(self, name: str, consumer: str = "preview") -> bytes | None:
        camera = self.cameras[name]
        frame = camera.get_current_frame()
        render_frame = frame.get("rendering_frame")
        key = (name, consumer)
        if render_frame is None:
            return None
        identity = render_identity(render_frame)
        previous_identity = self.last_render_frame.get(key)
        if previous_identity == identity:
            return None
        if previous_identity is not None and identity[0] < previous_identity[0]:
            raise ValueError("The native camera rendering identity moved backwards.")
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        if not np.any(rgba[:, :, :3]):
            self.empty_frames[name] = self.empty_frames.get(name, 0) + 1
            if self.empty_frames[name] > 30:
                raise RuntimeError(f"Camera {name} kept returning empty RGB after scene loading.")
            return None
        self.empty_frames[name] = 0
        previous_base = self.camera_timebases.get(name)
        if previous_base is not None and previous_base != identity[1]:
            raise ValueError("The native camera frame timebase changed within the scene.")
        self.camera_timebases[name] = identity[1]
        output = BytesIO()
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(
            output, format="PNG", compress_level=1
        )
        self.last_render_frame[key] = identity
        return output.getvalue()

    def stop(self) -> None:
        if self.world is not None:
            self.world.pause()
        self.controller = None
        self.control_done = True
        if self.paused_driver is not None:
            self.paused_driver.stop()
        if self.policy_executor is not None:
            self.policy_executor.stop()
        try:
            self._restore_paused_gripper_servo()
        except (ValueError, RuntimeError, TypeError, OSError) as exc:
            logging.getLogger(__name__).exception(
                "Owned paused gripper restoration failed; scene remains stopped"
            )
            if self.control_core is None:
                raise
            if self.paused_driver is not None:
                self.paused_driver.fail(str(exc))
            self.control_core.fail_scene(
                self.paused_driver.episode.failure if self.paused_driver is not None else str(exc),
                epoch=self.scene_epoch,
            )
