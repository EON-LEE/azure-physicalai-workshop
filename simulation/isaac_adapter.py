"""Imported only after SimulationApp is initialized in the pinned Isaac container."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from math import dist
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
from learning.contract import AppliedControl, CameraSample, FrameSample, Scope
from learning.inference import PolicyObservation
from learning.paused import FrozenCameraSample, FrozenPolicyObservation, InitialFrozenPublication
from simulation.asset_references import validate_usd_bundle
from simulation.camera_observation import camera_evidence, observation_barrier, render_identity
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
from simulation.paused_observation import PausedPublication, PausedPublicationCache
from simulation.physics_scheduling import physics_scheduling_readback, require_control_scheduling
from simulation.policy_executor import PolicyExecutor
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

    def load(self, spec: SceneSpec) -> None:
        self._validate_asset_bundle()
        self.stop()
        self.paused_driver = None
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
        self.parked_state = None
        self.policy_executor = None
        self.issued_targets = None
        self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=0.0)
        self._configure_cameras()
        configuration = interface_config_loader.load_supported_lula_kinematics_solver_config(
            "Franka"
        )
        self.kinematics = LulaKinematicsSolver(**configuration)
        self.articulation_kinematics = ArticulationKinematicsSolver(
            self.robot, self.kinematics, "right_gripper"
        )
        self.controller = RMPFlowController(
            name="paused-reference-expert",
            robot_articulation=self.robot,
            physics_dt=1 / core.paused_profile.control_sim_hz,
        )
        self.last_effector_position = self._measured_tcp()
        _, rotation = self.articulation_kinematics.compute_end_effector_pose()
        self.orientation_target = tuple(float(value) for value in rot_matrix_to_quat(rotation))
        self.task_watchdog = TaskWatchdog(self.position(), self.spec.station(self.target).position)
        self.grasp_verified = False
        self.peak_tcp_speed = 0.0
        self.paused_measured_success = False
        self.world.play()

    def paused_reference_targets(self, point, closed: bool) -> tuple[float, ...]:
        if self.control_mode != "paused_simulation" or self.controller is None:
            raise RuntimeError(
                "The explicitly authorized paused reference controller is unavailable."
            )
        joints = tuple(float(value) for value in self.robot.get_joint_positions())
        self.orientation_target = rotate_toward(
            self.orientation_target, (0.0, 0.0, 1.0, 0.0), 0.5 / 10
        )
        proposed = self.controller.forward(
            target_end_effector_position=np.array(point),
            target_end_effector_orientation=np.array(self.orientation_target),
        )
        if proposed.joint_positions is None:
            raise RuntimeError("The reference expert produced no actual position command.")
        targets = list(joints)
        selected = (
            range(len(proposed.joint_positions))
            if proposed.joint_indices is None
            else proposed.joint_indices
        )
        for index, value in zip(selected, proposed.joint_positions, strict=True):
            if index < 7 and value is not None:
                targets[index] = float(value)
        targets[7:] = move_toward(joints[7:], (0.0, 0.0) if closed else (0.04, 0.04), 0.025 / 10)
        return validate_position_target(joints, tuple(targets), self.issued_targets)

    def apply_paused_tick(self, targets: tuple[float, ...]) -> AppliedControl:
        self._check_control_scheduling("before_control_tick")
        if not self.control_core.actuation_allowed(self.control_binding):
            raise RuntimeError("The paused episode authority is no longer active.")
        efforts = self._compensate_gravity()
        self._issue_command(targets, (0.0,) * 9)
        self._step_control_physics(render=False)
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
        previous = {
            name: render_identity(camera.get_current_frame().get("rendering_frame"))
            for name, camera in self.cameras.items()
        }
        rendered_ns = observation_barrier(
            self.world,
            self.cameras,
            dt=self.dt,
            physics_step=before.physics_step,
            clock_ns=core.clock_ns,
            deadline_ns=deadline_ns,
            previous_identities=previous,
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
        if self.frozen_physics_state(core.epoch) != before:
            raise RuntimeError("Physics changed while publishing the frozen camera pair.")
        published_ns = core.clock_ns()
        if published_ns >= deadline_ns:
            raise RuntimeError(
                "Actual frozen camera publication exceeded its original wall deadline."
            )
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
