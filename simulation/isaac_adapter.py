"""Imported only after SimulationApp is initialized in the pinned Isaac container."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

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
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from apps.api.models import MotionPhase
from learning.capture import resolve_joint_targets
from learning.contract import CameraSample, FrameSample
from simulation.asset_references import validate_usd_bundle
from simulation.extensions import SceneSpec
from simulation.motion import (
    GripperRamp,
    InspectionRoute,
    arm_gravity_efforts,
    move_toward,
    rotate_toward,
)


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

    def load(self, spec: SceneSpec) -> None:
        self._validate_asset_bundle()
        self.stop()
        for camera in self.cameras.values():
            camera.destroy()
        self.cameras.clear()
        self.last_render_frame.clear()
        self.empty_frames.clear()
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
                position=np.array(spec.station(spec.source_id).position),
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
        if spec.defective:
            crack = UsdGeom.Cube.Define(stage, "/World/Part/SurfaceDefect")
            crack.CreateSizeAttr(1)
            crack.CreateDisplayColorAttr([Gf.Vec3f(0.015, 0.015, 0.015)])
            transform = UsdGeom.Xformable(crack.GetPrim())
            transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.501))
            transform.AddScaleOp().Set(Gf.Vec3f(0.08, 0.8, 0.01))
            defect_material = UsdShade.Material.Define(stage, "/World/DefectMaterial")
            defect_shader = UsdShade.Shader.Define(stage, "/World/DefectMaterial/Surface")
            defect_shader.CreateIdAttr("UsdPreviewSurface")
            defect_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.015, 0.015, 0.015)
            )
            defect_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            defect_material.CreateSurfaceOutput().ConnectToSource(
                defect_shader.ConnectableAPI(), "surface"
            )
            parent_binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Part"))
            # The default cube binding otherwise overrides the child's dark material.
            UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
                parent_binding.GetDirectBindingRel(), UsdShade.Tokens.weakerThanDescendants
            )
            UsdShade.MaterialBindingAPI.Apply(crack.GetPrim()).Bind(defect_material)
            bound_material, _ = UsdShade.MaterialBindingAPI(crack.GetPrim()).ComputeBoundMaterial()
            if bound_material.GetPath() != defect_material.GetPath():
                raise RuntimeError("The reference surface-defect material is not visibly bound.")
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
        self.robot.gripper.set_joint_positions(self.robot.gripper.joint_opened_positions)
        self.world.play()
        self.steps = 0
        self.target = None
        self.route = None
        self.last_effector_position = None
        for _ in range(18):
            self._compensate_gravity()
            self.world.step(render=True)
            self.steps += 1

    @staticmethod
    def _validate_asset_bundle() -> None:
        validate_usd_bundle(
            Path(os.environ["FRANKA_ASSET_ROOT"]), Path(os.environ["FRANKA_USD_PATH"])
        )

    def start(self, target_id: str, recording=None) -> None:
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
                image, int(metadata["rendering_frame"]), self.steps, time.monotonic_ns()
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
            sample = self._sample_before_command(
                self.issued_targets, terminated=not truncated, truncated=truncated
            )
            recorder.append(sample)
        finally:
            self.recording = None
        return recorder.finalize_and_upload()

    def position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.part.get_world_pose()[0])

    def motion_phase(self) -> MotionPhase:
        if self.world is None or not self.world.is_playing():
            return "stopped"
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

    def _compensate_gravity(self) -> None:
        gravity = self.dynamics.get_generalized_gravity_forces()
        if gravity is None:
            raise RuntimeError("PhysX gravity compensation is unavailable for the reference arm.")
        self.robot.set_joint_efforts(np.array(arm_gravity_efforts(tuple(gravity[0]))))

    def advance(self) -> bool:
        if self.world is None:
            return False
        if not self.world.is_playing():
            self.world.render()
            return False
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
            self.robot.apply_action(actions)
            if self.recording is not None:
                self.issued_targets = targets
                self.recording.append(sample)
        render = self.spec.record_demonstration or self.steps % 6 == 0
        self.world.step(render=render)
        self.steps += 1
        if self.controller is not None:
            current, _ = self.controller.rmp_flow.get_end_effector_pose(
                self.robot.get_joint_positions()[:7]
            )
            if self.last_effector_position is not None:
                speed = np.linalg.norm(current - self.last_effector_position) / self.dt
                self.peak_tcp_speed = max(self.peak_tcp_speed, float(speed))
                if speed > self.spec.requested_speed:
                    self.stop()
                    raise RuntimeError(
                        f"Reference Cartesian-speed watchdog exceeded: {speed:.3f} m/s; "
                        f"limit {self.spec.requested_speed:.3f}; phase {self.route.index}."
                    )
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
                self.robot.apply_action(
                    ArticulationAction(
                        joint_positions=self.robot.get_joint_positions(),
                        joint_velocities=np.zeros(9),
                    )
                )
                self.controller = None
                return True
        return False

    def capture(self, name: str, consumer: str = "preview") -> bytes | None:
        camera = self.cameras[name]
        frame = camera.get_current_frame()
        render_frame = frame.get("rendering_frame")
        key = (name, consumer)
        if render_frame is None or self.last_render_frame.get(key) == render_frame:
            return None
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        if not np.any(rgba[:, :, :3]):
            self.empty_frames[name] = self.empty_frames.get(name, 0) + 1
            if self.empty_frames[name] > 30:
                raise RuntimeError(f"Camera {name} kept returning empty RGB after scene loading.")
            return None
        self.empty_frames[name] = 0
        output = BytesIO()
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(
            output, format="PNG", compress_level=1
        )
        self.last_render_frame[key] = render_frame
        return output.getvalue()

    def stop(self) -> None:
        self.controller = None
        if self.world is not None:
            self.world.pause()
