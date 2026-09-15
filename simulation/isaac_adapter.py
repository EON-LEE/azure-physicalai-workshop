"""Imported only after SimulationApp is initialized in the pinned Isaac container."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux

from simulation.asset_references import local_reference
from simulation.extensions import SceneSpec


class IsaacWorkcell:
    dt = 1 / 60

    def __init__(self) -> None:
        self.world = None
        self.cameras = {}
        self.last_render_frame = {}
        self.controller = None
        self.target = None
        self.phase = 0
        self.steps = 0
        self.last_effector_position = None

    def load(self, spec: SceneSpec) -> None:
        self._validate_asset_bundle()
        self.stop()
        for camera in self.cameras.values():
            camera.pause()
        self.cameras.clear()
        self.last_render_frame.clear()
        if self.world is not None:
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
        for index, station in enumerate(spec.stations):
            x, y, z = station.position
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Station{index}",
                    name=f"station-{index}",
                    position=np.array([x, y, z - 0.045]),
                    scale=np.array([0.16, 0.16, 0.04]),
                    color=np.array(spec.platform_color),
                )
            )
        self.robot = self.world.scene.add(
            Franka(
                prim_path="/World/Robot",
                name="reference-arm",
                usd_path=os.environ["FRANKA_USD_PATH"],
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
            )
        )
        if spec.defective:
            crack = UsdGeom.Cube.Define(stage, "/World/Part/SurfaceDefect")
            crack.CreateSizeAttr(1)
            crack.CreateDisplayColorAttr([Gf.Vec3f(0.015, 0.015, 0.015)])
            transform = UsdGeom.Xformable(crack.GetPrim())
            transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.501))
            transform.AddScaleOp().Set(Gf.Vec3f(0.08, 0.8, 0.01))
        for name, eye, target in (
            ("overview", (1.4, 1.4, 1.6), (0, 0, 0.2)),
            ("inspection", (0, 0, 1.7), (0, 0, 0.2)),
        ):
            camera = Camera(
                prim_path=f"/World/{name}",
                name=name,
                frequency=10,
                resolution=(1280, 720),
            )
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0))
            quaternion = view.GetInverse().ExtractRotationQuat()
            orientation = np.array([quaternion.GetReal(), *quaternion.GetImaginary()])
            camera.set_world_pose(
                position=np.array(eye), orientation=orientation, camera_axes="usd"
            )
            camera.set_clipping_range(0.01, 100)
            self.cameras[name] = camera
        self.world.reset()
        for camera in self.cameras.values():
            camera.initialize()
        self.robot.gripper.set_joint_positions(self.robot.gripper.joint_opened_positions)
        self.world.play()
        self.steps = 0
        self.target = None
        self.last_effector_position = None

    @staticmethod
    def _validate_asset_bundle() -> None:
        root = Path(os.environ["FRANKA_ASSET_ROOT"])
        pending = [Path(os.environ["FRANKA_USD_PATH"])]
        seen = set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise ValueError("The approved robot USD layer could not be opened.")

            def check(reference, layer_path=path):
                resolved = local_reference(root, layer_path, reference)
                if resolved is not None and resolved.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    pending.append(resolved)

            for reference in layer.GetExternalReferences():
                check(reference)

            def attribute(spec_path, current_layer=layer, validator=check):
                item = current_layer.GetObjectAtPath(spec_path)
                if not isinstance(item, Sdf.AttributeSpec):
                    return
                values = [item.default]
                values.extend(
                    current_layer.QueryTimeSample(spec_path, time)
                    for time in current_layer.ListTimeSamplesForPath(spec_path)
                )
                for value in values:
                    if isinstance(value, Sdf.AssetPath):
                        validator(value.path)
                    elif item.typeName == Sdf.ValueTypeNames.AssetArray and value is not None:
                        for asset in value:
                            validator(asset.path)

            layer.Traverse(Sdf.Path.absoluteRootPath, attribute)

    def start(self, target_id: str) -> None:
        self.target = target_id
        self.phase = 0
        self.controller = PickPlaceController(
            name="guarded-pick-place",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )
        self.robot.gripper.set_joint_positions(self.robot.gripper.joint_opened_positions)
        self.last_effector_position = None
        self.world.play()

    def position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.part.get_world_pose()[0])

    def advance(self) -> bool:
        if self.world is None:
            return False
        if self.controller is not None:
            destination = self.spec.inspection_id if self.phase == 0 else self.target
            actions = self.controller.forward(
                picking_position=np.array(self.position()),
                placing_position=np.array(self.spec.station(destination).position),
                current_joint_positions=self.robot.get_joint_positions(),
            )
            self.robot.apply_action(actions)
        self.world.step(render=True)
        self.steps += 1
        if self.controller is not None:
            current = self.robot.end_effector.get_world_pose()[0]
            if self.last_effector_position is not None:
                speed = np.linalg.norm(current - self.last_effector_position) / self.dt
                if speed > self.spec.requested_speed:
                    self.stop()
                    raise RuntimeError("Reference simulation Cartesian-speed watchdog exceeded.")
            self.last_effector_position = current.copy()
            if self.controller.is_done():
                destination = self.spec.inspection_id if self.phase == 0 else self.target
                goal = self.spec.station(destination).position
                if any(abs(a - b) > 0.04 for a, b in zip(self.position(), goal, strict=True)):
                    self.stop()
                    raise RuntimeError("The part did not reach the commanded station volume.")
                if self.phase == 0:
                    self.phase = 1
                    self.controller.reset()
                    self.last_effector_position = None
                else:
                    self.controller = None
                    return True
        return False

    def capture(self, name: str) -> bytes | None:
        camera = self.cameras[name]
        frame = camera.get_current_frame()
        render_frame = frame.get("rendering_frame")
        if render_frame is None or self.last_render_frame.get(name) == render_frame:
            return None
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        output = BytesIO()
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(output, format="PNG")
        self.last_render_frame[name] = render_frame
        return output.getvalue()

    def stop(self) -> None:
        self.controller = None
        if self.world is not None:
            self.world.pause()
