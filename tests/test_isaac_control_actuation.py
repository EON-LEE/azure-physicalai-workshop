"""Execute the real adapter methods against CPU SDK/physics doubles, never GPU proof."""

import importlib
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
from runtime_support import ACTOR, PNG
from test_policy_executor import CountingPolicy
from test_policy_runtime import request_for
from test_position_hold_control import Recorder
from test_teaching_runtime import begin, jog
from test_teaching_runtime import teaching as teaching

from learning.contract import CameraSample, FrameSample, Scope
from learning.inference import ControlContext, GuardedPolicyAdapter, PolicyObservation
from simulation.policy_executor import PolicyExecutor


class Array(list):
    def copy(self):
        return Array(self)

    def tolist(self):
        return list(self)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Array(value) if isinstance(index, slice) else value


class Action:
    def __init__(self, joint_positions=None, joint_velocities=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_indices = joint_indices


@pytest.fixture
def hardware(teaching, monkeypatch):
    core, request, clock = teaching

    class RMP:
        def __init__(self, **kwargs):
            self.robot = kwargs["robot_articulation"]
            self.forward_calls = 0

        def forward(self, **kwargs):
            self.forward_calls += 1
            joints = self.robot.get_joint_positions()
            return Action(Array([joints[0] + 0.001, *joints[1:7]]), joint_indices=range(7))

    class Kinematics:
        def set_robot_base_pose(self, position, orientation):
            pass

    class ArticulationKinematics:
        def __init__(self, robot, solver, frame):
            assert frame == "right_gripper"

        def compute_end_effector_pose(self):
            return Array([0.35, 0.25, 0.3]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    modules = {
        "numpy": {
            "array": Array,
            "zeros": lambda size: Array([0.0] * size),
            "concatenate": lambda rows: Array(v for row in rows for v in row),
        },
        "isaacsim.core.api": {"World": object},
        "isaacsim.core.api.materials": {"PhysicsMaterial": object},
        "isaacsim.core.api.objects": {"DynamicCuboid": object, "FixedCuboid": object},
        "isaacsim.core.prims": {"Articulation": object},
        "isaacsim.core.utils.rotations": {"rot_matrix_to_quat": lambda matrix: (1, 0, 0, 0)},
        "isaacsim.core.utils.stage": {"create_new_stage": lambda: None},
        "isaacsim.core.utils.types": {"ArticulationAction": Action},
        "isaacsim.robot.manipulators.examples.franka": {"Franka": object},
        "isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller": {
            "RMPFlowController": RMP,
        },
        "isaacsim.robot_motion.motion_generation": {
            "ArticulationKinematicsSolver": ArticulationKinematics,
            "LulaKinematicsSolver": Kinematics,
            "interface_config_loader": SimpleNamespace(
                load_supported_lula_kinematics_solver_config=lambda robot: {}
            ),
        },
        "isaacsim.sensors.camera": {"Camera": object},
        "pxr": {name: SimpleNamespace() for name in ("Gf", "Sdf", "UsdGeom", "UsdLux", "UsdShade")},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    previous = sys.modules.pop("simulation.isaac_adapter", None)
    adapter = importlib.import_module("simulation.isaac_adapter")
    monkeypatch.setattr(adapter, "observation_barrier", lambda *args, **kwargs: clock[1])
    monkeypatch.setattr(
        adapter,
        "camera_evidence",
        lambda *args, **kwargs: {
            "physics_step": kwargs["physics_step"],
            "cameras": {},
        },
    )
    cell = adapter.IsaacWorkcell()
    joints = Array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.02, 0.02])

    class Robot:
        def __init__(self):
            self.actions = []
            self.efforts = []
            self.joints = joints
            self.end_effector = SimpleNamespace(
                get_world_pose=lambda: (Array([0.35, 0.25, 0.3]), Array([1, 0, 0, 0]))
            )

        def get_joint_positions(self):
            return self.joints.copy()

        def get_world_pose(self):
            return Array([0, 0, 0]), Array([1, 0, 0, 0])

        def set_joint_efforts(self, efforts):
            self.efforts.append(tuple(efforts))

        def apply_action(self, action):
            assert isinstance(action, Action), "Must use the SDK ArticulationAction boundary"
            self.actions.append(action)
            self.joints = Array(action.joint_positions)

    class World:
        def __init__(self):
            self.playing = True

        def is_playing(self):
            return self.playing

        def play(self):
            self.playing = True

        def pause(self):
            self.playing = False

        def render(self):
            pass

        def step(self, *, render):
            clock[1] += 5_000_000

    cell.robot = Robot()
    cell.world = World()
    cell.dynamics = SimpleNamespace(
        get_generalized_gravity_forces=lambda: [Array([1.0] * 7 + [0.0, 0.0])]
    )
    cell.spec = core.spec
    cell.part = SimpleNamespace(
        get_world_pose=lambda: (Array([0.35, 0.25, 0.2]), Array([1, 0, 0, 0]))
    )

    def observation(targets, **kwargs):
        return FrameSample(
            captured_at_utc=(clock[0] + timedelta(seconds=cell.steps / 60))
            .isoformat()
            .replace("+00:00", "Z"),
            monotonic_ns=clock[1],
            physics_step=cell.steps,
            joint_positions=tuple(cell.robot.get_joint_positions()),
            commanded_joint_targets=tuple(targets),
            images={
                name: CameraSample(PNG, cell.steps + 1, cell.steps, clock[1])
                for name in ("inspection", "overview")
            },
            **kwargs,
        )

    monkeypatch.setattr(cell, "_sample_before_command", observation)
    try:
        yield cell, core, request, clock
    finally:
        sys.modules.pop("simulation.isaac_adapter", None)
        if previous is not None:
            sys.modules["simulation.isaac_adapter"] = previous


def test_actual_adapter_holds_without_jog_and_records_exact_zero_velocity_ticks(hardware):
    cell, core, request, clock = hardware
    begin(core, request)
    recorder = Recorder()
    cell.start_teaching(request, core, recorder)
    controller = cell.controller
    original = tuple(cell.robot.get_joint_positions())
    started = clock[1]
    for index in range(12):
        if index == 6:
            clock[1] = started + 100_000_000
        assert not cell.advance()
    cell.stop()
    cell.finish_recording(truncated=True)
    assert controller.forward_calls == 0
    assert len(cell.robot.actions) == 12
    assert all(tuple(action.joint_positions) == original for action in cell.robot.actions)
    assert all(tuple(action.joint_velocities) == (0.0,) * 9 for action in cell.robot.actions)
    assert recorder.sealed and len(recorder.frames) == 2
    assert all(len(frame.applied_controls) == 6 for frame in recorder.frames)
    assert all(
        control.gravity_efforts == (1.0,) * 7 + (0.0, 0.0)
        for frame in recorder.frames
        for control in frame.applied_controls
    )


def test_actual_teaching_jog_does_not_run_a_route_or_recompute_targets_during_hold(hardware):
    cell, core, request, clock = hardware
    begin(core, request)
    core.teaching_input(ACTOR.owner_key, request.session_id, jog(request, clock))
    recorder = Recorder()
    cell.start_teaching(request, core, recorder)
    for _ in range(6):
        cell.advance()
    assert cell.route is None
    assert cell.controller.forward_calls == 1
    assert {tuple(action.joint_positions) for action in cell.robot.actions} == {
        (0.001, 0, 0, -1.57, 0, 1.57, 0, 0.02, 0.02)
    }
    clock[1] += 100_000_000
    cell.advance()
    core.cancel(ACTOR.owner_key, request.command_id)
    cell.stop()
    with pytest.raises(RuntimeError, match="partial"):
        cell.finish_recording(truncated=True)
    count = len(cell.robot.actions)
    assert not cell.advance()
    assert len(cell.robot.actions) == count
    assert recorder.invalid and not recorder.sealed


def test_actual_learned_adapter_applies_model_targets_with_no_reference_controller(hardware):
    cell, core, teaching_request, clock = hardware
    core, request, _ = request_for((core, teaching_request, clock))
    core.dispatch_policy(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command.command_id)
    binding = core.binding(request.command.command_id)
    model = CountingPolicy()
    model.model_sha256 = request.model_sha256
    port = GuardedPolicyAdapter(model, clock_ns=core.clock_ns)
    executor = PolicyExecutor(
        port,
        profile=core.control_profile,
        release_id=request.policy_release_id,
        policy_type=request.policy_type,
        expected_model_sha256=request.model_sha256,
        context=lambda: core.policy_context(binding),
        clock_ns=core.clock_ns,
        apply_guard=lambda submit: core.apply_guarded(binding, submit),
    )
    cell.start_learned(request, core, executor, None)
    for _ in range(6):
        cell.advance()
    assert cell.controller is None and cell.route is None
    assert len(cell.robot.actions) == 6
    assert all(
        tuple(action.joint_positions) == (0.001, 0, 0, -1.57, 0, 1.57, 0, 0.02, 0.02)
        for action in cell.robot.actions
    )
    assert executor.metrics().applied_action_count == 6
    assert executor.metrics().applied_model_sha == request.model_sha256
    assert executor.metrics().policy_predict_calls == 1
    assert executor.metrics().reference_route_calls == 0
    core.cancel(ACTOR.owner_key, request.command.command_id)
    with pytest.raises((ValueError, RuntimeError)):
        cell.advance()
    assert len(cell.robot.actions) == 6


def test_part_placement_is_applied_only_during_a_stopped_scene_reset(hardware):
    from test_learning_scene import environment

    from simulation.extensions import SceneRegistry

    cell, _, _, _ = hardware
    cell.spec = SceneRegistry(load_installed=False).build(environment(37))
    placements, velocities = [], []
    cell.part.set_world_pose = lambda *, position: placements.append(tuple(position))
    cell.part.set_linear_velocity = lambda value: velocities.append(tuple(value))
    cell.part.set_angular_velocity = lambda value: velocities.append(tuple(value))
    cell._place_part_at_reset()
    assert placements == [cell.spec.part_position]
    assert velocities == [(0, 0, 0), (0, 0, 0)]
    cell.control_mode, cell.control_done = "learned", False
    with pytest.raises(RuntimeError, match="reset"):
        cell._place_part_at_reset()
    assert len(placements) == 1


def test_model_reset_failure_cannot_prevent_the_physics_stop(hardware):
    cell, _, _, _ = hardware

    def broken_stop():
        raise RuntimeError("Fixture model reset failed")

    cell.policy_executor = SimpleNamespace(stop=broken_stop)
    with pytest.raises(RuntimeError, match="model reset"):
        cell.stop()
    assert not cell.world.is_playing()


def test_profile_waits_for_its_real_control_period_without_advancing_extra_physics(hardware):
    cell, core, request, clock = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    start = clock[1]
    for _ in range(6):
        cell.advance()
    assert cell.steps == 6
    assert not cell.advance()
    assert cell.steps == 6
    clock[1] = start + 100_000_000
    cell.advance()
    assert cell.steps == 7


def test_teacher_contact_pressure_and_deadman_hold_pass_the_unchanged_student_guard(
    hardware, monkeypatch
):
    cell, core, request, clock = hardware
    begin(core, request)
    original = cell.robot.get_joint_positions

    def contact_positions():
        positions = original()
        positions[7:] = (0.02, 0.02)
        return positions

    monkeypatch.setattr(cell.robot, "get_joint_positions", contact_positions)
    core.teaching_input(
        ACTOR.owner_key,
        request.session_id,
        jog(request, clock, delta_xyz_m=(0, 0, 0), gripper="close"),
    )
    recorder = Recorder()
    cell.start_teaching(request, core, recorder)
    start = clock[1]
    for interval in range(4):
        clock[1] = start + interval * 100_000_000
        for _ in range(6):
            cell.advance()
    frames = [*recorder.frames, cell.hold_capture.completed]
    pressure = frames[0].commanded_joint_targets[7:]
    assert all(0 < 0.02 - target <= 0.004 for target in pressure)
    assert frames[-1].commanded_joint_targets[7:] == pressure
    assert core.teaching_intent(core.binding(request.command_id)) is None

    class Student:
        scope = Scope(str(ACTOR.tenant_id), ACTOR.owner_key)
        fps, physics_hz, chunk_size, n_action_steps = 10, 60, 1, 1
        model_sha256 = "e" * 64

        def reset(self):
            pass

        def predict_chunk(self, observation):
            return (self.teacher_targets,)

    student = Student()
    now = [frames[0].monotonic_ns]
    adapter = GuardedPolicyAdapter(student, clock_ns=lambda: now[0])
    context = ControlContext(
        student.scope,
        request.environment_id,
        request.revision,
        "cross-profile",
        str(request.epoch),
        "cross-profile-command",
        request.target_station_id,
        True,
        True,
        start + 30_000_000_000,
    )
    adapter.reset(context)
    for frame in frames:
        now[0] = frame.monotonic_ns
        student.teacher_targets = frame.commanded_joint_targets
        result = adapter.step(
            PolicyObservation(
                context.scope,
                context.environment_id,
                context.revision,
                context.episode_id,
                frame.captured_at_utc,
                frame.monotonic_ns,
                frame.physics_step,
                frame.joint_positions,
                frame.images,
            ),
            context,
        )
        assert result.targets == frame.commanded_joint_targets


def test_teaching_uses_the_reviewed_right_gripper_frame_not_an_arbitrary_finger_prim(hardware):
    cell, core, request, _ = hardware
    begin(core, request)
    cell.robot.end_effector.get_world_pose = lambda: (
        Array([0.35, 0.25, 0.15]),
        Array([1, 0, 0, 0]),
    )
    cell.start_teaching(request, core, Recorder())
    assert cell._measured_tcp() == (0.35, 0.25, 0.3)


def test_profile_camera_acquires_every_scheduled_render_without_a_second_frequency_gate(hardware):
    cell, _, _, _ = hardware

    class Sensor:
        frequency = 60

        def get_resolution(self):
            return (320, 320)

        def get_frequency(self):
            return self.frequency

        def set_frequency(self, value):
            self.frequency = value

    cell.cameras = {name: Sensor() for name in ("inspection", "overview")}
    cell.control_mode = "human_teaching"
    assert cell._configure_cameras()
    assert all(sensor.frequency == -1 for sensor in cell.cameras.values())
    assert not cell._configure_cameras()


def test_profile_publishes_sensor_state_on_the_sixth_actual_physics_tick(hardware):
    cell, core, request, _ = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    original = cell.world.step
    rendered = []

    def step(*, render):
        rendered.append(render)
        original(render=render)

    cell.world.step = step
    for _ in range(6):
        cell.advance()
    assert cell.steps == 6
    assert rendered == [False, False, False, False, False, True]


def test_control_trace_measures_render_observation_planning_and_six_physics_ticks(
    hardware, monkeypatch
):
    cell, core, request, clock = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    module = sys.modules["simulation.isaac_adapter"]
    sample = cell._sample_before_command
    targets = cell._teaching_targets

    def render(*args, **kwargs):
        clock[1] += 2_000_000
        return clock[1]

    def observe(*args, **kwargs):
        clock[1] += 3_000_000
        return sample(*args, **kwargs)

    def plan(*args, **kwargs):
        clock[1] += 4_000_000
        return targets(*args, **kwargs)

    monkeypatch.setattr(module, "observation_barrier", render)
    monkeypatch.setattr(cell, "_sample_before_command", observe)
    monkeypatch.setattr(cell, "_teaching_targets", plan)
    for _ in range(6):
        cell.advance()
    timing = cell.control_timings[-1]
    assert timing["observation_render_ms"] == 2
    assert timing["observation_read_ms"] == 3
    assert timing["policy_or_teacher_ms"] == 4
    assert timing["physics_and_publish_ms"] == 30
    assert timing["capture_queue_ms"] == 0
    assert timing["control_cycle_ms"] == 39
