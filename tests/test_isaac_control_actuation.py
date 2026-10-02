"""Execute the real adapter methods against CPU SDK/physics doubles, never GPU proof."""

import importlib
import sys
from dataclasses import asdict, replace
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
from runtime_support import ACTOR, PNG
from test_paused_dispatch import paused_core as paused_core
from test_policy_executor import CountingPolicy
from test_policy_runtime import request_for
from test_position_hold_control import Recorder
from test_teaching_runtime import begin, jog
from test_teaching_runtime import teaching as teaching

from learning.contract import CameraSample, FrameSample, Scope
from learning.inference import ControlContext, GuardedPolicyAdapter, PolicyObservation
from simulation.extensions import PausedSceneAuthority
from simulation.paused_gripper_servo import DriveReadback
from simulation.physics_scheduling import PHYSICS_THREAD_SETTING
from simulation.policy_executor import PolicyExecutor
from simulation.reference_targets import GRASP_ASSET_SHA256


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
    settings_values = {
        PHYSICS_THREAD_SETTING: 0,
        "/rtx/hydra/supportMultiTickRate": True,
    }

    class RMP:
        def __init__(self, **kwargs):
            self.robot = kwargs["robot_articulation"]
            self.forward_calls = 0
            self.physics_dt = kwargs["physics_dt"]
            self.rmp_flow = SimpleNamespace(
                ignore_robot_state_updates=False,
                maximum_substep_size=0.00334,
                get_end_effector_pose=lambda joints: (
                    Array([0.35 + 0.1 * joints[0], 0.25, 0.3]),
                    None,
                ),
            )

        def get_articulation_motion_policy(self):
            return SimpleNamespace(get_default_physics_dt=lambda: self.physics_dt)

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
        "carb": {
            "settings": SimpleNamespace(
                get_settings=lambda: SimpleNamespace(get=settings_values.get)
            ),
        },
        "numpy": {
            "array": Array,
            "zeros": lambda size: Array([0.0] * size),
            "concatenate": lambda rows: Array(v for row in rows for v in row),
        },
        "isaacsim.core.api": {"World": object},
        "isaacsim.core.simulation_manager": {
            "SimulationManager": SimpleNamespace(
                get_simulation_time=lambda: cell.world.current_time,
                get_num_physics_steps=lambda: cell.world.current_time_step_index,
            ),
        },
        "isaacsim.core.experimental.utils.stage": {
            "get_current_stage": lambda *, backend: SimpleNamespace(
                GetPrimAtPath=lambda path: SimpleNamespace(
                    GetAttribute=lambda name: SimpleNamespace(Get=lambda: cell.world.current_time)
                )
            ),
        },
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
    cell.test_settings_values = settings_values
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

        def get_joint_velocities(self):
            return Array([0.0] * 9)

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
            self.physics_dt = 1 / 60
            self.rendering_dt = 1 / 60
            self.fabric_flags = []
            self.current_time = 0.0
            self.current_time_step_index = 0

        def set_simulation_dt(self, *, physics_dt, rendering_dt):
            self.physics_dt, self.rendering_dt = physics_dt, rendering_dt

        def get_physics_context(self):
            return SimpleNamespace(device="cpu", is_gpu_dynamics_enabled=lambda: False)

        def is_playing(self):
            return self.playing

        def play(self):
            self.playing = True

        def pause(self):
            self.playing = False

        def render(self):
            pass

        def step(self, *, render, update_fabric=False):
            self.fabric_flags.append(update_fabric)
            self.current_time += self.physics_dt
            self.current_time_step_index += 1
            clock[1] += 5_000_000

    cell.robot = Robot()
    cell.world = World()
    drive = [
        DriveReadback(
            (22918.3125,) * 7 + (400.0, 0.0),
            (4583.66259765625,) * 7 + (80.0, 0.0),
            (87.0,) * 4 + (12.0,) * 3 + (7.199999809265137, 0.0),
        )
    ]

    def write_stiffness(value):
        drive[0] = replace(drive[0], stiffness=drive[0].stiffness[:7] + (value, 0.0))

    # These tests isolate actuation/capture; the native gain setter has its own SDK-bound tests.
    cell._read_paused_drive_snapshot = lambda: drive[0]
    cell._write_paused_finger_stiffness = write_stiffness
    cell._read_paused_gripper_asset = lambda: {
        "archive_sha256": GRASP_ASSET_SHA256,
        "joint_names": tuple(f"panda_joint{i}" for i in range(1, 8))
        + ("panda_finger_joint1", "panda_finger_joint2"),
        "driven_joint_index": 7,
        "driven_joint_type": "PhysicsPrismaticJoint",
        "drive_type": "force",
        "authored_stiffness": 400.0,
        "authored_damping": 80.0,
        "authored_max_force": 7.199999809265137,
        "passive_joint_index": 8,
        "passive_has_drive": False,
        "mimic_axis": "rotX",
        "mimic_gearing": -1.0,
        "mimic_reference": "panda_finger_joint1",
    }
    cell.dynamics = SimpleNamespace(
        get_generalized_gravity_forces=lambda: [Array([1.0] * 7 + [0.0, 0.0])]
    )
    cell.spec = core.spec
    cell.scene_epoch = core.epoch
    cell.part = SimpleNamespace(
        get_world_pose=lambda: (Array([0.35, 0.25, 0.2]), Array([1, 0, 0, 0])),
        get_linear_velocity=lambda: Array([0.0] * 3),
        get_angular_velocity=lambda: Array([0.0] * 3),
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


@pytest.mark.parametrize("unsafe", [False, True])
def test_actual_learned_adapter_applies_model_targets_with_no_reference_controller(
    hardware, monkeypatch, unsafe
):
    cell, core, teaching_request, clock = hardware
    core, request, _ = request_for((core, teaching_request, clock))
    core.dispatch_policy(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command.command_id)
    binding = core.binding(request.command.command_id)
    model = CountingPolicy()
    model.model_sha256 = request.model_sha256
    adapter = sys.modules["simulation.isaac_adapter"]

    def forbidden_planner(*args, **kwargs):
        raise AssertionError("A learned target must never use the reference trajectory planner")

    monkeypatch.setattr(adapter, "plan_reference_targets", forbidden_planner)
    monkeypatch.setattr(adapter, "reference_tcp_target", forbidden_planner)
    monkeypatch.setattr(adapter, "reference_contact_point", forbidden_planner)
    if unsafe:
        measured = tuple(cell.robot.get_joint_positions())
        model.predict_chunk = lambda observation: ((0.08,) + measured[1:],) * 16
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
    if unsafe:
        with pytest.raises(ValueError):
            cell.advance()
        assert not cell.robot.actions and cell.world.current_time_step_index == 0
        assert cell.controller is None
        assert executor.metrics().reference_route_calls == 0
        return
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

    def step(*, render, **kwargs):
        rendered.append(render)
        original(render=render, **kwargs)

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


def test_profile_uses_explicit_physics_clock_and_publishes_fabric_for_every_tick(hardware):
    cell, core, request, _ = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    for _ in range(6):
        cell.advance()
    assert cell.world.physics_dt == 1 / 60
    assert cell.world.rendering_dt == 0
    assert cell.world.fabric_flags == [True] * 6


def test_fixed_unarmed_warmup_is_bounded_and_never_claims_control_intervals(hardware):
    cell, core, _, _ = hardware
    assert core.active_command is None
    cell.prime_control_profile()
    assert cell.steps == 60
    assert len(cell.control_warmup_timings) == 60
    assert not getattr(cell, "control_timings", [])
    assert cell.controller is None
    assert cell.recording is None
    assert all(tuple(action.joint_velocities) == (0.0,) * 9 for action in cell.robot.actions)


def test_fixed_unarmed_warmup_yields_heartbeat_after_each_actual_tick(hardware):
    cell, _, _, _ = hardware
    called = []
    cell.prime_control_profile(on_tick=lambda: called.append(cell.world.current_time_step_index))
    assert called == list(range(1, 61))


def test_world_override_of_control_thread_count_fails_before_unarmed_or_armed_actuation(hardware):
    cell, core, request, _ = hardware
    cell.test_settings_values[PHYSICS_THREAD_SETTING] = 8
    with pytest.raises(RuntimeError, match="numThreads"):
        cell.prime_control_profile()
    assert cell.robot.actions == []
    assert cell.world.fabric_flags == []
    assert cell.physics_scheduling["scene_ready"]["observed_num_threads"] == 8
    begin(core, request)
    with pytest.raises(RuntimeError, match="numThreads"):
        cell.start_teaching(request, core, Recorder())
    assert cell.robot.actions == []


def test_thread_setting_drift_is_detected_before_the_next_control_tick(hardware):
    cell, core, request, _ = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    cell.advance()
    count = len(cell.robot.actions)
    cell.test_settings_values[PHYSICS_THREAD_SETTING] = 8
    with pytest.raises(RuntimeError, match="numThreads"):
        cell.advance()
    assert len(cell.robot.actions) == count
    assert cell.steps == 1


def test_scheduling_readback_time_remains_inside_the_original_control_budget(hardware, monkeypatch):
    cell, core, request, clock = hardware
    begin(core, request)
    cell.start_teaching(request, core, Recorder())
    check = cell._check_control_scheduling

    def observed(phase):
        if phase == "before_control_tick":
            clock[1] += 10_000_000
        check(phase)

    monkeypatch.setattr(cell, "_check_control_scheduling", observed)
    for _ in range(6):
        cell.advance()
    assert cell.control_timings[-1]["control_cycle_ms"] == 90


def test_opted_in_paused_scene_idle_never_advances_unsupervised_physics(hardware):
    cell, _, _, _ = hardware
    cell.spec = replace(
        cell.spec,
        learning_execution=PausedSceneAuthority(
            "physicalai.paused-simulation/v1",
            "paused_simulation",
            "franka-position-hold-10hz-paused-v1",
            30,
            600,
        ),
    )
    before = cell.world.current_time_step_index
    for _ in range(10):
        assert cell.advance() is False
    assert cell.world.current_time_step_index == before
    assert cell.robot.actions == []


def test_legacy_reference_idle_still_advances_as_before(hardware):
    cell, _, _, _ = hardware
    assert cell.spec.learning_execution is None
    before = cell.world.current_time_step_index
    cell.advance()
    assert cell.world.current_time_step_index == before + 1


def test_hidden_parked_scene_mutation_is_not_accepted_as_a_frozen_preview(hardware):
    cell, _, _, _ = hardware
    cell.spec = replace(
        cell.spec,
        learning_execution=PausedSceneAuthority(
            "physicalai.paused-simulation/v1",
            "paused_simulation",
            "franka-position-hold-10hz-paused-v1",
            30,
            600,
        ),
    )
    cell.world.render = lambda: cell.robot.joints.__setitem__(0, 0.001)
    with pytest.raises(RuntimeError, match="frozen"):
        cell.advance()


@pytest.fixture
def paused_hardware(
    hardware,
    paused_core,
):
    cell, original_core, _, _ = hardware
    core, _, request = paused_core
    core.clock_ns = original_core.clock_ns
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command_id)
    cell.spec, cell.scene_epoch = core.spec, core.epoch
    cell.prepare_paused_reference(request, core)
    return cell, core, request


def test_actual_paused_reference_servo_uses_articulation_actions_and_one_explicit_tick(
    paused_hardware,
):
    cell, _, _ = paused_hardware
    targets = cell.paused_reference_targets(
        (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=0
    )
    applied = cell.apply_paused_tick(targets)
    assert len(cell.robot.actions) == 1
    assert tuple(cell.robot.actions[0].joint_positions) == targets
    assert tuple(cell.robot.actions[0].joint_velocities) == (0.0,) * 9
    assert applied.physics_step == cell.world.current_time_step_index == 1
    assert applied.gravity_efforts == (1.0,) * 7 + (0.0, 0.0)
    assert cell.world.fabric_flags == [True]
    assert cell.paused_goal_reached() is False


def test_paused_reference_plans_an_oversized_rmp_endpoint_before_six_exact_applied_targets(
    paused_hardware,
):
    from test_reference_targets import MEASURED_FRAME_1, SECOND_ISSUED

    from simulation.control import validate_position_target

    cell, _, _ = paused_hardware
    cell.robot.joints = Array(MEASURED_FRAME_1)
    cell.issued_targets = SECOND_ISSUED
    requested = list(SECOND_ISSUED[:7])
    requested[5] -= 0.08  # Synthetic endpoint, not the unrecorded third GPU proposal.
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        return Action(Array(requested), Array([0.1] * 7), range(7))

    cell.controller.forward = forward
    targets = cell.paused_reference_targets(
        (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=2
    )
    validate_position_target(MEASURED_FRAME_1, targets, SECOND_ISSUED)
    for _ in range(6):
        applied = cell.apply_paused_tick(targets)
        assert applied.commanded_joint_targets == targets
    assert len(calls) == 1 and cell.controller.physics_dt == 0.1
    assert targets != tuple(requested) + (0.04, 0.04)
    assert len(cell.robot.actions) == 6
    assert all(tuple(item.joint_positions) == targets for item in cell.robot.actions)
    assert all(tuple(item.joint_velocities) == (0.0,) * 9 for item in cell.robot.actions)
    assert cell.world.current_time_step_index == 6
    assert abs(cell.world.current_time - 0.1) < 1e-12
    diagnostic = cell.reference_target_diagnostic
    assert diagnostic["phase"] == "source_approach" and diagnostic["control_tick"] == 2
    assert diagnostic["plan"]["targets"] == targets
    assert diagnostic["last_issued_targets"] == targets
    assert diagnostic["actual_hold_steps"] == 6
    assert diagnostic["rmp_ignores_state_updates"] is False
    assert diagnostic["rmp_maximum_substep_size"] == 0.00334
    assert diagnostic["raw_rmp_joint_velocities"] == (0.1,) * 7


def test_paused_reference_rejection_keeps_bounded_joint_diagnostic_before_any_actuation(
    paused_hardware,
    capsys,
):
    cell, _, _ = paused_hardware
    measured = tuple(cell.robot.get_joint_positions())
    cell.issued_targets = (measured[0] - 0.2,) + measured[1:]
    with pytest.raises((ValueError, RuntimeError)):
        cell.paused_reference_targets(
            (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=2
        )
    diagnostic = cell.reference_target_diagnostic
    assert diagnostic["measured_joint_positions"] == measured
    assert diagnostic["previous_issued_targets"] == cell.issued_targets
    assert diagnostic["rmp_dt"] == 0.1
    assert diagnostic["raw_rmp_joint_positions"][0] == 0.001
    assert diagnostic["limit_violations"][0]["joint_name"] == "panda_joint1"
    assert diagnostic["limit_violations"][0]["constraint"] == "slew"
    assert diagnostic["failure"]
    assert len(capsys.readouterr().out) < 12_000
    assert not cell.robot.actions and cell.world.current_time_step_index == 0


def test_paused_reference_keeps_immutable_successful_interval_trace_separate_from_next_plan(
    paused_hardware,
):
    cell, _, _ = paused_hardware
    targets = cell.paused_reference_targets(
        (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=0
    )
    assert cell.reference_target_evidence()["intervals"] == []
    for _ in range(6):
        cell.apply_paused_tick(targets)
    original = cell.reference_target_evidence()
    assert len(original["intervals"]) == 1
    assert original["intervals"][0]["last_issued_targets"] == targets
    assert original["intervals"][0]["actual_hold_steps"] == 6
    cell.paused_reference_targets(
        (0.35, 0.25, 0.32), False, phase="source_approach", control_tick=1
    )
    later = cell.reference_target_evidence()
    assert later["latest"]["control_tick"] == 1
    assert later["intervals"] == original["intervals"]
    original["intervals"][0]["plan"]["path_fraction"] = -1
    assert cell.reference_target_evidence()["intervals"][0]["plan"]["path_fraction"] > 0


def test_paused_reference_trace_capacity_fails_before_a_new_plan_without_discarding_evidence(
    paused_hardware, monkeypatch
):
    cell, _, _ = paused_hardware
    assert cell.reference_target_evidence()["max_retained_intervals"] == 300
    monkeypatch.setattr(cell, "_reference_trace_limit", lambda: 1)
    targets = cell.paused_reference_targets(
        (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=0
    )
    for _ in range(6):
        cell.apply_paused_tick(targets)
    prior = cell.reference_target_evidence()["intervals"]
    with pytest.raises(ValueError, match="cannot discard"):
        cell.paused_reference_targets(
            (0.35, 0.25, 0.32), False, phase="source_approach", control_tick=1
        )
    assert cell.reference_target_evidence()["intervals"] == prior
    assert cell.world.current_time_step_index == 6
    assert cell.controller.forward_calls == 1


@pytest.mark.parametrize("version,frames,limit_mib", [(1, 300, 4), (2, 600, 8)])
def test_maximum_private_reference_trace_and_all_six_tick_controls_fit_the_receipt_limit(
    paused_hardware,
    tmp_path,
    version,
    frames,
    limit_mib,
):
    from simulation.paused_profiles import paused_profile, read_paused_attempt
    from simulation.probe_control import _persist_receipt

    cell, _, _ = paused_hardware
    measured = (
        0.12345678901234567,
        -0.5123456789012345,
        0.12345678901234567,
        -2.8123456789012345,
        0.12345678901234567,
        3.0123456789012345,
        0.7123456789012345,
        0.02,
        0.02,
    )
    cell.robot.joints = Array(measured)
    cell.robot.get_joint_velocities = lambda: Array([0.12345678901234567] * 9)
    cell.issued_targets = measured
    cell.dynamics.get_generalized_gravity_forces = lambda: [
        Array([86.12345678901234] * 4 + [11.123456789012345] * 3 + [0.0, 0.0])
    ]
    cell.controller.forward = lambda **kw: Action(
        Array([value + 0.12345678901234567 for value in measured[:7]]),
        Array([0.9234567890123456] * 7),
        range(7),
    )
    targets = cell.paused_reference_targets(
        (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=0
    )
    controls = [asdict(cell.apply_paused_tick(targets)) for _ in range(6)]
    trace = cell.reference_target_evidence()["intervals"][0]
    assert len(trace["limit_violations"]) == 14
    # Synthetic serialization stress only: full-length records with all
    # seven arm joints violating both raw-proposal envelopes, never GPU data.
    profile_id = f"franka-position-hold-10hz-paused-v{version}"
    profile = paused_profile("a" * 64, profile_id)
    cell.control_profile = profile
    assert cell.reference_target_evidence()["max_retained_intervals"] == frames
    interval = {
        "freeze_id": "12345678-1234-1234-1234-123456789012",
        "observation_physics_step": 1860,
        "completed_physics_step": 1866,
        "observation_wall_ms": 1234.1234567890123,
        "policy_wall_ms": 1234.1234567890123,
        "hold_wall_ms": 1234.1234567890123,
        "interval_wall_ms": 3702.370370367037,
        "simulated_seconds": 0.10000000000000001,
        "applied_controls": controls,
    }
    report = {
        "schema": "physicalai.paused-reference-attempt/v1",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "control_profile": asdict(profile),
        "control_profile_sha256": profile.sha256,
        "physical_status": "failed",
        "metrics": {"intervals": [interval] * frames},
        "reference_target_evidence": {
            "latest": trace,
            "intervals": [trace] * frames,
            "max_retained_intervals": frames,
        },
        "reserved_other_receipt_metadata": "x" * (64 * 1024),
    }
    path = tmp_path / "bounded-receipt.json"
    _persist_receipt(path, report)
    size = path.stat().st_size
    assert size < limit_mib * 1024 * 1024
    decoded = read_paused_attempt(path, expected_profile_id=profile_id)
    assert len(decoded["reference_target_evidence"]["intervals"]) == frames
    print(f"V{version}_MAX_REFERENCE_RECEIPT_BYTES={size}")


@pytest.mark.parametrize(
    "setting,value",
    [
        ("physics_dt", 1 / 60),
        ("ignore_robot_state_updates", True),
    ],
)
def test_paused_reference_cannot_mask_rmp_horizon_or_measured_feedback_drift(
    paused_hardware, setting, value
):
    cell, _, _ = paused_hardware
    target = cell.controller if setting == "physics_dt" else cell.controller.rmp_flow
    setattr(target, setting, value)
    with pytest.raises(ValueError, match="feedback.*horizon"):
        cell.paused_reference_targets(
            (0.35, 0.25, 0.31), False, phase="source_approach", control_tick=0
        )
    assert cell.controller.forward_calls == 0
    assert not cell.robot.actions


def test_paused_warmup_publishes_original_first_frame_after_the_last_real_tick(
    hardware,
    paused_core,
    monkeypatch,
):
    cell, original_core, _, clock = hardware
    core, _, request = paused_core
    core.clock_ns = original_core.clock_ns
    core.tenant_id = str(ACTOR.tenant_id)
    cell.spec, cell.scene_epoch, cell.paused_scene_core = core.spec, core.epoch, core
    cell.part.get_world_pose = lambda: (Array(cell.spec.part_position), Array([1, 0, 0, 0]))

    class Sensor:
        def __init__(self):
            self.frame = {
                "rendering_frame": {
                    "referenceTimeNumerator": 0,
                    "referenceTimeDenominator": 1_000_000_000,
                },
                "rendering_time": 0.0,
                "rgb": PNG,
            }

        def get_current_frame(self):
            return self.frame

        def get_resolution(self):
            return (320, 320)

        def get_frequency(self):
            return -1

    cell.cameras = {name: Sensor() for name in ("inspection", "overview")}
    phase_order = []

    def render():
        phase_order.append(("render", cell.world.current_time_step_index))
        clock[1] += 1_000_000
        for camera in cell.cameras.values():
            camera.frame.update(
                rendering_time=cell.world.current_time,
                rendering_frame={
                    "referenceTimeNumerator": round(cell.world.current_time * 1_000_000_000),
                    "referenceTimeDenominator": 1_000_000_000,
                },
            )

    original_step = cell.world.step

    def step(**kwargs):
        original_step(**kwargs)
        phase_order.append(("step", cell.world.current_time_step_index))

    cell.world.render, cell.world.step = render, step
    monkeypatch.setattr(cell, "_encode_published_rgb", lambda frame: frame["rgb"])
    # Use the production synchronization helper, not the generic fixture's observation stub.
    from simulation.camera_observation import observation_barrier

    monkeypatch.setattr(
        sys.modules["simulation.isaac_adapter"], "observation_barrier", observation_barrier
    )
    cell.prime_control_profile()
    publication = cell.paused_publications.publication
    assert publication is not None
    assert publication.frozen_state.physics_step == 60
    assert len(cell.control_warmup_timings) == 60
    assert phase_order[-2:] == [("step", 60), ("render", 60)]
    assert publication.freeze_established_ns <= publication.joint_sample_ns
    assert dict(publication.images)["inspection"].png == PNG
    before = cell.world.current_time_step_index
    core.dispatch_simulation_episode(ACTOR.owner_key, request)
    core.next_action()
    core.begin_motion(request.command_id)
    from simulation.paused_control import PausedEpisode

    initial = cell.frozen_physics_state(core.epoch)
    episode = PausedEpisode(
        initial,
        wall_deadline_ns=core.monotonic_deadlines[core.active_command],
        max_simulation_steps=1800,
        authorized=lambda: True,
        clock_ns=core.clock_ns,
    )
    episode.begin_observation(initial)
    observed = cell.paused_observation(request, core, episode, 0)
    assert observed.initial_publication is not None
    assert observed.monotonic_ns == publication.published_ns
    assert (
        observed.images["inspection"].monotonic_ns
        == dict(publication.images)["inspection"].monotonic_ns
    )
    assert cell.world.current_time_step_index == before
