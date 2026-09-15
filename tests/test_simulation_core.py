import base64
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from runtime_support import ACTOR, OTHER, PNG, document, service

from apps.api.errors import Problem
from apps.api.models import MotionCommand, SaveEnvironment, utcnow
from simulation.core import LoadScene, SimulationCore, StartMotion, StopMotion
from simulation.extensions import SceneBuilder, SceneRegistry


@pytest.fixture
def ready_core():
    backend = service()
    environment = backend.save_environment(
        ACTOR, SaveEnvironment(document_json=json.dumps(document()))
    )
    core = SimulationCore(SceneRegistry(load_installed=False))
    core.activate(ACTOR.owner_key, environment)
    assert isinstance(core.next_action(), LoadScene)
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (-0.5, 0, 0.2), 5)
    return core, environment


def command(core, environment):
    observation = core.observe(
        ACTOR.owner_key, environment.environment_id, environment.revision, "inspection"
    )
    return MotionCommand(
        command_id=uuid4(),
        environment_id=environment.environment_id,
        revision=environment.revision,
        epoch=core.epoch,
        state_revision=core.state_revision,
        observation_id=observation.observation_id,
        object_id="part-001",
        target_station_id="rejected",
        deadline=utcnow() + timedelta(seconds=20),
    )


def test_owning_scene_and_camera_data_are_isolated(ready_core):
    core, environment = ready_core
    assert core.status(OTHER.owner_key).status == "occupied"
    assert core.status(OTHER.owner_key).environment_id is None
    with pytest.raises(Problem):
        core.observe(OTHER.owner_key, environment.environment_id, environment.revision, "overview")
    actual = core.observe(
        ACTOR.owner_key, environment.environment_id, environment.revision, "overview"
    )
    assert base64.b64decode(actual.image_base64) == PNG
    assert "defective" not in actual.model_dump()


def test_idempotent_dispatch_and_changed_payload_rejection(ready_core):
    core, environment = ready_core
    motion = command(core, environment)
    assert core.dispatch(ACTOR.owner_key, motion).status == "queued"
    assert core.dispatch(ACTOR.owner_key, motion).status == "queued"
    assert isinstance(core.next_action(), StartMotion)
    assert core.next_action() is None
    with pytest.raises(Problem, match="reused"):
        core.dispatch(ACTOR.owner_key, motion.model_copy(update={"target_station_id": "accepted"}))


@pytest.mark.parametrize(
    "update",
    [
        {"epoch": uuid4()},
        {"state_revision": 999},
        {"observation_id": uuid4()},
        {"target_station_id": "inspection"},
        {"object_id": "another-part"},
        {"deadline": utcnow() - timedelta(seconds=1)},
        {"deadline": utcnow() + timedelta(hours=1)},
    ],
)
def test_invalid_motion_never_enters_the_execution_queue(ready_core, update):
    core, environment = ready_core
    with pytest.raises(Problem):
        core.dispatch(ACTOR.owner_key, command(core, environment).model_copy(update=update))
    assert core.next_action() is None
    assert core.active_command is None


def test_cancellation_preempts_a_queued_motion(ready_core):
    core, environment = ready_core
    motion = command(core, environment)
    core.dispatch(ACTOR.owner_key, motion)
    assert core.cancel(ACTOR.owner_key, motion.command_id).status == "cancelling"
    assert isinstance(core.next_action(), StopMotion)
    assert not core.begin_motion(motion.command_id)
    core.finish("cancelled", None)
    assert core.command(ACTOR.owner_key, motion.command_id).status == "cancelled"
    core.finish("succeeded", (0.5, -0.4, 0.2))
    assert core.command(ACTOR.owner_key, motion.command_id).status == "cancelled"


def test_new_scene_invalidates_old_observations(ready_core):
    core, environment = ready_core
    motion = command(core, environment)
    core.activate(ACTOR.owner_key, environment)
    core.next_action()
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (-0.5, 0, 0.2), 5)
    with pytest.raises(Problem, match="world state"):
        core.dispatch(ACTOR.owner_key, motion)


def test_command_capacity_fails_closed_instead_of_forgetting_idempotency(ready_core):
    core, environment = ready_core
    core.max_commands = 1
    motion = command(core, environment)
    core.dispatch(ACTOR.owner_key, motion)
    core.next_action()
    core.begin_motion(motion.command_id)
    core.finish("succeeded", (0.5, -0.4, 0.2))
    for camera in ("overview", "inspection"):
        core.publish_frame(camera, PNG, (-0.5, 0, 0.2), 10)
    with pytest.raises(Problem, match="capacity"):
        core.dispatch(ACTOR.owner_key, command(core, environment))


def test_unknown_template_and_unsupported_robot_fail_instead_of_selecting_defaults():
    backend = service()
    for field, value in (("template_id", "unknown"), ("robot_profile", "unknown")):
        doc = document()
        doc["scene"][field] = value
        record = (
            backend.save_environment(
                ACTOR,
                SaveEnvironment(document_json=json.dumps(doc)),
            )
            if field == "template_id"
            else service().save_environment(
                ACTOR,
                SaveEnvironment(document_json=json.dumps(doc)),
            )
        )
        with pytest.raises(Problem):
            SceneRegistry(load_installed=False).build(record)


def test_reviewed_extension_api_version_is_enforced():
    class Incompatible(SceneBuilder):
        api_version = "999"

        def build(self, environment):
            raise AssertionError("An incompatible builder must never run")

    with pytest.raises(ValueError, match="Unsupported"):
        SceneRegistry(load_installed=False).register("custom-v999", Incompatible())
