from datetime import timedelta
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_models import (
    ArmTeaching,
    CreateDataset,
    JogIntent,
    StartTeaching,
    TeachingControl,
)
from apps.api.learning_ports import RuntimeCapture, TeachingRuntimeState
from apps.api.learning_service import LearningService
from apps.api.models import DemonstrationResult, Execution, utcnow
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import ACTOR


class TeachingRuntimeStub:
    def __init__(self):
        self.starts = []
        self.inputs = []
        self.controls = []
        self.state = None
        self.error = None

    def start_teaching(self, owner, body):
        self.starts.append(body)
        self.state = TeachingRuntimeState(
            session_id=body.session_id,
            lease_id=body.lease_id,
            epoch=body.epoch,
            command_id=body.command_id,
            control_profile_id=body.control_profile_id,
            status="running",
            last_sequence=0,
            execution=Execution(command_id=body.command_id, status="running"),
        )
        return self.state

    def teaching(self, owner, session_id):
        return self.state

    def capture(self, owner, command_id):
        assert command_id == self.state.command_id
        return self.state.capture

    def teaching_input(self, owner, session_id, body):
        self.inputs.append(body)
        if self.error:
            raise self.error
        self.state = self.state.model_copy(update={"last_sequence": body["sequence"]})
        return self.state

    def finish_teaching(self, owner, session_id, body):
        self.controls.append(("finish", body))
        self.state = self.state.model_copy(
            update={
                "status": "succeeded",
                "execution": Execution(
                    command_id=self.state.command_id,
                    status="succeeded",
                    completed_at=utcnow(),
                    final_position=(0.42, -0.22, 0.2),
                ),
                "capture": RuntimeCapture(
                    capture_id=uuid4(),
                    command_id=self.state.command_id,
                    epoch=self.state.epoch,
                    status="finalizing",
                ),
            }
        )
        return self.state

    def cancel_teaching(self, owner, session_id, body):
        self.controls.append(("cancel", body))
        self.state = self.state.model_copy(update={"status": "cancelling"})
        return self.state


def setup(source="human_teleop"):
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    project, _, _ = seed_project_and_dataset(store, request)
    runtime = TeachingRuntimeStub()
    service = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
        allowed_policy_types=("gr00t_n1_5",),
        runtime=runtime,
    )
    started = service.start_teaching(
        ACTOR,
        project.value.id,
        StartTeaching(request_id=uuid4(), source=source, motion_approved=True),
        project.etag,
    )
    return service, project, started, runtime


def intent(service, stored, **overrides):
    session = stored.value
    body = ArmTeaching(
        request_id=uuid4(),
        lease_id=session.lease_id,
        epoch=session.epoch,
        sequence=session.last_sequence + 1,
        deadman=True,
        delta_xyz_m=(0.005, 0, 0),
        gripper="hold",
    )
    grant = service.arm(ACTOR, session.id, body, stored.etag)
    return JogIntent(
        request_id=uuid4(),
        lease_id=session.lease_id,
        epoch=session.epoch,
        sequence=session.last_sequence + 1,
        deadman=True,
        delta_xyz_m=(0.005, 0, 0),
        gripper="hold",
        grant_id=grant.value.id,
        **overrides,
    )


def test_teaching_session_is_separate_from_thirty_second_motion_and_preserves_real_source():
    service, project, started, runtime = setup("reference_controller")
    body = runtime.starts[0]
    assert body.demonstrator_kind == "reference_controller"
    assert "command" not in body.model_dump() and "deadline" not in body.model_dump()
    assert 115 < (body.session_expires_at - utcnow()).total_seconds() <= 120
    assert started.value.expires_at == body.session_expires_at
    assert body.task.instruction == project.value.instruction
    assert body.task.goal_id == project.value.goal_station_id
    assert started.value.status == "recording"
    assert runtime.inputs == []
    assert service.store.get_learning(ACTOR.owner_key, "teaching", started.value.id)


def test_jog_expiry_is_server_issued_once_and_retry_does_not_move_or_extend_authority():
    service, _, started, runtime = setup()
    request = intent(service, started)
    before = utcnow()
    moved = service.jog(ACTOR, started.value.id, request, started.etag)
    after = utcnow()
    deadline = moved.value.input_expires_at
    assert before + timedelta(milliseconds=250) <= deadline <= after + timedelta(milliseconds=250)
    assert runtime.inputs[0]["expires_at"] == deadline.isoformat().replace("+00:00", "Z")
    repeated = service.jog(ACTOR, started.value.id, request, started.etag)
    assert repeated.value.input_expires_at == deadline
    assert len(runtime.inputs) == 1
    with pytest.raises(Problem) as failure:
        service.jog(
            ACTOR, started.value.id, request.model_copy(update={"request_id": uuid4()}), moved.etag
        )
    assert failure.value.code == "teaching_sequence"


def test_uncertain_input_is_never_resent_and_stale_lease_never_revived():
    service, _, started, runtime = setup()
    request = intent(service, started)
    runtime.error = Problem(503, "dependency_unavailable", "Input acknowledgement lost")
    with pytest.raises(Problem):
        service.jog(ACTOR, started.value.id, request, started.etag)
    with pytest.raises(Problem) as failure:
        service.jog(ACTOR, started.value.id, request, started.etag)
    assert failure.value.code == "learning_operation_unconfirmed"
    assert len(runtime.inputs) == 1
    current = service.get(ACTOR, "teaching", started.value.id)
    service.store.put_learning(
        ACTOR.owner_key,
        current.value.model_copy(update={"expires_at": utcnow() - timedelta(seconds=1)}),
        current.etag,
    )
    with pytest.raises(Problem) as failure:
        service.jog(
            ACTOR,
            started.value.id,
            request.model_copy(
                update={
                    "request_id": uuid4(),
                    "sequence": current.value.last_sequence + 1,
                }
            ),
            current.etag,
        )
    assert failure.value.code == "teaching_expired"
    assert len(runtime.inputs) == 1


def test_delayed_first_jog_cannot_gain_fresh_authority_on_arrival(monkeypatch):
    service, _, started, runtime = setup()
    request = intent(service, started)
    grant = service.get(ACTOR, "control_grant", request.grant_id)
    delayed = grant.value.expires_at + timedelta(milliseconds=1)
    monkeypatch.setattr("apps.api.learning_service.utcnow", lambda: delayed)
    with pytest.raises(Problem) as failure:
        service.jog(ACTOR, started.value.id, request, started.etag)
    assert failure.value.code == "teaching_grant_expired"
    assert runtime.inputs == []
    assert service.get(ACTOR, "teaching", started.value.id).value.last_sequence == 0


def test_arm_retry_cannot_renew_old_nonce_or_move_the_robot(monkeypatch):
    service, _, started, runtime = setup()
    body = ArmTeaching(
        request_id=uuid4(),
        lease_id=started.value.lease_id,
        epoch=started.value.epoch,
        sequence=1,
        deadman=True,
        delta_xyz_m=(0.005, 0, 0),
        gripper="hold",
    )
    grant = service.arm(ACTOR, started.value.id, body, started.etag)
    monkeypatch.setattr(
        "apps.api.learning_service.utcnow", lambda: grant.value.expires_at + timedelta(seconds=1)
    )
    repeated = service.arm(ACTOR, started.value.id, body, started.etag)
    assert repeated.value.id == grant.value.id
    assert repeated.value.expires_at == grant.value.expires_at
    assert runtime.inputs == []


def test_deadman_release_preempts_delayed_lower_sequence_without_a_motion_grant():
    service, _, started, runtime = setup()
    delayed = intent(service, started)
    released = JogIntent(
        request_id=uuid4(),
        lease_id=started.value.lease_id,
        epoch=started.value.epoch,
        sequence=2,
        deadman=False,
        delta_xyz_m=(0, 0, 0),
        gripper="hold",
    )
    result = service.jog(ACTOR, started.value.id, released, "stale-stop-only-etag")
    assert result.value.last_sequence == 2
    assert runtime.inputs[0]["deadman"] is False
    with pytest.raises(Problem) as failure:
        service.jog(ACTOR, started.value.id, delayed, result.etag)
    assert failure.value.code == "teaching_sequence"
    assert len(runtime.inputs) == 1


def test_physical_finish_is_visible_without_blocking_on_upload_and_cannot_enter_a_dataset():
    service, project, started, runtime = setup()
    finish = TeachingControl(
        request_id=uuid4(),
        lease_id=started.value.lease_id,
        epoch=started.value.epoch,
    )
    result = service.control_teaching(ACTOR, started.value.id, finish, started.etag, "finish")
    assert result.value.physical_status == "succeeded"
    assert result.value.status == "finalizing"
    assert result.value.capture is None
    with pytest.raises(Problem) as failure:
        service.dataset(
            ACTOR,
            project.value.id,
            CreateDataset(
                request_id=uuid4(),
                teaching_session_ids=(started.value.id,),
            ),
            project.etag,
        )
    assert failure.value.code == "capture_not_ready"
    runtime.state = runtime.state.model_copy(
        update={
            "capture": runtime.state.capture.model_copy(update={"status": "uploading"}),
        }
    )
    assert service.get_teaching(ACTOR, started.value.id).value.status == "uploading"


def test_ready_capture_must_match_owner_task_command_epoch_and_source():
    service, _, started, runtime = setup("reference_controller")
    runtime.state = runtime.state.model_copy(
        update={
            "capture": RuntimeCapture(
                capture_id=uuid4(),
                command_id=uuid4(),
                epoch=started.value.epoch,
                status="ready",
                receipt=DemonstrationResult(
                    status="uploaded",
                    manifest_uri="https://test.blob.core.windows.net/private/manifest.json",
                    manifest_sha256="a" * 64,
                    episode_id=uuid4(),
                    frame_count=20,
                ),
            ),
        }
    )
    with pytest.raises(Problem) as failure:
        service.get_teaching(ACTOR, started.value.id)
    assert failure.value.code == "capture_scope_mismatch"
    assert service.get(ACTOR, "teaching", started.value.id).value.status != "ready"


def test_completed_capture_is_reconciled_by_historical_command_after_another_scene_started():
    service, _, started, runtime = setup()
    finish = TeachingControl(
        request_id=uuid4(),
        lease_id=started.value.lease_id,
        epoch=started.value.epoch,
    )
    result = service.control_teaching(ACTOR, started.value.id, finish, started.etag, "finish")
    assert result.value.physical_status == "succeeded"

    def different_active_scene(*args):
        raise Problem(409, "scene_changed", "Scene B replaced the live teaching scene.")

    runtime.teaching = different_active_scene
    runtime.state = runtime.state.model_copy(
        update={
            "capture": runtime.state.capture.model_copy(update={"status": "uploading"}),
        }
    )
    upload = service.get_teaching(ACTOR, started.value.id)
    assert upload.value.status == "uploading"
    assert upload.value.physical_status == "succeeded"
    assert upload.value.epoch == started.value.epoch
