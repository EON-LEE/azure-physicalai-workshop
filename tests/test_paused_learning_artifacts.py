import json
import shutil
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import CaptureReceipt, CreateProject, LearningProject, TrainingParent
from apps.learning_worker.artifacts import VerifiedArtifacts
from learning.common import file_digest
from learning.contract import DemonstrationSource, EpisodeSpec, Scope
from learning.paused.capture import PausedEpisodeBudget, PausedEpisodeWriter
from tests.learning.test_paused_capture import shifted_sample
from tests.learning.test_paused_contract import profile
from tests.learning.test_paused_model_artifacts import fixture as model_fixture
from tests.runtime_support import ACTOR
from tests.test_paused_learning_api import paused_project_payload


def project():
    value = paused_project_payload()
    value.update(
        control_profile_sha256=profile().sha256,
        criteria_sha256="d" * 64,
        frozen_plan_sha256="e" * 64,
        instruction="Move the part to the approved tray.",
        goal_station_id="rejected",
    )
    return LearningProject.create(ACTOR, CreateProject.model_validate(value))


def raw_episode(root, project, *, purpose="demonstration"):
    from learning.checks.fixtures import PROVENANCE

    case = project.teaching_cases[0]
    scope = Scope(str(ACTOR.tenant_id), ACTOR.owner_key)
    episode_id = str(uuid4())
    writer = PausedEpisodeWriter(
        root,
        dataset_id=episode_id,
        scope=scope,
        episode=EpisodeSpec(
            episode_id,
            case.environment_id,
            case.revision,
            case.seed,
            "test" if purpose == "integration" else case.split,
        ),
        provenance=PROVENANCE,
        profile=profile(),
        demonstration=DemonstrationSource(
            kind="reference_controller",
            task_id=project.task_id,
            instruction=project.instruction,
            goal_id=project.goal_station_id,
        ),
        budget=PausedEpisodeBudget(1_000_000_000, 601_000_000_000, 60, 1860),
        purpose=purpose,
        criteria_sha256=project.criteria_sha256,
        frozen_plan_sha256=project.frozen_plan_sha256,
    )
    for index in range(2):
        sample = shifted_sample(index, terminal=index == 1)
        observation = replace(
            sample.observation,
            scope=scope,
            environment_id=case.environment_id,
            revision=case.revision,
            episode_id=episode_id,
        )
        writer.append(replace(sample, observation=observation))
    writer.finalize()
    return episode_id, file_digest(root / "manifest.json")


def capture_context(tmp_path, *, purpose="demonstration"):
    current = project()
    root = tmp_path / "native-v3"
    episode_id, digest = raw_episode(root, current, purpose=purpose)
    uploads = []
    registry = SimpleNamespace(upload=lambda *args: uploads.append(args))
    verifier = VerifiedArtifacts(
        registry,
        None,
        "https://testcapture.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    verifier._download_prefix = lambda _account, _container, _prefix, output, **_: shutil.copytree(
        root, output
    )
    session = SimpleNamespace(
        owner_key=ACTOR.owner_key,
        project_id=current.id,
        teaching_case=current.teaching_cases[0],
        source="reference_controller",
        command_id=UUID(episode_id),
    )
    receipt = {
        "status": "uploaded",
        "manifest_sha256": digest,
        "episode_id": episode_id,
        "frame_count": 2,
        "manifest_uri": f"https://testcapture.blob.core.windows.net/demonstrations/{ACTOR.owner_key}/{episode_id}/manifest.json",
    }
    return verifier, current, session, receipt, uploads, root


def test_paused_capture_receipt_requires_full_mode_provenance_and_keeps_reference_source():
    current = project()
    case = current.teaching_cases[0]
    fields = {
        "episode_id": uuid4(),
        "manifest_sha256": "f" * 64,
        "artifact_id": uuid4(),
        "frame_count": 2,
        "source": "reference_controller",
        "seed": case.seed,
        "case_id": case.case_id,
        "environment_id": case.environment_id,
        "revision": case.revision,
        "split": case.split,
        "task_id": current.task_id,
        "control_profile_id": current.control_profile_id,
        "execution_timing": current.execution_timing,
        "real_time_admission": False,
        "control_profile_sha256": current.control_profile_sha256,
        "criteria_sha256": current.criteria_sha256,
        "frozen_plan_sha256": current.frozen_plan_sha256,
    }
    capture = CaptureReceipt.model_validate(fields)
    assert capture.authorized_case(current) == case
    assert capture.source == "reference_controller"
    for changed in ({"real_time_admission": True}, {"criteria_sha256": None}):
        with pytest.raises(ValidationError):
            CaptureReceipt.model_validate(fields | changed)
    wrong = capture.model_copy(update={"frozen_plan_sha256": "c" * 64})
    with pytest.raises(Problem):
        wrong.authorized_case(current)


def test_actual_v3_cpu_fixture_cannot_be_admitted_as_live_capture(tmp_path):
    verifier, current, session, receipt, uploads, _ = capture_context(tmp_path)
    with pytest.raises(Problem) as failure:
        verifier.verify_capture(ACTOR, current, session, receipt)
    assert failure.value.code == "capture_invalid"
    assert uploads == []


def test_paused_capture_from_another_command_never_enters_this_teaching_session(tmp_path):
    verifier, current, session, receipt, uploads, _ = capture_context(tmp_path)
    session.command_id = uuid4()
    with pytest.raises(Problem) as failure:
        verifier.verify_capture(ACTOR, current, session, receipt)
    assert failure.value.code == "capture_provenance_mismatch"
    assert uploads == []


def test_worker_selects_real_v3_validator_with_live_and_demonstration_requirements(
    tmp_path, monkeypatch
):
    from learning.paused import capture as native

    verifier, current, session, receipt, uploads, _ = capture_context(tmp_path)
    validate = native.validate_dataset
    calls = []

    def cpu_fixture_only(root, **kwargs):
        calls.append(kwargs)
        assert kwargs["require_live"] is True
        assert kwargs["require_demonstrations"] is True
        return validate(root, **{**kwargs, "require_live": False})

    monkeypatch.setattr(native, "validate_dataset", cpu_fixture_only)
    verified = verifier.verify_capture(ACTOR, current, session, receipt)
    assert len(calls) == len(uploads) == 1
    assert verified.execution_timing == "paused_simulation"
    assert verified.real_time_admission is False
    assert verified.criteria_sha256 == current.criteria_sha256
    assert verified.frozen_plan_sha256 == current.frozen_plan_sha256
    assert verified.control_profile_sha256 == profile().sha256
    assert verified.source == "reference_controller"


@pytest.mark.parametrize("purpose", ["integration", "evaluation"])
def test_worker_never_admits_integration_or_evaluation_capture_as_training_data(
    tmp_path, monkeypatch, purpose
):
    from learning.paused import capture as native

    verifier, current, session, receipt, uploads, _ = capture_context(tmp_path, purpose=purpose)
    validate = native.validate_dataset
    monkeypatch.setattr(
        native,
        "validate_dataset",
        lambda root, **kwargs: validate(root, **{**kwargs, "require_live": False}),
    )
    with pytest.raises(Problem) as failure:
        verifier.verify_capture(ACTOR, current, session, receipt)
    assert failure.value.code == "capture_invalid" and uploads == []


def test_sealing_preserves_actual_v3_purpose_profile_and_both_clocks(tmp_path, monkeypatch):
    from learning.paused import capture as native

    verifier, current, session, receipt, uploads, root = capture_context(tmp_path)
    validate, assemble = native.validate_dataset, native.assemble_dataset
    monkeypatch.setattr(
        native,
        "validate_dataset",
        lambda path, **kwargs: validate(path, **{**kwargs, "require_live": False}),
    )
    assembled = []

    def test_only_assembly(roots, output, **kwargs):
        assert kwargs["require_live"] is True
        result = assemble(roots, output, **{**kwargs, "require_live": False})
        assembled.append(json.loads(result.read_text()))
        return result

    monkeypatch.setattr(native, "assemble_dataset", test_only_assembly)
    verified = verifier.verify_capture(ACTOR, current, session, receipt)
    registry = verifier.registry

    def download(actor, artifact_id, output):
        shutil.copytree(root, output)
        return {"manifest_sha256": receipt["manifest_sha256"]}

    registry.download = download
    dataset_id = uuid4()
    assert verifier.seal_dataset(ACTOR, current, dataset_id, (verified,))[0] == dataset_id
    assert len(assembled) == 1 and len(uploads) == 2
    assert assembled[0]["schema"] == "physicalai.demonstrations/v3"
    assert assembled[0]["timestamp_basis"] == "simulation_time"
    assert assembled[0]["purpose"] == "demonstration"
    assert assembled[0]["criteria_sha256"] == current.criteria_sha256
    assert assembled[0]["frozen_plan_sha256"] == current.frozen_plan_sha256


@pytest.mark.parametrize(
    "field", ["criteria_sha256", "frozen_plan_sha256", "control_profile_sha256"]
)
def test_capture_cannot_adopt_a_different_project_provenance(tmp_path, monkeypatch, field):
    from learning.paused import capture as native

    verifier, current, session, receipt, uploads, _ = capture_context(tmp_path)
    validate = native.validate_dataset
    monkeypatch.setattr(
        native,
        "validate_dataset",
        lambda root, **kwargs: validate(root, **{**kwargs, "require_live": False}),
    )
    changed = current.model_copy(update={field: "a" * 64})
    with pytest.raises(Problem) as failure:
        verifier.verify_capture(ACTOR, changed, session, receipt)
    assert failure.value.code == "capture_provenance_mismatch"
    assert uploads == []


def test_worker_does_not_route_a_paused_checkpoint_through_legacy_validator(tmp_path):
    digest = model_fixture(tmp_path)
    value = json.loads((tmp_path / "model.json").read_text())
    value["scope"] = {"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key}
    (tmp_path / "model.json").write_text(json.dumps(value))
    digest = file_digest(tmp_path / "model.json")
    verifier = VerifiedArtifacts(
        None,
        None,
        "https://testcapture.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    with pytest.raises(Problem):
        verifier._model(ACTOR, tmp_path, digest, "smolvla")
    model = verifier._model(
        ACTOR, tmp_path, digest, "smolvla", execution_timing="paused_simulation"
    )
    assert model["schema"] == "physicalai.smolvla-checkpoint/v2"
    assert model["training"]["raw_schema"] == "physicalai.demonstrations/v3"
    assert model["real_time_admission"] is False


def test_paused_train_only_parent_record_retains_mode_pins_without_changing_legacy_shape():
    from apps.api.models import utcnow

    current = project()
    fields = {
        "id": uuid4(),
        "owner_key": ACTOR.owner_key,
        "actor_id": ACTOR.object_id,
        "tenant_id": ACTOR.tenant_id,
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "fingerprint": "a" * 64,
        "policy_type": "smolvla",
        "artifact_id": uuid4(),
        "model_sha256": "a" * 64,
        "processor_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "model_revision": "d" * 40,
        "registered_by": ACTOR.object_id,
    }
    original = TrainingParent.model_validate(fields)
    assert "execution_timing" not in original.model_dump(mode="json")
    paused = TrainingParent.model_validate(
        fields
        | {
            "execution_timing": "paused_simulation",
            "real_time_admission": False,
            "control_profile_id": current.control_profile_id,
            "control_profile_sha256": current.control_profile_sha256,
            "criteria_sha256": current.criteria_sha256,
            "frozen_plan_sha256": current.frozen_plan_sha256,
        }
    )
    assert paused.role == "pretrained_train_only"
    assert paused.real_time_admission is False
    assert paused.control_profile_sha256 == profile().sha256
    with pytest.raises(ValidationError):
        TrainingParent.model_validate(
            paused.model_dump(mode="json") | {"real_time_admission": True}
        )


def test_paused_registration_rejects_fake_vendor_bytes_and_records_only_injected_verified_metadata(
    tmp_path,
    monkeypatch,
):
    model_fixture(tmp_path)
    value = json.loads((tmp_path / "model.json").read_text())
    value.update(
        role="pretrained",
        training=None,
        scope={"tenant_id": str(ACTOR.tenant_id), "owner_id": ACTOR.owner_key},
    )
    (tmp_path / "model.json").write_text(json.dumps(value))
    digest = file_digest(tmp_path / "model.json")
    stored, uploads = {}, []

    def put(actor, path, record):
        if path in stored:
            assert stored[path] == record
        stored[path] = record
        return True

    registry = SimpleNamespace(
        get=lambda actor, key: stored.get(key),
        put=put,
        upload=lambda *args: uploads.append(args),
    )
    verifier = VerifiedArtifacts(
        registry,
        None,
        "https://testcapture.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    with pytest.raises(Problem):
        verifier.register_parent(ACTOR, tmp_path, digest, ACTOR.object_id)
    with pytest.raises(Problem) as failure:
        verifier.register_parent(
            ACTOR, tmp_path, digest, ACTOR.object_id, execution_timing="paused_simulation"
        )
    assert failure.value.code == "invalid_model_artifact"
    assert uploads == []

    def metadata_fixture(actor, root, expected_sha, policy_type, *, inference, execution_timing):
        assert execution_timing == "paused_simulation" and inference is False
        assert expected_sha == digest and policy_type == "smolvla"
        return value

    monkeypatch.setattr(verifier, "_model", metadata_fixture)
    first = verifier.register_parent(
        ACTOR, tmp_path, digest, ACTOR.object_id, execution_timing="paused_simulation"
    )
    second = verifier.register_parent(
        ACTOR, tmp_path, digest, ACTOR.object_id, execution_timing="paused_simulation"
    )
    assert first == second
    assert first.control_profile_sha256 == profile().sha256
    assert first.criteria_sha256 == value["criteria_sha256"]
    assert first.frozen_plan_sha256 == value["frozen_plan_sha256"]
    assert first.role == "pretrained_train_only"
    assert len(uploads) == 1


def test_original_frozen_job_config_controls_output_not_a_later_project_approval(tmp_path):
    from tests.test_learning_worker import specification

    spec, _ = specification()
    prefix = f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/outputs"
    config = {
        "schema": "physicalai.smolvla-azure/v2",
        "output_prefix": prefix,
        "owner_id": ACTOR.owner_key,
        "storage_account_name": "testoutput",
        "blob_container": "artifacts",
    }
    registry = SimpleNamespace(
        job_configuration=lambda *_: config,
        approved_plan=lambda *_: pytest.fail("Do not reopen a changed per-project approval."),
    )
    verifier = VerifiedArtifacts(
        registry, None, "https://testcapture.blob.core.windows.net", "demonstrations"
    )
    calls = []
    verifier._download_prefix = lambda *args, **kwargs: calls.append(args)
    assert verifier._output(ACTOR, spec, "model", tmp_path) is config
    assert calls[0][2] == f"{prefix}/{spec.run.backend_job_name}/model/"
