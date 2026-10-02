import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import (
    CaptureReceipt,
    CreateDataset,
    CreateProject,
    LearningProject,
    StartTeaching,
    StartTraining,
)
from apps.api.learning_service import LearningService
from apps.api.models import SaveEnvironment
from tests.learning_api_support import learning_setup, seed_project_and_dataset
from tests.runtime_support import ACTOR, OTHER, document
from tests.test_learning_teaching import TeachingRuntimeStub

TASK = "manufacturing-part-placement-v1"
INSTRUCTION = (
    "Pick up the synthetic part from the source platform and place it in the quarantine tray."
)


def varied_context():
    factory, store, jobs, artifacts, catalog, template = learning_setup()

    def saved_case(seed, split, *, actor=ACTOR):
        value = document()
        value["environment_id"] = f"{split}-{seed}"
        value["scene"].update(template_id="inspection-cell-learning-v1", seed=seed)
        value["execution"].update(record_demonstration=True, demonstration_split=split)
        current = store.get_environment(actor.owner_key, value["environment_id"])
        saved = factory.save_environment(
            actor,
            SaveEnvironment(
                document_json=json.dumps(value),
                expected_revision=current.value.revision if current else None,
            ),
        )
        return {
            "case_id": f"{split}-{seed}",
            "environment_id": saved.environment_id,
            "revision": saved.revision,
            "seed": seed,
            "split": split,
        }

    cases = [
        saved_case(10001, "train"),
        saved_case(10002, "train"),
        saved_case(20001, "validation"),
    ]
    held_out = [saved_case(seed, "test") for seed in range(30001, 30021)]
    plan = {
        **template.evaluation_plan.model_dump(mode="json"),
        "seeds": [item["seed"] for item in held_out],
        "cases": [
            {key: item[key] for key in ("environment_id", "revision", "seed")} for item in held_out
        ],
    }
    raw = {
        **template.model_dump(mode="json"),
        "task_id": TASK,
        "instruction": INSTRUCTION,
        "goal_station_id": "rejected",
        "environment_id": cases[0]["environment_id"],
        "revision": cases[0]["revision"],
        "teaching_cases": cases,
        "evaluation_plan": plan,
    }
    catalog.record = catalog.record.model_copy(
        update={
            "task_id": TASK,
            "instruction": INSTRUCTION,
            "goal_station_id": "rejected",
        }
    )
    runtime = TeachingRuntimeStub()
    service = LearningService(
        factory,
        store,
        jobs,
        artifacts,
        catalog,
        enabled=True,
        runtime=runtime,
        allowed_policy_types=("gr00t_n1_5",),
    )
    return service, raw, cases, saved_case


def create_varied():
    service, raw, cases, saved_case = varied_context()
    project = service.create_project(ACTOR, CreateProject.model_validate(raw))
    return service, project, cases, saved_case


def start(service, project, case_id=None, request_id=None):
    return service.start_teaching(
        ACTOR,
        project.value.id,
        StartTeaching.model_validate(
            {
                "request_id": str(request_id or uuid4()),
                "source": "reference_controller",
                "motion_approved": True,
                **({"case_id": case_id} if case_id is not None else {}),
            }
        ),
        project.etag,
    )


def activate_case(service, case):
    service.factory.activate(ACTOR, case["environment_id"], case["revision"])


def test_legacy_project_anchor_is_not_implicitly_authorized_as_training_data():
    factory, store, jobs, artifacts, catalog, request = learning_setup()
    project, _, _ = seed_project_and_dataset(store, request)
    legacy = project.value.model_dump()
    legacy.pop("teaching_cases", None)
    stored = store.put_learning(
        ACTOR.owner_key, LearningProject.model_validate(legacy), project.etag
    )
    runtime = TeachingRuntimeStub()
    service = LearningService(
        factory, store, jobs, artifacts, catalog, enabled=True, runtime=runtime
    )
    with pytest.raises(Problem) as failure:
        start(service, stored)
    assert failure.value.code == "teaching_case_unapproved"
    assert runtime.starts == []


@pytest.mark.parametrize("index", [0, 1, 2])
def test_selected_case_reaches_runtime_response_and_durable_session_exactly(index):
    service, project, cases, _ = create_varied()
    chosen = cases[index]
    activate_case(service, chosen)
    result = start(service, project, chosen["case_id"])
    wire = service.runtime.starts[-1]
    assert (wire.environment_id, wire.revision, wire.split) == (
        chosen["environment_id"],
        chosen["revision"],
        chosen["split"],
    )
    assert result.value.teaching_case.model_dump(mode="json") == chosen
    assert result.value.public()["teaching_case"] == chosen
    assert wire.task.task_id == TASK and wire.task.instruction == INSTRUCTION
    assert wire.task.goal_id == "rejected"
    assert wire.control_profile_id == project.value.control_profile_id
    assert wire.epoch == result.value.epoch
    assert wire.demonstrator_kind == "reference_controller"
    assert "seed" not in wire.model_dump(), "Runtime must derive seed from the saved scene."


def test_default_selects_only_an_explicit_anchor_and_case_changes_need_new_request_id():
    service, project, cases, _ = create_varied()
    activate_case(service, cases[0])
    request_id = uuid4()
    first = start(service, project, request_id=request_id)
    assert first.value.teaching_case.case_id == cases[0]["case_id"]
    with pytest.raises(Problem) as failure:
        start(service, project, cases[1]["case_id"], request_id)
    assert failure.value.code == "request_id_reused"
    assert len(service.runtime.starts) == 1


@pytest.mark.parametrize("case_id", ["test-30001", "unknown-case", "private-10003"])
def test_test_unknown_or_unapproved_private_selector_never_dispatches(case_id):
    service, project, _, _ = create_varied()
    with pytest.raises(Problem) as failure:
        start(service, project, case_id)
    assert failure.value.code == "teaching_case_unapproved"
    assert service.runtime.starts == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_id", "another-scene"),
        ("revision", "e" * 64),
        ("seed", 30001),
        ("split", "train"),
    ],
)
def test_start_request_does_not_accept_caller_scene_seed_or_split_overrides(field, value):
    with pytest.raises(ValidationError):
        StartTeaching.model_validate(
            {
                "request_id": str(uuid4()),
                "source": "human_teleop",
                "motion_approved": True,
                "case_id": "train-10001",
                field: value,
            }
        )


@pytest.mark.parametrize(
    "change", ["test-seed", "same-split-seed", "cross-split", "duplicate-id", "integration-seed"]
)
def test_project_separates_frozen_teaching_validation_and_test_conditions(change):
    _, raw, cases, _ = varied_context()
    mutated = dict(cases[1])
    if change == "test-seed":
        mutated["seed"] = 30001
    elif change == "same-split-seed":
        mutated["seed"] = cases[0]["seed"]
    elif change == "cross-split":
        mutated.update(seed=cases[0]["seed"], split="validation")
    elif change == "duplicate-id":
        mutated["case_id"] = cases[0]["case_id"]
    else:
        mutated["seed"] = 900002
    with pytest.raises(ValidationError):
        CreateProject.model_validate({**raw, "teaching_cases": [cases[0], mutated, cases[2]]})


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "private",
        "wrong-seed",
        "wrong-revision",
        "no-goal",
        "wrong-builder",
        "corrupt-content",
        "split-relabel",
        "recording-disabled",
    ],
)
def test_every_saved_case_is_verified_before_project_authorization(change):
    service, raw, cases, saved_case = varied_context()
    broken = dict(cases[1])
    if change == "unknown":
        broken["environment_id"] = "unknown-owner-scene"
    elif change == "private":
        broken = saved_case(10003, "train", actor=OTHER)
    elif change == "wrong-seed":
        broken["seed"] = 10003
    elif change == "wrong-revision":
        broken["revision"] = "f" * 64
    else:
        current = service.factory.store.get_environment(ACTOR.owner_key, broken["environment_id"])
        content = json.loads(json.dumps(current.value.document))
        if change == "no-goal":
            content["stations"][-1]["id"] = "another-tray"
            content["workflow"]["reject_station"] = "another-tray"
        elif change == "wrong-builder":
            content["scene"]["template_id"] = "inspection-cell-v1"
        elif change == "split-relabel":
            content["execution"]["demonstration_split"] = "validation"
        elif change == "recording-disabled":
            content["execution"]["record_demonstration"] = False
        else:
            content["scene"]["seed"] = 10009
        if change == "corrupt-content":
            changed = current.value.model_copy(update={"document": content})
            service.factory.store.put_environment(ACTOR.owner_key, changed, current.etag)
        else:
            saved = service.factory.save_environment(
                ACTOR,
                SaveEnvironment(
                    document_json=json.dumps(content),
                    expected_revision=current.value.revision,
                ),
            )
            broken["revision"] = saved.revision
    candidate = CreateProject.model_validate(
        {**raw, "teaching_cases": [cases[0], broken, cases[2]]}
    )
    with pytest.raises(Problem):
        service.create_project(ACTOR, candidate)
    assert service.store.list_learning(ACTOR.owner_key, "project") == []
    assert service.runtime.starts == []


def test_case_pins_are_checked_again_at_start_not_rebound_to_a_changed_saved_record():
    service, project, cases, _ = create_varied()
    current = service.factory.store.get_environment(ACTOR.owner_key, cases[1]["environment_id"])
    changed = json.loads(json.dumps(current.value.document))
    changed["scene"]["seed"] = 10004
    service.factory.save_environment(
        ACTOR,
        SaveEnvironment(
            document_json=json.dumps(changed),
            expected_revision=current.value.revision,
        ),
    )
    with pytest.raises(Problem) as failure:
        start(service, project, cases[1]["case_id"])
    assert failure.value.code == "revision_conflict"
    assert service.runtime.starts == []


def ready_capture(service, project, selected):
    activate_case(service, selected)
    session = start(service, project, selected["case_id"])
    case = session.value.teaching_case
    receipt = CaptureReceipt(
        episode_id=uuid4(),
        artifact_id=uuid4(),
        manifest_sha256="b" * 64,
        frame_count=20,
        source=session.value.source,
        seed=case.seed,
        task_id=project.value.task_id,
        control_profile_id=project.value.control_profile_id,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        split=case.split,
    )
    saved = service.store.put_learning(
        ACTOR.owner_key,
        session.value.model_copy(
            update={
                "status": "ready",
                "physical_status": "succeeded",
                "capture": receipt,
            }
        ),
        session.etag,
    )
    return saved


def test_sealed_dataset_keeps_varied_case_seed_split_and_source_provenance():
    service, project, cases, _ = create_varied()
    sessions = [ready_capture(service, project, case) for case in cases]
    sealed = service.dataset(
        ACTOR,
        project.value.id,
        CreateDataset(
            request_id=uuid4(),
            teaching_session_ids=tuple(item.value.id for item in sessions),
        ),
        project.etag,
    )
    dataset = sealed.value
    assert dataset.seeds == (10001, 10002, 20001)
    assert [
        (item.environment_id, item.revision, item.seed, item.split) for item in dataset.captures
    ] == [(case["environment_id"], case["revision"], case["seed"], case["split"]) for case in cases]
    assert dataset.human_teleop_count == 0 and dataset.reference_controller_count == 3
    assert all(
        item.task_id == TASK and item.control_profile_id == project.value.control_profile_id
        for item in dataset.captures
    )
    service.train(
        ACTOR,
        project.value.id,
        StartTraining(
            request_id=uuid4(),
            dataset_id=dataset.id,
            parent_release_id=project.value.baseline_release_id,
            policy_type=project.value.policy_type,
            optimizer_steps=10,
            paid_approved=True,
            maximum_cost_usd="1.00",
        ),
        project.etag,
    )
    assert service.jobs.submissions[0].dataset.captures[-1].split == "validation"


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "test-30001"),
        ("seed", 30001),
        ("seed", 900002),
        ("split", "validation"),
        ("environment_id", "private-case"),
        ("revision", "d" * 64),
        ("task_id", "another-task"),
        ("control_profile_id", "wrong-profile"),
    ],
)
def test_dataset_sealing_rejects_corrupt_ready_receipts_before_artifact_assembly(field, value):
    service, project, cases, _ = create_varied()
    ready = ready_capture(service, project, cases[0])
    altered = ready.value.capture.model_copy(update={field: value})
    service.store.put_learning(
        ACTOR.owner_key,
        ready.value.model_copy(update={"capture": altered}),
        ready.etag,
    )
    with pytest.raises(Problem):
        service.dataset(
            ACTOR,
            project.value.id,
            CreateDataset(
                request_id=uuid4(),
                teaching_session_ids=(ready.value.id,),
            ),
            project.etag,
        )
    assert service.store.list_learning(ACTOR.owner_key, "dataset") == []


def test_validation_only_dataset_is_not_optimizer_data_and_mislabeled_mix_is_rejected():
    service, project, cases, _ = create_varied()
    validation = ready_capture(service, project, cases[2])
    sealed = service.dataset(
        ACTOR,
        project.value.id,
        CreateDataset(
            request_id=uuid4(),
            teaching_session_ids=(validation.value.id,),
        ),
        project.etag,
    )
    train = StartTraining(
        request_id=uuid4(),
        dataset_id=sealed.value.id,
        parent_release_id=project.value.baseline_release_id,
        policy_type=project.value.policy_type,
        optimizer_steps=10,
        paid_approved=True,
        maximum_cost_usd="1.00",
    )
    with pytest.raises(Problem) as failure:
        service.train(ACTOR, project.value.id, train, project.etag)
    assert failure.value.code == "training_split_empty"
    assert service.jobs.submissions == []
    capture = sealed.value.captures[0].model_copy(update={"split": "train"})
    service.store.put_learning(
        ACTOR.owner_key,
        sealed.value.model_copy(update={"captures": (capture,)}),
        sealed.etag,
    )
    with pytest.raises(Problem) as failure:
        service.train(ACTOR, project.value.id, train, project.etag)
    assert failure.value.code == "capture_case_mismatch"
    assert service.jobs.submissions == []


def test_project_without_an_anchor_case_requires_an_explicit_approved_selector():
    service, raw, cases, _ = varied_context()
    raw["teaching_cases"] = cases[1:]
    project = service.create_project(ACTOR, CreateProject.model_validate(raw))
    with pytest.raises(Problem) as failure:
        start(service, project)
    assert failure.value.code == "teaching_case_unapproved"
    activate_case(service, cases[1])
    assert (
        start(service, project, cases[1]["case_id"]).value.teaching_case.case_id
        == cases[1]["case_id"]
    )


def test_parent_frozen_seventy_case_partition_can_authorize_both_training_ranges():
    service, raw, _, saved_case = varied_context()
    teaching = [
        saved_case(seed, "train") for seed in (*range(10001, 10021), *range(11001, 11021))
    ] + [saved_case(seed, "validation") for seed in range(20001, 20011)]
    request = CreateProject.model_validate({**raw, "teaching_cases": teaching})
    project = service.create_project(ACTOR, request)
    assert len(project.value.teaching_cases) == 50
    assert len(project.value.evaluation_plan.cases) == 20
    assert {case.seed for case in project.value.teaching_cases if case.split == "train"} == {
        *range(10001, 10021),
        *range(11001, 11021),
    }
    assert not {case.seed for case in project.value.teaching_cases} & set(
        project.value.evaluation_plan.seeds
    )
    chosen = next(case for case in teaching if case["seed"] == 11020)
    activate_case(service, chosen)
    result = start(service, project, chosen["case_id"])
    assert result.value.teaching_case.seed == 11020
    assert service.runtime.starts[-1].environment_id == chosen["environment_id"]
