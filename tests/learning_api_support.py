import json
from datetime import timedelta
from uuid import uuid4

from apps.api.errors import Problem
from apps.api.learning_models import (
    PROFILE_ID,
    CaptureReceipt,
    DatasetVersion,
    LearningProject,
    PolicyRelease,
    StartTraining,
    fingerprint,
)
from apps.api.learning_ports import BackendJob
from apps.api.models import SaveEnvironment, utcnow
from tests.runtime_support import ACTOR, MemoryStore, document, service
from tests.test_learning_lifecycle_contracts import project_request


class MemoryLearningStore(MemoryStore):
    """Explicitly injected tests only; production must use durable Cosmos storage."""

    def get_learning(self, owner, kind, resource_id):
        return self._get(owner, f"learning:{kind}", resource_id)

    def put_learning(self, owner, record, etag):
        if record.owner_key != owner:
            raise Problem(403, "learning_owner_mismatch", "Mismatched test owner.")
        return self._put(owner, f"learning:{record.kind}", record.id, record, etag)

    def list_learning(self, owner, kind, project_id=None):
        return [
            stored.model_copy(deep=True)
            for (partition, item_kind, _), stored in self.items.items()
            if partition == owner
            and item_kind == f"learning:{kind}"
            and (project_id is None or stored.value.project_id == project_id)
        ][:50]


class TestJobs:
    __test__ = False

    def __init__(self):
        self.submissions = []
        self.cancellations = []
        self.status_calls = []
        self.receipts = {}
        self.preflight_error = None
        self.submit_error = None
        self.after_submit = None

    def preflight(self, actor, specification):
        if self.preflight_error:
            raise self.preflight_error

    def submit(self, actor, specification):
        self.submissions.append(specification)
        receipt = BackendJob(
            job_name=specification.run.backend_job_name,
            azure_job_id=(
                "/subscriptions/11111111-1111-4111-8111-111111111111"
                "/resourceGroups/test/providers/Microsoft.MachineLearningServices"
                f"/workspaces/test/jobs/{specification.run.backend_job_name}"
            ),
            owner_key=actor.owner_key,
            specification_sha256=specification.run.specification_sha256,
            status="submitted",
        )
        self.receipts[receipt.job_name] = receipt
        if self.after_submit:
            self.after_submit(specification)
        if self.submit_error:
            raise self.submit_error
        return receipt

    def status(self, actor, run):
        self.status_calls.append(run.id)
        return self.receipts.get(run.backend_job_name)

    def cancel(self, actor, run):
        self.cancellations.append(run.id)
        result = self.receipts[run.backend_job_name].model_copy(update={"status": "cancelling"})
        self.receipts[run.backend_job_name] = result
        return result


class TestArtifacts:
    __test__ = False

    def __init__(self):
        self.candidate_checks = []
        self.report_checks = []
        self.error = None

    def verify_candidate(self, actor, project, run, candidate):
        if self.error:
            raise self.error
        self.candidate_checks.append(candidate.id)

    def verify_report(self, actor, project, run, report):
        if self.error:
            raise self.error
        self.report_checks.append(report.report_sha256)

    def seal_dataset(self, actor, project, dataset_id, captures):
        if self.error:
            raise self.error
        return dataset_id, fingerprint([capture.model_dump(mode="json") for capture in captures])

    def verify_capture(self, actor, project, session, receipt):
        if self.error:
            raise self.error
        from apps.api.learning_models import CaptureReceipt

        return CaptureReceipt.model_validate(receipt)


class TestCatalog:
    __test__ = False

    def __init__(self, record):
        self.record = record

    def resolve(self, actor, release_id):
        if self.record.id != release_id or self.record.owner_key != actor.owner_key:
            raise Problem(404, "policy_release_missing", "No approved test baseline.")
        return self.record


def learning_setup():
    factory = service()
    store = MemoryLearningStore()
    factory.store = store
    anchor = document()
    anchor["scene"]["template_id"] = "inspection-cell-learning-v1"
    anchor["execution"].update(record_demonstration=True, demonstration_split="train")
    saved = factory.save_environment(ACTOR, SaveEnvironment(document_json=json.dumps(anchor)))
    factory.activate(ACTOR, saved.environment_id, saved.revision)
    request = project_request()
    request = request.model_copy(
        update={
            "revision": saved.revision,
            "teaching_cases": tuple(
                case.model_copy(update={"revision": saved.revision})
                for case in request.teaching_cases
            ),
        }
    )
    now = utcnow()
    baseline = PolicyRelease(
        id=request.baseline_release_id,
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
        fingerprint="a" * 64,
        project_id=uuid4(),
        candidate_id=uuid4(),
        evaluation_run_id=uuid4(),
        policy_type="gr00t_n1_5",
        model_sha256="b" * 64,
        processor_sha256="c" * 64,
        manifest_sha256="d" * 64,
        artifact_id=uuid4(),
        environment_id=saved.environment_id,
        revision=saved.revision,
        task_id=request.task_id,
        goal_station_id=request.goal_station_id,
        instruction=request.instruction,
        control_profile_id=PROFILE_ID,
        evaluation_plan_sha256="f" * 64,
        reviewed_by=ACTOR.object_id,
    )
    return factory, store, TestJobs(), TestArtifacts(), TestCatalog(baseline), request


def seed_project_and_dataset(store, request):
    project = LearningProject.create(ACTOR, request)
    stored = store.put_learning(ACTOR.owner_key, project, None)
    now = utcnow()
    case = project.teaching_cases[0]
    capture = CaptureReceipt(
        episode_id=uuid4(),
        manifest_sha256="e" * 64,
        artifact_id=uuid4(),
        frame_count=20,
        source="human_teleop",
        seed=case.seed,
        task_id=project.task_id,
        control_profile_id=project.control_profile_id,
        case_id=case.case_id,
        environment_id=case.environment_id,
        revision=case.revision,
        split=case.split,
    )
    dataset = DatasetVersion(
        id=uuid4(),
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="a" * 64,
        project_id=project.id,
        artifact_id=uuid4(),
        manifest_sha256="e" * 64,
        episode_ids=(capture.episode_id,),
        seeds=(capture.seed,),
        captures=(capture,),
        human_teleop_count=1,
        reference_controller_count=0,
        learned_policy_count=0,
        evaluation_plan_sha256=project.evaluation_plan.sha256,
    )
    store.put_learning(ACTOR.owner_key, dataset, None)
    train = StartTraining(
        request_id=uuid4(),
        dataset_id=dataset.id,
        parent_release_id=project.baseline_release_id,
        optimizer_steps=100,
        policy_type="gr00t_n1_5",
        paid_approved=True,
        maximum_cost_usd="10.00",
    )
    return stored, dataset, train
