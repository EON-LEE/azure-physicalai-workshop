from types import SimpleNamespace
from uuid import uuid4

from azure.core.exceptions import ResourceNotFoundError

from apps.api.learning_ports import JobSpecification
from apps.learning_worker.backend import PolicyLearningWorker
from learning.checks.smolvla_aml_check import example_config
from learning.smolvla import azure
from tests.runtime_support import ACTOR
from tests.test_learning_results import evaluated_setup
from tests.test_learning_worker import specification
from tests.test_worker_deadlines import utc_text


def test_worker_calls_actual_smol_client_factory_with_keyword_only_identity(monkeypatch):
    spec, _ = specification()
    config = example_config()
    config.update(
        schema="physicalai.smolvla-azure/v2", job_deadline_utc=utc_text(spec.run.deadline)
    )
    observed = []

    def factory(value, *, caller_client_id):
        observed.append((value, caller_client_id))
        return object(), object()

    monkeypatch.setattr(azure, "clients_for_managed_identity", factory)
    worker_id = uuid4()
    worker = PolicyLearningWorker(None, None, worker_id, allowed_policy_types=("smolvla",))
    jobs, create = worker._sdk(config, "smolvla", model_use=True)
    assert isinstance(jobs, azure.PolicyJobs)
    assert create is azure.create_plan
    assert observed == [(config, str(worker_id))]


def test_worker_generates_and_revalidates_a_real_native_plan_without_submitting_to_azure(
    monkeypatch,
):
    spec, _ = specification()
    config = example_config()
    config.update(
        schema="physicalai.smolvla-azure/v2", job_deadline_utc=utc_text(spec.run.deadline)
    )
    observed = []

    class Jobs:
        def submit(self, path, *, approved_plan_sha256, deterministic_job_name):
            plan = azure.read_plan(path)
            assert plan["plan_sha256"] == approved_plan_sha256
            assert plan["job_name"] == deterministic_job_name
            observed.append(plan)
            return {
                "job_name": deterministic_job_name,
                "azure_job_id": (
                    "/subscriptions/test/providers/Microsoft.MachineLearningServices"
                    f"/workspaces/test/jobs/{deterministic_job_name}"
                ),
                "owner_key": ACTOR.owner_key,
                "specification_sha256": spec.run.specification_sha256,
                "status": "submitted",
            }

    registry = SimpleNamespace(claim_job=lambda *_: True)
    worker = PolicyLearningWorker(
        registry,
        None,
        uuid4(),
        allowed_policy_types=("smolvla",),
        sdk_factory=lambda _: (Jobs(), azure.create_plan),
    )
    monkeypatch.setattr(worker, "preflight", lambda *_: None)
    monkeypatch.setattr(worker, "_configuration", lambda *_: config)
    monkeypatch.setattr(worker, "_enrollment", lambda *_: None)
    receipt = worker.submit(ACTOR, spec)
    assert receipt.status == "submitted"
    assert receipt.candidate is None
    assert len(observed) == 1


def test_worker_reads_a_final_report_even_when_native_quality_gate_exits_failed():
    _, project, baseline, candidate, evaluation = evaluated_setup(10, 10)
    expected = evaluation.value.report
    active = evaluation.value.model_copy(update={"status": "running", "report": None})
    spec = JobSpecification(
        owner_key=ACTOR.owner_key,
        project=project,
        baseline=baseline,
        candidate=candidate,
        run=active,
    )
    calls = []

    class Artifacts:
        def completed_report(self, actor, specification, azure_job_id, *, required=True):
            calls.append((azure_job_id, required))
            return expected

    worker = PolicyLearningWorker(None, Artifacts(), uuid4())
    result = worker._receipt(
        ACTOR,
        spec,
        {
            "job_name": active.backend_job_name,
            "azure_job_id": active.azure_job_id,
            "owner_key": ACTOR.owner_key,
            "specification_sha256": active.specification_sha256,
            "status": "failed",
        },
    )
    assert result.status == "failed"
    assert result.report == expected
    assert calls == [(active.azure_job_id, False)]


def test_missing_native_named_job_is_unconfirmed_not_recreated_or_faked():
    spec, _ = specification()

    class Jobs:
        def status(self, name):
            raise ResourceNotFoundError(status_code=404)

    registry = SimpleNamespace(
        job=lambda *_: spec,
        job_configuration=lambda *_: None,
        approved_plan=lambda *_: {"config": {"schema": "physicalai.smolvla-azure/v1"}},
    )
    worker = PolicyLearningWorker(
        registry,
        None,
        uuid4(),
        sdk_factory=lambda _: (Jobs(), None),
    )
    assert worker.status(ACTOR, spec.run) is None
