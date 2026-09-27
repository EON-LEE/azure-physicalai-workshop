"""Managed import boundary tests; CPU fixtures never establish physical/model quality."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_api import api as api
from test_api import headers
from test_batch_learned import learned_spec as learned_spec
from test_learned_managed_lifecycle import authority as authority
from test_learned_rescore import evidence as evidence
from test_paired_artifacts import recording as recording
from test_paired_evaluation import mapping_value as mapping_value
from test_simulation_batch import spec as spec

from apps.api.errors import Problem
from apps.api.learning_models import EvaluationRun
from learning.common import canonical, file_digest, read_json
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_results import evaluated_setup


def pending_run():
    _, _, _, _, stored = evaluated_setup()
    return {
        **stored.value.model_dump(mode="json"),
        "provider": "managed_batch",
        "status": "awaiting_import",
        "azure_job_id": None,
        "report": None,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "control_profile_id": "franka-position-hold-10hz-paused-v1",
        "control_profile_sha256": "a" * 64,
        "criteria_sha256": "b" * 64,
        "frozen_plan_sha256": "c" * 64,
    }


def test_managed_evaluation_waits_for_verified_import_without_an_azure_ml_identity():
    value = EvaluationRun.model_validate(pending_run())
    assert value.provider == "managed_batch"
    assert value.azure_job_id is None and value.report is None
    for change in (
        {"provider": "azure_ml"},
        {"azure_job_id": "/fake/azureml/job"},
        {"azure_status": "Completed"},
        {"status": "succeeded"},
    ):
        with pytest.raises(ValidationError):
            EvaluationRun.model_validate(pending_run() | change)


def test_managed_registration_and_completion_are_separate_closed_records():
    from apps.learning_worker.managed_reports import ManagedImportCompletion

    value = {
        "schema": "physicalai.managed-evaluation-import/v1",
        "binding_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
        "report_sha256": "c" * 64,
        "created_at": pending_run()["created_at"],
        "operation_id": str(uuid4()),
    }
    parsed = ManagedImportCompletion.model_validate(value)
    assert parsed.evidence_sha256 == "b" * 64
    with pytest.raises(ValidationError):
        ManagedImportCompletion.model_validate(value | {"report_url": "https://foreign.invalid"})
    with pytest.raises(ValidationError):
        ManagedImportCompletion.model_validate(value | {"binding_sha256": ""})


def test_legacy_evaluation_dump_retains_its_original_providerless_contract():
    _, _, _, _, stored = evaluated_setup()
    raw = stored.value.model_dump(mode="json")
    assert "provider" not in raw and "import_operation_id" not in raw
    assert EvaluationRun.model_validate(raw).model_dump(mode="json") == raw


@pytest.fixture(scope="module")
def managed_bundle(tmp_path_factory, request):
    from tests.managed_report_support import bundle

    directory = tmp_path_factory.mktemp("managed-import-codecs")
    with pytest.MonkeyPatch.context() as patch:
        base = spec.__wrapped__()
        learned = learned_spec.__wrapped__(base)
        approved = authority.__wrapped__(learned, directory, patch, request)
        measured = evidence.__wrapped__(approved, directory)
        recorded = recording.__wrapped__(measured, mapping_value.__wrapped__(), directory)
        yield bundle(recorded, directory / "managed")


def test_complete_forty_native_rescores_keep_failures_and_physical_ids(managed_bundle):
    from apps.learning_worker.managed_reports import verify_local
    from tests.test_paired_artifacts import inventory

    bundle = managed_bundle
    original = inventory(bundle.root)
    receipt = verify_local(bundle.verifier, ACTOR, bundle.context, bundle.root, bundle.models)
    assert receipt.provider == "managed_batch"
    assert len(receipt.report.trials) == 40
    assert not receipt.report.quality_gate_passed
    assert receipt.report.counts["before"].success == receipt.report.counts["after"].success == 0
    assert all(str(row.physical_attempt_id) == row.episode_id for row in receipt.report.trials)
    assert len({row.logical_case_id for row in receipt.report.trials}) == 20
    assert receipt.report.results_sha256 is None
    assert inventory(bundle.root) == original


def test_managed_study_window_is_explicit_bounded_and_separate_from_legacy_budget(managed_bundle):
    from apps.learning_worker.managed_reports import ManagedEvaluationBinding

    original = managed_bundle.context.binding
    values = original.model_dump(mode="json", by_alias=True)
    values["study_max_wall_seconds"] = 28800
    run = original.specification.run
    values["specification"]["run"]["deadline"] = (
        run.created_at + timedelta(seconds=28800)
    ).isoformat()
    accepted = ManagedEvaluationBinding.model_validate(values)
    assert accepted.study_max_wall_seconds == 28800
    assert accepted.specification.project.budget.evaluation_seconds <= 21600
    assert original.specification.run.deadline == run.deadline
    assert accepted.study_approved_cost_usd == run.approved_cost_usd
    for field in ("study_max_wall_seconds", "study_approved_cost_usd"):
        missing = deepcopy(values)
        missing.pop(field)
        with pytest.raises(ValidationError):
            ManagedEvaluationBinding.model_validate(missing)
    for changes in (
        {"study_max_wall_seconds": 28801},
        {"study_max_wall_seconds": 28799},
        {"study_max_wall_seconds": 28800.0},
        {"study_approved_cost_usd": "20.01"},
        {"study_approved_cost_usd": "9.99"},
    ):
        with pytest.raises(ValidationError):
            ManagedEvaluationBinding.model_validate(values | changes)
    too_costly = deepcopy(values)
    too_costly["study_approved_cost_usd"] = "20"
    too_costly["specification"]["run"]["approved_cost_usd"] = "20"
    too_costly["specification"]["project"]["budget"]["maximum_cost_usd"] = "19.99"
    with pytest.raises(ValidationError):
        ManagedEvaluationBinding.model_validate(too_costly)
    with pytest.raises(ValidationError):
        type(original.specification.project.budget).model_validate(
            original.specification.project.budget.model_dump() | {"evaluation_seconds": 28800}
        )


@contextmanager
def rewritten(path, value):
    original = path.read_bytes()
    path.write_bytes(canonical(value) + b"\n")
    try:
        yield
    finally:
        path.write_bytes(original)


def repinned(bundle):
    return replace(
        bundle.context,
        completion=bundle.context.completion.model_copy(
            update={
                "evidence_sha256": file_digest(bundle.root / "evidence.json"),
                "report_sha256": file_digest(bundle.root / "report.json"),
            }
        ),
    )


@pytest.mark.parametrize("change", ["green", "model", "mapping", "physical-id"])
def test_rehashed_report_cannot_change_rescored_quality_or_identity(managed_bundle, change):
    from apps.learning_worker.managed_reports import verify_local

    bundle = managed_bundle
    report = read_json(bundle.root / "report.json", max_bytes=32 * 1024**2)
    if change == "green":
        report.update(quality_gate_passed=True, conclusion="improved")
        report["comparison"].update(quality_gate_passed=True, conclusion="improved")
    elif change == "model":
        report["comparison"]["policy_after_sha256"] = "f" * 64
    elif change == "mapping":
        report["mapping_sha256"] = "f" * 64
    else:
        report["attempts"][0]["physical_trial"]["episode_id"] = str(uuid4())
    with rewritten(bundle.root / "report.json", report), pytest.raises(Problem) as failure:
        verify_local(bundle.verifier, ACTOR, repinned(bundle), bundle.root, bundle.models)
    assert failure.value.code == "managed_import_rescore"


def test_only_thirty_nine_received_or_preempted_attempts_cannot_produce_a_report(managed_bundle):
    from apps.learning_worker.managed_reports import verify_local
    from simulation import paired_evaluation
    from tests.test_paired_artifacts import inventory

    bundle = managed_bundle
    evidence = read_json(bundle.root / "evidence.json", max_bytes=32 * 1024**2)
    directory = bundle.root / "attempts" / evidence["attempts"][-1]["physical_attempt_id"]
    hidden = directory.with_name(directory.name + "-not-in-this-snapshot")
    directory.rename(hidden)
    moved = bundle.root.parent / hidden.name
    hidden.rename(moved)
    try:
        shorter = {**evidence, "attempts": evidence["attempts"][:-1]}
        with rewritten(bundle.root / "evidence.json", shorter):
            result = paired_evaluation.aggregate(
                bundle.root,
                bundle.mapping,
                paired_evaluation.PairingEvidence.model_validate(shorter),
                mapping_sha256=bundle.context.binding.mapping_sha256,
                evidence_sha256=file_digest(bundle.root / "evidence.json"),
                before_root=bundle.models["before"],
                after_root=bundle.models["after"],
            )
            assert not result["complete"] and result["verified_trial_count"] == 39
            with rewritten(bundle.root / "report.json", result), pytest.raises(Problem) as failure:
                verify_local(bundle.verifier, ACTOR, repinned(bundle), bundle.root, bundle.models)
            assert failure.value.code == "managed_import_incomplete"
    finally:
        moved.rename(directory)
    completion = directory / "completion.json"
    hidden_completion = bundle.root.parent / "cpu-preempted-completion.json"
    completion.rename(hidden_completion)
    try:
        interrupted = deepcopy(evidence)
        interrupted["attempts"][-1]["files"] = inventory(directory)
        with rewritten(bundle.root / "evidence.json", interrupted):
            result = paired_evaluation.aggregate(
                bundle.root,
                bundle.mapping,
                paired_evaluation.PairingEvidence.model_validate(interrupted),
                mapping_sha256=bundle.context.binding.mapping_sha256,
                evidence_sha256=file_digest(bundle.root / "evidence.json"),
                before_root=bundle.models["before"],
                after_root=bundle.models["after"],
            )
            assert result["attempts"][-1]["status"] == "incomplete"
            assert "physical_trial" not in result["attempts"][-1]
            with rewritten(bundle.root / "report.json", result), pytest.raises(Problem) as failure:
                verify_local(bundle.verifier, ACTOR, repinned(bundle), bundle.root, bundle.models)
            assert failure.value.code == "managed_import_incomplete"
    finally:
        hidden_completion.rename(completion)


@pytest.mark.parametrize("change", ["owner", "model", "profile", "heldout"])
def test_foreign_or_rehashed_mapping_cannot_replace_the_predeclared_project(managed_bundle, change):
    from apps.learning_worker.managed_reports import verify_local

    bundle = managed_bundle
    mapping = read_json(bundle.root / "mapping.json")
    native = mapping["evaluation_plan"]
    if change == "owner":
        native["scope"]["owner_id"] = "f" * 64
    elif change == "model":
        native["policy_after_sha256"] = "f" * 64
    elif change == "profile":
        native["control_profile_sha256"] = "f" * 64
    else:
        native["cases"][0]["revision"] = "f" * 64
    with rewritten(bundle.root / "mapping.json", mapping):
        changed = replace(
            bundle.context,
            binding=bundle.context.binding.model_copy(
                update={"mapping_sha256": file_digest(bundle.root / "mapping.json")}
            ),
        )
        with pytest.raises((Problem, ValueError)):
            verify_local(bundle.verifier, ACTOR, changed, bundle.root, bundle.models)


def test_backdated_binding_field_cannot_waive_blob_observed_registration_time(managed_bundle):
    from apps.learning_worker.managed_reports import verify_local

    bundle = managed_bundle
    changed = replace(
        bundle.context, registered_at=bundle.context.registered_at + timedelta(seconds=10)
    )
    with pytest.raises(Problem) as failure:
        verify_local(bundle.verifier, ACTOR, changed, bundle.root, bundle.models)
    assert failure.value.code == "managed_import_late_binding"


def test_rehashed_trial_summary_cannot_replace_raw_and_heartbeat_rescore(managed_bundle):
    from apps.learning_worker.managed_reports import verify_local
    from tests.test_paired_artifacts import inventory

    bundle = managed_bundle
    evidence = read_json(bundle.root / "evidence.json", max_bytes=32 * 1024**2)
    directory = bundle.root / "attempts" / evidence["attempts"][0]["physical_attempt_id"]
    probe = read_json(directory / "probe.json")
    probe["trial"]["latencies_wall_ms"][0] += 0.5
    with rewritten(directory / "probe.json", probe):
        completion = read_json(directory / "completion.json")
        for item in completion["artifacts"]:
            if item["path"] == "probe.json":
                item.update(
                    sha256=file_digest(directory / "probe.json"),
                    bytes=(directory / "probe.json").stat().st_size,
                )
        with rewritten(directory / "completion.json", completion):
            evidence["attempts"][0]["files"] = inventory(directory)
            with rewritten(bundle.root / "evidence.json", evidence), pytest.raises(ValueError):
                verify_local(bundle.verifier, ACTOR, repinned(bundle), bundle.root, bundle.models)


def test_validation_candidate_cannot_enter_the_predeclared_test_pairing(managed_bundle):
    from apps.learning_worker.managed_reports import verify_local
    from tests.test_paired_artifacts import inventory

    bundle = managed_bundle
    evidence = read_json(bundle.root / "evidence.json", max_bytes=32 * 1024**2)
    directory = bundle.root / "attempts" / evidence["attempts"][0]["physical_attempt_id"]
    specification = read_json(directory / "inputs" / "spec.json")
    specification.update(evaluation_split="validation", role="candidate")
    specification.pop("pairing_plan_sha256")
    with rewritten(directory / "inputs" / "spec.json", specification):
        evidence["attempts"][0]["files"] = inventory(directory)
        with rewritten(bundle.root / "evidence.json", evidence), pytest.raises(ValueError):
            verify_local(bundle.verifier, ACTOR, repinned(bundle), bundle.root, bundle.models)


@pytest.mark.parametrize(
    "field,value",
    [
        ("azure_job_id", "/a-different-training-job"),
        ("optimizer_steps", 999),
        ("processor_sha256", "f" * 64),
    ],
)
def test_candidate_record_cannot_relabel_actual_native_training_metadata(
    managed_bundle, tmp_path, field, value
):
    from apps.learning_worker.artifacts import VerifiedArtifacts
    from apps.learning_worker.managed_reports import _models
    from tests.managed_report_support import registry_for

    bundle = managed_bundle
    registry = registry_for(bundle)
    verifier = VerifiedArtifacts(
        registry, None, registry.client.url, "demonstrations", allowed_policy_types=("smolvla",)
    )
    changed = bundle.spec.model_copy(
        update={"candidate": bundle.spec.candidate.model_copy(update={field: value})}
    )
    context = replace(
        bundle.context, binding=bundle.context.binding.model_copy(update={"specification": changed})
    )
    with pytest.raises(Problem) as failure:
        _models(verifier, ACTOR, context, tmp_path)
    assert failure.value.code == "managed_import_model"


def test_operator_binding_requires_exact_scope_original_spec_and_later_completion(managed_bundle):
    from apps.learning_worker.managed_reports import context
    from tests.managed_report_support import registry_for

    bundle = managed_bundle
    registry = registry_for(bundle)
    assert context(registry, ACTOR, bundle.spec.project.id, bundle.spec.run.id) == bundle.context
    with pytest.raises(Problem):
        context(registry, OTHER, bundle.spec.project.id, bundle.spec.run.id)
    key = registry.key(ACTOR, f"{bundle.context.prefix}/binding.json")
    data, etag = registry.container.items[key]
    registry.container.items[key] = (
        data.replace(
            b'"provider":"managed_batch"',
            b'"provider":"managed_batch","provider":"managed_batch"',
            1,
        ),
        etag,
    )
    with pytest.raises(Problem):
        context(registry, ACTOR, bundle.spec.project.id, bundle.spec.run.id)
    registry.container.items[key] = data, etag
    registry.container.modified[key] = bundle.context.completion.created_at + timedelta(seconds=2)
    with pytest.raises(Problem) as failure:
        context(registry, ACTOR, bundle.spec.project.id, bundle.spec.run.id)
    assert failure.value.code == "managed_import_binding"


def test_protected_api_to_private_artifact_worker_imports_only_after_actual_rescore(
    managed_bundle, api, monkeypatch
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import httpx
    from fastapi.testclient import TestClient

    from apps.api.learning_gateway import ManagedLearningGateway
    from apps.api.learning_service import LearningService
    from apps.learning_worker import artifact_task, artifacts
    from apps.learning_worker.artifact_operations import ArtifactOperations
    from apps.learning_worker.artifacts import VerifiedArtifacts
    from apps.learning_worker.main import WorkerSettings, create_worker
    from simulation import paired_evaluation
    from tests.learning_api_support import MemoryLearningStore
    from tests.managed_report_support import registry_for
    from tests.runtime_support import AUDIENCE, service

    bundle = managed_bundle
    registry = registry_for(bundle)

    class BlobService:
        def __init__(self, account, **kwargs):
            assert account == registry.client.url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_container_client(self, name):
            assert name == registry.container.container_name
            return registry.container

    monkeypatch.setattr(artifacts, "BlobServiceClient", BlobService)
    monkeypatch.setattr(artifact_task, "BlobServiceClient", BlobService)
    monkeypatch.setattr(artifact_task, "BlobRegistry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(
        artifact_task, "ManagedIdentityCredential", lambda **kwargs: nullcontext(None)
    )
    verifier = VerifiedArtifacts(
        registry, None, registry.client.url, "demonstrations", allowed_policy_types=("smolvla",)
    )
    ops = ArtifactOperations(
        registry,
        enabled=True,
        actor_ids=frozenset({ACTOR.object_id}),
        managed_evaluations_enabled=True,
        verifier=verifier,
    )
    settings = WorkerSettings(
        tenant_id=ACTOR.tenant_id,
        audience=AUDIENCE,
        managed_identity_client_id=uuid4(),
        registry_account_url=registry.client.url,
        registry_container=registry.container.container_name,
        capture_account_url=registry.client.url,
        allowed_policy_types=("smolvla",),
        artifact_ops_enabled=True,
        artifact_actor_ids=frozenset({ACTOR.object_id}),
        paused_evaluation_enabled=True,
    )

    class Identity:
        def controller(self, authorization, allowed):
            if authorization != "Bearer cpu-worker-token":
                raise Problem(
                    401, "authentication_required", "Worker application identity required."
                )

    worker = SimpleNamespace(registry=registry, artifacts=verifier, artifact_operations=ops)
    rescores = []
    aggregate = paired_evaluation.aggregate

    def measured_rescore(*args, **kwargs):
        rescores.append(True)
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(paired_evaluation, "aggregate", measured_rescore)
    with TestClient(create_worker(settings, worker, Identity())) as private:

        def transport(request):
            path = request.url.path
            if request.url.query:
                path += "?" + request.url.query.decode()
            response = private.request(
                request.method, path, content=request.content, headers=dict(request.headers)
            )
            return httpx.Response(
                response.status_code, content=response.content, headers=response.headers
            )

        gateway = ManagedLearningGateway(
            "https://private.invalid",
            SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="cpu-worker-token")),
            "api://cpu-worker/.default",
            transport=httpx.MockTransport(transport),
        )
        store = MemoryLearningStore()
        for record in (
            bundle.spec.project,
            bundle.spec.baseline_candidate,
            bundle.spec.candidate,
            bundle.spec.run,
        ):
            store.put_learning(ACTOR.owner_key, record, None)
        learning = LearningService(
            service(),
            store,
            gateway,
            gateway,
            enabled=True,
            allowed_policy_types=("smolvla",),
            paused_evaluation_enabled=True,
        )
        client, _, token = api
        client.app.state.learning = learning
        run = bundle.spec.run
        path = f"/api/learning/jobs/{run.id}/managed-import"
        body = {"request_id": str(bundle.context.reference.operation_id)}
        original = store.get_learning(ACTOR.owner_key, "evaluation", run.id)
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers=headers(token)).status_code == 428
        learning.paused_evaluation_enabled = False
        assert (
            client.post(
                path, json=body, headers={**headers(token), "If-Match": original.etag}
            ).status_code
            == 503
        )
        learning.paused_evaluation_enabled = True
        accepted = client.post(
            path, json=body, headers={**headers(token), "If-Match": original.etag}
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["item"]["status"] == "queued"
        assert rescores == []
        pending = learning.get_job(ACTOR, run.id)
        assert pending.value.status == "awaiting_import" and pending.value.azure_job_id is None
        repeat = client.post(path, json=body, headers={**headers(token), "If-Match": "stale-retry"})
        assert repeat.status_code == 202 and rescores == []
        assert ops.pending(ACTOR) == [bundle.context.reference.operation_id]
        claim_id = uuid4()
        ops.claim(ACTOR, bundle.context.reference.operation_id, claim_id)
        assert (
            artifact_task.execute(
                bundle.context.reference.operation_id, ACTOR.object_id, claim_id, settings
            )
            == "report_committed"
        )
        result = client.get(
            f"/api/learning/artifact-operations/{bundle.context.reference.operation_id}",
            headers=headers(token),
        )
        assert result.status_code == 200, result.text
        assert result.json()["item"]["status"] == "ready"
        assert result.json()["item"]["phase"] == "report_committed"
        imported = learning.get_job(ACTOR, run.id)
        assert imported.value.status == "succeeded"
        assert imported.value.azure_job_id is imported.value.azure_status is None
        assert imported.value.provider == "managed_batch"
        assert not imported.value.report.quality_gate_passed
        assert len(rescores) == 1
        download = client.get(f"/api/learning/jobs/{run.id}/report", headers=headers(token))
        assert download.status_code == 200, download.text
        assert download.content == (bundle.root / "report.json").read_bytes()
        assert download.json()["schema"] == "physicalai.managed-paired-report/v1"
        assert (
            client.post(
                f"/api/learning/jobs/{run.id}/cancel",
                json={"request_id": str(uuid4())},
                headers={**headers(token), "If-Match": imported.etag},
            ).status_code
            == 409
        )
        learning.paused_release_enabled = True
        denied = client.post(
            "/api/policy-releases",
            json={
                "request_id": str(uuid4()),
                "candidate_id": str(bundle.spec.candidate.id),
                "evaluation_run_id": str(run.id),
                "release_approved": True,
            },
            headers={**headers(token), "If-Match": imported.etag},
        )
        assert denied.status_code == 409, denied.text
        assert store.list_learning(ACTOR.owner_key, "release") == []
        from apps.api.public_learning import learning_publication
        from tests.runtime_support import settings as api_settings
        from tests.test_learning_worker import specification as worker_specification

        assert learning_publication(api_settings(), learning)["status"] == "not_published"
        prototype, _ = worker_specification()
        dataset = prototype.dataset.model_copy(
            update={
                "id": bundle.spec.candidate.dataset_id,
                "project_id": bundle.spec.project.id,
                **bundle.spec.project.timing_fields(),
            }
        )
        training = prototype.run.model_copy(
            update={
                "id": bundle.spec.candidate.training_run_id,
                "project_id": bundle.spec.project.id,
                "dataset_id": dataset.id,
                "pretrained_artifact_id": bundle.spec.project.pretrained_artifact_id,
                "candidate_id": bundle.spec.candidate.id,
                "status": "succeeded",
                "azure_job_id": bundle.spec.candidate.azure_job_id,
                **bundle.spec.project.timing_fields(),
            }
        )
        store.put_learning(ACTOR.owner_key, dataset, None)
        store.put_learning(ACTOR.owner_key, training, None)
        configuration = api_settings().model_copy(
            update={
                "public_learning_owner_id": ACTOR.object_id,
                "public_learning_project_id": bundle.spec.project.id,
                "public_learning_evaluation_id": run.id,
            }
        )
        publication = learning_publication(configuration, learning)
        assert (
            publication["publication"]["comparison"]["native_schema"]
            == "physicalai.managed-paired-report/v1"
        )
        assert publication["publication"]["comparison"]["quality_gate_passed"] is False
        assert len(publication["publication"]["comparison"]["trials"]) == 40
        assert "artifact_id" not in publication["publication"]["comparison"]
        first = bundle.evidence.attempts[0].physical_attempt_id
        source = registry.key(ACTOR, f"{bundle.context.prefix}/files/attempts/{first}/probe.json")
        content, _ = registry.container.items[source]
        registry.container.items[source] = content + b" ", '"tampered"'
        with pytest.raises(Problem):
            gateway.verify_report(ACTOR, bundle.spec.project, imported.value, imported.value.report)
        assert len(rescores) == 1
        gateway.close()
