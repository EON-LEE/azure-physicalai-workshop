"""Post-hoc native training imports use CPU fixtures, never actual training/quality claims."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_api import api as api
from test_api import headers

from apps.api.errors import Problem
from apps.api.learning_models import PolicyCandidate, TrainingRun
from learning.common import canonical, digest
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_results import evaluated_setup


def candidate_value():
    _, _, _, candidate, _ = evaluated_setup()
    return candidate.model_dump(mode="json")


def test_external_candidate_is_explicit_and_cannot_invent_an_api_training_run():
    value = candidate_value() | {
        "training_origin": "external_native_import",
        "external_import_id": str(uuid4()),
        "training_run_id": None,
        "parent_release_id": None,
        "pretrained_artifact_id": None,
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "control_profile_id": "franka-position-hold-10hz-paused-v1",
        "control_profile_sha256": "a" * 64,
        "criteria_sha256": "b" * 64,
        "frozen_plan_sha256": "c" * 64,
        "policy_type": "smolvla",
    }
    candidate = PolicyCandidate.model_validate(value)
    assert candidate.training_run_id is None
    assert candidate.external_import_id is not None
    for change in (
        {"training_run_id": str(uuid4())},
        {"external_import_id": None},
        {"training_origin": "api_training_run"},
        {"parent_release_id": str(uuid4())},
        {"pretrained_artifact_id": str(uuid4())},
    ):
        with pytest.raises(ValidationError):
            PolicyCandidate.model_validate(value | change)


def test_api_training_contract_and_legacy_candidate_serialization_are_unchanged():
    value = candidate_value()
    assert "training_origin" not in value and "external_import_id" not in value
    assert PolicyCandidate.model_validate(value).model_dump(mode="json") == value
    with pytest.raises(ValidationError):
        PolicyCandidate.model_validate(value | {"training_run_id": None})
    field = TrainingRun.model_fields["backend_job_name"]
    assert any(
        getattr(item, "pattern", None) == r"^learning-[a-f0-9-]+$" for item in field.metadata
    )


@pytest.fixture(scope="module")
def original_native_import(tmp_path_factory):
    from tests.external_training_support import make_fixture

    with pytest.MonkeyPatch.context() as patch:
        yield make_fixture(tmp_path_factory.mktemp("external-native-codecs"), patch)


@pytest.fixture
def native_import(original_native_import, monkeypatch):
    from tests.external_training_support import clone_fixture

    return clone_fixture(original_native_import, monkeypatch)


def test_full_posthoc_import_verifies_original_native_artifacts_without_api_training_record(
    native_import,
):
    from apps.learning_worker import external_training

    sample = native_import
    imported = external_training.complete(sample.verifier, ACTOR, sample.work)
    assert imported.record.native_job_name == sample.request.native_job_name
    assert imported.candidate.azure_job_id == sample.job_id
    assert imported.candidate.training_run_id is None
    assert imported.candidate.pretrained_artifact_id is None
    assert imported.candidate.parent_release_id is None
    assert imported.candidate.external_import_id == sample.work.id
    assert imported.candidate.optimizer_steps == 1000
    assert sample.result["candidate"] == "candidates/step-001000"
    assert len(imported.dataset.episode_ids) == 20
    assert imported.dataset.reference_controller_count == 20
    assert imported.record.imported_at > imported.record.native_created_at
    assert imported.record.learning_quality_verified is False
    checked, _ = external_training.verified_result(
        sample.verifier, ACTOR, sample.project, sample.context.reference
    )
    assert checked == imported
    assert not any("/jobs/" in name for name in sample.registry.container.items)


@pytest.mark.parametrize(
    "missing", ["approval.json", "claim.json", "plan.tar.gz", "image-qualification.json"]
)
def test_missing_original_pre_execution_evidence_never_registers_candidate(native_import, missing):
    from apps.learning_worker import external_training

    sample = native_import
    del sample.registry.container.items[
        sample.registry.key(ACTOR, f"{sample.context.native_prefix}/{missing}")
    ]
    with pytest.raises(Problem):
        external_training.complete(sample.verifier, ACTOR, sample.work)
    assert not any("/artifacts/" in key for key in sample.registry.container.items)


@pytest.mark.parametrize("target", ["approval", "claim", "job", "result"])
def test_server_observed_chronology_cannot_be_backdated_by_payload_fields(native_import, target):
    from apps.learning_worker import external_training

    sample = native_import
    if target in ("approval", "claim"):
        name = sample.registry.key(ACTOR, f"{sample.context.native_prefix}/{target}.json")
        sample.registry.container.modified[name] = sample.request.created_at
    elif target == "job":
        sample.job.creation_context.created_at = sample.context.approval_modified
    else:
        name = (
            sample.config["output_prefix"]
            + "/"
            + str(sample.request.native_job_name)
            + "/model/result.json"
        )
        sample.registry.container.modified[name] = sample.request.created_at
    with pytest.raises(Problem):
        external_training.complete(sample.verifier, ACTOR, sample.work)
    assert not any("/artifacts/" in key for key in sample.registry.container.items)


@pytest.mark.parametrize(
    "change",
    ["Running", "Failed", "name", "owner", "type", "parent", "plan", "deadline", "identity"],
)
def test_actual_root_must_be_the_completed_original_command(native_import, change):
    from apps.learning_worker import external_training

    sample = native_import
    if change in ("Running", "Failed"):
        sample.job.status = change
    elif change == "name":
        sample.job.name = str(uuid4())
    elif change == "type":
        sample.job.type = "pipeline"
    elif change == "parent":
        sample.job.parent_job_name = "fictional-pipeline"
    elif change == "identity":
        sample.job.identity.client_id = str(uuid4())
    else:
        sample.job.tags[
            {"owner": "scope_owner", "plan": "plan_sha256", "deadline": "job_deadline_utc"}[change]
        ] = "changed"
    with pytest.raises(Problem):
        external_training.complete(sample.verifier, ACTOR, sample.work)
    assert not any("/artifacts/" in key for key in sample.registry.container.items)


@pytest.mark.parametrize("target", ["raw", "parent", "backbone", "result"])
def test_original_payload_tampering_never_yields_an_external_import(native_import, target):
    from apps.learning_worker import external_training
    from learning.paused.blob_transfer import input_prefix

    sample = native_import
    if target == "result":
        key = (
            sample.config["output_prefix"]
            + "/"
            + str(sample.request.native_job_name)
            + "/model/result.json"
        )
    elif target == "raw":
        start = input_prefix(sample.config, "demonstrations") + "/"
        key = next(
            key
            for key in sample.registry.container.items
            if key.startswith(start) and key.endswith(".png")
        )
    else:
        start = (
            input_prefix(sample.config, "parent_model" if target == "parent" else "backbone") + "/"
        )
        key = next(
            key
            for key in sample.registry.container.items
            if key.startswith(start) and key.endswith(".safetensors")
        )
    body, etag = sample.registry.container.items[key]
    sample.registry.container.items[key] = body + b"tamper", etag
    with pytest.raises(Problem):
        external_training.complete(sample.verifier, ACTOR, sample.work)
    assert not any("/artifacts/" in key for key in sample.registry.container.items)


def test_wrong_owner_missing_result_and_rehashed_mixed_provenance_fail_closed(native_import):
    from apps.api.artifact_models import ArtifactWork
    from apps.learning_worker import external_training

    sample = native_import
    with pytest.raises(ValidationError):
        ArtifactWork.model_validate(sample.work.model_dump() | {"actor": OTHER})
    with pytest.raises(Problem):
        external_training.context(sample.registry, OTHER, sample.project, sample.work.id)
    output = sample.config["output_prefix"] + "/" + str(sample.request.native_job_name)
    key = output + "/model/result.json"
    value = json.loads(sample.registry.container.items[key][0])
    value["azure_pipeline_job_id"] = sample.job_id
    body = canonical(value)
    sample.registry.container.items[key] = body, '"changed"'
    completion_key = sample.registry.key(ACTOR, sample.context.prefix + "/completion.json")
    updated = sample.request.model_copy(update={"result_sha256": digest(body)})
    sample.registry.container.items[completion_key] = (
        canonical(updated.model_dump(mode="json", by_alias=True)),
        '"new-request"',
    )
    with pytest.raises(Problem):
        external_training.complete(sample.verifier, ACTOR, sample.work)


def test_candidate_path_is_exact_native_six_digit_format():
    from apps.learning_worker.candidate_provenance import CommandTrainingResult
    from tests.test_command_candidate_provenance import command_result

    original = command_result() | {"candidate": "candidates/step-001000", "optimizer_steps": 1000}
    assert CommandTrainingResult.model_validate(original).candidate == original["candidate"]
    for value in (
        "step-1000",
        "step-0001000",
        "step-+01000",
        "step-000000",
        "step-100001",
        "../step-001000",
    ):
        with pytest.raises(ValidationError):
            CommandTrainingResult.model_validate(original | {"candidate": "candidates/" + value})


def test_public_projection_does_not_invent_training_for_external_candidate():
    from apps.api.public_learning import learning_publication
    from tests.runtime_support import settings

    _, project, _, candidate, evaluation = evaluated_setup()
    external = candidate.model_copy(
        update={
            "training_origin": "external_native_import",
            "training_run_id": None,
            "external_import_id": uuid4(),
        }
    )
    records = {"project": project, "evaluation": evaluation.value, "candidate": external}

    def get(actor, kind, identifier):
        assert kind in records, (
            "External publication must fail before looking up an API training run."
        )
        return SimpleNamespace(value=records[kind])

    configuration = settings().model_copy(
        update={
            "public_learning_owner_id": ACTOR.object_id,
            "public_learning_project_id": project.id,
            "public_learning_evaluation_id": evaluation.value.id,
        }
    )
    with pytest.raises(Problem) as failure:
        learning_publication(configuration, SimpleNamespace(get=get))
    assert failure.value.code == "external_learning_not_publishable"


def test_original_archive_is_verified_without_substituting_current_worker_source(
    native_import, monkeypatch, tmp_path
):
    from apps.learning_worker import candidate_provenance, external_training
    from learning.smolvla import embedded_source

    sample = native_import
    original = embedded_source.static_inventory
    inspected = []

    def archived_only(root, **kwargs):
        assert str(root).endswith("plan/code")
        inspected.append(root)
        return original(root, **kwargs)

    monkeypatch.setattr(embedded_source, "static_inventory", archived_only)
    monkeypatch.setattr(
        candidate_provenance,
        "command_snapshot",
        lambda *_: pytest.fail("Historical imports must not hash current worker source."),
    )
    plan = external_training._plan(sample.registry, ACTOR, sample.context, tmp_path)
    assert plan["job_name"] == str(sample.request.native_job_name)
    assert inspected


@pytest.mark.parametrize("target", ["candidate", "dataset", "receipt"])
def test_verified_external_receipt_rechecks_published_etags_and_provenance(native_import, target):
    from apps.learning_worker import external_training

    sample = native_import
    result = external_training.complete(sample.verifier, ACTOR, sample.work)
    if target == "receipt":
        key = sample.registry.key(ACTOR, sample.context.prefix + "/verified.json")
        data, etag = sample.registry.container.items[key]
        value = json.loads(data)
        value["result"]["record"]["plan_sha256"] = "f" * 64
        sample.registry.container.items[key] = canonical(value), etag
    else:
        artifact = (
            result.candidate.artifact_id if target == "candidate" else result.dataset.artifact_id
        )
        prefix = sample.registry.key(ACTOR, f"artifacts/{artifact}/files/")
        key = next(key for key in sample.registry.container.items if key.startswith(prefix))
        data, _ = sample.registry.container.items[key]
        sample.registry.container.items[key] = data, '"replaced"'
    with pytest.raises(Problem):
        external_training.verified_result(
            sample.verifier, ACTOR, sample.project, sample.context.reference
        )


def test_private_api_import_uses_bounded_worker_and_creates_no_api_training_or_release(
    native_import, api, monkeypatch, tmp_path
):
    from contextlib import nullcontext

    import httpx
    from fastapi.testclient import TestClient

    from apps.api.learning_gateway import ManagedLearningGateway
    from apps.api.learning_service import LearningService
    from apps.learning_worker import artifact_task
    from apps.learning_worker.artifact_operations import ArtifactOperations
    from apps.learning_worker.main import WorkerSettings, create_worker
    from tests.learning_api_support import MemoryLearningStore
    from tests.runtime_support import AUDIENCE, service

    sample = native_import
    ops = ArtifactOperations(
        sample.registry,
        enabled=True,
        actor_ids=frozenset({ACTOR.object_id}),
        external_training_enabled=True,
        external_operator_ids=frozenset({ACTOR.object_id}),
        verifier=sample.verifier,
    )
    settings = WorkerSettings(
        tenant_id=ACTOR.tenant_id,
        audience=AUDIENCE,
        managed_identity_client_id=uuid4(),
        registry_account_url=sample.registry.client.url,
        registry_container=sample.config["blob_container"],
        capture_account_url=sample.registry.client.url,
        allowed_policy_types=("smolvla",),
        bootstrap_owner_ids=frozenset({ACTOR.object_id}),
        artifact_ops_enabled=True,
        artifact_actor_ids=frozenset({ACTOR.object_id}),
        paused_training_enabled=True,
    )
    monkeypatch.setattr(
        artifact_task, "ManagedIdentityCredential", lambda **kwargs: nullcontext(None)
    )
    monkeypatch.setattr(artifact_task, "BlobRegistry", lambda *args, **kwargs: sample.registry)

    def verifier(*args, **kwargs):
        sample.verifier.budget = kwargs["budget"]
        return sample.verifier

    monkeypatch.setattr(artifact_task, "VerifiedArtifacts", verifier)

    class Identity:
        def controller(self, authorization, allowed):
            if authorization != "Bearer cpu-import-app-token":
                raise Problem(401, "authentication_required", "Private API identity required.")

    worker = SimpleNamespace(
        registry=sample.registry, artifacts=sample.verifier, artifact_operations=ops
    )
    with TestClient(create_worker(settings, worker, Identity())) as private:

        def transport(request):
            path = request.url.path + (
                "?" + request.url.query.decode() if request.url.query else ""
            )
            response = private.request(
                request.method, path, content=request.content, headers=dict(request.headers)
            )
            return httpx.Response(
                response.status_code, content=response.content, headers=response.headers
            )

        gateway = ManagedLearningGateway(
            "https://private.invalid",
            SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="cpu-import-app-token")),
            "api://worker/.default",
            transport=httpx.MockTransport(transport),
        )
        store = MemoryLearningStore()
        project = store.put_learning(ACTOR.owner_key, sample.project, None)
        backend = LearningService(
            service(),
            store,
            artifacts=gateway,
            enabled=True,
            allowed_policy_types=("smolvla",),
            paused_training_enabled=True,
            bootstrap_principal_ids=frozenset({ACTOR.object_id}),
        )
        client, _, token = api
        client.app.state.learning = backend
        path = f"/api/learning/projects/{sample.project.id}/external-imports/{sample.work.id}"
        body = {"request_id": str(sample.work.id)}
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers=headers(token)).status_code == 428
        backend.bootstrap_principal_ids = frozenset()
        assert (
            client.post(
                path, json=body, headers={**headers(token), "If-Match": project.etag}
            ).status_code
            == 403
        )
        backend.bootstrap_principal_ids = frozenset({ACTOR.object_id})
        accepted = client.post(
            path, json=body, headers={**headers(token), "If-Match": project.etag}
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["item"]["status"] == "queued"
        assert accepted.json()["item"]["operation"] == "external_training"
        assert "work_document" not in accepted.json()["item"]
        assert sample.calls == []
        repeated = client.post(path, json=body, headers={**headers(token), "If-Match": "old-etag"})
        assert repeated.status_code == 202 and ops.pending(ACTOR) == [sample.work.id]
        claim_id = uuid4()
        ops.claim(ACTOR, sample.work.id, claim_id)
        assert (
            artifact_task.execute(sample.work.id, ACTOR.object_id, claim_id, settings)
            == "import_committed"
        )
        result = client.get(
            f"/api/learning/artifact-operations/{sample.work.id}", headers=headers(token)
        )
        assert result.status_code == 200, result.text
        assert result.json()["item"]["phase"] == "import_committed"
        receipt = result.json()["item"]["result"]["external_training"]
        record = backend.get(ACTOR, "external_import", sample.work.id).value
        candidate = backend.get(ACTOR, "candidate", record.candidate_id).value
        dataset = backend.get(ACTOR, "dataset", record.dataset_id).value
        assert receipt["record"]["schema"] == "physicalai.external-training-import/v1"
        assert candidate.training_run_id is None and candidate.external_import_id == sample.work.id
        assert dataset.reference_controller_count == 20 and len(dataset.episode_ids) == 20
        assert store.list_learning(ACTOR.owner_key, "training") == []
        assert store.list_learning(ACTOR.owner_key, "release") == []
        listing = client.get(
            f"/api/learning/projects/{sample.project.id}/records?kind=external_import",
            headers=headers(token),
        )
        assert listing.status_code == 200 and len(listing.json()["items"]) == 1
        assert "owner_key" not in listing.json()["items"][0]["item"]
        assert client.get(f"/api/learning/external-imports/{sample.work.id}").status_code == 401
        with pytest.raises(Problem):
            backend.get(OTHER, "external_import", sample.work.id)
        root = tmp_path / "registered-model"
        index = sample.registry.download(ACTOR, candidate.artifact_id, root)
        model = sample.verifier.registered_model(
            ACTOR, candidate, root, index, execution_timing="paused_simulation"
        )
        assert model["training"]["azure_job_id"] == sample.job_id
        assert sample.job.name == str(sample.request.native_job_name)
        assert not sample.job.name.startswith("learning-")
        from apps.learning_worker.managed_reports import _inventory as report_inventory

        spec = SimpleNamespace(baseline_candidate=candidate, baseline=None, candidate=candidate)
        report_context = SimpleNamespace(
            prefix="projects/test/managed-evaluations/test",
            binding=SimpleNamespace(specification=spec),
        )
        report_inventory(sample.verifier, ACTOR, report_context)
        native_key = sample.registry.key(ACTOR, f"{sample.context.native_prefix}/approval.json")
        raw, etag = sample.registry.container.items[native_key]
        sample.registry.container.items[native_key] = raw + b" ", etag
        with pytest.raises(Problem):
            report_inventory(sample.verifier, ACTOR, report_context)
        gateway.close()
