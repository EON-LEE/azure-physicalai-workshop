from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.artifact_models import ArtifactWork
from apps.api.errors import Problem
from apps.api.learning_models import CreateProject, LearningProject
from apps.api.simulation_reports import ManagedImportReference
from tests.runtime_support import ACTOR
from tests.test_artifact_operations import operation_spec, operations
from tests.test_paused_learning_api import paused_project_payload
from tests.test_worker_blob_sdk import setup


def managed_work():
    work = operation_spec()
    project = LearningProject.create(ACTOR, CreateProject.model_validate(paused_project_payload()))
    reference = ManagedImportReference(
        operation_id=work.id,
        owner_key=ACTOR.owner_key,
        project_id=project.id,
        evaluation_run_id=uuid4(),
        specification_sha256="a" * 64,
        binding_sha256="b" * 64,
        completion_sha256="c" * 64,
    )
    return ArtifactWork.model_validate(
        {
            **work.model_dump(),
            "operation": "managed_evaluation",
            "project": project,
            "target_id": reference.evaluation_run_id,
            "managed_import": reference,
            "captures": (),
        }
    )


def test_managed_artifact_stage_defaults_off_before_any_registry_read_or_queue_write():
    ops, registry = operations()
    with pytest.raises(Problem) as failure:
        ops.begin(ACTOR, managed_work())
    assert failure.value.code == "paused_learning_unavailable"
    assert registry.container.items == {}


def test_managed_work_cannot_reuse_legacy_project_or_retarget_the_import():
    work = managed_work()
    for changes in (
        {"project": operation_spec().project.model_copy(update={"id": work.project.id})},
        {"target_id": uuid4()},
        {"id": uuid4()},
    ):
        with pytest.raises(ValidationError):
            ArtifactWork.model_validate(work.model_dump() | changes)


def test_actual_blob_sdk_returns_exact_bytes_and_same_response_registration_timestamp():
    raw = b'{ "operator-record": true }\n'
    registry, container, transport, _ = setup(
        [
            {"status": 206, "etag": '"original-binding"', "body": raw},
        ]
    )
    with container:
        data, modified = registry.import_document(ACTOR, "projects/test/binding.json")
    assert data == raw
    assert modified == datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    assert [request.method for request in transport.requests] == ["GET"]
    assert not transport.responses


def test_actual_blob_sdk_enforces_metadata_size_before_parsing():
    registry, container, transport, _ = setup(
        [
            {"status": 206, "etag": '"too-large"', "body": b"0123456789"},
        ]
    )
    with container, pytest.raises(Problem) as failure:
        registry.import_document(ACTOR, "projects/test/binding.json", max_bytes=4)
    assert failure.value.code == "managed_import_size"
    assert len(transport.requests) == 1


def test_managed_registration_cannot_fall_through_to_legacy_aml_configuration():
    from types import SimpleNamespace

    from apps.learning_worker.backend import PolicyLearningWorker

    class NoAzureConfiguration:
        def job_configuration(self, *args):
            pytest.fail("Managed imports must not resolve an Azure ML configuration.")

    worker = PolicyLearningWorker(NoAzureConfiguration(), None, uuid4())
    with pytest.raises(Problem) as failure:
        worker._job_configuration(
            ACTOR, SimpleNamespace(run=SimpleNamespace(provider="managed_batch"))
        )
    assert failure.value.code == "managed_import_required"


def test_verification_inventory_also_pins_registered_model_indexes():
    import json
    from types import SimpleNamespace

    from apps.learning_worker.managed_reports import _inventory

    _, registry = operations()
    registry.client = SimpleNamespace(url="https://test.blob.core.windows.net")
    registry.container.container_name = "artifacts"
    before = SimpleNamespace(artifact_id=uuid4(), model_sha256="b" * 64)
    after = SimpleNamespace(artifact_id=uuid4(), model_sha256="c" * 64)
    context = SimpleNamespace(
        prefix=f"projects/{uuid4()}/managed-evaluations/{uuid4()}",
        binding=SimpleNamespace(
            specification=SimpleNamespace(
                baseline=None,
                baseline_candidate=before,
                candidate=after,
            )
        ),
    )
    for record in (before, after):
        registry.put(
            ACTOR,
            f"artifacts/{record.artifact_id}/index.json",
            {
                "owner_key": ACTOR.owner_key,
                "manifest_sha256": record.model_sha256,
                "role": "trained_candidate",
                "files": {"model.json": record.model_sha256},
            },
        )
    verifier = SimpleNamespace(registry=registry, _blob_inventory=lambda *args: "d" * 64)
    original = _inventory(verifier, ACTOR, context)
    key = registry.key(ACTOR, f"artifacts/{after.artifact_id}/index.json")
    raw, etag = registry.container.items[key]
    value = json.loads(raw)
    value["role"] = "different-role"
    registry.container.items[key] = json.dumps(value).encode(), etag
    assert _inventory(verifier, ACTOR, context) != original
