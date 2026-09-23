import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from apps.api.azure_data import CosmosStore
from apps.api.errors import Problem
from apps.api.learning_models import LearningProject
from tests.learning_api_support import seed_project_and_dataset
from tests.runtime_support import ACTOR, OTHER
from tests.test_learning_lifecycle_contracts import project_request


class ConditionalContainer:
    """SDK-shaped test transport; never used by production."""

    def __init__(self):
        self.items = {}
        self.version = 0
        self.lock = threading.Lock()
        self.replacements = []
        self.queries = []

    def read_item(self, item, partition_key):
        with self.lock:
            if (partition_key, item) not in self.items:
                raise CosmosResourceNotFoundError(status_code=404)
            return copy.deepcopy(self.items[(partition_key, item)])

    def create_item(self, body):
        with self.lock:
            key = body["owner_key"], body["id"]
            if key in self.items:
                raise CosmosHttpResponseError(status_code=409)
            self.version += 1
            item = {**copy.deepcopy(body), "_etag": f'"version-{self.version}"'}
            self.items[key] = item
            return copy.deepcopy(item)

    def replace_item(self, item, body, etag, match_condition):
        with self.lock:
            assert match_condition == MatchConditions.IfNotModified
            self.replacements.append((item, etag))
            key = body["owner_key"], item
            if key not in self.items or self.items[key]["_etag"] != etag:
                raise CosmosHttpResponseError(status_code=412)
            self.version += 1
            self.items[key] = {**copy.deepcopy(body), "_etag": f'"version-{self.version}"'}
            return copy.deepcopy(self.items[key])

    def query_items(self, **kwargs):
        self.queries.append(kwargs)
        return []


def store():
    result = CosmosStore.__new__(CosmosStore)
    result.container = ConditionalContainer()
    return result


def test_cosmos_learning_storage_is_available_without_an_in_memory_production_alternative():
    actual = store()
    assert callable(getattr(actual, "get_learning", None))
    assert callable(getattr(actual, "put_learning", None))
    assert callable(getattr(actual, "list_learning", None))


def test_conditional_create_claims_one_winner_and_survives_service_reconstruction():
    actual = store()
    project = LearningProject.create(ACTOR, project_request())

    def write(_):
        try:
            return actual.put_learning(ACTOR.owner_key, project, None)
        except Problem as exc:
            assert exc.code == "revision_conflict"
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(write, range(4)))
    assert sum(result is not None for result in results) == 1
    reconstructed = CosmosStore.__new__(CosmosStore)
    reconstructed.container = actual.container
    assert reconstructed.get_learning(ACTOR.owner_key, "project", project.id).value == project
    assert reconstructed.get_learning(OTHER.owner_key, "project", project.id) is None


def test_immutable_scope_and_project_pins_cannot_be_overwritten_even_with_current_etag():
    actual = store()
    project = LearningProject.create(ACTOR, project_request())
    stored = actual.put_learning(ACTOR.owner_key, project, None)
    for changes in ({"revision": "9" * 64}, {"instruction": "Replace immutable task"}):
        with pytest.raises(Problem) as failure:
            actual.put_learning(ACTOR.owner_key, project.model_copy(update=changes), stored.etag)
        assert failure.value.code == "immutable_learning_record"
    with pytest.raises(Problem) as failure:
        actual.put_learning(OTHER.owner_key, project, None)
    assert failure.value.status == 403
    assert actual.container.replacements == []


def test_mutable_job_state_uses_if_not_modified_and_conflicts_instead_of_last_writer_wins():
    from apps.api.learning_models import TrainingRun
    from apps.api.models import utcnow

    actual = store()
    request = project_request()
    project, dataset, train = seed_project_and_dataset(actual, request)
    now = utcnow()
    run = TrainingRun(
        id=train.request_id,
        owner_key=ACTOR.owner_key,
        actor_id=ACTOR.object_id,
        tenant_id=ACTOR.tenant_id,
        created_at=now,
        updated_at=now,
        fingerprint="e" * 64,
        project_id=project.value.id,
        status="submitting",
        backend_job_name=f"learning-{train.request_id.hex}",
        deadline=now,
        approved_cost_usd="10.00",
        specification_sha256="f" * 64,
        dataset_id=dataset.id,
        parent_release_id=request.baseline_release_id,
        optimizer_steps=100,
    )
    initial = actual.put_learning(ACTOR.owner_key, run, None)
    updated = run.model_copy(update={"status": "submission_unknown"})
    actual.put_learning(ACTOR.owner_key, updated, initial.etag)
    with pytest.raises(Problem) as failure:
        actual.put_learning(ACTOR.owner_key, updated, initial.etag)
    assert failure.value.code == "revision_conflict"
    assert actual.container.replacements[0][1] == initial.etag


def test_list_is_bounded_and_owner_partitioned_without_cross_partition_enumeration():
    actual = store()
    actual.list_learning(ACTOR.owner_key, "training", project_request().request_id)
    query = actual.container.queries[0]
    assert "TOP 50" in query["query"]
    assert query["partition_key"] == ACTOR.owner_key
    assert "enable_cross_partition_query" not in query
    assert any(param["name"] == "@project" for param in query["parameters"])
