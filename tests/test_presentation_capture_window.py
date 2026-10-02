import json
from datetime import datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from presentation_support import prepare
from runtime_support import ACTOR
from test_public_presentation import planned

from apps.api.azure_data import CosmosStore
from apps.api.errors import Problem
from apps.api.presentation import validate_run
from apps.api.public_demo import PublicDemo
from scripts.run_reference_demo import run_cycles


class JsonContainer:
    """Cosmos wire/ETag double: every read and write crosses a JSON boundary."""

    def __init__(self):
        self.items = {}
        self.version = 0

    def read_item(self, *, item, partition_key):
        key = (partition_key, item)
        if key not in self.items:
            raise CosmosResourceNotFoundError(status_code=404, message="Test missing item")
        return json.loads(json.dumps(self.items[key]))

    def create_item(self, *, body):
        key = (body["owner_key"], body["id"])
        assert key not in self.items
        return self.save(body)

    def replace_item(self, *, item, body, etag, match_condition):
        assert item == body["id"]
        assert match_condition == MatchConditions.IfNotModified
        assert self.items[(body["owner_key"], item)]["_etag"] == etag
        return self.save(body)

    def save(self, body):
        self.version += 1
        result = json.loads(json.dumps({**body, "_etag": str(self.version)}))
        self.items[(body["owner_key"], body["id"])] = result
        return json.loads(json.dumps(result))


@pytest.mark.parametrize("age_ms", [3.938, 100, 2000])
def test_fresh_precreated_capture_survives_full_cosmos_roundtrip(monkeypatch, age_ms):
    backend, settings, clock = prepare(monkeypatch)
    store = CosmosStore.__new__(CosmosStore)
    store.container = JsonContainer()
    for environment_id in (
        settings.public_demo_environment_id,
        settings.public_demo_defect_environment_id,
    ):
        store.put_environment(
            ACTOR.owner_key,
            backend.environment(ACTOR, environment_id).value,
            None,
        )
    backend.store = store
    observe = backend.bridge.observe

    def cached(*args):
        return observe(*args).model_copy(
            update={
                "captured_at": clock.now() - timedelta(milliseconds=age_ms),
            }
        )

    backend.bridge.observe = cached
    report = run_cycles(backend, settings, cycles=1, maximum_seconds=60, authorized=True)
    assert report["status"] == "completed" and report["successes"] == 1
    record = store.get_presentation(ACTOR.owner_key, settings.public_demo_presentation_id).value
    run = store.get_run(ACTOR.owner_key, record.run_id).value
    assert run.evidence.captured_at == run.created_at - timedelta(milliseconds=age_ms)
    payload = PublicDemo(settings, backend).snapshot()["presentation"]
    assert datetime.fromisoformat(payload["decision"]["captured_at"]) == run.evidence.captured_at
    assert payload["result"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "age_ms,limit,accepted",
    [
        (2000, 2000, True),
        (2000.001, 2000, False),
        (500, 500, True),
        (500.001, 500, False),
        (2000.001, 3000, False),
    ],
)
def test_publication_capture_bound_is_configured_age_capped_at_two_seconds(
    monkeypatch,
    age_ms,
    limit,
    accepted,
):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    environment = runner.environments[0].model_copy(deep=True)
    environment.document["execution"]["max_observation_age_ms"] = limit
    run.environment_document = environment.document
    run.evidence.captured_at = run.created_at - timedelta(milliseconds=age_ms)
    if accepted:
        validate_run(runner.record, run, environment)
    else:
        with pytest.raises(Problem, match="observation scope"):
            validate_run(runner.record, run, environment)


@pytest.mark.parametrize(
    "mutation", ["future", "plan_epoch", "evidence_epoch", "path", "observation"]
)
def test_freshness_fix_does_not_weaken_other_evidence_guards(monkeypatch, mutation):
    prepared = prepare(monkeypatch)
    runner, run = planned(prepared)
    run.evidence.captured_at = run.created_at - timedelta(milliseconds=3.938)
    if mutation == "future":
        run.evidence.captured_at = run.updated_at + timedelta(microseconds=1)
    elif mutation == "plan_epoch":
        run.plan.epoch = uuid4()
    elif mutation == "evidence_epoch":
        run.evidence.epoch = uuid4()
    elif mutation == "path":
        run.evidence.blob_name = "private/input.png"
    else:
        run.evidence.observation_id = uuid4()
    with pytest.raises(Problem, match="observation scope"):
        validate_run(runner.record, run, runner.environments[0])


@pytest.mark.parametrize("untrusted_identity", [False, True])
def test_rejected_run_cleanup_marks_claim_failed_without_recursive_validation(
    monkeypatch,
    untrusted_identity,
):
    backend, settings, _ = prepare(monkeypatch)
    start = backend.start

    def rejected(*args):
        run = start(*args)
        stored = backend.store.get_run(ACTOR.owner_key, run.id)
        if untrusted_identity:
            run.instruction = "untrusted private request"
        else:
            run.evidence.captured_at = run.created_at - timedelta(milliseconds=2000.001)
        backend.store.put_run(ACTOR.owner_key, run, stored.etag)
        return run

    backend.start = rejected
    cancel = Mock(wraps=backend.cancel)
    backend.cancel = cancel
    report = run_cycles(backend, settings, cycles=1, maximum_seconds=60, authorized=True)
    record = backend.store.get_presentation(
        ACTOR.owner_key, settings.public_demo_presentation_id
    ).value
    run = backend.store.get_run(ACTOR.owner_key, record.run_id).value
    assert record.status == report["status"] == "failed"
    assert backend.bridge.dispatches == 0
    assert report["cycles"] == [] and report["successes"] == 0
    if untrusted_identity:
        cancel.assert_not_called()
        assert run.status == "awaiting_approval"
    else:
        cancel.assert_called_once_with(ACTOR, run.id)
        assert run.status == "cancelled"
    with pytest.raises(Problem):
        PublicDemo(settings, backend).snapshot()
