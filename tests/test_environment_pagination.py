import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_api import api as api
from test_api import headers

from apps.api.azure_data import CosmosStore
from apps.api.models import EnvironmentRecord
from apps.api.service import content_hash
from tests.runtime_support import ACTOR, OTHER, document
from tests.test_learning_cosmos import ConditionalContainer


class PagedContainer(ConditionalContainer):
    def __init__(self):
        super().__init__()
        self.default_ttl = -1

    def read(self):
        return {"defaultTtl": self.default_ttl}

    def query_items(self, **kwargs):
        self.queries.append(kwargs)
        assert kwargs["partition_key"] == ACTOR.owner_key
        assert "enable_cross_partition_query" not in kwargs
        params = {item["name"]: item["value"] for item in kwargs["parameters"]}
        assert params["@kind"] == "environment"
        rows = [
            copy.deepcopy(row)
            for (owner, _), row in self.items.items()
            if owner == kwargs["partition_key"] and row["kind"] == "environment"
        ]
        if "@after" in params:
            rows = [row for row in rows if row["id"] < params["@after"]]
        key = "id" if "ORDER BY c.id DESC" in kwargs["query"] else "updated_at"
        return sorted(rows, key=lambda row: row[key], reverse=True)[: params.get("@limit", 50)]


def record(index, now):
    value = document()
    value["environment_id"] = f"case-{index:03d}"
    value["scene"]["template_id"] = "inspection-cell-learning-v1"
    value["scene"]["seed"] = (
        10001 + index
        if index < 20
        else 11001 + index - 20
        if index < 40
        else 20001 + index - 40
        if index < 50
        else 30001 + index - 50
    )
    value["execution"].update(
        record_demonstration=True,
        demonstration_split="train" if index < 40 else "validation" if index < 50 else "test",
    )
    return EnvironmentRecord(
        environment_id=value["environment_id"],
        display_name=f"Test-only saved case {index}",
        revision=content_hash(value),
        document=value,
        created_at=now - timedelta(minutes=1),
        updated_at=now - timedelta(seconds=70 - index),
    )


def setup(api, monkeypatch, count=70):
    client, backend, token = api
    store = CosmosStore.__new__(CosmosStore)
    store.container = PagedContainer()
    backend.store = store
    clock = SimpleNamespace(now=datetime(2026, 9, 23, 12, tzinfo=UTC))
    monkeypatch.setattr("apps.api.azure_data.utcnow", lambda: clock.now, raising=False)
    records = [record(index, clock.now) for index in range(count)]
    for item in records:
        store.put_environment(ACTOR.owner_key, item, None)
    return client, store, token, clock, records


def test_legacy_fifty_and_two_bounded_pages_expose_exactly_seventy_owned_records(api, monkeypatch):
    client, store, token, _, records = setup(api, monkeypatch)
    legacy = client.get("/api/environments", headers=headers(token)).json()
    assert set(legacy) == {"items"} and len(legacy["items"]) == 50
    assert not {item["environment_id"] for item in legacy["items"]} & {
        item.environment_id for item in records[:20]
    }
    first = client.get("/api/environments?page_size=50", headers=headers(token))
    assert first.status_code == 200
    assert len(first.json()["items"]) == 50
    cursor = first.json()["next_cursor"]
    second = client.get("/api/environments", params={"cursor": cursor}, headers=headers(token))
    assert second.status_code == 200
    assert len(second.json()["items"]) == 20
    assert second.json()["next_cursor"] is None
    items = first.json()["items"] + second.json()["items"]
    assert [item["environment_id"] for item in items] == [
        item.environment_id for item in reversed(records)
    ]
    assert {item["revision"] for item in items} == {item.revision for item in records}
    assert all(query["partition_key"] == ACTOR.owner_key for query in store.container.queries)
    cursors = [row for row in store.container.items.values() if row["kind"] == "environment_cursor"]
    assert len(cursors) == 1 and cursors[0]["ttl"] == 600
    assert all("ttl" not in row for row in store.container.items.values() if row not in cursors)


@pytest.mark.parametrize("bad", ["malformed", "missing", "foreign", "expired", "page-size"])
def test_invalid_or_foreign_cursor_is_rejected_before_any_environment_query(api, monkeypatch, bad):
    client, store, token, clock, _ = setup(api, monkeypatch)
    first = client.get("/api/environments?page_size=50", headers=headers(token)).json()
    params = {"cursor": first["next_cursor"]}
    auth = headers(token)
    if bad == "malformed":
        params["cursor"] = "not-a-cursor"
    elif bad == "missing":
        params["cursor"] = str(uuid4())
    elif bad == "foreign":
        auth = {"Authorization": f"Bearer {token(oid=str(OTHER.object_id))}"}
    elif bad == "expired":
        clock.now += timedelta(minutes=10)
    else:
        params["page_size"] = 25
    queries = len(store.container.queries)
    response = client.get("/api/environments", params=params, headers=auth)
    assert response.status_code == 422
    assert len(store.container.queries) == queries


def test_equal_update_times_use_unique_id_order_without_a_composite_index(api, monkeypatch):
    client, store, token, clock, records = setup(api, monkeypatch)
    for item in records:
        stored = store.get_environment(ACTOR.owner_key, item.environment_id)
        store.put_environment(
            ACTOR.owner_key, item.model_copy(update={"updated_at": clock.now}), stored.etag
        )
    first = client.get("/api/environments?page_size=50", headers=headers(token)).json()
    second = client.get(
        "/api/environments", params={"cursor": first["next_cursor"]}, headers=headers(token)
    ).json()
    ids = [item["environment_id"] for item in first["items"] + second["items"]]
    assert ids == sorted((item.environment_id for item in records), reverse=True)
    assert all("ORDER BY c.id DESC" in query["query"] for query in store.container.queries)


def test_cutoff_excludes_later_creation_but_does_not_pretend_to_freeze_existing_content(
    api, monkeypatch
):
    client, store, token, clock, records = setup(api, monkeypatch, count=69)
    first = client.get("/api/environments?page_size=50", headers=headers(token)).json()
    clock.now += timedelta(seconds=1)
    later_document = {**record(69, clock.now).document, "environment_id": "case-018-later"}
    created = record(69, clock.now).model_copy(
        update={
            "created_at": clock.now,
            "environment_id": "case-018-later",
            "document": later_document,
            "revision": content_hash(later_document),
        }
    )
    store.put_environment(ACTOR.owner_key, created, None)
    old = store.get_environment(ACTOR.owner_key, records[0].environment_id)
    changed_document = {**old.value.document, "display_name": "Test-only concurrent update"}
    changed = old.value.model_copy(
        update={
            "document": changed_document,
            "revision": content_hash(changed_document),
            "updated_at": clock.now,
        }
    )
    store.put_environment(ACTOR.owner_key, changed, old.etag)
    second = client.get(
        "/api/environments", params={"cursor": first["next_cursor"]}, headers=headers(token)
    ).json()
    items = first["items"] + second["items"]
    assert len(items) == 69
    assert created.environment_id not in {item["environment_id"] for item in items}
    updated = next(item for item in items if item["environment_id"] == changed.environment_id)
    assert updated["revision"] == changed.revision


def test_filtered_new_records_still_advance_a_bounded_empty_page(api, monkeypatch):
    client, store, token, clock, _ = setup(api, monkeypatch, count=3)
    first = client.get("/api/environments?page_size=1", headers=headers(token)).json()
    clock.now += timedelta(seconds=1)
    later_document = {**record(3, clock.now).document, "environment_id": "case-001-later"}
    later = record(3, clock.now).model_copy(
        update={
            "environment_id": "case-001-later",
            "document": later_document,
            "revision": content_hash(later_document),
            "created_at": clock.now,
        }
    )
    store.put_environment(ACTOR.owner_key, later, None)
    filtered = client.get(
        "/api/environments", params={"cursor": first["next_cursor"]}, headers=headers(token)
    ).json()
    assert filtered["items"] == [] and filtered["next_cursor"]
    third = client.get(
        "/api/environments", params={"cursor": filtered["next_cursor"]}, headers=headers(token)
    ).json()
    assert third["items"][0]["environment_id"] == "case-001"


@pytest.mark.parametrize("ttl", [None, 0, 600])
def test_pagination_refuses_cursor_creation_until_nonexpiring_default_ttl_is_enabled(
    api, monkeypatch, ttl
):
    client, store, token, _, _ = setup(api, monkeypatch)
    store.container.default_ttl = ttl
    response = client.get("/api/environments?page_size=50", headers=headers(token))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "environment_pagination_unavailable"
    assert store.container.queries == []
    assert all("ttl" not in row for row in store.container.items.values())
    assert client.get("/api/environments", headers=headers(token)).status_code == 200


@pytest.mark.parametrize("size", [0, 51, -1, "invalid"])
def test_page_size_bounds_are_not_silently_coerced(api, monkeypatch, size):
    client, store, token, _, _ = setup(api, monkeypatch)
    response = client.get("/api/environments", params={"page_size": size}, headers=headers(token))
    assert response.status_code == 422 and store.container.queries == []


def test_paged_environment_route_is_not_an_anonymous_viewer_surface(api, monkeypatch):
    client, store, _, _, _ = setup(api, monkeypatch)
    assert client.get("/api/environments?page_size=50").status_code == 401
    assert store.container.queries == []
