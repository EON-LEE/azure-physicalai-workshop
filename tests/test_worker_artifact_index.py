"""Bounded artifact-index/readback tests; all Blob data and HTTP responses are local."""

import hashlib
import json
from collections import Counter
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError

from apps.api.errors import Problem
from apps.learning_worker.registry import BlobRegistry
from tests.runtime_support import ACTOR, OTHER
from tests.test_worker_deadlines import ConditionalBlobs

INDEX_LIMIT = 16 * 1024**2


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class ArtifactBlobs(ConditionalBlobs):
    def __init__(self):
        super().__init__()
        self.reads = Counter()
        self.readall_calls = Counter()
        self.download_options = []
        self.list_calls = 0
        self.before_download = None
        self.before_upload = None

    def _upload(self, name, data, **kwargs):
        if self.before_upload:
            self.before_upload(name)
        return super()._upload(name, data.read() if hasattr(data, "read") else data, **kwargs)

    def download_blob(self, name, **kwargs):
        assert not kwargs.keys() - {
            "offset",
            "length",
            "etag",
            "match_condition",
            "max_concurrency",
            "retry_total",
        }, kwargs
        self.reads[name] += 1
        self.download_options.append((name, kwargs))
        if self.before_download:
            self.before_download(name)
        if name not in self.items:
            raise ResourceNotFoundError(status_code=404)
        content, etag = self.items[name]
        if kwargs.get("etag") is not None:
            assert kwargs.get("match_condition") == MatchConditions.IfNotModified
            if etag != kwargs["etag"]:
                raise ResourceModifiedError(status_code=412)
        start = kwargs.get("offset", 0)
        length = kwargs.get("length")
        body = content[start:] if length is None else content[start : start + length]

        def readall():
            self.readall_calls[name] += 1
            return body

        return SimpleNamespace(
            readall=readall,
            chunks=lambda: (body[i : i + 65536] for i in range(0, len(body), 65536)),
            properties=SimpleNamespace(etag=etag, size=len(content)),
        )

    def list_blobs(self, *, name_starts_with):
        self.list_calls += 1
        return super().list_blobs(name_starts_with=name_starts_with)


def setup():
    registry = BlobRegistry.__new__(BlobRegistry)
    registry.container = ArtifactBlobs()
    registry.budget = None
    return registry, registry.container


def source_tree(tmp_path, count=3):
    entries = {"manifest.json": b'{"source":"CPU-only artifact fixture"}'}
    for index in range(count - 1):
        episode = UUID(int=index % 20 + 1)
        camera = "inspection" if index % 2 else "overview"
        entries[f"episodes/{episode}/{camera}/{index:08d}.png"] = b"CPU-only payload"
    for name, content in entries.items():
        path = tmp_path.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    value = {
        "owner_key": ACTOR.owner_key,
        "manifest_sha256": checksum(entries["manifest.json"]),
        "role": "sealed_dataset",
        "files": {name: checksum(content) for name, content in entries.items()},
    }
    return entries, value


def install(registry, artifact_id, entries, index):
    for name, content in entries.items():
        registry.container._upload(
            registry.key(ACTOR, f"artifacts/{artifact_id}/files/{name}"),
            content,
            overwrite=False,
        )
    registry.container._upload(
        registry.key(ACTOR, f"artifacts/{artifact_id}/index.json"), encoded(index), overwrite=False
    )
    registry.container.writes.clear()


def test_real_sized_17195_file_index_has_its_own_bounded_reader(tmp_path):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 17195)
    body = encoded(value)
    assert 2 * 1024**2 < len(body) < INDEX_LIMIT
    install(registry, artifact_id, entries, value)
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    assert registry.artifact_index(ACTOR, artifact_id) == value
    assert blobs.reads[key] == 1 and blobs.readall_calls[key] == 0
    options = blobs.download_options[-1][1]
    assert options["offset"] == 0 and options["length"] == INDEX_LIMIT + 1


def test_existing_deterministic_large_dataset_is_verified_and_reused_without_per_file_index_reads(
    tmp_path,
):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 17195)
    install(registry, artifact_id, entries, value)
    original = dict(blobs.items)
    metadata = {key: value[key] for key in ("role", "manifest_sha256")}
    registry.upload(ACTOR, artifact_id, tmp_path, metadata)
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    assert 1 <= blobs.reads[key] <= 3
    assert blobs.writes == []
    assert blobs.items == original
    assert all(
        blobs.reads[registry.key(ACTOR, f"artifacts/{artifact_id}/files/{name}")] >= 1
        for name in entries
    ), "Existing payloads must be read back, not trusted because their index is unchanged."


@pytest.mark.parametrize(
    "mutation",
    ["changed-payload", "missing-payload", "extra-payload", "changed-metadata", "missing-index"],
)
def test_existing_payloads_and_complete_index_must_match_before_reuse(tmp_path, mutation):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path)
    install(registry, artifact_id, entries, value)
    index_key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    payload_name = next(name for name in entries if name != "manifest.json")
    payload_key = registry.key(
        ACTOR,
        f"artifacts/{artifact_id}/files/{payload_name}",
    )
    if mutation == "changed-payload":
        content, etag = blobs.items[payload_key]
        blobs.items[payload_key] = b"X" * len(content), etag
    elif mutation == "missing-payload":
        del blobs.items[payload_key]
    elif mutation == "extra-payload":
        blobs.items[registry.key(ACTOR, f"artifacts/{artifact_id}/files/unlisted.txt")] = (
            b"x",
            '"extra"',
        )
    elif mutation == "missing-index":
        del blobs.items[index_key]
    else:
        changed = value | {"role": "another-dataset"}
        blobs.items[index_key] = encoded(changed), blobs.items[index_key][1]
    original = dict(blobs.items)
    with pytest.raises(Problem):
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert blobs.items == original
    assert blobs.writes == []


def test_generic_and_lifecycle_limits_do_not_grow_for_artifact_indexes():
    from tests.test_worker_blob_sdk import setup as sdk_setup

    data = b" " * (2 * 1024**2 + 1)
    registry, container, _, record = sdk_setup([{"status": 206, "etag": '"generic"', "body": data}])
    with container, pytest.raises(Problem) as failure:
        registry.get(ACTOR, "ordinary.json")
    assert failure.value.code == "registry_document_size"
    registry, container, _, _ = sdk_setup(
        [{"status": 206, "etag": '"lifecycle"', "body": b" " * 65537}]
    )
    with container, pytest.raises(Problem) as failure:
        registry._read_record(ACTOR, "lifecycle.json", type(record))
    assert failure.value.code == "registry_document_size"


def test_artifact_index_path_is_fixed_owner_and_uuid_scoped_before_io():
    registry, blobs = setup()
    for artifact_id in ("../another-owner", "id/index.json", "a%2fb", ""):
        with pytest.raises(Problem):
            registry.artifact_index(ACTOR, artifact_id)
    assert not blobs.reads


def test_index_owner_cannot_be_rebound():
    registry, blobs = setup()
    artifact_id = uuid4()
    index = {
        "owner_key": OTHER.owner_key,
        "manifest_sha256": "a" * 64,
        "files": {"manifest.json": "a" * 64},
    }
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    blobs.items[key] = encoded(index), '"original"'
    with pytest.raises(Problem):
        registry.artifact_index(ACTOR, artifact_id)


@pytest.mark.parametrize(
    "change",
    [
        {"files": []},
        {"files": {}},
        {"manifest_sha256": "a" * 63},
        {"files": {"manifest.json": "a" * 63}},
        {"files": {"manifest.json": 123}},
        {"files": {"../manifest.json": "a" * 64}},
        {"files": {"/manifest.json": "a" * 64}},
        {"files": {"C:/manifest.json": "a" * 64}},
        {"files": {"folder\\manifest.json": "a" * 64}},
        {"files": {"a//manifest.json": "a" * 64}},
        {"files": {"a/./manifest.json": "a" * 64}},
        {"files": {"manifest.json": "a" * 64, "manifest.json/child": "a" * 64}},
        {"files": {"payload.bin": "a" * 64}},
        {"role": {"arbitrary": "object"}},
        {"unapproved": "metadata"},
    ],
)
def test_index_schema_paths_and_hashes_are_validated_before_use(change):
    registry, blobs = setup()
    artifact_id = uuid4()
    value = {
        "owner_key": ACTOR.owner_key,
        "manifest_sha256": "a" * 64,
        "files": {"manifest.json": "a" * 64},
    } | change
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    blobs.items[key] = encoded(value), '"invalid-index"'
    with pytest.raises(Problem):
        registry.artifact_index(ACTOR, artifact_id)
    assert sum(blobs.reads.values()) == 1


@pytest.mark.parametrize(
    "body",
    [
        b'{"files":{},"files":{}}',
        b'{"files":NaN}',
        b"[]",
        b'{"files":',
    ],
)
def test_index_parser_rejects_duplicates_nonfinite_and_invalid_json(body):
    registry, blobs = setup()
    artifact_id = uuid4()
    blobs.items[registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")] = body, '"invalid"'
    with pytest.raises(Problem):
        registry.artifact_index(ACTOR, artifact_id)


def test_more_than_one_hundred_thousand_index_entries_is_rejected():
    registry, blobs = setup()
    artifact_id = uuid4()
    files = {f"p/{i}": "a" * 64 for i in range(100000)}
    files["manifest.json"] = "a" * 64
    body = encoded({"owner_key": ACTOR.owner_key, "manifest_sha256": "a" * 64, "files": files})
    assert len(body) < INDEX_LIMIT
    blobs.items[registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")] = (
        body,
        '"oversized-count"',
    )
    with pytest.raises(Problem) as failure:
        registry.artifact_index(ACTOR, artifact_id)
    assert failure.value.code == "artifact_index_invalid"


def test_oversized_index_read_is_capped_and_stops_at_first_byte_over_limit():
    registry, blobs = setup()
    artifact_id = uuid4()
    consumed = []

    def chunks():
        consumed.append("limit")
        yield b" " * INDEX_LIMIT
        consumed.append("over")
        yield b" "
        pytest.fail("The index reader must stop after the first byte over its bound.")

    def bounded_download(name, **kwargs):
        assert name == registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
        assert kwargs["offset"] == 0 and kwargs["length"] == INDEX_LIMIT + 1
        return SimpleNamespace(chunks=chunks, properties=SimpleNamespace(etag='"oversized"'))

    blobs.download_blob = bounded_download
    with pytest.raises(Problem) as failure:
        registry.artifact_index(ACTOR, artifact_id)
    assert failure.value.code == "artifact_index_size"
    assert consumed == ["limit", "over"]


def test_same_index_is_not_reused_if_its_etag_changes_during_payload_readback(tmp_path):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path)
    install(registry, artifact_id, entries, value)
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")

    def replace_index(name):
        if name == key and blobs.reads[key] == 2:
            blobs.items[key] = encoded(value), '"new-index-version"'

    blobs.before_download = replace_index
    with pytest.raises(Problem) as failure:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert failure.value.code == "artifact_index_changed"
    assert blobs.writes == []


def test_payload_replacement_between_listing_and_get_is_conditionally_rejected(tmp_path):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path)
    install(registry, artifact_id, entries, value)
    key = registry.key(ACTOR, f"artifacts/{artifact_id}/files/manifest.json")

    def replace_payload(name):
        if name == key:
            blobs.items[key] = entries["manifest.json"], '"new-payload-version"'

    blobs.before_download = replace_payload
    with pytest.raises(Problem) as failure:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert failure.value.code == "artifact_payload_changed"
    assert blobs.writes == []


@pytest.mark.parametrize("conflicting", [False, True])
def test_index_publication_conflict_uses_dedicated_readback_without_generic_get(
    tmp_path, conflicting
):
    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 17195)
    index_key = registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    competitor = value | ({"role": "different-artifact"} if conflicting else {})

    def publish_competitor(name):
        if name == index_key:
            blobs.items[name] = encoded(competitor), '"concurrent-complete-index"'

    blobs.before_upload = publish_competitor
    registry.get = lambda *_: pytest.fail("Large index publication must not use generic 2 MiB get.")
    metadata = {key: value[key] for key in ("role", "manifest_sha256")}
    if conflicting:
        with pytest.raises(Problem) as failure:
            registry.upload(ACTOR, artifact_id, tmp_path, metadata)
        assert failure.value.code == "immutable_registry_conflict"
    else:
        registry.upload(ACTOR, artifact_id, tmp_path, metadata)
    assert blobs.items[index_key] == (encoded(competitor), '"concurrent-complete-index"')
    assert blobs.reads[index_key] <= 3
    assert len(blobs.writes) == len(entries)
    assert all(not overwrite for _, overwrite, _, _ in blobs.writes)


def test_index_and_reused_payload_io_remain_charged_to_the_existing_budget(tmp_path):
    from datetime import timedelta

    from apps.api.models import utcnow
    from apps.learning_worker.artifact_operations import ArtifactBudget

    registry, blobs = setup()
    artifact_id = uuid4()
    entries, value = source_tree(tmp_path)
    install(registry, artifact_id, entries, value)
    registry.budget = ArtifactBudget(
        utcnow() + timedelta(seconds=60), max_bytes=len(encoded(value)) + 1, max_files=100000
    )
    with pytest.raises(Problem) as failure:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert failure.value.code == "artifact_byte_budget"
    assert blobs.writes == []


def sdk_listing(prefix, entries):
    from xml.sax.saxutils import escape

    items = "".join(
        f"<Blob><Name>{escape(name)}</Name><Properties>"
        "<Last-Modified>Wed, 23 Sep 2026 12:00:00 GMT</Last-Modified>"
        f"<Etag>{escape(etag)}</Etag><Content-Length>{size}</Content-Length>"
        "<BlobType>BlockBlob</BlobType></Properties></Blob>"
        for name, size, etag in entries
    )
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<EnumerationResults ServiceEndpoint="https://offline.blob.core.windows.net/" '
        f'ContainerName="test-only"><Prefix>{escape(prefix)}</Prefix><Blobs>{items}</Blobs>'
        "<NextMarker /></EnumerationResults>"
    ).encode()
    return {"status": 200, "etag": '"list"', "body": body, "content_type": "application/xml"}


def test_actual_sdk_caps_index_range_before_buffering_large_response(tmp_path):
    from tests.test_worker_blob_sdk import setup as sdk_setup

    _, value = source_tree(tmp_path, 17195)
    body = encoded(value)
    registry, container, transport, _ = sdk_setup(
        [
            {"status": 206, "etag": '"large-index"', "body": body},
        ]
    )
    artifact_id = uuid4()
    with container:
        assert registry.artifact_index(ACTOR, artifact_id) == value
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "GET"
    assert request.url.split("?", 1)[0].endswith(
        registry.key(ACTOR, f"artifacts/{artifact_id}/index.json")
    )
    assert request.headers["x-ms-range"] == f"bytes=0-{INDEX_LIMIT}"


def test_actual_sdk_oversized_index_response_is_rejected_at_the_dedicated_limit():
    from tests.test_worker_blob_sdk import setup as sdk_setup

    registry, container, transport, _ = sdk_setup(
        [
            {"status": 206, "etag": '"large-index"', "body": b" " * (INDEX_LIMIT + 1)},
        ]
    )
    with container, pytest.raises(Problem) as failure:
        registry.artifact_index(ACTOR, uuid4())
    assert failure.value.code == "artifact_index_size"
    assert len(transport.requests) == 1
    assert transport.requests[0].headers["x-ms-range"] == f"bytes=0-{INDEX_LIMIT}"


def test_actual_sdk_retry_reads_payload_with_if_match_and_preserves_completed_index(tmp_path):
    from tests.test_worker_blob_sdk import setup as sdk_setup

    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 1)
    registry, container, transport, _ = sdk_setup([])
    prefix = registry.key(ACTOR, f"artifacts/{artifact_id}/files/")
    data = entries["manifest.json"]
    listing = sdk_listing(prefix, [(prefix + "manifest.json", len(data), '"payload"')])
    transport.responses.extend(
        [
            {"status": 206, "etag": '"complete-index"', "body": encoded(value)},
            listing,
            {"status": 206, "etag": '"payload"', "body": data},
            listing,
            {"status": 206, "etag": '"complete-index"', "body": encoded(value)},
        ]
    )
    with container:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert not transport.responses
    assert all(request.method == "GET" for request in transport.requests)
    payload = transport.requests[2]
    assert payload.headers["If-Match"] == '"payload"'
    assert payload.headers["x-ms-range"] == f"bytes=0-{len(data)}"
    assert transport.requests[-1].headers["If-Match"] == '"complete-index"'
    assert transport.requests[-1].headers["x-ms-range"] == f"bytes=0-{INDEX_LIMIT}"


def test_actual_sdk_rejects_replaced_payload_instead_of_repairing_it(tmp_path):
    from tests.test_worker_blob_sdk import setup as sdk_setup

    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 1)
    registry, container, transport, _ = sdk_setup([])
    prefix = registry.key(ACTOR, f"artifacts/{artifact_id}/files/")
    transport.responses.extend(
        [
            {"status": 206, "etag": '"complete-index"', "body": encoded(value)},
            sdk_listing(
                prefix, [(prefix + "manifest.json", len(entries["manifest.json"]), '"old"')]
            ),
            {"status": 412, "etag": '"replacement"', "code": "ConditionNotMet"},
        ]
    )
    with container, pytest.raises(Problem) as failure:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert failure.value.code == "artifact_payload_changed"
    assert len(transport.requests) == 3
    assert transport.requests[-1].headers["If-Match"] == '"old"'
    assert all(request.method == "GET" for request in transport.requests)


def test_actual_sdk_fresh_upload_commits_index_only_after_payload_readback(tmp_path):
    from tests.test_worker_blob_sdk import setup as sdk_setup

    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 1)
    registry, container, transport, _ = sdk_setup([])
    prefix = registry.key(ACTOR, f"artifacts/{artifact_id}/files/")
    data = entries["manifest.json"]
    listing = sdk_listing(prefix, [(prefix + "manifest.json", len(data), '"written-payload"')])
    transport.responses.extend(
        [
            {"status": 404, "etag": '"missing"', "code": "BlobNotFound"},
            sdk_listing(prefix, []),
            {"status": 201, "etag": '"written-payload"'},
            listing,
            {"status": 206, "etag": '"written-payload"', "body": data},
            listing,
            {"status": 201, "etag": '"written-index"'},
            {"status": 206, "etag": '"written-index"', "body": encoded(value)},
        ]
    )
    with container:
        registry.upload(
            ACTOR, artifact_id, tmp_path, {key: value[key] for key in ("role", "manifest_sha256")}
        )
    assert not transport.responses
    assert [request.method for request in transport.requests] == [
        "GET",
        "GET",
        "PUT",
        "GET",
        "GET",
        "GET",
        "PUT",
        "GET",
    ]
    assert transport.requests[2].headers["If-None-Match"] == "*"
    assert transport.requests[4].headers["If-Match"] == '"written-payload"'
    assert transport.requests[6].headers["If-None-Match"] == "*"


@pytest.mark.parametrize("conflicting", [False, True])
def test_actual_sdk_index_publication_conflict_is_bounded_and_never_overwritten(
    tmp_path, conflicting
):
    from tests.test_worker_blob_sdk import setup as sdk_setup

    artifact_id = uuid4()
    entries, value = source_tree(tmp_path, 1)
    registry, container, transport, _ = sdk_setup([])
    prefix = registry.key(ACTOR, f"artifacts/{artifact_id}/files/")
    data = entries["manifest.json"]
    listing = sdk_listing(prefix, [(prefix + "manifest.json", len(data), '"payload"')])
    published = value | ({"role": "different"} if conflicting else {})
    transport.responses.extend(
        [
            {"status": 404, "etag": '"missing"', "code": "BlobNotFound"},
            sdk_listing(prefix, []),
            {"status": 201, "etag": '"payload"'},
            listing,
            {"status": 206, "etag": '"payload"', "body": data},
            listing,
            {"status": 409, "etag": '"concurrent"', "code": "BlobAlreadyExists"},
            {"status": 206, "etag": '"concurrent"', "body": encoded(published)},
        ]
    )
    if not conflicting:
        transport.responses.extend(
            [
                listing,
                {"status": 206, "etag": '"payload"', "body": data},
                listing,
                {"status": 206, "etag": '"concurrent"', "body": encoded(published)},
            ]
        )
    with container:
        if conflicting:
            with pytest.raises(Problem) as failure:
                registry.upload(
                    ACTOR,
                    artifact_id,
                    tmp_path,
                    {key: value[key] for key in ("role", "manifest_sha256")},
                )
            assert failure.value.code == "immutable_registry_conflict"
        else:
            registry.upload(
                ACTOR,
                artifact_id,
                tmp_path,
                {key: value[key] for key in ("role", "manifest_sha256")},
            )
    assert not transport.responses
    writes = [request for request in transport.requests if request.method == "PUT"]
    assert len(writes) == 2
    assert all(request.headers["If-None-Match"] == "*" for request in writes)
    reads = [
        request
        for request in transport.requests
        if "/index.json" in request.url and request.method == "GET"
    ]
    assert len(reads) <= 3
    assert all(request.headers["x-ms-range"] == f"bytes=0-{INDEX_LIMIT}" for request in reads)
