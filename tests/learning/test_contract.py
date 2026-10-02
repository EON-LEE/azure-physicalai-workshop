from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from learning.capture import EpisodeWriter, assemble_dataset, resolve_joint_targets
from learning.checks.fixtures import JOINTS, PROVENANCE, SCOPE, edit_frames, frame, overwrite, png
from learning.common import ContractError, canonical, read_json, safe_path
from learning.contract import EpisodeSpec, Scope, png_dimensions, validate_dataset

pytest_plugins = ("learning.checks.pytest_fixtures",)


def test_customer_demonstrations_round_trip_without_labels(make_dataset):
    root = make_dataset()
    data = validate_dataset(root, expected_scope=SCOPE)
    assert len(data.episodes) == 3
    assert len(data.split("test")) == 1
    assert data.episodes[0].metadata["environment_id"] == "customer-line"
    assert (
        data.episodes[0].metadata["provenance"]["scene_builder_id"] == "inspection-cell-custom-v1"
    )
    assert len(data.episodes[0].frames[0]["joint_positions"]) == 9
    assert "defect" not in canonical(data.manifest).decode()
    assert "success" not in canonical(data.manifest).decode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0", None])
def test_writer_rejects_nonfinite_or_non_numeric_joints(tmp_path, value):
    writer = EpisodeWriter(
        tmp_path / "run",
        dataset_id="demo",
        scope=SCOPE,
        episode=EpisodeSpec("episode", "line", "f" * 64, 1, "train"),
        provenance=PROVENANCE,
    )
    changed = (value, *JOINTS[1:])
    with pytest.raises(ContractError):
        writer.append(replace(frame(0), joint_positions=changed))
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize("length", [0, 7, 8, 10])
def test_wrong_action_dimensions_rejected(make_capture, length):
    root = make_capture()
    edit_frames(root, lambda frames: frames[0].update(commanded_joint_targets=[0] * length))
    with pytest.raises(ContractError, match="expected 9"):
        validate_dataset(root, expected_scope=SCOPE)


@pytest.mark.parametrize(
    "field",
    [
        "captured_at_utc",
        "monotonic_ns",
        "physics_step",
        "joint_positions",
        "commanded_joint_targets",
        "images",
        "terminated",
        "truncated",
    ],
)
def test_missing_frame_fields_rejected(make_capture, field):
    root = make_capture()
    edit_frames(root, lambda frames: frames[0].pop(field))
    with pytest.raises(ContractError, match="missing or unexpected"):
        validate_dataset(root, expected_scope=SCOPE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("monotonic_ns", 1_000_000_000),
        ("captured_at_utc", "2026-09-20T00:00:00Z"),
        ("physics_step", 5),
        ("frame_index", 7),
        ("terminated", 1),
    ],
)
def test_bad_sequence_rejected(make_capture, field, value):
    root = make_capture()
    edit_frames(root, lambda frames: frames[1].update({field: value}))
    with pytest.raises(ContractError):
        validate_dataset(root, expected_scope=SCOPE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rendering_frame", 0),
        ("monotonic_ns", 1_000_000_000),
        ("physics_step", 0),
        ("monotonic_ns", 9_000_000_000),
        ("physics_step", 100),
        ("width", 10),
    ],
)
def test_stale_future_or_wrong_shaped_images_rejected(make_capture, field, value):
    root = make_capture()
    edit_frames(root, lambda frames: frames[1]["images"]["inspection"].update({field: value}))
    with pytest.raises(ContractError):
        validate_dataset(root, expected_scope=SCOPE)


@pytest.mark.parametrize("label", ["success", "defect_label", "reward", "destination"])
def test_ground_truth_is_not_allowed_in_policy_data(make_capture, label):
    root = make_capture()
    edit_frames(root, lambda frames: frames[0].update({label: True}))
    with pytest.raises(ContractError, match="unexpected"):
        validate_dataset(root, expected_scope=SCOPE)


def test_extra_evaluator_artifact_rejected(make_capture):
    root = make_capture()
    (root / "truth.json").write_text('{"success":true}')
    with pytest.raises(ContractError, match="keep evaluator labels separate"):
        validate_dataset(root, expected_scope=SCOPE)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.png",
        "/absolute.png",
        "C:/secret.png",
        "C:\\secret.png",
        "//host/share.png",
        "x/../../secret.png",
        "x//file.png",
        "x/./file.png",
        "x/file.png:secret",
        "x/%2e%2e/file.png",
        "",
        "https://example.com/image.png",
    ],
)
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(ContractError):
        safe_path(tmp_path, path, must_exist=False)


def test_symlink_paths_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "file").write_text("not data")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ContractError, match="Symlink"):
        safe_path(tmp_path, "link/file")


@pytest.mark.parametrize("kind", ["manifest", "episode", "image"])
def test_checksums_are_verified(make_capture, kind):
    root = make_capture()
    manifest = read_json(root / "manifest.json")
    if kind == "episode":
        (root / manifest["episodes"][0]["path"]).write_text("{}\n")
    elif kind == "image":
        (root / "episodes" / "demo" / "inspection" / "00000000.png").write_bytes(png(color=4))
    with pytest.raises(ContractError, match="[Cc]hecksum"):
        validate_dataset(
            root,
            expected_scope=SCOPE,
            expected_manifest_sha256="0" * 64 if kind == "manifest" else None,
        )


def test_scope_checked_before_observations_are_opened(make_capture):
    root = make_capture()
    (root / "episodes" / "demo" / "frames.jsonl").unlink()
    with pytest.raises(ContractError, match="Tenant/owner"):
        validate_dataset(root, expected_scope=replace(SCOPE, owner_id="b" * 64))


def test_fixture_cannot_be_live_evidence(make_capture):
    with pytest.raises(ContractError, match="not live"):
        validate_dataset(make_capture(), expected_scope=SCOPE, require_live=True)


@pytest.mark.parametrize("end", ["unfinished", "both", "early"])
def test_termination_is_exact(make_capture, end):
    root = make_capture()
    if end == "unfinished":
        edit_frames(root, lambda frames: frames[-1].update(terminated=False))
    elif end == "both":
        edit_frames(root, lambda frames: frames[-1].update(truncated=True))
    else:
        edit_frames(root, lambda frames: frames[0].update(terminated=True))
    with pytest.raises(ContractError):
        validate_dataset(root, expected_scope=SCOPE)


def test_truncated_episode_is_preserved_not_promoted_to_success(make_capture):
    root = make_capture()
    edit_frames(root, lambda frames: frames[-1].update(terminated=False, truncated=True))
    last = validate_dataset(root, expected_scope=SCOPE).episodes[0].frames[-1]
    assert last["truncated"] is True and last["terminated"] is False


def test_cross_split_seed_leakage_rejected_before_assembly(make_capture, tmp_path):
    roots = [make_capture("train", seed=4), make_capture("test", seed=4, split="test")]
    with pytest.raises(ContractError, match="Cross-split seed"):
        assemble_dataset(
            roots,
            tmp_path / "combined",
            dataset_id="combined",
            expected_scope=SCOPE,
            require_live=False,
        )
    assert not (tmp_path / "combined").exists()


def test_duplicate_episode_rejected(make_capture):
    root = make_capture()
    manifest = read_json(root / "manifest.json")
    manifest["episodes"].append({**manifest["episodes"][0], "split": "test", "seed": 2})
    overwrite(root / "manifest.json", manifest)
    with pytest.raises(ContractError, match="episode leakage"):
        validate_dataset(root, expected_scope=SCOPE)


def test_empty_dataset_rejected(make_capture):
    root = make_capture()
    manifest = read_json(root / "manifest.json")
    manifest["episodes"] = []
    overwrite(root / "manifest.json", manifest)
    with pytest.raises(ContractError, match="Empty"):
        validate_dataset(root, expected_scope=SCOPE)


def test_unfinished_empty_capture_never_publishes(tmp_path):
    writer = EpisodeWriter(
        tmp_path / "run",
        dataset_id="demo",
        scope=SCOPE,
        episode=EpisodeSpec("episode", "line", "f" * 64, 1, "train"),
        provenance=PROVENANCE,
    )
    with pytest.raises(ContractError, match="Empty"):
        writer.finalize()
    writer.append(frame(0))
    writer.append(frame(1))
    with pytest.raises(ContractError, match="terminal"):
        writer.finalize()
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize("limit", ["frames", "bytes", "thread"])
def test_writer_is_bounded_and_single_threaded(tmp_path, limit):
    writer = EpisodeWriter(
        tmp_path / "run",
        dataset_id="demo",
        scope=SCOPE,
        episode=EpisodeSpec("episode", "line", "f" * 64, 1, "train"),
        provenance=PROVENANCE,
        max_frames=2,
        max_bytes=1024 if limit == "bytes" else 1024 * 1024,
    )
    with pytest.raises(ContractError):
        if limit == "thread":
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(writer.append, frame(0)).result()
        elif limit == "bytes":
            writer.append(frame(0))
            writer.append(frame(1))
        else:
            writer.append(frame(0))
            writer.append(frame(1))
            writer.append(frame(2, terminal=True))


def test_sparse_issued_actions_never_use_measured_position():
    with pytest.raises(ContractError, match="not yet known"):
        resolve_joint_targets(None, (0.02, 0.02), (7, 8))
    full = resolve_joint_targets(None, JOINTS)
    resolved = resolve_joint_targets(full, (0.03, None), (7, 8))
    assert resolved == (*JOINTS[:7], 0.03, JOINTS[8])
    with pytest.raises(ContractError, match="Duplicate"):
        resolve_joint_targets(full, (0.02, 0.02), (7, 7))


def test_png_validation_rejects_crc_and_truncated_pixels():
    image = png()
    assert png_dimensions(image) == (32, 32)
    for invalid in (b"not-a-png", image[:-1], image + b"hidden", image[:30] + b"\0" + image[31:]):
        with pytest.raises(ContractError):
            png_dimensions(invalid)


def test_duplicate_json_keys_are_not_silently_accepted(make_capture):
    root = make_capture()
    path = root / "manifest.json"
    document = json.loads(path.read_text())
    path.write_text('{"schema":"wrong",' + json.dumps(document)[1:])
    with pytest.raises(ContractError, match="Duplicate JSON"):
        validate_dataset(root, expected_scope=SCOPE)


def test_bad_utc_and_isaac_identity_fail_closed(make_capture):
    root = make_capture()
    edit_frames(root, lambda frames: frames[0].update(captured_at_utc="2026-09-20T00:00:00+09:00"))
    with pytest.raises(ContractError, match="UTC"):
        validate_dataset(root, expected_scope=SCOPE)
    replace(PROVENANCE, simulator_version="6.0.0").validate()
    with pytest.raises(ContractError, match="5.1.0"):
        replace(PROVENANCE, simulator_version="mock").validate()
    with pytest.raises(ContractError, match="Azure GPU"):
        replace(PROVENANCE, source_kind="isaac_sim").validate()
    with pytest.raises(ContractError, match="opaque owner"):
        Scope(SCOPE.tenant_id, "raw-user-oid").validate()
