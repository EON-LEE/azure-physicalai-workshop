import shutil
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.models import DemonstrationResult
from apps.learning_worker.artifacts import VerifiedArtifacts
from learning import contract
from learning.capture import EpisodeWriter
from learning.checks.fixtures import PROVENANCE
from learning.common import file_digest
from tests.learning.test_groot_capture import sample
from tests.runtime_support import ACTOR
from tests.test_learning_teaching_cases import activate_case, create_varied, start


def fixture_capture(root, project, case, *, split=None, seed=None):
    episode_id = uuid4()
    writer = EpisodeWriter(
        root,
        dataset_id=str(episode_id),
        scope=contract.Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
        episode=contract.EpisodeSpec(
            str(episode_id),
            case.environment_id,
            case.revision,
            case.seed if seed is None else seed,
            case.split if split is None else split,
        ),
        provenance=replace(PROVENANCE, source_kind="test_fixture", simulator_version="6.0.0"),
        control_profile=contract.ControlProfile(servo_profile_sha256="d" * 64),
        demonstration=contract.DemonstrationSource(
            kind="reference_controller",
            task_id=project.task_id,
            instruction=project.instruction,
            goal_id=project.goal_station_id,
        ),
    )
    writer.append(sample(0))
    writer.append(sample(1, terminal=True))
    writer.finalize()
    return DemonstrationResult(
        status="uploaded",
        episode_id=episode_id,
        frame_count=2,
        manifest_uri=f"https://test.blob.core.windows.net/demonstrations/{ACTOR.owner_key}/{episode_id}/manifest.json",
        manifest_sha256=file_digest(root / "manifest.json"),
    )


def verifier(tmp_path, monkeypatch):
    uploaded = {}
    sources = {}
    production_validation = contract.validate_dataset

    def explicitly_test_only_validation(root, **kwargs):
        if "require_live" in kwargs:
            assert kwargs["require_live"] is True, (
                "Worker verification must still demand real capture."
            )
        return production_validation(root, **{**kwargs, "require_live": False})

    # Test-fixture images cannot claim Azure/Isaac provenance. Only this injected validator
    # permits them, while exercising the real native inventory/hold/split checks.
    monkeypatch.setattr(contract, "validate_dataset", explicitly_test_only_validation)
    monkeypatch.setattr("learning.capture.validate_dataset", explicitly_test_only_validation)

    def upload(actor, artifact_id, root, metadata):
        assert actor == ACTOR
        saved = tmp_path / f"registered-{artifact_id}"
        shutil.copytree(root, saved)
        uploaded[artifact_id] = (saved, metadata)

    def download(actor, artifact_id, root):
        assert actor == ACTOR
        saved, metadata = uploaded[artifact_id]
        shutil.copytree(saved, root)
        return metadata

    registry = SimpleNamespace(upload=upload, download=download)
    worker = VerifiedArtifacts(
        registry,
        None,
        "https://test.blob.core.windows.net",
        "demonstrations",
    )

    def download_original(account, container, prefix, output, **kwargs):
        assert account == "https://test.blob.core.windows.net" and container == "demonstrations"
        shutil.copytree(sources[prefix], output)

    monkeypatch.setattr(worker, "_download_prefix", download_original)
    return worker, sources, uploaded, production_validation


def test_nonanchor_validation_capture_keeps_actual_case_and_native_mixed_splits(
    tmp_path, monkeypatch
):
    service, project, cases, _ = create_varied()
    worker, sources, uploaded, native_validate = verifier(tmp_path, monkeypatch)
    captures = []
    for index, selected in enumerate(cases):
        activate_case(service, selected)
        session = start(service, project, selected["case_id"]).value
        original = tmp_path / f"original-{index}"
        receipt = fixture_capture(original, project.value, session.teaching_case)
        sources[f"{ACTOR.owner_key}/{receipt.episode_id}/"] = original
        capture = worker.verify_capture(
            ACTOR, project.value, session, receipt.model_dump(mode="json")
        )
        assert (
            capture.case_id,
            capture.environment_id,
            capture.revision,
            capture.seed,
            capture.split,
        ) == (
            selected["case_id"],
            selected["environment_id"],
            selected["revision"],
            selected["seed"],
            selected["split"],
        )
        captures.append(capture)
    dataset_id = uuid4()
    artifact_id, digest = worker.seal_dataset(ACTOR, project.value, dataset_id, tuple(captures))
    root = uploaded[artifact_id][0]
    validated = native_validate(
        root,
        expected_scope=contract.Scope(str(ACTOR.tenant_id), ACTOR.owner_key),
        require_live=False,
    )
    assert digest == validated.manifest_sha256
    assert [episode.metadata["split"] for episode in validated.episodes] == [
        "train",
        "train",
        "validation",
    ]
    assert [episode.metadata["seed"] for episode in validated.split("train")] == [10001, 10002]
    assert [episode.metadata["seed"] for episode in validated.split("validation")] == [20001]
    assert all(
        episode.metadata["demonstration"]["instruction"] == project.value.instruction
        for episode in validated.episodes
    )


@pytest.mark.parametrize(
    "split,seed", [("test", None), ("train", None), ("validation", 30001), ("validation", 900002)]
)
def test_worker_refuses_actual_test_or_relabelled_split_seed_before_registering_capture(
    tmp_path, monkeypatch, split, seed
):
    service, project, cases, _ = create_varied()
    worker, sources, uploaded, _ = verifier(tmp_path, monkeypatch)
    activate_case(service, cases[2])
    session = start(service, project, cases[2]["case_id"]).value
    root = tmp_path / "wrong-original"
    receipt = fixture_capture(root, project.value, session.teaching_case, split=split, seed=seed)
    sources[f"{ACTOR.owner_key}/{receipt.episode_id}/"] = root
    with pytest.raises(Problem) as failure:
        worker.verify_capture(ACTOR, project.value, session, receipt.model_dump(mode="json"))
    assert failure.value.code == "capture_provenance_mismatch"
    assert uploaded == {}


def test_worker_rechecks_downloaded_case_metadata_at_sealing_not_just_a_receipt_hash(
    tmp_path, monkeypatch
):
    service, project, cases, _ = create_varied()
    worker, sources, uploaded, _ = verifier(tmp_path, monkeypatch)
    activate_case(service, cases[2])
    session = start(service, project, cases[2]["case_id"]).value
    root = tmp_path / "validation"
    receipt = fixture_capture(root, project.value, session.teaching_case)
    sources[f"{ACTOR.owner_key}/{receipt.episode_id}/"] = root
    capture = worker.verify_capture(ACTOR, project.value, session, receipt.model_dump(mode="json"))
    wrong = tmp_path / "wrong-split"
    altered = fixture_capture(wrong, project.value, session.teaching_case, split="train")
    forged = capture.model_copy(
        update={
            "episode_id": altered.episode_id,
            "manifest_sha256": altered.manifest_sha256,
        }
    )
    uploaded[capture.artifact_id] = (wrong, {"manifest_sha256": altered.manifest_sha256})
    with pytest.raises(Problem) as failure:
        worker.seal_dataset(ACTOR, project.value, uuid4(), (forged,))
    assert failure.value.code == "capture_provenance_mismatch"
