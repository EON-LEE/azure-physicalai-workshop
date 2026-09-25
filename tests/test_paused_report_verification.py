import json
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.api.errors import Problem
from apps.api.learning_models import fingerprint
from apps.api.models import utcnow
from apps.learning_worker import paused_reports
from apps.learning_worker.artifacts import VerifiedArtifacts
from learning.azure import datastore_prefix
from learning.checks.smolvla_aml_check import example_config
from learning.common import canonical, digest, file_digest
from tests.runtime_support import ACTOR
from tests.test_paused_report_projection import specimen


class Record(SimpleNamespace):
    def model_copy(self, *, update):
        return Record(**(vars(self) | update))


def setup(monkeypatch, *, stub_metadata=True, stub_rescore=True):
    original, plan, verified, output = specimen()
    run = Record(**vars(original.run), backend_job_name=f"learning-{uuid4().hex}")
    specification = Record(
        project=original.project,
        owner_key=ACTOR.owner_key,
        run=run,
        candidate=Record(model_sha256=original.candidate.model_sha256, artifact_id=uuid4()),
        baseline=Record(model_sha256=original.baseline.model_sha256, artifact_id=uuid4()),
    )
    config = example_config()
    config.update(
        schema="physicalai.smolvla-azure/v2",
        execution_timing="paused_simulation",
        real_time_admission=False,
        job_deadline_utc=(utcnow() - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
        kind="compare",
        parameters={"timeout_seconds": 300},
        tenant_id=str(ACTOR.tenant_id),
        owner_id=ACTOR.owner_key,
        output_prefix=f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/output",
        specification_sha256=run.specification_sha256,
        control_profile_sha256=original.project.control_profile_sha256,
        criteria_sha256=original.project.criteria_sha256,
        frozen_plan_sha256=original.project.frozen_plan_sha256,
    )
    prefix = datastore_prefix(config) + f"tenants/{ACTOR.tenant_id}/owners/{ACTOR.owner_key}/"
    config["inputs"] = {
        key: {
            "name": key,
            "version": "1",
            "uri": prefix + key,
            "sha256": checksum,
            "type": "uri_file" if key == "plan" else "uri_folder",
        }
        for key, checksum in {
            "plan": output["evaluation_plan_sha256"],
            "evidence": output["results_sha256"],
            "policy_before": specification.baseline.model_sha256,
            "policy_after": specification.candidate.model_sha256,
        }.items()
    }
    saved, uploads, rescores = {}, [], []
    registry = SimpleNamespace(
        client=SimpleNamespace(url="https://registry.blob.core.windows.net"),
        container=SimpleNamespace(container_name="artifacts"),
        key=lambda actor, path: (
            f"tenants/{actor.tenant_id}/owners/{actor.owner_key}/learning/{path}"
        ),
        get=lambda actor, key: saved.get(key),
        job_configuration=lambda *_: config,
    )

    def put(actor, key, value):
        if key in saved:
            assert saved[key] == value
            return False
        saved[key] = deepcopy(value)
        return True

    def upload(actor, artifact_id, root, metadata):
        assert (root / "report.json").is_file()
        assert file_digest(root / "report.json") == metadata["manifest_sha256"]
        saved[f"artifacts/{artifact_id}/index.json"] = {
            **metadata,
            "owner_key": actor.owner_key,
            "files": {"report.json": metadata["manifest_sha256"]},
        }
        uploads.append(artifact_id)

    registry.put, registry.upload = put, upload
    verifier = VerifiedArtifacts(
        registry,
        None,
        "https://capture.blob.core.windows.net",
        "demonstrations",
        allowed_policy_types=("smolvla",),
    )
    inventory = {"etag": "original"}
    verifier._blob_inventory = lambda *args, **kwargs: fingerprint(inventory)
    verifier._download_file = lambda _account, _container, _key, target: target.write_text(
        json.dumps(output)
    )
    if stub_metadata:
        monkeypatch.setattr(paused_reports, "_read_metadata", lambda *_: None)
    monkeypatch.setattr(paused_reports, "verifier_version", lambda _: "f" * 64)

    def rescore(*args):
        rescores.append(True)
        return deepcopy(verified)

    if stub_rescore:
        monkeypatch.setattr(paused_reports, "_source_rescore", rescore)
    return verifier, specification, config, plan, output, saved, uploads, rescores, inventory


def test_content_bound_cache_avoids_repeated_rescore_and_upload(monkeypatch):
    verifier, spec, _, _, _, _, uploads, rescores, _ = setup(monkeypatch)
    first = verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    second = verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert first == second
    assert len(rescores) == len(uploads) == 1
    assert first.real_time_admission is False


def test_artifact_etag_change_invalidates_verification_not_just_job_identity(monkeypatch):
    verifier, spec, _, _, _, _, _, rescores, inventory = setup(monkeypatch)
    verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    inventory["etag"] = "changed-model-bytes"
    verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert len(rescores) == 2


def test_pinned_verifier_change_requires_new_verification(monkeypatch):
    verifier, spec, _, _, _, _, _, rescores, _ = setup(monkeypatch)
    verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    monkeypatch.setattr(paused_reports, "verifier_version", lambda _: "e" * 64)
    verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert len(rescores) == 2


def test_failure_persists_unverified_diagnostic_and_never_retries_heavy_work(monkeypatch):
    verifier, spec, _, _, _, saved, uploads, rescores, _ = setup(monkeypatch)

    def invalid(*args):
        rescores.append(True)
        raise ValueError("Test-only missing per-tick source/camera proof")

    monkeypatch.setattr(paused_reports, "_source_rescore", invalid)
    with pytest.raises(Problem):
        verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    with pytest.raises(Problem) as second:
        verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert second.value.code == "paused_report_unverified"
    assert len(rescores) == 1 and uploads == []
    assert any(
        value.get("verified") is False
        for key, value in saved.items()
        if key.endswith("failure.json")
    )
    assert not any(key.endswith("verified.json") for key in saved)


def test_process_loss_after_claim_requires_explicit_recovery_without_fake_success(monkeypatch):
    verifier, spec, _, _, _, _, uploads, rescores, _ = setup(monkeypatch)

    def lost(*args):
        rescores.append(True)
        raise SystemExit("Test-only interrupted source verification")

    monkeypatch.setattr(paused_reports, "_source_rescore", lost)
    with pytest.raises(SystemExit):
        verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    with pytest.raises(Problem) as second:
        verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert second.value.code == "paused_report_verification_unconfirmed"
    assert len(rescores) == 1 and uploads == []


def test_artifact_change_during_verification_never_gets_a_certificate(monkeypatch):
    verifier, spec, _, _, output, saved, uploads, rescores, inventory = setup(monkeypatch)

    def changed(*args):
        rescores.append(True)
        inventory["etag"] = "mutated-during-read"
        return {key: value for key, value in output.items() if key not in paused_reports.JOB_FIELDS}

    monkeypatch.setattr(paused_reports, "_source_rescore", changed)
    with pytest.raises(Problem) as failure:
        verifier.completed_report(ACTOR, spec, spec.run.azure_job_id)
    assert failure.value.code == "paused_report_source_changed"
    assert uploads == [] and not any(key.endswith("verified.json") for key in saved)


def test_real_native_evaluator_entry_is_used_and_missing_source_never_passes(monkeypatch, tmp_path):
    verifier, spec, config, plan, _, _, _, _, _ = setup(monkeypatch)
    from learning.paused.evaluation import evaluate_pair

    with pytest.raises((ValueError, OSError)):
        evaluate_pair(
            plan,
            tmp_path / "no-recording",
            tmp_path / "no-p0",
            tmp_path / "no-p1",
            scope=verifier._scope(ACTOR),
            expected_plan_sha256=digest(canonical(plan)),
            expected_results_sha256=config["inputs"]["evidence"]["sha256"],
        )


def test_historical_metadata_check_is_read_only_and_rechecks_datastore_assets_parent_and_child(
    monkeypatch,
):
    verifier, spec, config, _, output, _, _, _, _ = setup(monkeypatch, stub_metadata=False)
    from learning.gr00t.azure import workspace_id

    check = paused_reports._read_metadata
    parent_id = workspace_id(config) + "/jobs/" + spec.run.backend_job_name
    child_id = workspace_id(config) + "/jobs/component"
    spec.run.azure_job_id = parent_id
    output.update(azure_job_id=parent_id, azure_component_job_id=child_id)
    tags = {
        "scope_owner": ACTOR.owner_key,
        "scope_tenant": str(ACTOR.tenant_id),
        "specification_sha256": spec.run.specification_sha256,
        "config_schema": config["schema"],
        "job_deadline_utc": config["job_deadline_utc"],
        "execution_timing": "paused_simulation",
        "real_time_admission": "false",
        "criteria_sha256": config["criteria_sha256"],
        "frozen_plan_sha256": config["frozen_plan_sha256"],
    }
    calls = []
    datastore = SimpleNamespace(
        type="AzureBlob",
        account_name=config["storage_account_name"],
        container_name=config["blob_container"],
        credentials=SimpleNamespace(type="None"),
    )
    parent = SimpleNamespace(id=parent_id, status="Failed", tags=tags)
    child = SimpleNamespace(
        id=child_id, parent_job_name=spec.run.backend_job_name, status="Failed", tags=tags
    )

    def asset(name, *, version):
        calls.append(("asset", name, version))
        value = next(item for item in config["inputs"].values() if item["name"] == name)
        return SimpleNamespace(type=value["type"], path=value["uri"])

    def job(name):
        calls.append(("job", name))
        return child if name == "component" else parent

    verifier._metadata_client = lambda _: SimpleNamespace(
        datastores=SimpleNamespace(get=lambda _: datastore),
        data=SimpleNamespace(get=asset),
        jobs=SimpleNamespace(get=job),
    )
    check(verifier, config, spec, output)
    assert len(calls) == 6
    datastore.container_name = "wrong-container"
    with pytest.raises(Problem) as failure:
        check(verifier, config, spec, output)
    assert failure.value.code == "paused_datastore_changed"
    datastore.container_name = config["blob_container"]
    child.parent_job_name = "another-pipeline"
    with pytest.raises(Problem) as failure:
        check(verifier, config, spec, output)
    assert failure.value.code == "paused_job_mismatch"


def test_input_locations_reject_noncanonical_or_foreign_datastore_paths(monkeypatch):
    verifier, spec, config, _, _, _, _, _, _ = setup(monkeypatch)
    original = config["inputs"]["evidence"]["uri"]
    for uri in (
        "https://another.blob.core.windows.net/private/results",
        original.replace(f"/owners/{ACTOR.owner_key}/", "/owners/" + "a" * 64 + "/"),
        original + "/../escape",
        original + "%2Fescape",
    ):
        altered = deepcopy(config)
        altered["inputs"]["evidence"]["uri"] = uri
        with pytest.raises((Problem, ValueError)):
            paused_reports._locations(verifier, ACTOR, altered, spec)


@pytest.mark.parametrize(
    "changed_timing", [None, "profile", "criteria_sha256", "frozen_plan_sha256"]
)
def test_source_rescore_requires_original_plan_results_and_actual_model_timing(
    monkeypatch, tmp_path, changed_timing
):
    from learning.paused import evaluation

    verifier, spec, config, plan, _, _, _, _, _ = setup(monkeypatch, stub_rescore=False)
    source_rescore = paused_reports._source_rescore
    task = {
        "task_id": spec.project.task_id,
        "instruction": spec.project.instruction,
        "goal_id": spec.project.goal_station_id,
    }
    config["task_sha256"] = digest(canonical(task))
    verifier._download_file = lambda _account, _container, _key, target: target.write_text(
        json.dumps(plan, indent=4)
    )
    verifier._download_prefix = lambda _account, _container, _key, target: target.mkdir()
    model = {
        "task": task,
        "control_profile": plan["control_profile"],
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "criteria_sha256": plan["criteria_sha256"],
        "frozen_plan_sha256": plan["frozen_plan_sha256"],
    }
    if changed_timing == "profile":
        model["control_profile"] = plan["control_profile"] | {
            "profile_id": "franka-position-hold-10hz-paused-v2",
            "max_simulation_steps": 3600,
        }
    elif changed_timing:
        model[changed_timing] = "f" * 64
    verifier._model = lambda *args, **kwargs: model
    verifier.registry.artifact_index = lambda actor, artifact_id: {
        "manifest_sha256": spec.candidate.model_sha256
        if artifact_id == spec.candidate.artifact_id
        else spec.baseline.model_sha256,
    }
    calls = []

    def native(plan_value, evidence, before, after, **kwargs):
        calls.append((plan_value, evidence, before, after, kwargs))
        assert all(path.is_dir() for path in (evidence, before, after))
        return {"test_only": "verified-entry-result"}

    monkeypatch.setattr(evaluation, "evaluate_pair", native)
    if changed_timing:
        with pytest.raises(Problem) as failure:
            source_rescore(
                verifier,
                ACTOR,
                spec,
                config,
                paused_reports._locations(verifier, ACTOR, config, spec),
                tmp_path,
            )
        assert failure.value.code == "paused_model_task"
        assert calls == []
        return
    result = source_rescore(
        verifier,
        ACTOR,
        spec,
        config,
        paused_reports._locations(verifier, ACTOR, config, spec),
        tmp_path,
    )
    assert result == {"test_only": "verified-entry-result"}
    assert len(calls) == 1 and calls[0][0] == plan
    assert calls[0][4] == {
        "scope": verifier._scope(ACTOR),
        "expected_plan_sha256": config["inputs"]["plan"]["sha256"],
        "expected_results_sha256": config["inputs"]["evidence"]["sha256"],
    }


def test_matching_hash_alone_cannot_label_a_v1_native_plan_as_v2(monkeypatch, tmp_path):
    verifier, spec, config, plan, _, _, _, _, _ = setup(monkeypatch, stub_rescore=False)
    spec.project = spec.project.model_copy(
        update={
            "control_profile_id": "franka-position-hold-10hz-paused-v2",
            "evaluation_plan": spec.project.evaluation_plan.model_copy(
                update={
                    "control_profile_id": "franka-position-hold-10hz-paused-v2",
                }
            ),
        }
    )
    verifier._download_file = lambda _account, _container, _key, target: target.write_text(
        json.dumps(plan)
    )
    verifier._download_prefix = lambda *_: pytest.fail(
        "Reject crossed profile ID before bulk input I/O."
    )
    with pytest.raises(Problem) as failure:
        paused_reports._source_rescore(
            verifier,
            ACTOR,
            spec,
            config,
            paused_reports._locations(verifier, ACTOR, config, spec),
            tmp_path,
        )
    assert failure.value.code == "paused_plan_mismatch"
