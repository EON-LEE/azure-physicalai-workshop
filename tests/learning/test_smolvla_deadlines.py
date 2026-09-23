import copy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.checks.smolvla_aml_check import example_config
from learning.common import ContractError, read_json, write_json
from learning.gr00t.azure import workspace_id
from learning.smolvla import azure

NOW = datetime(2026, 9, 23, 12, tzinfo=UTC)
DEADLINE = "2026-09-23T12:01:00Z"
SCHEMA = "physicalai.smolvla-azure/v2"


@pytest.fixture
def config():
    return {**example_config(), "schema": SCHEMA, "job_deadline_utc": DEADLINE}


@pytest.fixture
def clock(monkeypatch):
    from learning import deadlines

    value = SimpleNamespace(utc=NOW, monotonic=10.0)
    monkeypatch.setattr(deadlines, "utcnow", lambda: value.utc)
    monkeypatch.setattr(deadlines, "time", SimpleNamespace(monotonic=lambda: value.monotonic))
    return value


def backend(config, *, state="Queued", cancel_error=None):
    job = SimpleNamespace(
        name="smol-deadline",
        id=workspace_id(config) + "/jobs/smol-deadline",
        status=state,
        tags={
            "scope_owner": config["owner_id"],
            "scope_tenant": config["tenant_id"],
            "specification_sha256": config["specification_sha256"],
            "policy_type": "smolvla",
            "config_schema": config["schema"],
            **(
                {"job_deadline_utc": config["job_deadline_utc"]}
                if "job_deadline_utc" in config
                else {}
            ),
        },
    )
    calls = []

    def get(name):
        assert name == job.name
        calls.append("get")
        return job

    def cancel(name):
        assert name == job.name
        calls.append("cancel")
        if cancel_error:
            raise cancel_error
        job.status = "CancelRequested"

    client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=get,
            cancel=cancel,
            create_or_update=lambda value: pytest.fail("Never recreate an expired/uncertain job"),
        )
    )
    return azure.PolicyJobs(client, config, storage_client=None), job, calls


def test_versioned_deadline_changes_runtime_snapshot_plan_and_all_job_tags(config, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_sha = azure.create_plan(config, first, deterministic_job_name="smol-deadline")
    changed = {**config, "job_deadline_utc": "2026-09-23T12:00:59Z"}
    second_sha = azure.create_plan(changed, second, deterministic_job_name="smol-deadline")
    assert first_sha != second_sha
    plan = azure.read_plan(first)
    assert plan["schema"] == "physicalai.smolvla-azure-plan/v2"
    assert plan["snapshot_sha256"] != azure.read_plan(second)["snapshot_sha256"]
    assert read_json(first / "code" / "run-config.json") == config
    job = read_json(first / "job.json")
    for node in (job, *job["jobs"].values()):
        assert node["tags"]["job_deadline_utc"] == DEADLINE
        assert node["tags"]["config_schema"] == SCHEMA


@pytest.mark.parametrize(
    "value",
    [None, "", "2026-09-23T12:01:00", "2026-09-23T12:01:00+00:00", "2026-09-23 12:01:00Z"],
)
def test_new_configuration_requires_explicit_canonical_utc_deadline(config, value):
    if value is None:
        config.pop("job_deadline_utc")
    else:
        config["job_deadline_utc"] = value
    with pytest.raises(ContractError):
        azure.validate_config(config)


def test_legacy_plan_is_inspectable_but_never_acquires_new_queue_authority(tmp_path):
    config = example_config()
    plan = tmp_path / "legacy"
    azure.create_plan(config, plan, deterministic_job_name="smol-deadline")
    assert azure.read_plan(plan)["schema"] == "physicalai.smolvla-azure-plan/v1"
    jobs, _, calls = backend(config)
    assert jobs.status("smol-deadline")["status"] == "submitted"
    with pytest.raises(ContractError, match="deadline|v2"):
        jobs.preflight()
    with pytest.raises(ContractError, match="deadline|v2"):
        jobs.reconcile_deadline("smol-deadline")
    assert calls == ["get"]


@pytest.mark.parametrize("offset", [60, 61])
def test_expired_preflight_and_submit_fail_before_any_azure_call(config, clock, tmp_path, offset):
    jobs, _, calls = backend(config)
    clock.utc += timedelta(seconds=offset)
    with pytest.raises(ContractError, match="deadline"):
        jobs.preflight()
    with pytest.raises(ContractError, match="deadline"):
        jobs.submit(
            tmp_path / "not-read",
            approved_plan_sha256="a" * 64,
            deterministic_job_name="smol-deadline",
        )
    assert calls == []


def test_submit_rechecks_expiry_after_preflight_before_create(config, clock, tmp_path, monkeypatch):
    from azure.core.exceptions import ResourceNotFoundError

    plan = tmp_path / "plan"
    checksum = azure.create_plan(config, plan, deterministic_job_name="smol-deadline")
    jobs, _, calls = backend(config)

    def missing(name):
        calls.append("get")
        raise ResourceNotFoundError("Not submitted")

    jobs.client.jobs.get = missing
    monkeypatch.setattr(
        jobs, "preflight", lambda: setattr(clock, "utc", NOW + timedelta(seconds=60))
    )
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.ml",
        SimpleNamespace(load_job=lambda **kwargs: SimpleNamespace(tags={})),
    )
    with pytest.raises(ContractError, match="deadline"):
        jobs.submit(plan, approved_plan_sha256=checksum, deterministic_job_name="smol-deadline")
    assert calls == ["get"]


def test_expired_status_is_read_only_and_reconciliation_preserves_cancel_requested(config, clock):
    jobs, _, calls = backend(config)
    clock.utc += timedelta(seconds=61)
    assert jobs.status("smol-deadline")["status"] == "submitted"
    assert calls == ["get"]
    receipt = jobs.reconcile_deadline("smol-deadline")
    assert receipt["status"] == "cancelling"
    assert receipt["deadline_expired"] is True
    assert receipt["cancellation_requested"] is True
    assert receipt["job_deadline_utc"] == DEADLINE
    restarted = azure.PolicyJobs(jobs.client, copy.deepcopy(config), storage_client=None)
    assert restarted.reconcile_deadline("smol-deadline")["cancellation_requested"] is False
    assert calls.count("cancel") == 1


def test_reconciliation_before_deadline_does_not_cancel(config, clock):
    jobs, _, calls = backend(config)
    receipt = jobs.reconcile_deadline("smol-deadline")
    assert receipt["status"] == "submitted"
    assert receipt["deadline_expired"] is False
    assert receipt["cancellation_requested"] is False
    assert calls == ["get"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("Completed", "succeeded"),
        ("Canceled", "cancelled"),
        ("Failed", "failed"),
        ("CancelRequested", "cancelling"),
    ],
)
def test_reconciliation_does_not_fabricate_or_repeat_terminal_cancellation(
    config, clock, state, expected
):
    jobs, _, calls = backend(config, state=state)
    clock.utc += timedelta(seconds=60)
    receipt = jobs.reconcile_deadline("smol-deadline")
    assert receipt["status"] == expected
    assert receipt["deadline_expired"] is True
    assert receipt["cancellation_requested"] is False
    assert "cancel" not in calls


@pytest.mark.parametrize("field", ["scope_owner", "job_deadline_utc", "config_schema"])
def test_deadline_reconciliation_rejects_wrong_owner_or_rebound_authority(config, clock, field):
    jobs, job, calls = backend(config)
    job.tags[field] = "different"
    clock.utc += timedelta(seconds=61)
    with pytest.raises(ContractError):
        jobs.reconcile_deadline("smol-deadline")
    assert calls == ["get"]


def test_uncertain_cancel_propagates_and_never_creates_or_claims_terminal(config, clock):
    jobs, _, calls = backend(config, cancel_error=TimeoutError("cancel acknowledgement unknown"))
    clock.utc += timedelta(seconds=61)
    with pytest.raises(TimeoutError, match="acknowledgement unknown"):
        jobs.reconcile_deadline("smol-deadline")
    assert calls.count("cancel") == 1


def test_terminal_race_is_reread_without_cancel(config, clock):
    jobs, job, calls = backend(config)
    original = jobs.client.jobs.get

    def get(name):
        if calls:
            job.status = "Completed"
        return original(name)

    jobs.client.jobs.get = get
    clock.utc += timedelta(seconds=61)
    result = jobs.reconcile_deadline("smol-deadline")
    assert result["status"] == "succeeded"
    assert result["cancellation_requested"] is False
    assert "cancel" not in calls


@pytest.mark.parametrize("state", ["NotResponding", "Paused", "Unknown"])
def test_unhealthy_azure_state_is_not_evidence_of_terminal_allocation(config, clock, state):
    jobs, _, calls = backend(config, state=state)
    clock.utc += timedelta(seconds=60)
    assert jobs.status("smol-deadline")["azure_status"] == state
    result = jobs.reconcile_deadline("smol-deadline")
    assert result["status"] == "cancelling"
    assert result["cancellation_requested"] is True
    assert calls.count("cancel") == 1


@pytest.mark.parametrize("target", ["parent", "component"])
def test_actual_azure_binding_requires_both_component_and_parent_deadline_tags(
    config, clock, monkeypatch, target
):
    from learning.gr00t import azure as shared

    _, parent, _ = backend(config, state="Running")
    component = SimpleNamespace(
        name="smol-component",
        id=workspace_id(config) + "/jobs/smol-component",
        status="Running",
        parent_job_name=parent.name,
        tags=copy.deepcopy(parent.tags),
    )
    client = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda name: parent if name == parent.name else component)
    )
    monkeypatch.setattr(shared, "azure_job_identity", lambda value: component.id)
    assert shared.running_job_binding(client, config)["azure_component_job_id"] == component.id
    (parent if target == "parent" else component).tags["job_deadline_utc"] = "different"
    with pytest.raises(ContractError, match="deadline"):
        shared.running_job_binding(client, config)


@pytest.mark.parametrize("command", ["export", "train", "compare", "bootstrap"])
def test_expired_component_cannot_touch_data_models_or_managed_identity(
    config, clock, command, tmp_path, monkeypatch
):
    from learning.smolvla import components

    config_path = tmp_path / "runtime.json"
    write_json(config_path, config)
    clock.utc += timedelta(seconds=61)
    monkeypatch.setattr(
        components,
        "clients_for_managed_identity",
        lambda *args, **kwargs: pytest.fail("Expired component must not acquire identity"),
    )
    monkeypatch.setattr(
        components, "verify_code", lambda *args: pytest.fail("Reject before workload reads")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "components",
            command,
            "--runtime-config",
            str(config_path),
            "--snapshot-sha256",
            "a" * 64,
            "--input",
            str(tmp_path / "absent-data"),
            "--output",
            str(tmp_path / "never-written"),
        ],
    )
    with pytest.raises(ContractError, match="deadline"):
        components.main()
    assert not (tmp_path / "never-written").exists()


def test_direct_training_rejects_expiry_before_dataset_or_torch(config, clock, tmp_path):
    from learning.contract import Scope
    from learning.smolvla.train import TrainOptions, run_training

    clock.utc += timedelta(seconds=61)
    with pytest.raises(ContractError, match="deadline"):
        run_training(
            Path("unread-dataset"),
            Path("unread-model"),
            Path("unread-backbone"),
            tmp_path / "unwritten",
            scope=Scope(config["tenant_id"], config["owner_id"]),
            parent_model_sha256="a" * 64,
            conversion_sha256="b" * 64,
            code_snapshot_sha256="c" * 64,
            config=config,
            client=None,
            options=TrainOptions(**config["parameters"]),
        )
    assert not (tmp_path / "unwritten").exists()
