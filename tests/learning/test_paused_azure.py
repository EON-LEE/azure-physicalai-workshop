import pytest

from learning.checks.smolvla_aml_check import example_config
from learning.common import ContractError
from learning.smolvla import azure


def config():
    return {
        **example_config(),
        "schema": "physicalai.smolvla-azure/v2",
        "job_deadline_utc": "2030-01-01T00:00:00Z",
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
    }


def test_closed_paused_variant_keeps_utc_job_deadline_and_distinct_component_command(tmp_path):
    value = config()
    job = azure.build_job(value, "a" * 64, "paused-train")
    for step in job["jobs"].values():
        assert "learning.paused.components" in step["command"]
        assert step["tags"]["execution_timing"] == "paused_simulation"
        assert step["tags"]["real_time_admission"] == "false"
        assert step["tags"]["job_deadline_utc"] == "2030-01-01T00:00:00Z"
    checksum = azure.create_plan(value, tmp_path / "plan", deterministic_job_name="paused-train")
    assert azure.read_plan(tmp_path / "plan")["plan_sha256"] == checksum
    assert (tmp_path / "plan" / "code" / "learning" / "paused" / "components.py").exists()


@pytest.mark.parametrize(
    "field", ["execution_timing", "real_time_admission", "criteria_sha256", "frozen_plan_sha256"]
)
def test_partial_paused_configuration_is_rejected(field):
    value = config()
    value.pop(field)
    with pytest.raises(ContractError):
        azure.validate_config(value)


def test_legacy_v2_does_not_implicitly_become_paused():
    value = example_config()
    value.update(schema="physicalai.smolvla-azure/v2", job_deadline_utc="2030-01-01T00:00:00Z")
    job = azure.build_job(value, "a" * 64, "realtime-train")
    assert "learning.smolvla.components" in job["jobs"]["train"]["command"]
    assert "execution_timing" not in job["tags"]


def test_paused_training_entry_rejects_expired_wall_authority_before_data(tmp_path):
    from learning.contract import Scope
    from learning.paused.train import run_training
    from learning.smolvla.train import TrainOptions

    value = config()
    value["job_deadline_utc"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ContractError, match="deadline"):
        run_training(
            tmp_path / "data",
            tmp_path / "parent",
            tmp_path / "backbone",
            tmp_path / "output",
            scope=Scope(value["tenant_id"], value["owner_id"]),
            parent_model_sha256="a" * 64,
            conversion_sha256="b" * 64,
            code_snapshot_sha256="c" * 64,
            config=value,
            client=None,
            options=TrainOptions(**value["parameters"]),
        )


def test_native_job_cannot_be_downgraded_to_legacy_mode_by_dropping_config_fields():
    from types import SimpleNamespace

    from learning.gr00t.azure import job_tags, workspace_id

    paused = config()
    job = SimpleNamespace(
        name="bound-mode",
        id=workspace_id(paused) + "/jobs/bound-mode",
        status="Queued",
        tags=job_tags(paused, "a" * 64, policy_type="smolvla"),
    )
    legacy = {name: value for name, value in paused.items() if name not in azure.PAUSED_FIELDS}
    backend = azure.PolicyJobs(
        SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job)), legacy, storage_client=None
    )
    with pytest.raises(ContractError, match="mode"):
        backend.status("bound-mode")


def test_paused_training_preserves_real_cli_but_injects_new_version_validators(
    tmp_path, monkeypatch
):
    from learning.contract import Scope
    from learning.paused import artifacts, dataset, train
    from learning.paused.contract import PausedControlProfile
    from learning.smolvla.train import TrainOptions

    value = config()
    calls = []
    monkeypatch.setattr(train, "_run_training", lambda *args, **kwargs: calls.append(kwargs))
    train.run_training(
        tmp_path / "data",
        tmp_path / "parent",
        tmp_path / "backbone",
        tmp_path / "out",
        scope=Scope(value["tenant_id"], value["owner_id"]),
        parent_model_sha256="a" * 64,
        conversion_sha256="b" * 64,
        code_snapshot_sha256="c" * 64,
        config=value,
        client=None,
        options=TrainOptions(**value["parameters"]),
    )
    assert calls[0]["conversion_validator"] is dataset.validate_conversion
    assert calls[0]["model_validator"] is artifacts.validate_model
    assert calls[0]["profile_type"] is PausedControlProfile
    assert calls[0]["mode_metadata"] == {
        "execution_timing": "paused_simulation",
        "real_time_admission": False,
        "timestamp_basis": "simulation_time",
        "criteria_sha256": "d" * 64,
        "frozen_plan_sha256": "e" * 64,
    }


def test_real_time_training_cannot_silently_accept_paused_config(tmp_path):
    from learning.contract import Scope
    from learning.smolvla.train import TrainOptions, run_training

    value = config()
    with pytest.raises(ContractError, match="paused"):
        run_training(
            tmp_path / "data",
            tmp_path / "parent",
            tmp_path / "backbone",
            tmp_path / "out",
            scope=Scope(value["tenant_id"], value["owner_id"]),
            parent_model_sha256="a" * 64,
            conversion_sha256="b" * 64,
            code_snapshot_sha256="c" * 64,
            config=value,
            client=None,
            options=TrainOptions(**value["parameters"]),
        )
