from types import SimpleNamespace

import pytest

from learning.checks.smolvla_aml_check import example_config
from learning.gr00t.azure import workspace_id
from learning.smolvla.azure import PolicyJobs


@pytest.mark.parametrize("observed_after_post", ["CancelRequested", "Queued"])
def test_owned_cancel_explicitly_disables_sdk_mutation_retries(observed_after_post):
    config = example_config()
    job = SimpleNamespace(
        name="smol-cancel-once",
        id=workspace_id(config) + "/jobs/smol-cancel-once",
        status="Queued",
        tags={
            "scope_owner": config["owner_id"],
            "scope_tenant": config["tenant_id"],
            "specification_sha256": config["specification_sha256"],
            "policy_type": "smolvla",
        },
    )
    calls = []

    def cancel(name, **kwargs):
        calls.append((name, kwargs))
        job.status = observed_after_post

    client = SimpleNamespace(jobs=SimpleNamespace(get=lambda name: job, begin_cancel=cancel))
    receipt = PolicyJobs(client, config, storage_client=None).cancel(job.name)
    assert receipt["status"] == (
        "cancelling" if observed_after_post == "CancelRequested" else "submitted"
    )
    assert calls == [(job.name, {"polling": False, "retry_total": 0})]
