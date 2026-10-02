"""One bounded, explicitly scoped deadline tick; never submits a learning job."""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import ExitStack

from pydantic import ValidationError

from apps.api.errors import Problem
from apps.api.learning_models import JOB_TERMINAL
from apps.api.models import Principal
from apps.learning_worker.backend import PolicyLearningWorker
from apps.learning_worker.main import WorkerSettings
from apps.learning_worker.registry import BlobRegistry

log = logging.getLogger(__name__)


def reconciled(item):
    return item["status"] in JOB_TERMINAL or (
        item["status"] != "error"
        and not item.get("error_code")
        and item.get("cancellation_state") not in ("claimed", "uncertain", "forbidden")
        and not (item["status"] == "unconfirmed" and item.get("deadline_expired"))
    )


def tick(settings: WorkerSettings, worker: PolicyLearningWorker) -> dict:
    if not settings.reconciliation_enabled:
        return {"enabled": False, "items": [], "training_verified": False}
    deadline = time.monotonic() + 90
    items = []
    for target in settings.reconciliation_targets:
        if time.monotonic() >= deadline:
            items.append(
                {
                    "job_name": target.job_name,
                    "status": "error",
                    "error_code": "tick_budget_exceeded",
                }
            )
            break
        actor = Principal(tenant_id=settings.tenant_id, object_id=target.actor_id)
        try:
            result = worker.reconcile_deadline(actor, target)
            if reconciled(result):
                worker.registry.record_heartbeat(actor, target, settings.managed_identity_client_id)
            else:
                log.error("Deadline cancellation remains unconfirmed for %s", target.job_name)
            items.append(result)
        except Problem as exc:
            log.error("Deadline reconciliation failed for %s: %s", target.job_name, exc.code)
            items.append(
                {
                    "job_name": target.job_name,
                    "status": "error",
                    "error_code": exc.code,
                }
            )
    return {"enabled": True, "items": items, "training_verified": False}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        settings = WorkerSettings()
    except ValidationError:
        print(json.dumps({"error": {"code": "invalid_reconciliation_configuration"}}))
        return 2
    if not settings.reconciliation_enabled:
        print(json.dumps({"enabled": False, "items": [], "training_verified": False}))
        return 0
    from azure.identity import ManagedIdentityCredential

    with ExitStack() as stack:
        credential = ManagedIdentityCredential(client_id=str(settings.managed_identity_client_id))
        stack.callback(credential.close)
        registry = BlobRegistry(
            settings.registry_account_url, settings.registry_container, credential
        )
        stack.callback(registry.close)
        worker = PolicyLearningWorker(
            registry,
            None,
            settings.managed_identity_client_id,
            reconciliation_enabled=True,
            reconciliation_actor_ids=settings.reconciliation_actor_ids,
            reconciliation_targets=settings.reconciliation_targets,
        )
        result = tick(settings, worker)
    print(json.dumps(result, sort_keys=True))
    return int(any(not reconciled(item) for item in result["items"]))


if __name__ == "__main__":
    raise SystemExit(main())
