"""Offline plans, read-only MI observation and explicitly confirmed operator transition."""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path
from uuid import UUID

from learning.common import canonical, parse_json, read_json, require, write_json
from learning.deadlines import JobDeadline
from learning.paused.pool_preparation import (
    _etag,
    make_plan,
    observe,
    pool_resource_id,
    reconcile_transition,
    submit_warmup_once,
    transition,
    validate_plan,
    verify_transition,
)


class PoolArm:
    def __init__(self, plan: dict, credential, *, allow_transition: bool = False) -> None:
        validate_plan(plan)
        from simulation.batch import allocation_formula

        self.url = (
            "https://management.azure.com" + pool_resource_id(plan) + "?api-version=2025-06-01"
        )
        self.credential = credential
        self.allow_transition = allow_transition
        self.deadline = JobDeadline(plan["cleanup_deadline_utc"])
        self.preparation_deadline = JobDeadline(plan["preparation_deadline_utc"])
        self.transition_body = {
            "properties": {
                "scaleSettings": {
                    "autoScale": {
                        "formula": allocation_formula(plan["pool_deadline_utc"]),
                        "evaluationInterval": "PT5M",
                    }
                }
            }
        }

    def _request(self, method: str, body=None, etag=None):
        seconds = self.deadline.check()
        require(
            method == "GET" or (method == "PATCH" and self.allow_transition),
            "Observer transport cannot mutate the pool",
        )
        require(
            method != "PATCH" or body == self.transition_body,
            "Operator transport may only restore the exact original canonical formula",
        )
        if method == "PATCH":
            self.preparation_deadline.check()
            etag = _etag(etag)
        headers = {
            "Authorization": "Bearer "
            + self.credential.get_token("https://management.azure.com/.default").token,
            "Content-Type": "application/json",
        }
        if etag is not None:
            headers["If-Match"] = etag
        request = urllib.request.Request(
            self.url,
            method=method,
            headers=headers,
            data=None if body is None else canonical(body),
        )
        with urllib.request.urlopen(request, timeout=min(20, seconds)) as response:
            raw = response.read(2 * 1024**2 + 1)
        require(len(raw) <= 2 * 1024**2, "Pool response exceeds fixed limit")
        self.deadline.check()
        return parse_json(raw)

    def get(self):
        return self._request("GET")

    def patch(self, body, *, etag):
        return self._request("PATCH", body, etag)


def watch(plan: dict, *, arm, batch, output: Path, publish) -> dict:
    require(not output.exists(), "Observation directory must be new")
    output.mkdir(parents=True)
    deadline = JobDeadline(plan["preparation_deadline_utc"])
    for index in range(224):
        result = observe(plan, arm=arm, batch=batch)
        write_json(output / f"sample-{index:04d}.json", result)
        publish(f"sample-{index:04d}.json", result)
        if result["state"] in ("ready", "rejected", "expired") or deadline.remaining_seconds() <= 0:
            publish("result.json", result)
            write_json(output / "result.json", result)
            return result
        time.sleep(min(15, max(0, deadline.remaining_seconds())))
    raise RuntimeError("Bounded preparation observation count exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("plan", help="Write a new offline plan and preparation pool body.")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    for action in ("observe", "transition", "reconcile", "verify", "warmup"):
        command = commands.add_parser(action)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if action in ("observe", "verify", "warmup"):
            command.add_argument("--client-id", required=True)
        if action in ("transition", "verify", "warmup"):
            command.add_argument("--evidence", type=Path, required=True)
        if action in ("transition", "warmup"):
            command.add_argument("--journal", type=Path, required=True)
            command.add_argument("--approved-plan-sha256", required=True)
        if action in ("reconcile", "verify", "warmup"):
            command.add_argument("--intent", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        plan = make_plan(read_json(args.config))
        require(not args.output.exists(), "Never overwrite a reviewed operation package")
        args.output.mkdir(parents=True)
        write_json(args.output / "plan.json", plan)
        write_json(args.output / "pool.json", plan["pool"])
        print(
            canonical(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "resource_id": pool_resource_id(plan),
                    "cloud_calls": 0,
                    "execution_authorized": False,
                }
            ).decode()
        )
        return
    plan = read_json(args.plan)
    validate_plan(plan)
    require(not args.output.exists(), "Never overwrite operational evidence")
    if args.action in ("transition", "reconcile"):
        from azure.identity import AzureCliCredential

        subscription = pool_resource_id(plan).split("/")[2]
        credential = AzureCliCredential(
            tenant_id=plan["scope"]["tenant_id"], subscription=subscription
        )
        arm = PoolArm(plan, credential, allow_transition=args.action == "transition")
        if args.action == "transition":
            result = transition(
                plan,
                read_json(args.evidence),
                arm=arm,
                journal=args.journal,
                approved_plan_sha256=args.approved_plan_sha256,
            )
        else:
            result = reconcile_transition(plan, read_json(args.intent), arm=arm)
    else:
        from azure.batch import BatchClient
        from azure.identity import ManagedIdentityCredential
        from azure.storage.blob import BlobServiceClient

        require(
            str(UUID(args.client_id)) == args.client_id == plan["observer_client_id"],
            "The original existing observer MI client ID is required",
        )
        with (
            ManagedIdentityCredential(client_id=args.client_id) as credential,
            BatchClient(
                endpoint=plan["platform"]["account_url"],
                credential=credential,
                retry_total=0,
                connection_timeout=10,
                read_timeout=20,
            ) as batch,
        ):
            arm = PoolArm(plan, credential)
            if args.action == "observe":
                from azure.core import MatchConditions

                from learning.common import digest

                prefix = (
                    f"tenants/{plan['scope']['tenant_id']}/owners/{plan['scope']['owner_id']}/"
                    f"learning/pool-preparation/{plan['operation_id']}/"
                )
                with BlobServiceClient(
                    plan["storage_account_url"],
                    credential=credential,
                    retry_total=0,
                    connection_timeout=10,
                    read_timeout=20,
                ) as storage:

                    def publish(name, record):
                        body = canonical(record) + b"\n"
                        blob = storage.get_blob_client(plan["artifact_container"], prefix + name)
                        blob.upload_blob(body, overwrite=False, retry_total=0)
                        properties = blob.get_blob_properties()
                        require(
                            properties.etag and properties.size == len(body),
                            "Observer evidence publication lacks exact bytes/ETag",
                        )
                        copied = blob.download_blob(
                            etag=properties.etag,
                            match_condition=MatchConditions.IfNotModified,
                            retry_total=0,
                        ).readall()
                        require(copied == body, "Observer evidence readback differs")
                        receipt = {
                            "name": prefix + name,
                            "sha256": digest(body),
                            "etag": properties.etag,
                        }
                        print(canonical({"evidence_published": receipt}).decode(), flush=True)
                        return receipt

                    result = watch(plan, arm=arm, batch=batch, output=args.output, publish=publish)
            elif args.action == "verify":
                result = verify_transition(
                    plan, read_json(args.evidence), read_json(args.intent), arm=arm, batch=batch
                )
            else:
                result = submit_warmup_once(
                    plan,
                    read_json(args.evidence),
                    read_json(args.intent),
                    arm=arm,
                    batch=batch,
                    journal=args.journal,
                    approved_plan_sha256=args.approved_plan_sha256,
                )
    if args.action != "observe":
        write_json(args.output, result)
    print(
        canonical(
            {
                "state": result.get("state"),
                "plan_sha256": plan["plan_sha256"],
                "operation_id": plan["operation_id"],
                "output": str(args.output),
                "physical_grant_issued": False,
            }
        ).decode()
    )
    require(
        result.get("state")
        not in ("unknown", "not_ready", "rejected", "expired", "preparation_unchanged"),
        "Preparation did not advance; preserve evidence and reconcile without renewing clocks",
    )


if __name__ == "__main__":
    main()
