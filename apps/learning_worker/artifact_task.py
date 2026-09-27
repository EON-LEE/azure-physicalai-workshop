"""Fixed one-shot artifact validator; no arbitrary modules, URLs or model/job submission."""

import argparse
import json
import tempfile
from uuid import UUID

from azure.core.exceptions import AzureError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient

from apps.api.artifact_models import ArtifactResult
from apps.api.errors import Problem
from apps.api.models import Principal, utcnow
from apps.learning_worker.artifact_operations import ArtifactBudget, ArtifactOperations
from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.main import WorkerSettings
from apps.learning_worker.registry import BlobRegistry


def estimate_input_size(work, registry, credential, settings):
    total, files = 0, 0
    if work.operation == "capture":
        locations = [
            (
                settings.capture_account_url,
                settings.capture_container,
                f"{work.actor.owner_key}/{work.receipt.episode_id}/",
            )
        ]
    elif work.operation == "managed_evaluation":
        from apps.learning_worker.managed_reports import context, input_locations

        value = context(
            registry, work.actor, work.project.id, work.target_id, expected=work.managed_import
        )
        locations = [
            (settings.registry_account_url, settings.registry_container, location)
            for location in input_locations(registry, work.actor, value)
        ]
    else:
        locations = [
            (
                settings.registry_account_url,
                settings.registry_container,
                registry.key(work.actor, f"artifacts/{capture.artifact_id}/files/"),
            )
            for capture in work.captures
        ]
    for account, container, prefix in locations:
        with BlobServiceClient(
            account,
            credential=credential,
            connection_timeout=5,
            read_timeout=10,
            retry_total=0,
        ) as client:
            for blob in client.get_container_client(container).list_blobs(name_starts_with=prefix):
                if not blob.name.startswith(prefix):
                    raise Problem(
                        403, "artifact_scope", "Input inventory escaped its owned prefix."
                    )
                files += 1
                total += blob.size
                if files > work.max_files or total > work.max_bytes:
                    raise Problem(422, "artifact_input_budget", "Input inventory exceeds approval.")
    return total


def execute(operation_id, actor_id, claim_id, settings):
    actor = Principal(tenant_id=settings.tenant_id, object_id=actor_id)
    if not settings.artifact_ops_enabled or actor_id not in settings.artifact_actor_ids:
        raise Problem(
            403, "artifact_owner_unapproved", "Resident artifact operation is not approved."
        )
    with ManagedIdentityCredential(
        client_id=str(settings.managed_identity_client_id)
    ) as credential:
        registry = BlobRegistry(
            settings.registry_account_url, settings.registry_container, credential
        )
        try:
            ops = ArtifactOperations(registry)
            work = ops.work(actor, operation_id)
            if work.operation == "managed_evaluation" and not settings.paused_evaluation_enabled:
                raise Problem(503, "paused_learning_unavailable", "Managed import is not admitted.")
            if (
                work.operation == "managed_evaluation"
                and "smolvla" not in settings.allowed_policy_types
            ):
                raise Problem(
                    503, "learning_policy_unapproved", "Model admission is not configured."
                )
            maximum_bytes = (
                settings.artifact_capture_bytes
                if work.operation == "capture"
                else settings.artifact_dataset_bytes
            )
            if (
                work.max_bytes > maximum_bytes
                or work.max_files > settings.artifact_max_files
                or (work.deadline - work.created_at).total_seconds() > settings.artifact_max_seconds
            ):
                raise Problem(
                    403, "artifact_budget_unapproved", "Deployment artifact limits changed."
                )
            state = ops.status(actor, operation_id)
            if state is None or state.status != "running" or state.claim_id != claim_id:
                raise Problem(
                    409, "artifact_claim_changed", "This subprocess does not own the claim."
                )
            budget = ArtifactBudget(
                work.deadline, max_bytes=work.max_bytes, max_files=work.max_files
            )
            registry.budget = budget
            input_size = estimate_input_size(work, registry, credential, settings)
            if input_size * 2 > work.max_bytes:
                raise Problem(
                    422,
                    "artifact_byte_budget",
                    "Input plus required publication transfer exceeds the total operation budget.",
                )
            budget.require_disk(input_size, tempfile.gettempdir())
            verifier = VerifiedArtifacts(
                registry,
                credential,
                settings.capture_account_url,
                settings.capture_container,
                allowed_policy_types=settings.allowed_policy_types,
                budget=budget,
            )
            if work.operation == "capture":
                capture = verifier.verify_capture(
                    actor, work.project, work.session, work.receipt.model_dump(mode="json")
                )
                result = ArtifactResult(
                    artifact_id=capture.artifact_id,
                    manifest_sha256=capture.manifest_sha256,
                    capture=capture,
                )
            elif work.operation == "managed_evaluation":
                from apps.learning_worker.managed_reports import complete

                receipt = complete(verifier, actor, work)
                result = ArtifactResult(
                    artifact_id=receipt.report.artifact_id,
                    manifest_sha256=receipt.report.report_sha256,
                    managed_evaluation=receipt,
                )
            else:
                artifact_id, digest = verifier.seal_dataset(
                    actor, work.project, work.target_id, work.captures
                )
                result = ArtifactResult(artifact_id=artifact_id, manifest_sha256=digest)
            budget.check()
            registry.put(
                actor,
                f"artifact-operations/{operation_id}/result.json",
                {
                    "work_sha256": work.sha256,
                    "result": result.model_dump(mode="json"),
                    "completed_at": utcnow().isoformat(),
                },
            )
            return (
                "report_committed"
                if work.operation == "managed_evaluation"
                else "manifest_committed"
            )
        except (Problem, ValueError, OSError, AzureError) as exc:
            registry.put(
                actor,
                f"artifact-operations/{operation_id}/failure.json",
                {
                    "error_code": exc.code
                    if isinstance(exc, Problem)
                    else "artifact_processing_failed",
                    "verified": False,
                },
            )
            raise
        finally:
            registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation-id", type=UUID, required=True)
    parser.add_argument("--actor-id", type=UUID, required=True)
    parser.add_argument("--claim-id", type=UUID, required=True)
    args = parser.parse_args()
    try:
        status = execute(args.operation_id, args.actor_id, args.claim_id, WorkerSettings())
    except (Problem, ValueError, OSError, AzureError) as exc:
        print(
            json.dumps(
                {
                    "error_code": exc.code
                    if isinstance(exc, Problem)
                    else "artifact_processing_failed"
                }
            )
        )
        return 1
    print(json.dumps({"status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
