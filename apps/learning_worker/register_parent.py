"""Explicit operator-only registration; this module never downloads vendor weights."""

import argparse
from pathlib import Path
from uuid import UUID

from azure.identity import ManagedIdentityCredential

from apps.api.errors import Problem
from apps.api.models import Principal
from apps.learning_worker.artifacts import VerifiedArtifacts
from apps.learning_worker.main import WorkerSettings
from apps.learning_worker.registry import BlobRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Verify existing licensed weights and register train-only provenance."
    )
    parser.add_argument("--owner-object-id", type=UUID, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--execution-timing", choices=["paused_simulation"])
    parser.add_argument("--approve-train-only-registration", action="store_true", required=True)
    args = parser.parse_args()
    settings = WorkerSettings()
    if (
        args.owner_object_id not in settings.bootstrap_owner_ids
        or not settings.allowed_policy_types
    ):
        raise Problem(
            403,
            "registration_unapproved",
            "Explicit operator and reviewed model-license admission are required.",
        )
    actor = Principal(tenant_id=settings.tenant_id, object_id=args.owner_object_id)
    with ManagedIdentityCredential(
        client_id=str(settings.managed_identity_client_id)
    ) as credential:
        registry = BlobRegistry(
            settings.registry_account_url, settings.registry_container, credential
        )
        try:
            validator = VerifiedArtifacts(
                registry,
                credential,
                settings.capture_account_url,
                settings.capture_container,
                allowed_policy_types=settings.allowed_policy_types,
            )
            record = validator.register_parent(
                actor,
                args.model_root.resolve(strict=True),
                args.manifest_sha256,
                actor.object_id,
                execution_timing=args.execution_timing,
            )
            print(record.model_dump_json(exclude={"owner_key"}))
        finally:
            registry.close()


if __name__ == "__main__":
    main()
