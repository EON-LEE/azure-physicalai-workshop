"""Provision only the dedicated demo's two secretless Entra apps after explicit approval."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID, uuid4


def azure(*args: str):
    completed = subprocess.run(
        ["az", *args, "--output", "json", "--only-show-errors"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout) if completed.stdout.strip() else None


def graph_patch(object_id: str, body: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="physicalai-identity-") as directory:
        path = Path(directory) / "patch.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        azure(
            "rest",
            "--method",
            "PATCH",
            "--url",
            f"https://graph.microsoft.com/v1.0/applications/{object_id}",
            "--body",
            f"@{path}",
        )


def provision(
    tenant_id: UUID, prefix: str, output: Path, include_cli: bool = False, resume: bool = False
) -> dict:
    current = azure("account", "show")
    if current["tenantId"] != str(tenant_id):
        raise ValueError("CLI tenant does not match the explicitly approved tenant.")
    names = [f"physicalai-{prefix}-api", f"physicalai-{prefix}-spa"]
    previous = None
    if output.exists():
        if not resume:
            raise ValueError("Identity manifest exists. Use --resume only for these owned apps.")
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous["tenant_id"] != str(tenant_id):
            raise ValueError("The saved identity manifest belongs to a different tenant.")
        records = previous["applications"]
        if len(records) != 2 or [item["name"] for item in records] != names:
            raise ValueError("Resume requires the exact two recorded dedicated applications.")
        for item in records:
            actual = azure("ad", "app", "show", "--id", item["client_id"])
            if actual["id"] != item["object_id"] or actual["displayName"] != item["name"]:
                raise ValueError("The live application does not match its ownership manifest.")
    else:
        for name in names:
            if azure("ad", "app", "list", "--display-name", name):
                raise ValueError(f"Dedicated app already exists: {name}. Use its reviewed IDs.")
        records = []
        for name in names:
            application = azure(
                "ad", "app", "create", "--display-name", name, "--sign-in-audience", "AzureADMyOrg"
            )
            records.append(
                {"name": name, "object_id": application["id"], "client_id": application["appId"]}
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {"tenant_id": str(tenant_id), "status": "incomplete", "applications": records},
                    indent=2,
                ),
                encoding="utf-8",
            )
    api, spa = records
    scope_id = previous.get("scope_id", str(uuid4())) if previous else str(uuid4())
    output.write_text(
        json.dumps(
            {
                "tenant_id": str(tenant_id),
                "status": "configuring",
                "scope_id": scope_id,
                "applications": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    clients = [{"appId": spa["client_id"], "delegatedPermissionIds": [scope_id]}]
    if include_cli:
        clients.append(
            {
                "appId": "04b07795-8ddb-461a-bbee-02f9e1bf7b46",
                "delegatedPermissionIds": [scope_id],
            }
        )
    graph_patch(
        api["object_id"],
        {
            "identifierUris": [f"api://{api['client_id']}"],
            "api": {
                "requestedAccessTokenVersion": 2,
                "oauth2PermissionScopes": [
                    {
                        "id": scope_id,
                        "value": "access_as_user",
                        "type": "User",
                        "isEnabled": True,
                        "adminConsentDisplayName": "Use the dedicated Physical AI demo",
                        "adminConsentDescription": "Use your Physical AI demo.",
                        "userConsentDisplayName": "Use the Physical AI demo",
                        "userConsentDescription": "Manage your factory tasks.",
                    }
                ],
            },
        },
    )
    # Graph resolves preauthorization against scopes that already exist, not ones in the same PATCH.
    graph_patch(api["object_id"], {"api": {"preAuthorizedApplications": clients}})
    graph_patch(
        spa["object_id"],
        {
            "requiredResourceAccess": [
                {
                    "resourceAppId": api["client_id"],
                    "resourceAccess": [{"id": scope_id, "type": "Scope"}],
                }
            ],
        },
    )
    for item in records:
        existing = azure("ad", "sp", "list", "--filter", f"appId eq '{item['client_id']}'")
        principal = (
            existing[0] if existing else azure("ad", "sp", "create", "--id", item["client_id"])
        )
        item["service_principal_id"] = principal["id"]
    result = {
        "tenant_id": str(tenant_id),
        "status": "created_redirect_uri_pending",
        "applications": records,
        "api_client_id": api["client_id"],
        "spa_client_id": spa["client_id"],
        "scope_id": scope_id,
        "cli_pre_authorized": include_cli,
        "secrets_created": False,
        "directory_api_permissions_added": False,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, type=UUID)
    parser.add_argument("--prefix", required=True, choices=["factory"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-cli-test", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "tenant_id": str(args.tenant_id),
                    "apps": [f"physicalai-{args.prefix}-api", f"physicalai-{args.prefix}-spa"],
                    "scope": "access_as_user on this dedicated API only",
                    "cli_test_pre_authorized": args.authorize_cli_test,
                    "azure_changed": False,
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            provision(
                args.tenant_id, args.prefix, args.output, args.authorize_cli_test, args.resume
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
