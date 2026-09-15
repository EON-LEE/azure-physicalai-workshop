# Azure-only runtime and staged deployment

## What runs where

Production uses Azure Container Apps for the actual web/API, Microsoft Foundry
for the deployed inspection agent, Cosmos DB for versioned state, Blob Storage
for image evidence and licensed assets, Key Vault for simulator TLS, and a
private Azure GPU VM for Isaac Sim. Azure Monitor/Application Insights receives
application telemetry. No production memory store or fixture provider exists.

WSL is only for authoring, dependency management and automated tests. The staged
deployment builds both images in ACR; it does not need a local Docker daemon.
Agent creation and TLS generation execute in an **Azure Container Apps Job**.
The API and simulator authenticate with managed identities. The web client
uses real Microsoft Entra MSAL login and delegated API tokens.

## Current verification boundary

The code, CPU tests, frontend build/browser harness and Bicep compilation can be
checked without cloud writes. **No authorized subscription has been selected for
live deployment in this session.** There is no verified Azure URL, Foundry live
result, driver/SKU compatibility result, or GPU-success claim yet.

The provided Isaac integration is pinned to Isaac Sim 5.1.0 and a reference
Franka inspection/sorting workcell. Runtime API compatibility, robot reach,
motion timing, camera metadata and speed-watchdog thresholds still require the
actual GPU acceptance gate. Unsupported profiles fail instead of falling back.
The simulation-only watchdog is not an industrial safety certification.

## Required approvals and inputs

Copy `infra/deployment.example.json` to a private deployment configuration.
Its zero UUIDs and placeholder regions are deliberately rejected.

Provide an explicitly authorized subscription/tenant, dedicated resource group,
deployment operator object ID, model name/version/region/deployment type, and an
Isaac-compatible GPU SKU with approved quota. A model-catalog listing is not
proof of deployment support. There is no default GPU or implicit subscription.

Supply existing, approved Entra app registrations:

- A public SPA registration using authorization-code flow with PKCE.
- An API registration with `api.requestedAccessTokenVersion = 2`,
  identifier URI `api://<api-client-id>`, and enabled `access_as_user` scope.
- Delegated permission/consent from the SPA to the API.
- The simulator accepts application tokens only from the explicitly allowlisted
  API managed-identity object ID. A delegated user token cannot control it directly.
- After deployment, register the exact returned HTTPS web URL as the approved
  SPA redirect URI. This code does not mutate tenant-wide app registrations.

Confirm NVIDIA terms and the license to upload a **self-contained** Franka USD
asset archive, including referenced meshes/materials/textures. The repository
does not redistribute these assets. Provide a public SSH key, never a private
key. SSH is not exposed publicly by the GPU network security group.

The asset archive is uploaded to Azure Blob, checksummed and extracted on the
GPU VM. Links, traversal paths, duplicate files and oversized archives are
rejected. `Franka` receives the explicit local USD path. There is no default
NVIDIA/S3 asset-server fallback. Prepare all referenced assets locally in the
licensed bundle; external USD references are not an approved runtime dependency.

## Deployment entry point

From WSL at the repository root:

```bash
source scripts/dev-env.sh
uv run --locked python -m scripts.deploy path-to-your-deployment.json
```

Without `--apply`, the command only prints an offline plan and changes no Azure
resources. The configuration must still contain valid, non-placeholder IDs.

After the inputs, costs, licenses and maintenance window have been approved:

```bash
uv run --locked python -m scripts.deploy path-to-your-deployment.json --apply
```

This is a billable write operation. It explicitly pins the subscription on every
CLI invocation and refuses a resource group not tagged for this environment.
The operator needs the resource/RBAC permissions required by the templates.
Pre-register required resource providers through your approved administration
process; the script does not silently broaden privileges or register providers.

Stages:

1. Bicep creates the dedicated data, identity, Foundry, ACR, network and logging resources.
2. Upload the approved asset archive with Entra authentication.
3. Build API/UI and simulator images in ACR; resolve immutable image digests.
4. Start the cloud bootstrap job to create a versioned Foundry agent and TLS material.
5. Only after bootstrap succeeds, deploy the private GPU VM and the actual web/API image.
6. Configure the SPA redirect URI and run real authenticated acceptance tests.

Role propagation or NGC base-image access can delay/fail bootstrap/build. Inspect
the actual job/build logs; do not substitute a generic image or an anonymous
endpoint. Failed stages can leave billable resources in the dedicated group.
The final deployment report explicitly records `live_verified: false` until
the live test gates are run. The current entry point is staged Bicep/CLI;
an `azd up` wrapper is deferred until its full bootstrap lifecycle is verified.

## Networking and security boundary

The GPU VM has no public IP. Its inbound rules allow port 8443 only from the
Container Apps subnet, then deny other inbound traffic. The bridge independently
validates Entra token tenant, audience and the allowlisted managed identity.
TLS is mandatory and the API validates the generated deployment CA; certificate
verification is never disabled. Private keys remain in Key Vault and VM tmpfs.

The reference baseline uses authenticated **public Azure data-service
endpoints**, not private endpoints for every PaaS service. Local/shared-key
authentication is disabled for Blob, Cosmos and Foundry. This is not a claim of
compliance with a customer's private-link, residency, CSAP or production policy.
Add and validate the required private endpoints/egress controls for that target.

NAT egress is provisioned for the private GPU VM. Initial OS/driver/toolkit setup
downloads trusted vendor packages; production images are pulled from ACR and
robot assets from Azure Blob. GPU base OS/driver versions must be frozen to the
combination that passes the actual compatibility test before customer release.

## Costs, maintenance and cleanup

GPU, OS disks, NAT, model use, Container Apps, storage and logs can all incur
charges. The hourly budget field is an **acknowledgment**, not an enforced spend
cap. The VM has a configurable daily UTC shutdown schedule; this does not remove
disks, NAT, logs or other resources. No service is described as free.

Use a dedicated VM; do not colocate unrelated containers. A controlled redeploy
restarts the simulator and rotates bootstrap state/TLS. Stop active demo runs
and use a maintenance window. Old observations cannot approve motion in the new
world epoch.

Cleanup must first compare the selected subscription/group and recorded resource
IDs with the live inventory. Do not delete an entire subscription or unrelated
resources. Key Vault purge protection/soft deletion and platform-managed
Container Apps resources require explicit attention. Automated live teardown
and residual-cost verification remain required G7 work, not a completed claim.
