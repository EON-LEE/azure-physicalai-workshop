# Azure-only runtime and staged deployment

## What runs where

Production uses Azure Container Apps for the actual web/API, Microsoft Foundry
for the deployed inspection agent, Cosmos DB for versioned state, Blob Storage
for image evidence and licensed assets, Key Vault for simulator TLS, and a
private Azure GPU VM for Isaac Sim. Azure Monitor/Application Insights receives
application telemetry. No production memory store or fixture provider exists.

WSL is only for authoring, dependency management and automated tests. The staged
deployment builds both images in ACR; it does not need a local Docker daemon.
Agent creation, TLS generation and a real connectivity probe execute in an
explicit Azure bootstrap runner (private Container Instance by default;
Container Apps Job is an alternative).
The API and simulator authenticate with managed identities. The web client
uses real Microsoft Entra MSAL login and delegated API tokens.

## Current verification boundary (2026-09-20/21)

The code, CPU tests, frontend build/browser harness and Bicep compilation can be
checked without cloud writes. Actual deployment was subsequently authorized and
performed: the ACR image build/execution, protected Container Apps web/API,
Entra delegated authentication, private Cosmos configuration persistence, and
a real Foundry agent connectivity response are verified.

GPU execution and the physical closed loop remain blocked. The A10 quota
request returned `ContactSupport`; the RTX PRO alternative returned
`QuotaNotAvailableForResource`. NVIDIA/asset approval is also pending. No
unsupported GPU or replay is substituted for those gates.

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

Choose `runtime_profile: "web"` for the genuine web/agent/data plane without
a GPU or NVIDIA assent. This mode reports the simulator unavailable, not
working. `runtime_profile: "physical"` requires the GPU, licensed assets and
an explicit full `source_commit`. See the two deployment example files.

The default `bootstrap_runner` is `container_instance`, VNet-connected and
one-shot; `container_app_job` is an explicit alternative. A completed ARM
deployment is insufficient: require the bootstrap process to exit zero.
`scripts.provision_identity` can create the two dedicated secretless apps after
explicit authorization. It does not add directory-wide API permissions and its
resume mode checks the recorded ownership/tenant rather than reusing arbitrary apps.

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

1. Bicep creates dedicated data, identity, Foundry, ACR, Private Link and networking.
2. For a physical deployment, upload the approved archive from a deployment host
   with private Blob connectivity, such as an approved VPN/private build agent.
   Ordinary off-VNet WSL cannot bypass a private endpoint.
3. Build allowlisted API/UI sources, and the simulator for a physical profile, in ACR.
4. Run the selected Azure bootstrap and retain its actual Foundry response ID.
5. After bootstrap exit zero, deploy the web/API and, for a physical profile, the GPU.
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

Blob, Cosmos and Key Vault have **public network access disabled** and use private
endpoints/DNS linked to the application, bootstrap and simulator VNet.
ACR and Foundry remain authenticated public Azure endpoints in this baseline.
Local/shared-key authentication is disabled for Blob, Cosmos and Foundry.
This is not a claim that every PaaS endpoint is private or that all customer
residency/CSAP/production policies have been satisfied.

Actual tenant policy forced the three data services private even when an earlier
template requested public access. The implementation was corrected to honor that
policy rather than weaken it. The subscription also required the Azure network
feature named in its deployment error. A failed initial Container Apps environment
later reported `Succeeded` while still having no ingress IP and failing to start
containers. A fresh environment and dedicated recovery subnet fixed this.
`app_environment_name`, `app_subnet_name` and `web_app_name` support an explicit
recovery; they do not silently mask a failed deployment.

NAT egress is provisioned for the private GPU VM. Initial OS/driver/toolkit setup
downloads trusted vendor packages; production images are pulled from ACR and
robot assets from Azure Blob. GPU base OS/driver versions must be frozen to the
combination that passes the actual compatibility test before customer release.

## Costs, maintenance and cleanup

GPU, OS disks, NAT, model use, Container Apps, storage and logs can all incur
charges. The hourly budget field is an **acknowledgment**, not an enforced spend
cap. The VM has a configurable daily UTC shutdown schedule; this does not remove
disks, NAT, logs or other resources. No service is described as free.
The web app scales to zero when idle, but Private Link, NAT, logs, registry and
stored data can still incur charges.

Use a dedicated VM; do not colocate unrelated containers. A controlled redeploy
restarts the simulator and rotates bootstrap state/TLS. Stop active demo runs
and use a maintenance window. Old observations cannot approve motion in the new
world epoch.

Cleanup must first compare the selected subscription/group and recorded resource
IDs with the live inventory. Do not delete an entire subscription or unrelated
resources. Key Vault purge protection/soft deletion and platform-managed
Container Apps resources require explicit attention. Automated live teardown
and residual-cost verification remain required G7 work, not a completed claim.
