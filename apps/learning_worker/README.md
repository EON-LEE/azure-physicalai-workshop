# Private learning worker

This is a separate Azure managed-identity service, not an anonymous job proxy or
the GPU training image. It imports no trainer or model at startup. The existing
API image and operator/reference workflows do not need the Azure ML SDK.

**Production policy admission defaults empty.** SmolVLA is the explicitly
selected commercial candidate, under its own pinned native implementation and
Apache-2.0 model/backbone provenance. It still requires real artifact, GPU and
physical verification before admission. N1.5 is not approved for the
customer production use case. N1.7 has a reported conflict between the pinned
model card and license; it is not admitted either. Do not populate the allowlist,
download weights, submit a GPU job, or publish a policy until the exact source,
model, license and hardware gates are resolved. Recognizing a versioned DTO is
not admitting that model or relabeling another checkpoint.

## Deployment configuration (operator-owned)

Use a private/internal managed Azure endpoint, private Blob/AML connectivity
and explicit API managed-identity caller principals. All `/v1/learning/*` routes
require application identity; delegated or anonymous tokens are not accepted.
The actor envelope is bound to the validated tenant and `X-Environment-Owner`.
That owner key is `sha256(tenant:object)`, not the object UUID.

Environment prefix is `LEARNING_WORKER_`:

- `TENANT_ID`, `AUDIENCE`, `MANAGED_IDENTITY_CLIENT_ID`.
- `ALLOWED_API_PRINCIPALS`: JSON array of approved caller object IDs; empty by
  default.
- `REGISTRY_ACCOUNT_URL`, `REGISTRY_CONTAINER`: fixed private Blob registry.
- `CAPTURE_ACCOUNT_URL`, `CAPTURE_CONTAINER`: the approved simulator upload
  origin/container. A capture URL must exactly match owner/episode/manifest.
- `ALLOWED_POLICY_TYPES`: JSON array; **empty until license/hardware approval**.
- `BOOTSTRAP_OWNER_IDS`: JSON array; empty by default.
- `RECONCILIATION_ENABLED=false`, `RECONCILIATION_ACTOR_IDS=[]` and
  `RECONCILIATION_TARGETS=[]`: independently deployed deadline monitor enrollment.
  A model allowlist alone cannot bypass this separate paid-admission gate.

The API has separate `LEARNING_ENABLED=false`, paired
`LEARNING_WORKER_ENDPOINT`/`LEARNING_WORKER_SCOPE`, empty
`LEARNING_POLICY_TYPES`, and empty `LEARNING_BOOTSTRAP_PRINCIPAL_IDS`.
The normal public viewer remains GET-only.

Build from the repository root with `apps/learning_worker/Dockerfile` and
operator-reviewed, digest-pinned `UV_IMAGE` and `PYTHON_IMAGE` build arguments.
The build intentionally has no mutable image defaults. `uv.lock` pins worker
dependencies separately; no Torch/Isaac packages are mixed into the API runtime.
This repository change does not deploy the service or assign identities.

### Internal ACA deployment source

[`infra/learning-worker.bicep`](../../infra/learning-worker.bicep) deploys only
the worker app into an **existing** ACA environment. It consumes an **existing
dedicated worker UAMI resource ID**, not the API identity that has simulator
control permissions. The deployment operator must create/verify that separate
identity before invoking this template. No role assignments, app registrations,
storage containers, retention rules, AML jobs or GPU resources are created here.

Required parameters are `appName`, `managedEnvironmentId`,
`workerIdentityResourceId`, `apiPrincipalId` (the one approved API MI **object
ID**, not its client ID), `entraTenantId`, `workerAudience`, `registryServer`,
`workerImageRepository`, `workerImageSha256`, and the fixed registry/capture
storage account URLs and containers. The worker client ID is resolved from
the existing UAMI. Image assembly always includes `@sha256:`; no mutable tag
default is provided.

Ingress is internal HTTPS-only (ACA terminates TLS, container port 8080).
`maxReplicas` is 1; `enabled=false` leaves `minReplicas=0`, while explicitly
setting `enabled=true` keeps one warm process. This flag does **not** admit a
model or mean training succeeded. Model and bootstrap lists default to empty
even when the process is warm. `allowedPolicyTypes` admits only explicit
`smolvla` values, and only after the separate real artifact/hardware gate.
The configured caller list is always the singleton API MI principal.
Startup/readiness/liveness use `/healthz` for **process health only**, not AML,
Foundry, GPU, model readiness or policy quality.

The parent deployment operator must grant the **worker** identity, at the
narrowest approved scopes:

| Scope | Required authorization |
|---|---|
| Approved AML workspace | AzureML Data Scientist |
| Approved GPU compute/job UAMI | Managed Identity Operator |
| Private capture container | Storage Blob Data Reader |
| Private registry/output containers or approved prefixes | Storage Blob Data Contributor (or reviewed equivalent scoped permissions) |
| Approved private ACR | AcrPull |

Do not copy the API identity's simulator-control app role to the worker. Verify
existing private DNS/network paths for AML, ACR and Blob separately. Existing
raw/registry/output containers and the approved seven-day output-prefix retention
policy remain parent-owned; this template neither widens nor recreates them.
Serving a healthy internal endpoint is not evidence that these grants or paths
work.

Compile and test this source without authentication or resource creation:

```bash
bicep build infra/learning-worker.bicep --outfile /tmp/learning-worker.json
python -m pytest tests/test_learning_worker_infra.py -q
```

## Durable claims and SDK adapter

`PolicyLearningWorker` wraps the learner-owned closed model registry:
`learning.smolvla.azure.PolicyJobs`/`create_plan` for explicit `smolvla`,
with explicit managed-identity clients. Historical GR00T identities remain
separate; model-use paths reject them rather than silently choosing SmolVLA.
The installed package's exact `POLICY_TYPE` must match the project.

Each owner has an immutable registered plan approval at
`tenants/<tenant>/owners/<opaque-owner>/learning/projects/<project-id>/<kind>-approval.json`.
Artifact files use the same tenant/opaque-owner prefix, compatible with native
identity-based Azure ML datastore scope checks. It includes:

```json
{
  "expires_at": "operator-approved UTC deadline",
  "maximum_cost_usd": "explicit ceiling",
  "gpu_hourly_usd": "reviewed price for the exact approved SKU",
  "config": {
    "schema": "physicalai.smolvla-azure/v2",
    "job_deadline_utc": "exact-reviewed-UTC-Z-not-after"
  }
}
```

The `config` is the **complete** family-specific learner-validated Azure plan, not the illustrative
fragment above. It must bind the exact specification hash, private registered
input assets, source/model revisions, owner, compute/identity and budget.
Configuration is registered by an authorized deployment workflow, never uploaded
by a public browser. Expired/absent approval, missing assets/SDK, unapproved
license or capacity returns an error before submission.
The registered complete-job timeout must fit both the project budget and the
remaining original job authorization; leave time for admission and submission.
An oversized timeout is rejected, never silently extended or rewritten.
The native `job_deadline_utc` must already equal the canonical UTC-Z rendering
of `min(original run.deadline, original approval.expires_at)`. Registration
must occur before approving its config/plan hashes. The worker does not invent,
repair or renew this timestamp. Native v1 plans remain readable/cancellable
but cannot start new paid work. Before submission the worker freezes the exact
configuration alongside its original job claim; subsequent status/cancel never
adopt a changed project-level approval.

The API's conditional Cosmos claim and the worker's create-only Blob claim
precede paid submission. An ambiguous request is reconciled by the existing
job name/tags. It is never retried with a new job name or an upsert of old work.
Status/cancel remain read/reconcile operations for existing claims rather than
requiring new model admission. Azure `Completed` alone is not a candidate.

### Independent, default-OFF deadline reconciliation

`python -m apps.learning_worker.reconcile` is a bounded one-shot server tick,
not a browser timer and not an HTTP endpoint. It accepts only deployment
configuration, with at most 20 unique exact targets:

```json
{
  "actor_id": "explicitly-allowed-actor-uuid",
  "job_name": "the-exact-deterministic-owned-job-name",
  "specification_sha256": "the-approved-request-specification-sha256",
  "configuration_sha256": "the-canonical-complete-native-config-sha256",
  "job_deadline_utc": "the-same-original-reviewed-UTC-not-after"
}
```

These are illustrative field descriptions, not valid enrollment records. The
actor must also occur in `RECONCILIATION_ACTOR_IDS`; tenant comes only from the
worker configuration. There are no wildcards, caller paths, owner/job listings,
history scans, asset downloads, model loads or job submissions in the tick.
It may record enrollment for a known exact target before its job claim exists,
but never creates that job.

Paid preflight and the final pre-submit check require a matching durable
heartbeat from this tick, the same dedicated worker identity, and an age of at
most 90 seconds (future skew over five seconds is rejected). Flags or a static
target list without a recent heartbeat do not count as protection. For the
controlled operator-enrolled workflow, freeze a fixed operator expiry earlier
than the project's wall horizon, leaving admission/submission margin, so the
effective deadline is known before the API request. The request ID, specification
hash, config hash and target must all be approved before paid submission.
Arbitrary future UI jobs are **not automatically enrolled** and remain blocked.

[`infra/learning-reconciler.bicep`](../../infra/learning-reconciler.bicep) is an
optional scheduled ACA job using the existing environment, dedicated worker MI,
and digest-pinned worker image. `enabled=false` creates no resource. If separately
authorized and deployed, its schedule is once per minute, parallelism/completion
count one, retry limit zero, CPU 0.5, memory 1 GiB and replica timeout 120 seconds.
The CLI stops starting additional target checks after 90 seconds. Overlapping
executions share durable cancellation claims. It exposes no ingress and creates
no role, identity, secret, network, GPU or AML resource. The HTTP worker must have
the same explicit targets and `reconciliationEnabled` configuration.

Cancellation reserves an owner/job/config-bound Blob record using conditional
creation before the only native cancel call, and persists its outcome by ETag.
Both API/manual cancellation and the independent tick use this same fence.
Claimed, ambiguous and forbidden attempts are never automatically replayed,
including after a process restart. A crash between claim and POST cannot be
distinguished from a lost POST; it remains explicit and needs operator
reconciliation, not a blind second mutation. An absent receipt does **not**
consume the cancellation attempt: a later queued receipt remains cancellable.
SDK cancellation is `begin_cancel(..., polling=False, retry_total=0)` followed by
a separate read, never treating a cancel ACK as terminal. `CancelRequested`
remains pending; `Queued`/`Running`/`Unknown`/`NotResponding`/`Paused` are not
fabricated completion. Uncertain/forbidden cancellation makes the tick fail
explicitly without retrying or refreshing that target's admission heartbeat.

This source does **not** provision, enable or verify a live scheduler. The parent
operator must deploy the reviewed image/template, verify actual timed execution
and MI access, then enroll the exact test-only target before any new paid job.
Losing the scheduler or permissions remains an operational fault; native v2
late-start and wall-deadline component guards provide a separate defense, not a
claim of an Azure spend cap or guaranteed cancellation after an ambiguous POST.

Outputs must come from the fixed private owner/job output prefix. The final
`result.json`, candidate path, model inventory, actual Azure job ID, raw data
digest, parent weights, optimizer evidence, task/profile and pinned model family
are verified before exposing a candidate. Artifact inventories are uploaded
last and downloaded with bounds, no path traversal or symlinks. Human and
reference-controller sources remain distinct. Historical capture is queried by
its original command/epoch even if a different scene is now active.

## Empty-store bootstrap

No fake P0 or hand-authored ready candidate is required. Once the license is
resolved and actual approved weights are already present on the trusted worker,
an explicitly authorized operator can register **train-only** provenance:

```text
python -m apps.learning_worker.register_parent \
  --owner-object-id <approved operator> \
  --model-root <reviewed local artifact directory> \
  --manifest-sha256 <expected model.json SHA256> \
  --approve-train-only-registration
```

This command does not download vendor weights, accept licenses or execute a
model. It fails with the default empty allowlists and verifies the exact native
artifact/inventory before immutable private upload. A registered train-only
parent cannot resolve through the policy-release/run API.

The approved bootstrap operator can then collect correctly labeled teaching
data, seal it, submit actual optimizer work, and evaluate the new Franka
candidate against the scripted reference under frozen pose cases. That
bootstrap quality/safety gate is not a P0/P1 improvement result. Only explicit
review after actual training and physical evidence creates the first P0 release.
Subsequent customer adaptation requires that genuine P0 and a paired P0/P1 gate.

## Remaining real integration gates

The root/API tests use explicit injected SDK/transport fixtures; they do not
prove AML job execution, a model license, inference latency or learned motion.
The worker projects the concrete native SmolVLA paired/bootstrap schemas,
retaining every measured trial and exact source/job/artifact binding. API
authorization and native file hashes are distinct and both retained. An absent,
malformed or unverified native report is 503, never inferred quality.
Actual train → model load → guarded action →
held-out physical comparison is mandatory before enabling production learning.

Native adapter integration is checked against the committed SmolVLA implementation,
including keyword-only `caller_client_id` and `deterministic_job_name`. Offline
tests generate and revalidate real plans and load train/compare/bootstrap plans
with Azure ML SDK 1.35.0 without requesting credentials or submitting jobs.
An actual SDK `ResourceNotFoundError` during named-job reconciliation remains
unconfirmed; it never causes a second submission. Complete verified reports are
retained even when the native physical quality gate exits with Azure `failed`.
