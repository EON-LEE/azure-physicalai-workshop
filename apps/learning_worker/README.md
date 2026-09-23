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

The API has separate `LEARNING_ENABLED=false`, paired
`LEARNING_WORKER_ENDPOINT`/`LEARNING_WORKER_SCOPE`, empty
`LEARNING_POLICY_TYPES`, and empty `LEARNING_BOOTSTRAP_PRINCIPAL_IDS`.
The normal public viewer remains GET-only.

Build from the repository root with `apps/learning_worker/Dockerfile` and
operator-reviewed, digest-pinned `UV_IMAGE` and `PYTHON_IMAGE` build arguments.
The build intentionally has no mutable image defaults. `uv.lock` pins worker
dependencies separately; no Torch/Isaac packages are mixed into the API runtime.
This repository change does not deploy the service or assign identities.

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
  "config": {"schema": "physicalai.gr00t-azure/v1"}
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

The API's conditional Cosmos claim and the worker's create-only Blob claim
precede paid submission. An ambiguous request is reconciled by the existing
job name/tags. It is never retried with a new job name or an upsert of old work.
Status/cancel remain read/reconcile operations for existing claims rather than
requiring new model admission. Azure `Completed` alone is not a candidate.

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
