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
- `REFERENCE_COLLECTIONS_ENABLED=false`: read-only lookup of an operator's
  exact reference authorization; not a grant-creation or runtime-installation API.
- `PAUSED_TRAINING_ENABLED=false`, `PAUSED_EVALUATION_ENABLED=false`: separate
  stage admission, still requiring original native config, artifact, cost,
  managed-identity, hardware and fresh monitor-enrollment checks.
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

### Managed Batch controller packaging

Managed report certificates hash the verifier source and the production
dependency lock, not generated `.venv`, bytecode, package caches or
`node_modules` contents. Source or lock changes invalidate certificates; linked
source paths are rejected. This keeps repeated certificate checks bounded to
the verifier rather than recursively scanning its installed SDK environment.

The worker image includes the `simulation` source and pins `azure-batch==15.1.0`
for the lightweight `python -m simulation.batch` controller. This is an
operator-invoked managed-job CLI, **not an always-live simulator bridge**.
Isaac runs in the separately approved GPU task image through
`simulation.batch_task`; copying its source does not install or start Isaac,
CUDA, Torch or a model in this CPU worker.

The existing non-root HTTP entrypoint, routes and default-OFF admission flags
are unchanged. Packaging the SDK grants no Batch authority and creates no pool,
node or job. Review `python -m simulation.batch --help` and the approved
platform/job specification before use; `warmup` and `submit` require explicit
`--confirm-submission`. Image build/deployment and any actual submission remain
the deployment operator's responsibility.

Build this packaging together with the runtime-owner's committed
`simulation.batch` and `simulation.batch_task` sources. The following smoke
check imports the real locked SDK and both controller modules, rejects
network/process attempts, and checks that GPU/model modules remain unloaded.
Run the same import boundary in the resulting immutable image before deployment:

```bash
python -m pytest tests/test_learning_worker_packaging.py -q
PYTHONPATH="$PWD" uv run --project apps/learning_worker --locked --no-dev python tests/check_worker_batch_sdk.py
```

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

### Post-hoc external-native training imports

An operator-native UUID-named Azure ML command is **not** an API `TrainingRun`.
Do not rename the job, populate `jobs/<learning-name>/specification.json` after
execution, or call trained weights a pretrained parent or policy release.
The external import is a distinct, post-hoc verification/registration operation.
It does not submit, resume, cancel, authorize, or extend a training job.

Associate an existing private app project with the exact original task/profile/
criteria and all twenty original TRAIN case descriptors. This association may
be made after training and is not represented as pre-execution approval.
The API/worker global, paused-training, model, bootstrap-operator, and artifact
actor allowlists must already permit the operation; all deployment defaults
remain off. No new cloud roles or public publication endpoint is introduced.
Creating the prospective project is metadata-only and can be admitted by the
paused-training stage without enabling reference motion. Saved scene revisions,
train-only parent registration, ownership and all later per-operation gates
remain mandatory.

All paths below are under the fixed private worker registry container and
`tenants/<tenant>/owners/<owner>/learning/`. The native configuration's account/
container must match that registry. Browser requests contain no Blob URLs.

| Path | Original or post-hoc contents |
| --- | --- |
| `native-operator/<actual-job-UUID>/approval.json` | Original `physicalai.native-training-approval/v1`, including scope, actual job name, original UTC deadline, one-GPU/cost limits, native plan/archive/config/image qualification hashes and execution/source selectors |
| `native-operator/<actual-job-UUID>/claim.json` | Original scope/job/type, approval/plan/config hashes, deadline and `claimed_at_utc`; no invented API claim |
| `native-operator/<actual-job-UUID>/plan.tar.gz` | Original approved `plan/plan.json`, `plan/job.json`, and complete `plan/code/` snapshot |
| `native-operator/<actual-job-UUID>/image-qualification.json` | Original qualification bytes identified by the approval, not a newly asserted ready flag |
| `projects/<project>/external-native-training/<import>/files/run-config.json` | Exact original native config bytes |
| `projects/<project>/external-native-training/<import>/files/specification.json` | Retained `physicalai.native-bootstrap-authorization/v1`; its canonical hash must equal the original config's specification hash |
| `projects/<project>/external-native-training/<import>/completion.json` | New, create-only post-hoc request described below |

The post-hoc request has this closed shape (descriptions are not valid values):

```json
{
  "schema": "physicalai.external-native-training-request/v1",
  "import_id": "new-import-UUID",
  "project_sha256": "canonical-hash-of-protected-project-GET-item",
  "created_at": "post-training-aware-UTC-time",
  "native_job_name": "unchanged-actual-native-job-UUID",
  "approval_sha256": "exact-original-approval-file-hash",
  "claim_sha256": "exact-original-claim-file-hash",
  "configuration_file_sha256": "exact-original-config-file-hash",
  "specification_file_sha256": "exact-retained-specification-file-hash",
  "result_sha256": "exact-native-model-result-file-hash",
  "transfer_sha256": "exact-native-transfer-completion-file-hash",
  "model_sha256": "exact-native-candidate-model-json-hash"
}
```

File hashes cover original bytes including whitespace. Canonical specification/
configuration hashes retain their native semantics. Uploading a retained
specification after training does not make its upload time an old approval.
For `project_sha256`, use the API fingerprint of the protected GET response's
`item`: UTF-8 JSON with `sort_keys=True`, `separators=(',', ':')`,
`ensure_ascii=False`, and `allow_nan=False`. This is not the native configuration
canonicalizer, which escapes non-ASCII characters.
The verifier reads server `Last-Modified` and ETag from the same Blob download:
qualification/archive may share the approval's one-second timestamp, but the
end of the approval's timestamp interval must precede the original claim, and
the end of the claim's interval must precede actual Azure ML creation.
The exact root must independently report `Completed` with the approved command,
MI, compute, image, source/config/plan tags and original deadline.

The archive is bounded to 32 MiB compressed, 16 MiB unpacked and 1,024 regular
file/directory entries, with no links or escaping paths. Its source is **never
executed**. Native `read_plan` checks the archived code inventory/static hash,
config, job payload and original plan hash. The archived snapshot—not current
worker source—is used to verify the historical root and model provenance.
The running verifier is pinned separately in the import certificate.

The existing one-process artifact worker checks total network/file/disk budgets
before downloading the original configured inputs and published outputs.
It invokes the actual native validators for live TRAIN20 raw frames/images,
the prepared/trained parent, licensed backbone, conversion and v3 candidate
weights/processors. It checks exact task/profile/criteria, original episode
IDs/cases and ancestor lineage, optimizer/config/source provenance and any
explicit checkpoint resume. Candidate paths retain the exact native six-digit
form, for example `candidates/step-001000`; other spellings are rejected.
Native `model/result.json` and `transfer/completion.json` must match the request,
the actual root, the original result ETag/readback marker, and the original
publication deadline. Missing evidence is not reconstructed.

`POST /api/learning/projects/<project>/external-imports/<import>` takes only
`{"request_id":"<same-import-UUID>"}` plus the loaded `If-Match`. HTTP 202 returns
the existing artifact operation with `operation=external_training`; poll its
existing status endpoint. No candidate/dataset record is created from this ACK.
Only full verification yields `phase=import_committed` and
`result.external_training` containing the immutable external import record,
verified `DatasetVersion`, and `PolicyCandidate`. Actual raw files are copied
unchanged to the private artifact registry; no per-episode capture receipt is
fabricated. A matching existing dataset registration is reused.

External candidates declare `training_origin=external_native_import`,
`training_run_id=null`, an `external_import_id`, and no fabricated release or
pretrained-parent ID. Their actual parent model hash remains recorded. Legacy
API candidates omit the default `training_origin=api_training_run` and retain
their original required training-run ID. Private project records expose
`kind=external_import`, also readable at
`GET /api/learning/external-imports/<import>`, with explicit `imported_at`.
The source/published ETags, source verifier and original model metadata are
rechecked before accepting the receipt or using the candidate in a managed
pairing. Public learning explicitly rejects external-origin candidates;
import completion is not model quality, a release, or public improvement.

The original command deadline is never renewed. The separate import CPU wall
budget still includes queue time and is at most 1,800 seconds; aggregate
transfer stays at most 20 GiB/100,000 files and actual free disk may be smaller.
Large real imports can therefore fail resource-budget checks and require
explicit operator sizing. No successful production import is implied by CPU
fixtures or source tests.

### Operator-bound managed paired evaluation imports

Managed Batch results use a separate artifact import, not an Azure ML job
receipt. There is no Batch submitter, scheduler, cancellation proxy or automatic
publication in this path. Both the API's existing paused-evaluation stage and
the worker's `PAUSED_EVALUATION_ENABLED` must be admitted; resident artifact
processing and its exact actor allowlist must also be enabled. Defaults remain
off. Adding `azure-batch` or a model allowlist grants none of these permissions.

Before **any** scored physical claim, the trusted registration operator must
already have an owned project, `EvaluationRun`, both real trained candidate
records (or an actual reviewed baseline release), and the complete original
`JobSpecification`. The managed run declares `provider=managed_batch`,
`status=awaiting_import`, and no `azure_job_id`, `azure_status` or backend status.
Its `before_candidate_id`/specification `baseline_candidate` is mutually
exclusive with a real `baseline_release_id`/`baseline`; a candidate pair never
creates provisional release authority. Native model validation still requires
P1's training-parent hash to match P0 and excludes held-out training data.
This does not change the training API's parent-selection contract.

Register the original specification at the existing
`jobs/<backend_job_name>/specification.json` path. That name is only the
existing local correlation key, **not** a fabricated cloud job ID. Under the
normal `tenants/<tenant>/owners/<owner>/learning/` registry prefix, create:

| Fixed path | Immutable contents |
| --- | --- |
| `projects/<project>/managed-evaluations/<run>/binding.json` | `physicalai.managed-evaluation-binding/v1`, `provider: managed_batch`, `created_at`, explicit `study_max_wall_seconds` and `study_approved_cost_usd`, exact `mapping_sha256`, and the original full `specification` |
| `projects/<project>/managed-evaluations/<run>/files/mapping.json` | Native `physicalai.managed-paired-plan/v1`, including all forty predeclared logical/physical assignments |
| `projects/<project>/managed-evaluations/<run>/completion.json` | Later create-only `physicalai.managed-evaluation-import/v1`: `created_at`, `operation_id`, `binding_sha256`, `evidence_sha256`, `report_sha256` |
| `projects/<project>/managed-evaluations/<run>/files/` | Exact later `evidence.json`, `report.json`, and `attempts/<physical UUID>/...` export described in [managed paired evaluation](../../docs/managed-paired-evaluation.md) |

The browser cannot write these records or choose Blob URLs. Binding and
completion digests cover exact UTF-8 file bytes, including whitespace.
Serialize typed records with `model_dump(mode="json", by_alias=True)` before
writing and hashing them; preserve those original bytes.
Duplicate keys/non-finite JSON are rejected. The same Blob download's
server-observed `last_modified` must put registration before **every** original
claim. The verifier conservatively uses the end of its one-second HTTP timestamp
interval; a same-second claim is not proven later. Backdating `created_at`
cannot authorize an old physical attempt.
The later completion does not renew grants or physical/ML time budgets.

Managed study time is separate from the offline scorer/import timeout. The
operator must explicitly declare `study_max_wall_seconds` (1..28800, at most
eight hours **including queue, preparation and gaps**) before the study.
There is no eight-hour default. The original `EvaluationRun.deadline` must fit
its original `created_at` plus this window. Explicit `study_approved_cost_usd`
must equal the original run approval and be no more than USD 20 or the project's
lower cost ceiling; fresh price review remains the operator's responsibility.
These are source admission ceilings, not authorization to start or spend.
All forty physical claims and final times must fit that original deadline,
with no renewal or restamping. Per-episode 600 seconds, the legacy project's
21600-second evaluation ceiling, Azure ML/HTTP timeouts and the separate
1800-second artifact CPU budget are unchanged.

`GET /v1/learning/projects/<project>/managed-evaluations/<run>` exposes only
the protected typed import reference to the API identity. The API's
`POST /api/learning/jobs/<run>/managed-import` takes only `request_id`, which
must equal that completion's `operation_id`, plus the original `If-Match`.
HTTP 202 is an existing `ArtifactOperation`, not a completed evaluation.

The existing single-process artifact runner inventories the fixed input and
registered model prefixes, checks aggregate transfer and available temporary
disk, downloads original files, and invokes
`simulation.paired_evaluation.aggregate`. It verifies full raw captures,
per-tick task/heartbeat evidence, all file hashes and native model lineage,
then compares the entire regenerated report with the supplied report.
Thirty-nine trials, preemption, missing terminal files or a rehashed green
summary cannot become a scorable report. Fully verified failed physical trials
remain in each twenty-case denominator.

An explicitly attested command-v3 pairing additionally includes the original
`files/model-runtime.json` (at most 65536 bytes). Its exact file SHA must match
the frozen mapping's `model_runtime_sha256`; the worker forwards these bytes
unchanged to the native aggregate. Only its closed new runtime/admission-v2
contract can select the new command model verifier/server/provider. An absent
descriptor keeps the legacy v2 path; the controller's unchanged profile hash
is not implicit v3 model admission. Do not reuse the old runtime descriptor.
The compact report relays the native-verified `model_admission` object with
its exact entrypoints, IPC versions, image/source and runtime checksums; it
does not synthesize this evidence or infer physical quality from its presence.

Only a complete forty-trial result becomes `report_committed` and a typed
`ManagedEvaluationReceipt`. The compact API report explicitly keeps the managed
schema, mapping/evidence hashes and both logical/physical IDs. It does not
invent native `results.json`. The original managed report bytes are published
as a private report artifact; certificate reads recheck the original binding,
completion, verifier source and input inventories before API acceptance,
download or release. Changed/uncertain verification needs operator review,
not automatic heavy replay.

The unchanged artifact ceilings (20 GiB aggregate transfer, 100,000 files,
1,800 seconds including queue time) and actual smaller free disk remain
authoritative; they do **not** promise sufficient capacity for forty captures
and two checkpoints. Resource-budget failures remain explicit. A completed
import is not a passing quality result, release, current LIVE motion, or
evidence that the production import has been exercised.

### Reference authority catalog

The protected lookup
`GET /v1/learning/projects/{project_id}/reference-authorizations/{case_id}`
is disabled by default. It accepts only a UUID project and validated case
identifier in the authenticated actor's owner partition. The fixed registry
path is `projects/<project_id>/reference-authorizations/<case_id>.json`
under that owner's normal prefix; no URL or caller blob path is accepted.
There is no write endpoint for authority.

The operator-provisioned envelope is `physicalai.reference-authorization/v1`
with `operator_grant` matching the runtime's `PausedOperatorGrant`,
`grant_document_json` retaining the exact installed UTF-8 file,
`runtime_catalog_record_sha256`, project/case/profile/criteria/frozen-plan
pins and optional paired `reference_g0_proof_artifact_id` /
`reference_g0_proof_sha256`. Runtime record hashes cover exact file bytes;
criteria/scene-plan hashes cover canonical parsed JSON. The service rejects
duplicate keys and mismatches rather than reserializing away the difference.

The simulator must already have independently loaded that same authority and
must validate the actual runtime source, image, tenant, case and original
expiry at dispatch. Publishing a registry record alone cannot make that true.
The optional proof pointers do not substitute for actual full reference-task
G0 acceptance or authorize learning. Reference, bounded training, candidate
evaluation and quality-reviewed release are distinct default-off stages;
there is no requirement to have a quality-passed P0 before bootstrap training
can produce one. No flag enables manual paused controls or relabels scripted
motion as human input.

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

For an explicitly reviewed `full_state` restart, the public request's
`optimizer_steps` counts **new remaining updates**. It must equal the native
configuration's unchanged global `max_steps` minus the exact approved source
checkpoint step. Missing, invalid or exhausted steps are rejected before
submission. New and weights-only jobs retain their original step-count semantics;
none of these modes silently retries a paid job.

The API's conditional Cosmos claim and the worker's create-only Blob claim
precede paid submission. An ambiguous request is reconciled by the existing
job name/tags. It is never retried with a new job name or an upsert of old work.
Status/cancel remain read/reconcile operations for existing claims rather than
requiring new model admission. Azure `Completed` alone is not a candidate.

### Explicit single-command training provenance

The separate paused, image-embedded, checkpoint-enabled command variant must
declare the closed `job_execution` object in its original
`physicalai.smolvla-azure/v2` configuration:

```json
{
  "schema": "physicalai.smolvla-command-execution/v1",
  "kind": "command",
  "data_transport": "private_blob_mi"
}
```

The actual Azure ML command name must equal the original `config.run_id`.
Registered asset names, versions, URIs and manifest hashes remain reviewed
configuration inputs, but are not submitted as Azure ML job inputs, code
assets or custom mounted outputs. The native task transfers approved private
artifacts with managed identity and publishes under the unchanged
`output_prefix/<job-name>/model/` path. This is not a Batch training job.

The worker accepts this variant only with a natively validated
`physicalai.smolvla-checkpoint/v3` candidate declaring
`training_execution: azureml_command`, plus the closed
`physicalai.smolvla-command-training-result/v1` result. Both retain the one
actual root `azure_job_id` and exact `azure_job_type: command`. Pipeline and
component IDs must be absent, not copied from the root ID. Prepared parents
and old pipeline candidates keep their existing v2 contracts and validators;
the consumer never rewrites a v3 manifest into a v2 shape.
It selects the new, v3-only `learning.paused.command_artifacts.validate_model`
only after the original closed command configuration and actual root-job GET
have been verified. Merely detecting a v3 schema is not admission. The legacy
paused validator still rejects v3 and its controller-hashed bytes remain
unchanged; the new model-server/provider layer needs its own complete runtime
and image attestation, not a reused legacy descriptor.

Candidate acceptance independently checks the original immutable job/config,
model/data/backbone lineage, task/profile/criteria, new versus cumulative
optimizer updates, checkpoint source and original UTC deadline. It computes
the expected snapshot from the pinned static source and canonical config,
then uses the native read-only root-job verifier with `expected_status=Completed`.
The actual root must have the approved workspace/name, command type, managed
identity, compute, immutable environment image, exact embedded command and
source/config/deadline tags, with no parent, code, job inputs or custom outputs.
Historical completion verification does not renew the expired execution lease.

Reverification and managed paired-model import use the same original command
result and root metadata. An API candidate's existing `azure_job_id` remains
the real Azure ML ID; no synthetic pipeline identity, new status, public
checkpoint selector or inference/release permission is introduced. Command
checkpoint/resume provenance is checked in the operator-reviewed native config,
not accepted as browser-supplied JSON. Existing stage/model admission defaults,
physical evaluation criteria and release gates are unchanged.
The private registered artifact index retains explicit `training_execution`,
`azure_job_type` and the actual root ID for finding that original command
registration. These fields alone grant no authority: missing, mixed or changed
index/config/root metadata cannot select the new validator.

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
Lifecycle writes use `container.get_blob_client(key).upload_blob(...)` and retain
that write operation's returned ETag. `ContainerClient.upload_blob` returns a
`BlobClient`, not the write receipt; reading properties afterward would introduce
a version race. Offline tests exercise the installed Blob SDK over a local HTTP
transport with real conditional headers and 201/206/409/412 responses, including
reading and conditionally updating an existing heartbeat without changing its
original target/deadline. They do not count as Azure deployment verification.
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

### Explicit paused artifacts (not runtime admission)

The separate paused adapter selects `learning.paused.capture.validate_dataset`
only for an explicitly mode-bound project. It requires raw
`physicalai.demonstrations/v3`, `require_live=True`,
`require_demonstrations=True`, the original command/episode, the approved case
revision/seed/split, exact profile hash and matching frozen criteria/scene-plan
hashes. Integration/test-purpose G0 and evaluation captures cannot become
training data. Scripted source stays `reference_controller`; it never becomes
a human demonstration. V3 sealing uses the native paused assembler and retains
both original wall and simulation clocks. Old real-time data still uses the
original validator and is never relabeled.

Paused checkpoints use the separate committed
`learning.paused.artifacts.validate_model` for
`physicalai.smolvla-checkpoint/v2`. The default model path still rejects them.
Their validated mode/profile/criteria/scene-plan metadata remains on capture,
dataset, job, train-only parent and candidate records. Mixed or missing pins
fail rather than inheriting real-time qualification. Candidate output paths
come from the original immutable per-job config, not a mutable later project
approval.

Paused profile versions are explicit: v1 retains 1800 physics ticks / 300 frames
/ 30 SIM seconds, while declared v2 permits 3600 ticks / 600 frames / 60 SIM
seconds. The six-tick interval, 600-second wall ceiling, per-operation wall
budgets and physical quality requirements are unchanged. Raw v3, model v2 and
IPC v2 still bind their complete validated profile and SHA; an outer artifact
schema alone does not select the newer budget. Old manifests, cases, grants,
reports and failed v1 attempts are not rewritten. Operator-reviewed config,
dataset, parent/candidate and report must all match the same profile.

Registering an explicitly reviewed existing paused **train-only** artifact
requires `--execution-timing paused_simulation` in addition to the existing
bootstrap/operator/model approval checks below. It does not download a model,
grant GPU rights, create a released policy or enable a public operation.
Current API/worker paused execution remains supported-but-unadmitted while
end-to-end capture, optimizer, runtime and separate physical-report integration
are completed. Type/fixture/codec checks do not satisfy those live gates.

### Independently verified paused reports

Native paired/bootstrap report v2 is not admitted from a `live_gpu_verified`
or `quality_gate_passed` flag alone. The worker rereads the original closed
job config, checks the current identity-only datastore and exact registered
input versions using read-only Azure ML methods, and verifies the actual
parent/component job IDs, owner/specification, mode and deadline tags.
It never runs paid preflight or renews a historical deadline to read evidence.

Only canonical, owner-scoped `azureml://.../datastores/<configured>/paths/...`
locations are resolved to the approved Blob account/container. The plan hash
is over parsed canonical JSON; the results hash is over exact `results.json`
bytes. Actual model files must match the original registered model manifests
and approved native inputs. Native `evaluate_pair` / `evaluate_bootstrap`
independently verify the complete recording, every camera/control/heartbeat
artifact, and the per-physics-tick TaskWatchdog predicate before the entire
recomputed report is compared with the AML output (apart from its separately
verified job envelope).

One content-bound verification claim prevents repeated heavy rescoring from
browser polling. A completed certificate is keyed by owner, job, specification,
full config, exact report SHA, the running native/adapter verifier source hashes,
and hashes of every relevant Blob name/size/ETag inventory. Inputs are checked
before and after verification and again on cache reuse. A changed input or
verifier invalidates the cache. Failure records are explicitly unverified;
a lost process after claim requires explicit operator recovery rather than a
second hidden rescore or a fabricated successful cache entry.

Cosmos receives a compact projection capped at 512 KiB: all forty trial
identities, models, roles, failures and task-predicate summaries, separate WALL
and SIM durations, and per-phase sample count / nearest-rank p50 / p95 / maximum.
Full sample arrays and violation details remain intact in the verified native
Blob report. The authenticated job report download retrieves only that verified
artifact and checks its exact SHA; it accepts no caller URL or arbitrary path.
Summaries are not substitutes for the source proof or real-time qualification.

The separate capture/dataset resident operation path below does not provision a
timer or enable paid admission. Actual large-batch acceptance remains a separate
deployment test with real data and explicitly sized CPU, memory and disk.

### Resident asynchronous capture and dataset operations

`ARTIFACT_OPS_ENABLED=false` and `ARTIFACT_ACTOR_IDS=[]` are independent,
default-deny worker settings. The matching Bicep parameters are
`artifactOpsEnabled` and `artifactActorIds`; enabling this feature makes the
existing worker's minimum replica count one. It creates no new service, role,
GPU/AML job or public endpoint. An operator must approve this resident CPU cost,
verify deployment/readiness, and quiesce active artifact operations before a
worker revision roll. One fixed artifact subprocess runs per resident worker.

The private API exposes an owner-authorized budget read, immutable operation
POST, and status GET under `/v1/learning/artifact-policy` and
`/v1/learning/artifact-operations/{id}`. The POST only writes bounded request,
state and dedicated pending-queue metadata. The resident runner, not the HTTP
handler or browser, performs the transfer and native validation. Legacy inline
capture/dataset worker routes explicitly require this operation path instead
of silently falling back to long synchronous HTTP processing.

Before claiming work, the API freezes the owner, project, original capture or
capture tuple, request fingerprint, target ID, deadline and budgets. Default
maximums are 4 GiB capture / 20 GiB dataset aggregate transfer, 100000 files,
and 1800 wall seconds **including queue wait**; deployment may only lower them.
These are ceilings, not promises that the current replica has that capacity.
Input inventory is measured before transfer, total transfer includes required
publication, and free temporary disk must cover both input and sealed copies
plus reserve. Low disk fails explicitly before bulk processing; ongoing
transfer checks the remaining reserve and original byte/time bounds.

State is `queued -> running -> ready|failed|timed_out|uncertain`. Starting heavy
work requires a conditional ETag claim, and status reads perform only bounded
metadata/manifest checks. A complete result is written only after native
hash/live/source validators and manifest-last publication succeed. A crash
after claim is **uncertain**, not an automatic rerun; an already committed,
matching verified result can be reconciled from metadata. Partial files are
never reconstructed into a ready manifest. Queue pointer cleanup does not
delete request, state, source or artifact evidence.

The fixed subprocess receives only typed owner/operation/claim IDs and uses
the existing managed identity. No caller module, script, model URL or raw Python
is executed. A deadline watchdog and guarded process-group termination stop
only this child before reporting timeout/uncertainty. The separate short
learning-job cancellation tick is not placed behind this long operation.
An actual large-data transfer and restart test on the approved Azure replica
is still required before claiming the real UI batch workflow complete.

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
