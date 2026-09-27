# Managed simulator jobs (Azure Batch)

This source adapter executes **one explicitly authorized reference episode**, not an
always-running bridge. It uses the existing `simulation.paused_probe` and
`simulation.paused_acceptance` without changing physics, capture, task criteria or
policy code. A successful CPU test, image pull, Batch task exit or upload is **not**
an Isaac pick/place or model-quality result. The previous physical failures and
externally interrupted attempt remain unqualified.

Batch manages underlying GPU nodes; this is not GPU-free or serverless execution.
No manual VM start/stop, SSH, runCommand or public simulator endpoint is required.
Organizational shutdown, allocation and cost policies still apply. A100/H100
training compute is not a substitute for an RTX-capable Isaac renderer.

## Actual managed execution boundary

The private foundation, caller/node permissions and immutable worker/simulator
images were deployed and checked on 2026-09-27 KST. The actual Batch service
exposed two differences from synthetic SDK payloads: ARM-created no-public-IP
enum casing, and canonical duration strings. It also rejected the optional
`maxParallelTasks` account feature. These cases now have regressions; the adapter
preserves the same one-node/one-slot limits and original job deadlines without
enabling account features.

The East US 2 warm-up job and its single task were accepted. Batch then returned
`AllocationFailed`: insufficient regional capacity for the requested LowPriority
GPU size. No GPU node or `SimulationApp` ran. The original fixed allocation
window ended with current and target nodes both zero. A private West US 2
alternative was prepared using the existing connected network and initially
had zero LowPriority-core quota.

The next investigation found an important missing admission check: the actual
regional Batch catalog reports `LowPriorityCapable=False` for this full-A10 SKU
in East US 2, but `True` in West US 2. A large account quota cannot override that
regional restriction. An official account-only request increased West US 2
Spot quota to 36 without granting subscription-wide orchestration permissions.
`capacity-check`, `warmup` and `submit` now read both the exact regional SKU and
the account quota from ARM before submitting work. Passing this check permits
an allocation request; it does not guarantee available capacity or RTX readiness.

## Explicit platform and permission prerequisites

The pool-only `infra/simulation-batch.bicep` references an existing private Batch
account, private node subnet and dedicated node UAMI. It creates no account,
private endpoint, DNS zone, identity or role assignment. Its default
`provisionPool=false` creates **no pool or node**.

The reviewed host candidate is
`microsoft-dsvm:ubuntu-hpc:2404:24.04.2026092501`, x64/Gen2, with
`batch.node.ubuntu 24.04`. The parent independently observed this SKU as
`verified` and `DockerCompatible` in the authenticated Batch catalog. Those
catalog labels (including `SupportsRDMAOnly`) do not establish an installed NVIDIA
runtime, GRID compatibility, Vulkan rendering or Isaac task success. The earlier
Ubuntu-HPC 2204 candidate was unverified and is not selected.

The pool pins one `Standard_NV36ads_A10_v5` LowPriority node at most, one task slot,
no dedicated nodes, no public IP or configured inbound login endpoint, and no
named user/password. Non-admin task-scoped auto-users run containers. The reviewed
NVIDIA extension is `Microsoft.HpcCompute/NvidiaGpuDriverLinux`, handler request
`1.14`, with both upgrade flags disabled and these exact settings:

```json
{"driverVersion":"570.237","installCUDA":false,"updateOS":false}
```

The extension catalog lists a full package version such as `1.14.0.6`, but the
actual Compute provider rejected that four-component value for
`typeHandlerVersion`. The deployment uses the supported major/minor request
`1.14`; it does not claim an immutable patch-package pin. Actual driver
readback must still equal `570.237`, and Vulkan/RTX readiness remains mandatory.

The image/extension requires compatible kernel, Moby/NVIDIA container integration,
approved driver-download egress, and private Batch node-management/ACR/Blob
connectivity. Driver extension success alone is insufficient. Existing applicable
NVIDIA GRID/Isaac licensing authorization must already cover this deployment;
the adapter does not obtain or approve new marketplace or NVIDIA terms.

The **private caller** needs both **Azure Batch Job Submitter**
(`48e5e92e-a480-4e71-aa9c-2778f4c13781`) and **Azure Batch Account Reader**
(`11076f67-66f6-4be0-8f6b-f0609fd05cc9`), scoped to this Batch account. The parent
observed a real private-endpoint `403` for supported-image reads with Submitter
alone; adding Reader enabled the read. Neither role is a reason to grant
Data Contributor or resize permission. The caller also needs read access to the
private output artifacts for integrity-checked status. Input publication is a
separately authorized operator operation, not an implicit CLI upload.
The private worker deployment sets `AZURE_CLIENT_ID` to its existing worker
identity so `DefaultAzureCredential` selects that caller, not the separate node
identity. No client secret or interactive login is configured.

The **node UAMI** needs AcrPull on the selected registry, Blob Data Reader for the
approved inputs/assets, and Blob Data Contributor for the private `demonstrations`
output container. The opt-in `infra/simulation-batch-node.bicep` creates a dedicated
identity with precisely those three resource/container-scoped grants. It does not
reuse the legacy simulator identity's Key Vault TLS-secret access, and it grants
no Batch submission or control-plane permissions. No shared keys or SAS appear
in the task. Provisioning, RBAC,
image/platform admission and actual GPU execution remain operator-owned.

## Images and lightweight dependencies

The CPU controller entry point is **`python -m simulation.batch`**. Add only
`azure-batch==15.1.0` to the existing private learning-worker dependencies and copy
`simulation/` into that image, alongside its existing `apps/api/`, `learning/`,
`contracts/` and package initializers. Existing pydantic, azure-identity,
azure-storage-blob, Pillow and jsonschema dependencies cover the CPU paths.
The root project exposes the SDK separately as the `batch` optional dependency.
No Isaac, CUDA, Torch or model weights are imported by controller commands.

The digest-pinned Isaac image must contain **both** `simulation/batch.py` and
`simulation/batch_task.py` plus the existing runtime. Include these two new files
in immutable image/source verification; a historical runtime image lacks them.
The node entry points are:

```text
/isaac-sim/python.sh -m simulation.batch_task preflight --output preflight.json
/isaac-sim/python.sh -m simulation.batch_task run --spec attempt.json --spec-sha256 <raw-file-sha256>
```

The native image does **not** need the optional Batch SDK. It uses its existing
Blob/MI dependencies. Batch overrides the normal bridge ENTRYPOINT; it does not
launch `bootstrap_tls`, RPC, SSH or a web server. Batch automatically enables GPU
access for GPU container tasks: do not add `--gpus`. The native container needs
the image's existing `python.sh`, `/usr/bin/timeout`, nvidia-smi and Vulkan loader.
Writable bounded tmpfs/cache mounts accommodate the assigned non-admin UID.

## Bounded warm-up, then a fresh approved episode

An operator first reviews exact account/subnet/UAMI, capacity/quota, image,
licensing, extension, egress and organizational policy. The pool has autoscale
zero-to-one demand, a five-minute evaluation/sampling window, and an explicit
UTC allocation cutoff no later than 60 minutes after its approved start.
`allocationStartUtc` is required, with no `utcNow()` default: redeploying the
same reviewed parameters cannot silently renew the paid allocation window.
No demand eventually returns the target to zero. The cutoff forces target zero
even if demand persists; service evaluation/deallocation is not instantaneous
(evaluation can lag by five minutes). Redeploying a new cutoff is a new approval,
not an implicit renewal. Target-zero deallocation terminates node work; interrupted
work without complete proof remains incomplete.

The supplied [platform example](batch-platform.example.json) uses nonfunctional
fixture resource identifiers. Replace it with the reviewed private configuration.
These commands are operator actions, **not** part of CPU verification:

```bash
python -m simulation.batch capacity-check --platform /approved/platform.json
python -m simulation.batch warmup-plan --platform /approved/platform.json \
  --warmup-id <explicit-new-warmup-uuid>
python -m simulation.batch warmup --platform /approved/platform.json \
  --warmup-id <same-warmup-uuid> --confirm-submission
python -m simulation.batch preflight --platform /approved/platform.json
```

The separately submitted warm-up job creates pending demand without a motion
grant, case, dataset or `SimulationApp`. Its job is bounded to 900 seconds and its
task to 60 seconds. The pool StartTask additionally uses GNU timeout, 60 seconds
plus a five-second kill grace, because the Batch StartTask schema has no task
wall-clock constraint. It requires actual nvidia-smi evidence of exactly one
`NVIDIA A10-24Q` / `570.237`, and an actual NVIDIA A10 Vulkan device advertising
acceleration-structure and ray-tracing-pipeline extensions.

For private deployments, `infra/simulation-batch-warmup.bicep` provides a
default-off, manually started **Container Apps Job** to submit that same
deterministic warm-up through the existing worker identity. Its CPU controller
has one replica, zero retries and a 180-second timeout. It cannot dispatch a
physics episode. This avoids depending on an interactive exec connection to a
scale-to-zero web replica; an ambiguous submission still reconciles the original
warm-up UUID and never renews its constraints.

Read-only preflight checks the actual pool/image/identity/autoscale/StartTask,
idle single LowPriority node, successful exact driver extension and fresh
`startup/wd/preflight.json`. It never starts another job. A node that is absent,
busy, expiring, misconfigured or lacks graphics fails closed.
The actual service returns `NoPublicIPAddresses` for an ARM-created pool, while
SDK-created payloads use `nopublicipaddresses`. Both explicit no-public-IP values
are accepted; missing, Batch-managed and user-managed public addressing are not.

Only after readiness should the trusted operator mint a **new original grant**
with at most 600 seconds lifetime, binding the exact owner, tenant, saved case,
task, criteria, profile, source and image. Warm-up does not initialize Isaac or
prepare its scene; remaining cold setup and queue time still consume that
original grant. The task cannot mint, renew or extend authority.

Publish the approved spec and hashed inputs privately, then:

```bash
python -m simulation.batch plan --spec /approved/attempt.json \
  --spec-url https://<storage>.blob.core.windows.net/<inputs>/<approved-spec-name>
python -m simulation.batch submit --spec /approved/attempt.json \
  --spec-url https://<storage>.blob.core.windows.net/<inputs>/<approved-spec-name> \
  --confirm-submission
python -m simulation.batch status --spec /approved/attempt.json
```

Both submission kinds use deterministic IDs. Physical jobs are
`sim-<attempt UUID without hyphens>` with task `episode`; warm-up jobs are
`warm-<UUID without hyphens>` with task `preflight`. Job/task retry counts are zero.
Jobs are created with `onAllTasksComplete=noaction`, the single task is added,
then an ETag-conditional patch sets **only** `onAllTasksComplete=terminatejob`.
Otherwise Azure treats an empty job as already complete. Reconciliation refuses
changed bindings and never resets job constraints or deadlines. Status is GET-only.
Duration comparison uses the SDK's typed values: the service canonicalizes
`PT15M00S` to `PT15M` and `PT01M00S` to `PT1M`. That formatting difference does
not authorize a new deadline; unequal or missing durations are rejected.
The account rejected the optional job-level `maxParallelTasks` feature with
`Forbidden`, so jobs omit it and the related optional task-preemption setting.
Concurrency remains bounded by the verified pool's maximum one node, one task
slot, and exactly one deterministic task per submitted job. This does not disable
LowPriority node preemption or imply an account feature was enabled.

## Immutable task and completion contracts

`BatchPlatform` is the closed DTO shown in the platform example.
`BatchSimulationSpec` is the closed DTO in the
[episode example](batch-simulation.example.json). Each `BlobInput` has a safe
relative `name`, exact raw-file `sha256` and `size_bytes`. JSON inputs are bounded
to four MiB; the approved asset archive to 256 MiB. Criteria/conditions additionally
carry their distinct **parsed canonical JSON** hashes. Formatting/raw file hashes
must not be substituted for those canonical hashes.

The spec also pins the original `source_revision`, full container digest,
`control_profile_sha256`, v1/v2 `profile_id`, owner/tenant and asset root. Its
canonical DTO hash binds the job metadata; the exact spec-file byte hash binds
the MI ResourceFile and node command. No arbitrary executable, anonymous joint
targets, model substitution or client-supplied RPC command is accepted. This
adapter is reference-only; it does not pretend a missing learned adapter ran.

Before any `SimulationApp`, the task creates a create-only private
`<owner>/managed-simulation/<attempt UUID>/claim.json`. **Batch may internally
requeue preempted/recovered tasks even when maxTaskRetryCount is zero.** A claimed
attempt cannot execute physics again. Autoscale/service recovery may still
reallocate a node; that does not authorize replay. An explicit retry requires a
new attempt UUID, `previous_attempt_id` referring to the failed/incomplete attempt,
and a new approved grant. Physics always starts a newly approved scene, never a
fabricated partial-pose resume. Neural-network checkpoints are a separate learner
responsibility.

The task has a 900-second outer bound, reserving 30 seconds for proof publication.
Its owned Linux process group includes python.sh and all native descendants and
is terminated on timeout/cancellation. Existing 600-second grant, first-publication
age, two-second stage, 0.2 m/s speed, grasp, force and v1/v2 simulation limits remain
unchanged. Expiry is a failure, not a request for a longer grant.

The existing capture writer validates raw data and uploads its manifest last.
The wrapper runs existing native physical/capture acceptance, checks the uploaded
raw manifest against local bytes, then publishes the exact inputs, GPU preflight,
private native report/trace, logs, acceptance and raw-manifest copy. Only afterward
does it create `completion.json`. Failed upload or abrupt process/node loss leaves
no success-shaped terminal record. An incomplete terminal record may explain a
known error but cannot qualify data.

Status checks bounded artifact names/sizes/hashes, original spec/source/image/
profile, native physical semantics, receipt/episode identity and the current
private raw manifest. Batch Completed/exit 0, an acceptance flag alone, a copied
foreign receipt or object already near the goal is not sufficient.
`learning_quality_proven` remains false even for an accepted reference episode.

## Reproducible offline checks

The example hashes/identities do not authorize real execution. These two commands
construct SDK payloads without credentials, network, nodes or jobs:

```bash
python -m simulation.batch warmup-plan --platform docs/batch-platform.example.json \
  --warmup-id 11111111-1111-4111-8111-111111111111
python -m simulation.batch plan --spec docs/batch-simulation.example.json \
  --spec-url https://unitstorage.blob.core.windows.net/artifacts/approved/attempt.json
```

With the optional SDK installed in an isolated environment:

```bash
python -m pytest -q tests/test_simulation_batch.py tests/test_simulation_batch_sdk.py \
  tests/test_batch_simulation_task.py tests/test_batch_simulation_proof.py \
  tests/test_simulation_batch_infra.py tests/test_paused_acceptance.py
python -m ruff check simulation/batch.py simulation/batch_task.py tests/test*batch*.py
bicep build infra/simulation-batch.bicep --stdout
```

The SDK tests serialize real `azure-batch==15.1.0` HTTP requests through an
in-memory transport. Infrastructure tests compile actual ARM and compare the
autoscale/StartTask configuration. Optional-SDK tests skip when that extra is
absent; the dedicated validation command must run with it installed.
None of these checks is live GPU, dataset or trained-policy proof.

Official references: [Batch GPU pools](https://learn.microsoft.com/azure/batch/batch-pool-compute-intensive-sizes),
[containers](https://learn.microsoft.com/azure/batch/batch-docker-container-workloads),
[pool extensions](https://learn.microsoft.com/azure/batch/create-pool-extensions),
[NVIDIA Linux extension](https://learn.microsoft.com/azure/virtual-machines/extensions/hpccompute-gpu-linux),
[private nodes](https://learn.microsoft.com/azure/batch/simplified-node-communication-pool-no-public-ip),
[autoscale](https://learn.microsoft.com/azure/batch/batch-automatic-scaling),
[Spot recovery](https://learn.microsoft.com/azure/batch/batch-spot-vms),
[RBAC](https://learn.microsoft.com/azure/batch/batch-role-based-access-control),
and [pool ARM schema](https://learn.microsoft.com/azure/templates/microsoft.batch/batchaccounts/pools).
