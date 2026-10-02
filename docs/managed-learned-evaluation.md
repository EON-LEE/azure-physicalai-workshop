# One managed learned-policy evaluation

`simulation.batch_learned` adds a separate learned entry point to the existing
[private Batch workflow](managed-simulation.md). It does not replace the working
reference probe, alter robot physics, change a frozen profile/criterion, or open
an HTTP service. One task produces **one physical trial**, not the twenty-case
paired quality decision. `learning_quality_proven` remains false.

## Two real Python runtimes, one protected local socket

Isaac runs `/isaac-sim/python.sh` with its Python 3.12 environment. The actual
model runs **`/opt/smolvla-venv/bin/python`**, Python **3.11**, with native source
at **`/work/learning`**. Do not import LeRobot/Torch into Isaac Python or point
the model venv at Isaac's interpreter.

`simulation/Dockerfile.policy` assembles these separately from two immutable
base images. It copies the entire Python 3.11 installation into
`/opt/lerobot-python`, relocates the native venv's interpreter/base prefix,
and exposes the distinct libpython3.11 SONAME without replacing Isaac Python.
An actual ACR build verified Python 3.11.14, LeRobot 0.4.4, Torch 2.7.1+cu126
and the exact native source pins, followed by an independent Isaac Python 3.12
import. This is CPU/ABI qualification; it is not CUDA policy execution or
physical learned quality. New learned wrappers still need their own final
image-source verification and actual model-bound GPU trial.

Build each final code overlay directly from the qualified mixed **dependency**
image, not from the previous code overlay. Repeatedly stacking code overlays
eventually failed with Docker `max depth exceeded`. Reusing the qualified
dependency digest retains its runtime configuration and libraries while adding
only one current source layer group; it does not flatten or rewrite historical
images and their proofs.

The operator must assemble and independently qualify an immutable mixed image.
The model runtime needs its real interpreter, base prefix, standard library,
libpython and native ELF/CUDA dependencies, not merely a copied venv from
`python:3.11-slim-bookworm`. An isolated `/opt/lerobot-python` installation is
compatible with the fixed venv entry point; it must not replace Isaac's global
Python. Test both interpreters and actual Torch/LeRobot imports in the **final**
image. Source tests do not prove ABI, graphics, memory capacity or GPU inference.
The pool's current qualified Ubuntu-HPC 2204 host, driver bootstrap, GPU
job-preparation step, non-admin UID, Kit/Warp cache mounts and GPU-device
environment are inherited unchanged from `simulation.batch.build_job_task`.
Batch does not accept a `--runtime=nvidia` container override.

Provide a hash-pinned private `ModelRuntime` JSON descriptor:

```json
{
  "schema": "physicalai.paused-model-runtime/v1",
  "python_executable": "/opt/smolvla-venv/bin/python",
  "python_sha256": "<SHA256 of the resolved real Python 3.11 executable>",
  "python_version": "3.11",
  "code_root": "/work",
  "dependency_image": "<approved native dependency image>@sha256:<digest>",
  "code_image": "<approved native code image>@sha256:<digest>",
  "source_files": {
    "learning/paused/model.py": "<actual file SHA256>",
    "learning/paused/ipc.py": "<actual file SHA256>"
  }
}
```

The shown source entries are abbreviated: the actual descriptor must list
**every** `/work/learning/**/*.py` file, relative to `/work`, including package
initializers. The adapter checks the exact file set and every hash, the resolved
interpreter hash, and an actual subprocess's Python version/base/stdlib.
The dependency/code image digests identify the separately qualified assembly;
they are not required to equal the training GPU runtime hash or the Isaac image.

### Separately attested Azure ML command candidates

`ModelRuntime/v1` admits only the existing pipeline-provenance checkpoint v2.
It does **not** gain v3 support by inspecting a model header. For a genuine
standalone Azure ML command candidate, supply
`schema="physicalai.paused-model-runtime/v2"` with every v1 field above plus:

| Field | Closed value or binding |
| --- | --- |
| `admission_kind` | `azureml_command_v3` |
| `artifact_schema` | `physicalai.smolvla-checkpoint/v3` |
| `training_execution` | `azureml_command` |
| `server_entrypoint` | `learning.paused.command_model` |
| `provider_entrypoint` | `simulation.command_policy_deployment.CommandPausedPolicyProvider` |
| `request_schema` / `response_schema` | `physicalai.smolvla-request/v2` / `physicalai.smolvla-response/v2` |
| `legacy_servo_sha256` | Actual unchanged 39-file legacy control-bundle hash |
| `control_profile_sha256` | Original approved complete profile hash |
| `simulator_image` | Actual final immutable simulator image |
| `simulator_source_revision` | Actual simulator source revision, also in the original grant/spec |
| `simulator_source_files` | Complete `/app` Python source inventory described below |

Generate this **private sidecar after the final image is built and independently
read back**, not inside that image. This avoids an image/descriptor self-hash
cycle. The exact descriptor file SHA is still pinned by the original managed
spec, claim, preflight proof and paired mapping. Do not reuse descriptor
`553fb4...` or synthesize a descriptor from an intended build context after the
actual `/work` bytes or entry points change.

`source_files` must contain every `/work/learning/**/*.py`, including
`command_artifacts.py`, `command_model.py`, their imports and the byte-exact old
modules. `simulator_source_files` contains every `.py` under the literal `/app`
package roots **apps, contracts, learning, simulation**, including the new
provider and all admission wrappers. The only excluded directory names are
`__pycache__`, `.cache`, `.pytest_cache`, `.ruff_cache`, `.venv` and `venv`.
Do not put admission code in excluded paths. Symlinks are rejected, including
inside excluded-directory entries; actual extra/missing/source-changed files
fail verification. The complete `learning/` Python inventory must be identical
in `/app` and `/work`. Descriptor input remains bounded to 65,536 bytes and must
also fit the existing bounded preflight receipt after encoding.

The operator must prove the original 39 fingerprinted files, protocol table
and both v1/v2 numeric-limit dictionaries are byte-for-byte unchanged. The
recorded legacy hashes are servo
`d72db917989c3d55512241e1fd14b1a86be51317d5812bc3109a33b5ecf18351`
and profile
`851df47a362e4f62fcf0cbfa1b2761339ed5e346d1575123629c20355acb77dc`.
Those hashes identify the unchanged legacy control bundle, **not the newly
approved admission system**. Its additional authority is the new full runtime
descriptor plus immutable image/source/grant binding. No old hash list, model
schema constant, validator import or sealed TRAIN metadata is rewritten.

The command branch calls only
`learning.paused.command_artifacts.validate_model` and the new native
`learning.paused.command_model` server. It requires original v3 metadata,
`training_execution="azureml_command"` and one genuine command job identity;
pipeline/component aliases, v2 conversion views and fixture/untrained policies
are rejected. `CommandPausedPolicyProvider` performs new strict admission with
normal construction and inherits the unchanged legacy authorization, original
deadline checks, protected socket creation and guarded policy adapter. The
pinned `PausedLearnedRuntime`, `SocketChunkPolicy`, IPC schemas and actual
actuator/gravity/watchdogs are untouched. Legacy providers continue rejecting v3.

For this branch only, the model command below uses
`-m learning.paused.command_model` with the same arguments. The native probe
receives `--model-runtime <original-sidecar>` and verifies its hash and both
source contexts before selecting the new provider. The report and model-process
preflight contain the same explicit `model_admission`/`admission` proof; offline
rescore rejects missing, mixed-kind or rebound entry points/profile/source
inventories. No descriptor-supplied arbitrary Python module is imported.

The task starts the existing native server:

```text
/opt/smolvla-venv/bin/python -m learning.paused.model
  --model-root <private-task-disk>/model
  --backbone-root <private-task-disk>/backbone
  --model-sha256 <file-SHA256-of-model.json>
  --binding <private-task-disk>/binding.json
  --socket-path <0700-task-directory>/ipc/policy.sock
  --allowed-client-uid <actual-Batch-task-UID>
```

Model-server environment construction removes Isaac Python/loader overrides and
sets native `PYTHONPATH=/work`, offline model access and no user site packages.
The socket is fresh, protected and owned by the same task UID. Both directions
use existing `SO_PEERCRED` checks. The server and simulator share the same Linux
kernel/monotonic clock; a remote service or port mapping is not a substitute.
Socket readiness occurs only after `LocalPausedSmolVLAPolicy` has validated and
loaded actual CUDA weights. Startup is bounded to 120 seconds inside the original
grant; timeout/cancellation kills the owned model process group, with no fallback.

## Exact managed-attempt inputs

`BatchLearnedSpec` inherits the unchanged `BatchSimulationSpec` bindings and adds:

| Field | Required meaning |
| --- | --- |
| `schema` | `physicalai.batch-learned-evaluation/v1` |
| `role` | `candidate`, `before` or `after`; never `reference` |
| `evaluation_split` | `test` by default (omitted from the wire); explicit `validation` requires an unpaired `candidate` |
| `model` | `manifest: BlobInput` for `model.json`, plus every `checkpoint/` file |
| `backbone` | `manifest: BlobInput` for `backbone.json`, plus its entire native file inventory |
| `model_runtime` | `BlobInput` for the exact descriptor above |

Each bundle's `files` entry is `{path, sha256, size_bytes}`. Paths are relative to
the bundle directory. Blob names are constructed only beneath the manifest's
private prefix in the spec's existing storage account/input container. Model and
backbone manifests must be inside
`tenants/<tenant>/owners/<owner>/learning/`. There are at most 128 files per
bundle, three GiB per payload and eight GiB total. Weights stage on the mapped
Batch **task disk**, not the qualified two-GiB `/data` tmpfs.

Manifests are downloaded and hash-checked first, and their native inventories must
exactly match the spec. Every payload has an exact size/SHA check. Existing
`learning.paused.artifacts.validate_model` and
`learning.smolvla.artifacts.validate_backbone` validate the actual local files.
The selected model must be an actual changed, trained **candidate** with paused
provenance matching its explicitly selected admission version, nine-dimensional
action/state processors, the exact backbone, task, profile, criteria and
conditions. A vendor six-DOF base, `pretrained` preparation,
test fixture, pickle or dynamic processor is not an inference policy. In a
before/after comparison, both policies must satisfy the native inference contract.

The unchanged `PausedOperatorGrant` must authorize
`controller=learned`, `policy_type=smolvla`,
`authorization_kind=evaluation_grant`, `purpose=evaluation`, and the exact
**file SHA of `model.json`**, not merely a weights-file SHA.
Its original lifetime remains at most 600 seconds. By default the selected saved
scene must be a predeclared TEST case with seed 30001..30020, with exact frozen
initial/goal poses, revision and builder. Omitted or explicit
`evaluation_split="test"` preserves the existing spec serialization/hash.

Before freezing the final forty assignments, a trained candidate can be checked
on the already-frozen validation cohort by adding `"evaluation_split": "validation"`
to its one-trial spec. This requires `role="candidate"`, no
`pairing_plan_sha256`, and the original saved scene's
`execution.demonstration_split="validation"` with seed **20001..20010**. The same
case, split, seed, revision, builder and poses must appear in the hash-pinned
frozen conditions; nothing is added to or relabelled in that plan. Raw capture
metadata and the native report must agree with the selected cohort.

Validation uses the same model, grant, actuator, task predicates and timing
guards. Its report and acceptance explicitly say `evaluation_split="validation"`
and retain `learning_quality_proven=false`; acceptance means only that physical
trial passed. Validation must never enter final paired quality scoring. Do not
use held-out TEST outcomes to guide optimizer/debug decisions, or reuse TRAIN
seeds or the 900002/900004 integration cases by changing their labels.

All model downloads, model startup, Isaac initialization, original fixed
sixty-tick warm-up, motion and capture remain subject to the existing original
authority. There is no grant renewal, first-publication restamp, timing waiver or
partial-scene resume. The 900-second Batch outer limit and create-only attempt
claim remain unchanged. Internal Batch requeue cannot run the same attempt again.

## Operator entry points

Build/package `simulation/batch_learned.py`, `simulation/learned_probe.py` and the
updated `simulation/batch_task.py` into the qualified simulator and CPU controller
images. Include all three exact bytes in the new OCI source proof. The existing
paused servo fingerprint files are unchanged. CPU submission uses the already
pinned optional `azure-batch==15.1.0`; no new package is required.

```bash
python -m simulation.batch_learned plan --spec /approved/learned-attempt.json \
  --spec-url https://<storage>.blob.core.windows.net/<inputs>/<approved-spec>
python -m simulation.batch_learned submit --spec /approved/learned-attempt.json \
  --spec-url https://<storage>.blob.core.windows.net/<inputs>/<approved-spec> \
  --confirm-submission
python -m simulation.batch_learned status --spec /approved/learned-attempt.json
```

`plan` is offline. `submit` retains the current regional-capacity and real
managed-node readiness checks; it cannot create or resize pools. The task command
is `simulation.batch_learned run --spec attempt.json --spec-sha256 <raw-file-SHA>`.
Its original job/task IDs, non-admin environment, one slot, retries zero and
create-task-before-ETag-termination ordering are preserved.

The native probe installs `InstalledPausedPolicyProvider` as both the core
authorizer and policy provider, then selects the existing `PausedLearnedRuntime`.
That routes real `SocketChunkPolicy` predictions through the guarded
`ArticulationAction`/gravity/watchdog path. It never creates reference preparation,
RMPflow, scripted routes or replacement/clipped targets.

## Physical evidence, not a success flag

The wrapper samples actual `TaskState` after the native actuator/metrics boundary
and before preview or terminal I/O. It records one initial state and every actual
physics tick, original timestamps, measured part/TCP/joints, prediction counts,
applied model identity and zero reference-route calls. Missing ticks cannot be
backfilled. Evaluator-only ground truth never enters the policy's RGB2/state9/task
inputs.

After stop, the existing frozen-camera barrier supplies both actual 320x320 final
images without another physics tick. The raw-v3 validator checks the complete
local capture. `derive_trial`, `_trial` and the native task predicate rescore
grasp/lift, open/release, TCP retreat, object-speed settling and the unchanged
four-centimetre final volume. Actual gripper gain/restoration evidence, original
grant times and actuator-model metrics must agree.

The existing private publication order is retained: exact spec/inputs, GPU and
model-runtime proof, private report/trace, logs, native acceptance and raw-manifest
copy, then `completion.json` last. Status rechecks every artifact hash, native
task facts and current private raw manifest through the learned validator.
Batch Completed/exit zero, model loading, a reference report or an acceptance
boolean alone cannot qualify the trial.

The new CPU tests cover authority/owner/model/expiry failures, process cleanup,
anti-requeue claims, actual actuator doubles, skipped-state rejection, camera/
servo/metric tampering and private status verification. They are not proof of
mixed-image GPU inference or policy learning quality. Actual execution and the
complete frozen comparison remain operator-owned.

The physical command/raw `episode_id` is the unique Batch `attempt_id`; the
trial also retains the frozen seed/environment/revision and role. The explicit
[managed paired verifier](managed-paired-evaluation.md) predeclares all forty
logical-case/role/physical-UUID assignments and binds its file hash through
`pairing_plan_sha256` in each paired spec. It verifies complete private payloads,
original heartbeats and native task rescore without rewriting raw IDs. Standalone
specs omit that optional field and keep their original wire hash. Single-trial
files still cannot be passed off as the old all-attempt recorder's results.

### Diagnostic-only rejection evidence

The separately attested command-v3 server writes one
`PHYSICALAI_COMMAND_DIAGNOSTIC <JSON>` line to its existing stderr capture
(`probe.log`) when its guarded serving call fails. The record uses
`physicalai.paused-command-diagnostic/v1`, with `diagnostic_only=true`, and is
limited to **16 KiB including the prefix and newline**. There are no new CLI
flags, sockets, prediction wrappers or model-artifact fields.

A `kind=joint_guard_rejected` record is extracted only from the original
`ContractError` traceback with the exact code objects of the unchanged
`serve`, `make_response` and `bounded_joints`, in that order. The walk is capped
at 64 frames. Its closed primitive fields are:

| Field | Original evidence |
| --- | --- |
| `joint_guard` | Failing nine-target vector, joint name/index, value, nominal `low`/`high`, existing `1e-6` tolerance and label |
| `horizon_index` within `joint_guard` | Zero-based index only when the original row object appears exactly once in the original 50-row action object; otherwise `null`/`unknown` |
| `request` | Original request ID/sequence and model/profile/task/context/observation/freeze bindings |
| `context` | Allowlisted validated scope, command/episode/environment/revision, approval and original absolute timing/step bindings |
| `inference` | Original server `started_ns`, `ended_ns` and their actual elapsed milliseconds |

All 50 horizon rows are still checked by the pinned guard before a response;
a rejection does not imply that row zero failed. Missing or ambiguous frames,
malformed/nonfinite values and oversized diagnostics produce an explicit
`kind=unavailable`, never a reconstructed value, guessed row or clipped action.
No arbitrary locals, policy/tensor representations, credentials or PNG payloads
are copied. The original nonzero process exit is retained even if diagnostic I/O
fails. This observes the guard; it does not catch it into a usable policy result.

The learned probe snapshots the terminal `core.command` error/metrics and
initial/terminal observed states **before** optional final-camera collection.
For failure it logs `PHYSICALAI_LEARNED_DIAGNOSTIC <JSON>` and retains optional
`diagnostics` in `probe.json`, schema
`physicalai.paused-learned-diagnostics/v1`. The log line is also capped at
16 KiB; a failed diagnostic read is explicitly unavailable. A terminal snapshot
may share the initial physics step while showing a later prediction count. It
is **not appended** to the per-tick `task_states` array. A missing terminal
readback is `null`, not an invented zero-action measurement.

The original command error remains in `probe.json.error`; camera, capture,
teardown and receipt errors are recorded separately. If camera finalization
fails, `final_images` stays `null`: no final PNG or RAW manifest is fabricated.
The receipt is saved before application teardown; any later teardown error is
logged separately without rewriting that immutable receipt. A failed receipt
write leaves the earlier terminal snapshot in the existing captured log, not a
success-shaped publication.

The Batch wrapper reads available original evidence even when the model exits
or the probe times out. On failure, `completion.json.diagnostics` preserves
the command error/metrics, separate model exit code and secondary errors. It can
report a bound original joint rejection as the primary diagnostic, reading at
most the last **1 MiB** of `probe.log` and accepting exactly one bounded record
matching the attempt/model/owner/task/profile. Absent, malformed, duplicate or
foreign records remain unavailable; a bounded-log miss is not proof that no
inference occurred.

This does **not** admit a new kind of failed trial. The old model-exit/timeout
barrier still prevents native acceptance/raw-manifest publication. A zero-action
rejection still lacks the required two complete RAW frames and sequential
per-tick evidence, so it remains **incomplete and unscorable**. Complete normal
trial wire/defaults and all native scoring, safety counts and frozen criteria
are unchanged. Old incomplete attempts cannot be upgraded by these diagnostics.

There is deliberately **no new initial-RGB persistence or exact pre-prediction
wire-packet capture**: no existing observer callback guarantees durable storage
without adding timing risk to the unchanged 2-second stage, 5-second interval
and 600-second authority bounds. Request/observation hashes are bindings, not
input-image proof.

These logging/wrapper changes require a **new inference image and full source
runtime descriptor**; descriptor `591c5d...` and its old image cannot attest the
changed bytes. All 39 legacy control/IPC/model-reader/prediction files and the
`d72db9...`/`851df4...` identities remain unchanged. Existing P0/P1 training images,
training context identities, archives and import certificates are not rewritten
or forced to upgrade merely because their unused server logging differs.
