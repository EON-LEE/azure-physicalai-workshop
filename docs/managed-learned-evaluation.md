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
The selected model must be an actual changed, trained **candidate** with paused-v2
provenance, nine-dimensional action/state processors, the exact backbone, task,
profile, criteria and conditions. A vendor six-DOF base, `pretrained` preparation,
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
