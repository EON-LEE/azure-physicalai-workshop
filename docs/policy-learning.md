# Azure policy learning

**Explicit commercial default: SmolVLA (2026-09-23).** The coordinator selected
`policy_type: smolvla`, with Apache-2.0 model and backbone revisions, instead of
using ambiguously licensed GR00T weights. This is a visible model-family choice,
not a fallback or a claim of GR00T success. `learning/smolvla/` has its own frozen
environment, model manifests, importer, trainer, server and AML entry points.
No actual learned-policy accuracy or customer readiness follows from CPU tests.

**GR00T commercial model use is blocked.** The exact N1.5 and N1.6
weight licenses restrict use to non-commercial research/evaluation. N1.7's
model card permits commercial use and links the NVIDIA Open Model License,
but its **same pinned repository's `LICENSE` still contains the non-commercial
restriction**. A README claim does not resolve that conflicting primary grant.
`learning.gr00t.licensing.APPROVED_COMMERCIAL_MODELS` is therefore empty.
Customer weight import, paid-job preflight/submission, training, probing and
learned execution fail closed before accessing weights or launching work.
An assent checkbox cannot override this decision. Status/cancellation, raw
capture, CPU data conversion and source/API inspection remain available.
No GR00T weights have been acquired or executed by this workstream.

| Family | Exact inspected model revision | Decision |
| --- | --- | --- |
| N1.5 | `869830fc749c35f34771aa5209f923ac57e4564e` | Non-commercial; blocked |
| N1.6 | `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` | Non-commercial; blocked |
| N1.7 | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` | README/LICENSE conflict; blocked pending authoritative clarification |

Primary references: [N1.6 license](https://huggingface.co/nvidia/GR00T-N1.6-3B/raw/d0814e7ecb19202e7c8468b46098b0b7ef3a6d61/LICENSE),
[N1.7 model card](https://huggingface.co/nvidia/GR00T-N1.7-3B/raw/2fc962b973bccdd5d8ce4f67cc63b264d6886495/README.md),
[conflicting N1.7 license](https://huggingface.co/nvidia/GR00T-N1.7-3B/raw/2fc962b973bccdd5d8ce4f67cc63b264d6886495/LICENSE),
and the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
The latter includes conditional commercial rights and Cosmos attribution/
redistribution requirements; it cannot simply be assumed to replace a
contradictory artifact license. The deployment owner must resolve the exact
model grant and gated backbone access first. This is separate from EA maturity
or commercial-support availability.

## Explicit SmolVLA path

| Artifact | Exact identity |
| --- | --- |
| Native package | LeRobot 0.4.4, source `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| Policy weights | `lerobot/smolvla_base` at `d9f33c94a60fb382c90dea2164c96845bd955e28` |
| Backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Runtime | Python 3.11, PyTorch 2.7.1 / CUDA 12.6, Transformers 4.57.3 |
| Licenses | Apache-2.0 model/backbone cards and source license, with exact byte hashes in `UPSTREAM` |

The vendor policy declares six state/action features and three camera keys.
That is **not** a Franka policy. `adapted_config` explicitly changes physical
features to the actual seven arm radians plus two individual finger metres,
two real camera keys, absolute actions, no Aloha conversion, and one action per
10 Hz observation. Its actual native projections already use 32 internal
dimensions; nine measured values are preserved before documented internal
padding to 32. No missing physical joint measurement is invented. Native Smol
image preparation supports two cameras with `empty_cameras=0`; no fake third
observation is generated. Predicted chunks contain **50 x 9** physical values,
not GR00T's 16-step or N1.7's observed 40-step artifact configuration.

### Operator asset preparation and data

The authorized deployment owner acquires only the exact reviewed public model
and backbone files, separately from the offline job runtime. No customer data
is uploaded to Hugging Face. The importer performs no network calls:

```bash
python -m learning.smolvla.prepare \
  --model-source LOCAL_PINNED_SMOL_FILES --backbone-source LOCAL_PINNED_BACKBONE_FILES \
  --binding APPROVED_BINDING.json --output NEW_PRIVATE_ASSET_BUNDLE
```

For authorized private Azure preparation, `learning.smolvla.cloud_assets` verifies
the completed vendor inventory and exact allowlist through managed-identity Blob
reads, checks immutable ETags, byte counts, SHA-256s and pinned metadata revisions,
then invokes the same native importer. Inputs and outputs must be distinct paths
under the exact tenant/opaque-owner prefix; existing partial outputs are not
overwritten. The final prepared-inventory proof is written last. The job deadline
is bounded, and the runtime image must match its reviewed `code-manifest.json`.
`learning/smolvla/Dockerfile.runtime` layers only an allowlisted learning-code
build context over an explicitly digest-pinned dependency image; do not send the
whole worktree or test/credential files as its build context.

The September 23 authorized reconciliation observed asset-stage execution
`factory20-smol-assets-01-wyvpmlc` succeed, then independently re-read all **20
private files (2,941,579,286 bytes)** in bounded CPU execution
`factory20-learning-smol-prep-01-apbq8a5`. That second job checked every content
hash and metadata revision and published the private verification proof with SHA
`2c396c80cfa4cd2ddd41207569ffefcc16e2de87fcac268bd40147c3961220cc`.
It loaded **no model weights** and ran **zero optimizer steps**. This proves
staged asset integrity, not a prepared robot binding, learned policy or task
accuracy. Native bundle preparation still requires the actual deployed servo
profile, approved full task and independently verified real capture provenance.

Binding JSON contains `scope: {tenant_id, owner_id}`, `control_profile` and
`task: {task_id, instruction, goal_id}`. `owner_id` is the opaque 64-hex owner
hash, **not** the actor's object ID. `prepare.VENDOR_SHA256` and
`VENDOR_GIT_BLOBS` are the exact download allowlist; ONNX assets, notebooks and
arbitrary files are not copied. The importer verifies model weights, all used
backbone/tokenizer/config files, license cards and the installed source license.
It emits `model/` (train-only) and `backbone/` bundles and their manifest hashes.

```bash
python -m learning.smolvla.dataset --source REAL_V2_CAPTURE --output NEW_V3_DATASET \
  --binding APPROVED_BINDING.json --manifest-sha256 RAW_MANIFEST_SHA256
```

Only complete, scoped v2 10 Hz demonstrations enter this converter. It uses the
real pinned LeRobot v3 API and preserves all samples, two cameras and the
approved task instruction. The new `physicalai.lerobot-conversion/v2` sidecar
binds the servo profile and demonstrator provenance; existing v1 ACT conversion
artifacts remain valid. Both capture and inference share
`validate_joint_tracking`: 0.05 rad maximum arm target/tracking change and
0.004 m per finger per 10 Hz tick by default. A commanded full close from 0.02 m
to zero while contacting a part is rejected as an invalid training label.
Bad labels are not dropped, clipped, interpolated or padded into valid data.

### Actual training and inference

`learning.smolvla.train.run_training` verifies actual Azure component/parent job
identity and CUDA, initializes from real checked safetensors, then calls
`python -m lerobot.scripts.lerobot_train` with the Smol policy path. It builds
**new nine-dimensional normalization processors from the train split**, never
reusing vendor six-dimensional normalizers. Publishing/W&B and online Hub access
are disabled. Both the VLM and tokenizer point to a verified mounted backbone.
No homegrown optimization loop or ACT substitution is used.

Checkpoints are saved every approved `checkpoint_steps` on `rw_mount`.
`training-context.json` persists the original actual pipeline/component IDs,
specification/data/parent hashes, options and pre-update parameter fingerprint.
`recover_checkpoint` can seal a completed safe model checkpoint after verifying
the terminal Azure job, exact original context, native `training_step.json`,
parent fingerprint and changed trainable values. It neither loads optimizer
pickle nor submits/restarts a paid job. Continuing requires a new durable API
claim and reviewed `resume_mode: weights_only` job; optimizer/scheduler restart
is explicit and ancestry/cumulative step counts are retained.

The final `model` output contains `result.json` and
`candidates/step-XXXXXX/{model.json,checkpoint/...}`. The result binds
`azure_job_id` (parent pipeline), `azure_component_job_id` (actual optimizer
component), `specification_sha256`, actual optimizer steps and candidate manifest
SHA. The candidate's training fields retain the child `azure_job_id` and parent
`azure_pipeline_job_id` separately. The validator requires changed weight bytes
**and changed actual trainable projection samples**. `processor_sha256` binds
the canonical map of every saved pre/postprocessor JSON and normalization
safetensors checksum. `backbone_manifest_sha256` pins the separately supplied
backbone bundle. Loss is not fabricated: the upstream per-step training log is
retained, while an unparsed typed loss stays null.

The deployment-selected process is:

```bash
python -m learning.smolvla.inference \
  --model-root APPROVED_TRAINED_MODEL --backbone-root APPROVED_BACKBONE_BUNDLE \
  --binding APPROVED_BINDING.json --model-sha256 APPROVED_MODEL_MANIFEST_SHA256 \
  --socket-path /run/physicalai-policy/policy.sock --allowed-client-uid ISAAC_UID
```

The runtime uses `learning.smolvla.ipc.SocketChunkPolicy(socket_path, scope=...,
model_sha256=..., profile=..., task=..., expected_peer_uid=...)` with
`RemoteGuardedPolicyAdapter`. It preserves the existing main-thread actuator,
approval, source-case, owner, epoch, deadline, joint, speed and cancellation guards.
The transport schema is explicitly `physicalai.smolvla-request/v1` /
`physicalai.smolvla-response/v1`; it cannot accept a GR00T chunk or checkpoint.
The vendor initialization cannot execute as P0, even after its configuration
has been adapted. Only a genuinely trained candidate can enter privileged
evaluation; production execution additionally requires the parent's reviewed
release and measured timing attestation.

### Private AML worker and physical evaluation

`learning.smolvla.azure.create_plan(config, output, deterministic_job_name=...)`
is offline-only. `clients_for_managed_identity(config, caller_client_id=...)`
returns `(MLClient, StorageManagementClient)`; no interactive/default credential
is selected. `PolicyJobs(client, config, storage_client=...)` exposes `preflight`,
`submit`, `status`, `cancel` and explicit `reconcile_deadline`. The API owns the
durable paid-job and cancellation claims. An
existing deterministic job must match owner/specification/plan hashes and is
reconciled, not resubmitted. Metrics remain null until verified actual outputs
are read; Azure status `Completed` alone does not verify a model or its quality.

New admissions require config `physicalai.smolvla-azure/v2` and plan
`physicalai.smolvla-azure-plan/v2`, with explicit
`compute_tier`, resource/identity/private-store settings, immutable `UPSTREAM`,
specification/task/profile hashes, and the same bounded training options.
The required top-level `job_deadline_utc` is canonical UTC ending `Z`:
the **earlier of the original persisted API run deadline and the original
operator registration's `expires_at`**. The trusted worker binds this value
before approval; retries, process restarts and delayed allocation cannot replace
it with `now + timeout`. Config, code snapshot, plan SHA, and both pipeline and
component tags bind the exact value. A v1 plan remains available for offline
inspection and owned status/cancellation, but has no new queue authority and
cannot enter preflight, submission, reconciliation or component execution.
`kind: train` has registered inputs `demonstrations`, `parent_model`, `backbone`;
outputs are `dataset` and `model`. `kind: compare` has `policy_before`,
`policy_after`, `evidence`, `plan`; `kind: bootstrap_compare` has `candidate`,
`evidence`, `plan`. Both evaluation jobs output `report/report.json`.
Folder hashes bind their appropriate manifest files; evaluation-plan hashes
bind canonical native JSON. An API authorization-plan hash is a distinct value:
the immutable specification binds the reviewed mapping to the native plan asset.

For these new policy pipelines, `parameters.timeout_seconds` is the **total
sequential execution budget**, not training time plus an extra converter
allowance. Conversion receives at most 600 seconds and one fifth of that
budget; training gets the remainder. Both limits are explicit in the job graph.
Image preparation, allocation waits and actual dollar cost still require the
operator/API's separate bounded allocation deadline and price review.

Native v2 checks expiry before preflight and again immediately before job
creation. Every component checks before data/model access and verifies its real
Azure component/parent deadline tags; no AML identity is synthesized. The Linux
CLI supervises the complete component in its own process group with an absolute
UTC deadline and a monotonic budget that cannot be extended by a clock rollback.
This also bounds blocking artifact hashing, conversion and model-loading phases.
Expiry kills the component and its descendants, exits nonzero, and cannot seal
an acknowledged successful run. Training checks phase boundaries and caps its
actual upstream optimizer subprocess to the remaining approved budget.
Partial mounted files do not prove a successful candidate.

`status` is read-only and includes raw `azure_status`. Explicit
`reconcile_deadline(job_name)` requires the caller's **durable cancellation
claim**; it never creates or resubmits a job. Expired active jobs use the existing
owned `cancel` path, then return the actual Azure state. `CancelRequested` is
still `cancelling`, not `cancelled` or `timed_out`; a terminal race is reread
without another POST. `cancellation_requested` records whether this call sent
the request. `NotResponding`, `Paused` and `Unknown` are not evidence that
allocation ended. Permission/network failures propagate, retaining an uncertain
outcome rather than fabricating a terminal receipt.
The pinned AML SDK uses `jobs.begin_cancel(name, polling=False, retry_total=0)`,
not the nonexistent `jobs.cancel`. The returned LRO acknowledgement is ignored
as terminal evidence; an owned GET remains authoritative. The offline SDK check
uses the installed `JobOperations` autospec, actual public-method forwarding and
Azure Core retry configuration, in addition to lightweight port tests.

These source guards are **not a deployed durable queue reconciler**. Production
submission still requires the API/worker owner's independently hosted,
restart-safe deadline/cancellation claim and scheduling path; a CLI monitor or
UI GET alone cannot ensure unattended cancellation. The pinned SDK's command
execution timeout does not supply a queue TTL. No scheduler, role, network rule
or paid job is created by this native change.

### Separate paused-simulation contract

The coordinator froze a new **NON_REALTIME_SIMULATION** regime before new
collection or model selection. `learning.paused` contains its stdlib-only shared
types and version constants. Its canonical fields are
`execution_timing="paused_simulation"` and `real_time_admission=false`; it is not
a renamed or relaxed real-time release. `PausedControlProfile` binds an explicitly
versioned paused servo fingerprint, actual 60 Hz physics,
10 Hz **simulation-time** control and six actual held ticks. Frozen limits are
2,000 ms each for observation, policy and held-tick phases, intersected with
the **original** 5,000 ms whole interval and 600-second episode wall deadlines.
The simulation budget is a closed profile-ID/limit pair: the unchanged default
`franka-position-hold-10hz-paused-v1` requires exactly `max_simulation_steps=1800`
(30 simulation seconds), while the explicitly selected
`franka-position-hold-10hz-paused-v2` requires `max_simulation_steps=3600`
(60 simulation seconds). Unknown IDs and mixed ID/limit pairs fail closed.
`CONTROL_PROFILE_ID` remains v1; `CONTROL_PROFILE_V2_ID` names the new profile.
No other profile fields or wall-clock budgets change. The separate real-time
profile and both historical 30-second results retain their original meaning;
v2 does not promote or relabel an earlier failed/incomplete attempt.
The v2 redesign was frozen before any v2 capture in criteria revision 3
(`63b55c92a4d543c7b48d7c6ac0ed75f9526e5a8ec91852dac9f53b0c6804190a`).
Its new saved-environment revisions and model-independent conditions plan must
be bound explicitly before a real v2 run; the earlier v1 conditions hash is not
an alias for them.
Main-thread heartbeat gaps remain at most 2,000 ms; physical slew, tracking,
0.2 m/s speed and 4 cm goal limits are not relaxed.

`FrozenPolicyObservation` retains the original image/joint wall timestamps and
native simulation-time numerator/denominator, plus a unique freeze ID, actual
physics step, control tick, epoch, state revision and profile digest. Its digest
binds actual PNG checksums and metadata, never evaluator seed/object pose/success
features. `PausedControlContext` carries original episode, interval and operation
monotonic deadlines and the simulation-step deadline. `PausedFrameSample`
requires all six actual `AppliedControl` records, zero velocity targets and
measured arm-only gravity effort; partial or expired holds are not training
intervals. A single in-flight worker is necessary so the runtime main thread
can keep checking cancellation/heartbeat while **withholding physics steps**;
timeline pause/play/reset and timestamp reminting are not freeze mechanisms.

The new namespaces are raw/conversion v3, Smol checkpoint/IPC v2,
paired/bootstrap plan v2, results v3, reports v2 and rollout grant/recording v2.
They cannot be admitted by the old real-time data/model/IPC/report catalogues.
`learning.paused.capture.PausedEpisodeWriter` implements the separate raw-v3
producer. It accepts the real `PausedFrameSample` on its creating simulator
thread and publishes `manifest.json` **last**, after revalidating all files.
Frames go to `episodes/<episode_id>/frames.jsonl`; original PNGs go to
`episodes/<episode_id>/{inspection,overview}/<frame_index>.png`. The manifest
binds the new profile, frozen criteria/conditions-plan hashes, explicit capture
purpose and original `PausedEpisodeBudget`. The dataset records simulation-time
10 Hz cadence separately from variable original UTC/monotonic timestamps.
The writer's default frame limit is derived from the validated profile:
300 for v1 and 600 for v2. An explicit narrower limit remains allowed; a larger
one is rejected. The default byte budget remains 512 MiB. All six actual control
records and both original images are retained for every interval; no downsampling,
dropped frames or lossy resampling extends a capture. Raw v3, checkpoint v2 and
IPC v2 remain separate artifact-schema versions and continue to bind the whole
control profile/hash. A v1 capture cannot become v2 by editing its manifest.
Reference demonstrations have null policy timings, not invented zero-latency
neural calls. Every complete frame has exactly six real controls; failed append
attempts fault the writer rather than permit retry-until-success.

The matching `validate_dataset` and `assemble_dataset` retain scope, checksum,
safe-path, image, count and seed/episode split checks. They reject schema/profile/
criteria/plan/purpose mixtures and extra ground-truth artifacts. Integration
captures are explicitly TEST-only, reserved integration seeds cannot become
demonstrations, and `require_demonstrations=True` rejects integration/evaluation
purposes and truncated episodes. Incomplete streams remain unpublished evidence,
not usable datasets. Both v1/v2 and v3 reuse the same checked PNG/path verifier;
the old validator still refuses paused v3.

Before any paused-mode capture, the coordinator clarified first-frame causality
in frozen criteria revision 2 (`588fef92a96ace2dc5ffb5e606195a47f19e1e546b1ebf911220634708b3b1ef`;
new frozen conditions plan `c28dc6a4892f9a90c9215d746fe331c221d0fb737ad765f6230f24d01ef08854`).
Only control interval zero may carry `InitialFrozenPublication`: an original,
current-physics publication at most 2,000 ms before the new observation request,
with a real freeze established **before** its original joint/camera samples.
It retains original UTC/monotonic values and binds sensor-only capture content,
epoch/scope/physics/native-time identity, an opaque runtime record ID/hash and
the exact declared age. `observation_completed_ns` measures the **new**
observation work; `ready_ns` exposes that completion without reminting the old
publication timestamp. Subsequent observations keep the original strict fresh
post-hold chronology. All controls still follow episode admission, and none of
the 2/2/2/5-second or episode resource limits changed.

The private world/object/qvel state and its fingerprint stay in the trusted
runtime/evaluator record, never in model observations or neural features.
Offline hashes are not physical truth. For initial-publication inference the
adapter additionally requires a `publication_guard(publication, observation,
context) -> bool` callback authorizing the **known** live record both before and
after prediction; missing/unknown records and changed state fail closed. This
callback must use a trusted thread-safe ledger, not call the simulator SDK on
the inference worker. The main thread must still resolve and recheck actual
private frozen state immediately before applying any action. Missing, expired
or retrospectively invented proof requires explicit re-preparation or failure,
not timestamp copying, duplicate-frame rendering or an extra physics step.

`learning.paused.dataset.convert_dataset` is the explicit v3 -> pinned LeRobot
entry. It requires the approved raw manifest, criteria and new frozen-plan SHAs,
only selects complete train-split demonstrations, and uses the unchanged real
LeRobot image/state/action writer. The installed 0.4.4 API rejects a user-supplied
`timestamp` feature even though its implementation contains a timestamp pop.
Instead, conversion checks every actual native simulation-time delta against
the exact frame-index/10 Hz value **before** relying on the API's supported
timestamp generation. There is no wall-time resampling or interpolated frame.
`source-timing.jsonl` preserves original raw wall/native-SIM metadata, images'
original identities and held-control times outside neural features; its checksum
is part of conversion v3. The legacy conversion validator refuses v3.

The separated `python -m learning.checks.paused_conversion_check --output <new-dir>`
check used the actual installed LeRobot 0.4.4 and read actual Parquet timestamps
approximately `[0.0, 0.1, 0.2]` for explicitly test-only frames whose original
wall interval was two seconds. It loaded no policy weights, performed zero
optimizer steps/cloud calls and proves conversion/API compatibility only.

Capture storage does not by itself execute a model or establish physical quality.
Subsequent model/IPC/evaluation producers must explicitly consume these contracts;
real-time 80/100 ms gates remain blocked, and no actual paused training or quality
result is implied by contract or storage tests.

Conversion counts, episode grants and task-trace limits are derived from the
selected validated profile, not from a global replacement of 30 with 60.
Conversion retains 300/600 full records and their original timing sidecar.
The per-physics evaluator accepts at most 1,801 states for v1 and 3,601 for
explicit v2, including the initial state; a call without an explicit profile
retains its v1 ceiling. Task traces remain checksum-bound JSONL sidecars with
the existing 8 KiB row ceiling and a total limit derived from that state's
profile bound. Readers reject extra/missing/oversized records rather than
truncate them. No default `read_json` limit is increased. A separate private
runtime proof reader may use its reviewed v2-only 8 MiB envelope; that is not
permission to widen historical v1 or arbitrary JSON artifacts.

The separated CPU check supports
`--profile-version 2 --full-episode`: the actual pinned LeRobot API writes all
600 fixture frames and checks every Parquet timestamp against the exact
float32 representation of the recorded simulation-step cadence. This is
conversion compatibility, not physical training or task success. Evaluation
duration remains derived from integer applied ticks divided by 60 Hz; float
world-time drift is neither added to the cap nor hidden by a larger tolerance.

`learning.paused.inference.PausedGuardedPolicyAdapter` provides
`reset(context)`, `step(observation, context)` and immediate `stop()`. The caller
runs the single in-flight `step` on its bounded worker, not the simulator thread.
The adapter preserves the original episode authority across changing observation
freezes, consumes one fresh action from the unchanged 50-action Smol horizon, and
checks all nine targets without clipping or a reference-controller fallback.
Returned latency includes request queue/validation time and is never divided by
the chunk horizon. A generation change invalidates an in-flight reply; stop does
not wait for or reset a running model. The simulator must still recheck its
private frozen state and live authority before every actual physics tick.
The old real-time adapter now explicitly refuses paused policies/observations;
its original timing limits and successful v1 behavior are unchanged.

Paused checkpoints use the separate `learning.paused.artifacts` validator.
Checkpoint v2 retains the complete upstream weight/processor/CUDA/update lineage
checks and additionally binds simulation-time raw v3/conversion v3, the exact new
profile, frozen criteria and conditions plan. `pretrained` remains train-only;
it cannot be served as a nine-joint candidate. The real-time Smol validator and
IPC v1 refuse these v2 artifacts.

The real model entry is `learning.paused.model.LocalPausedSmolVLAPolicy`.
It shares the pinned actual `SmolVLAPolicy.from_pretrained(..., strict=True,
local_files_only=True)` implementation, local backbone and saved normalization
processors; it never creates a stand-in model. Only two decoded RGB tensors,
nine measured joints and the approved instruction become neural inputs. Private
freeze evidence, IDs, task outcomes and seeds remain outside those features.
The native 50-action output is checked as `[1, 50, 9]`, with one fresh action
consumed by the paused guard.

```bash
python -m learning.paused.model \
  --model-root /approved/paused-candidate \
  --backbone-root /approved/backbone \
  --model-sha256 "$MODEL_MANIFEST_SHA256" \
  --binding /approved/paused-binding.json \
  --socket-path /run/physicalai/paused-policy.sock \
  --allowed-client-uid "$SIMULATOR_UID"
```

The binding explicitly includes scope, the paused profile, criteria/conditions
SHAs and the canonical non-real-time flags. `learning.paused.ipc.SocketChunkPolicy`
uses the protected Linux socket and peer UID, independent Smol request/response
v2, exact freeze/context/content binding and the original two-second operation
deadline across serialization, connect, every read/write and validation.
The server rejects replayed sequences and exits on invalid/late requests rather
than supplying a reference action. This command path has CPU codec, actual
Linux socket and schema tests; a trained paused checkpoint/CUDA execution is
still required before claiming real model operation or quality.

The paused training producer uses a closed discriminated variant of
`physicalai.smolvla-azure/v2`: all of `execution_timing="paused_simulation"`,
`real_time_admission=false`, `criteria_sha256` and `frozen_plan_sha256` must be
present together. The latter is the **model-independent frozen scene/conditions
plan**, not a native evaluation-plan hash. `job_deadline_utc` remains the original
absolute wall-clock expiry. Snapshot/plan/job tags bind these fields and the
actual `learning.paused.components` code. Legacy v2 without the discriminator
keeps its previous real-time command; mixed/unknown fields and cross-mode job
tags fail closed.

```bash
python -m learning.paused.prepare \
  --model-source /approved/private-vendor/model \
  --backbone-source /approved/private-vendor/backbone \
  --binding /approved/paused-binding.json --output /approved/new-paused-initialization
python -m learning.smolvla.azure \
  --config /approved/paused-training-config.json \
  --plan-dir /approved/new-offline-plan --job-name "$DETERMINISTIC_JOB_NAME"
```

Preparation only verifies/copies the pinned local private vendor bytes into a
**train-only** checkpoint-v2 bundle; no inferred nine-dimensional vendor stats
or ready-to-act candidate is manufactured. The binding additionally supplies
the approved task, real new profile, current criteria and frozen conditions
hashes. Actual training explicitly dispatches to `learning.paused.train`,
validates raw-v3 conversion/profile/task/criteria lineage, and reuses the real
`lerobot.scripts.lerobot_train` command. Initialization derives new nine-dimensional
normalization from the train split; the existing pinned two-camera feature
mapping, 50-action horizon, one consumed action and local backbone remain.
Only actual changed parameter fingerprints and upstream optimizer markers seal
a candidate-v2 manifest. All long phases remain bounded by the original UTC
deadline supervisor; no automatic resume, publishing or paid submission occurs.
The paused comparison commands use the distinct recording/evaluation producer
below; its presence does not waive worker enrollment, actual data or operator
submission approval.

### Managed-job intermediate checkpoints and explicit restart

The optional, reviewed `checkpointing` object on a v2 Smol training configuration
enables the native checkpoint publisher. It contains
`schema: "physicalai.smolvla-checkpointing/v1"`, `limits` (the exact fields of
`learning.smolvla.checkpoints.CheckpointLimits`), and `resume: null` for an initial
run. The defaults are 64 files, 4 GiB per file, 8 GiB per complete checkpoint,
1 MiB per JSON file, a 4 MiB safetensors header, at most 32 checkpoints and
180 seconds per publication. Tighter reviewed limits are supported. Checkpoint
frequency must fit the count budget and the entire operation remains inside
the original absolute job deadline; this does not authorize extra time or cost.
Existing plans without this object do not acquire a durable-publication claim.

The runner calls the actual, source-hash-pinned LeRobot 0.4.4 training entry and
wraps its `save_checkpoint` boundary. It never implements a substitute optimizer
loop. At each configured save interval, native
`pretrained_model/model.safetensors`, policy/train configuration, processors,
optimizer safetensors, optimizer parameter-group JSON, scheduler JSON, RNG
safetensors and the actual step marker are verified together. The native step
file is written **before** the other state files and is not a complete marker;
the native `last` symlink is not restart authority.

Only after the native save returns and the complete allowlisted inventory passes
validation is `checkpoint.json` published locally. Checkpoints bind owner,
original data/conversion/model/profile/task/criteria/conditions, training
configuration, pinned code/runtime, actual source Azure component/pipeline,
original expiry, optimizer step and cumulative-step lineage. Pickle, executable,
unexpected and symlinked files are rejected without `torch.load`. Publication
uses the approved compute MI and storage/container beneath the existing
owner-scoped job output prefix. Every create-only Blob upload is reread with
its ETag and checked against exact byte count and SHA before the complete
manifest is uploaded last; the receipt records readback ETags. Mounted-output
existence or `rw_mount` alone does not prove that a completed checkpoint reached
Blob. A failed/torn publication has no usable complete marker.

`latest_remote_checkpoint` scans only a bounded, known owner/job prefix and
verifies the last complete bundle; newer unmarked files are not selected.
Corrupt marked checkpoints produce an explicit error, not a silent fallback.
`restore_checkpoint` downloads a named manifest-SHA-bound bundle into a new
directory and writes its local completion marker only after all readbacks pass.
Neither operation submits jobs, follows arbitrary URLs or creates a mutable
`latest` pointer.

The first publisher milestone (`3511e6d`) advertises
`resume_capability="weights_only"`. Although native safe optimizer/scheduler/RNG
files are preserved, native 0.4.4
does not persist a sampler cursor, discards Python's Gaussian cache and truncates
NumPy's cached Gaussian precision. Loading its `--resume` state alone is not
verified full-state continuation. A weights-only restart restores actual saved
model weights but initializes a **new optimizer/scheduler/RNG/data iterator**;
`optimizer_state_restored` and `bitwise_continuation_claimed` stay false.

The full-state runner adds `training_state/continuation.json` and
`continuation.safetensors`. These preserve Python and NumPy Gaussian caches at
full precision, Torch/CUDA RNG state, the exact shuffled sample permutation,
consumed cursor, epoch, batch count, independent sampler/loader generators and
the actual precision flags. Such checkpoints advertise
`resume_capability="full_state"` only after the complete native save and sidecar finish. Historical
weights-only checkpoints are not upgraded or backfilled into full-state proof.
Nonfinite model or optimizer tensors cannot be published as completed state.

Exact data continuation is explicitly restricted to the pinned single-process
Smol loop, one visible CUDA device, no AMP, zero data workers, no augmentation
and an immutable map-style dataset of at most 1,000,000 frames. The integration
wraps the native trainer's data-iteration boundary, rather than replacing its
optimizer loop. It retains the approved episode-aware eligible indices but uses
a dedicated, checkpointed permutation generator. A native `DataLoader` supplies
batches without Accelerate's one-batch read-ahead; this prevents saving a cursor
past the last completed update. On restart, native optimizer/scheduler state
loads through LeRobot's safetensors/JSON APIs, and the supplemental exact RNG
and data state restores **after** loader initialization. Checkpoint I/O restores
its entry RNG snapshot so storage-client bookkeeping cannot perturb training.

A restart is a distinct new v2 job with its own reviewed wall/cost approval.
Set `parameters.resume_mode` explicitly to `weights_only` or `full_state`, and
`checkpointing.resume` to the
exact `checkpoint_sha256`, `source_azure_job_id`, `source_azure_pipeline_job_id`
and `step`. Add immutable `resume_checkpoint` and `converted_dataset` folder
inputs in the same approved datastore. The latter is the original pipeline
dataset output containing `dataset/conversion.json`, not a new conversion of
the same raw data. Its manifest SHA and the checkpoint's original dataset
binding must match. The restart graph omits conversion, uses the explicit
checkpoint input and refuses the old source job/approval identity. The original
checkpoint expiry is retained as provenance, never renewed or substituted for
the new job's approval. New weights, cumulative updates and source checkpoint
links remain auditable; saved weights do not establish learned task quality.

For `full_state`, every original binding must match, including the approved
image digest, Python/package/driver/CUDA/device identity, reviewed code, batch,
seed, optimizer configuration, save interval and total native training horizon.
The CLI uses the original local `--config_path=.../pretrained_model/train_config.json`
with `--resume=true`; it does **not** use `--policy.path`, which takes a different
native parser branch. Only the new output location, unchanged dataset's mount
path and approved local backbone path are relocated. `max_steps` remains the
original global target, and it must exceed the source checkpoint step. This
is not an opportunity to silently lengthen a scheduler or renew an old job's
authority. The source expiry remains provenance; the new job has its own
approved absolute expiry and cost ceiling.

Final full-state model provenance records the exact source checkpoint manifest,
source component/pipeline, source step and source cumulative count. It requires
positive new updates, `checkpoint_step = source_step + new_updates`, and
`cumulative_optimizer_steps = source_cumulative + new_updates`. Only the verified
full-state branch sets `optimizer_state_restored=true`. Changes to data,
configuration, code, image, device or precision reject full-state continuation;
a separately approved weights-only restart is available but never presented as
identical training. CPU bitwise equivalence does not imply cross-device GPU
determinism or physical learning quality.

`python -m learning.checks.checkpoint_resume_check --output <new-local-dir>`
performs the separated actual native CPU serializer/optimizer/weight-roundtrip
check. Its storage transport is explicitly in-memory, not Azure durability
proof, and its tiny model is a test fixture, not a trained physical policy.
Add `--full-state` to compare uninterrupted training with interruptions at both
mid-epoch and epoch boundaries. The check uses actual native `update_policy`,
AdamW, LR scheduling, safe save/load and the decorated native `train` CLI entry,
including `--resume`. It compares model/optimizer/scheduler tensors, exact RNG
caches, every sample index and the next batch; `torch.load` is prohibited during
the check. The actual CPU result is a determinism test, not a GPU or cloud claim.

Deployment remains separate: build the pinned dependency image from
`learning/smolvla/Dockerfile` and the code layer from
`learning/smolvla/Dockerfile.runtime` with an approved `DEPENDENCY_IMAGE` digest.
The code-layer check verifies actual installed native source hashes without
loading model weights or contacting Azure. Generate the matching code manifest
and use a **new reviewed image/snapshot fingerprint**; original images/proofs
remain immutable. The API/worker must propagate the exact operator-approved
`checkpointing` object and both immutable resume inputs, install the updated
native model-provenance validator, and retain fresh deadline enrollment and
new-job mutation claims. For full-state jobs, the worker requires the public
request's `optimizer_steps` to equal `max_steps - checkpointing.resume.step`:
the approval covers new updates, not work already completed by the source job.
The original global horizon remains unchanged in the native configuration.
The same approval/deadline binding remains mandatory. This native slice adds no API route,
deployment, cloud resource or automatic paid retry.

An actual private Azure Blob diagnostic subsequently published all 12 safe
checkpoint files, verified ETags and hashes before the completion manifest, then
restored them in a separate Container Apps Job after the publisher terminated.
The restorer selected completed step 3 and rejected a deliberately torn newer
step 4. This proves the external checkpoint transport using a **tiny native CPU
fixture**; it is not SmolVLA robot training, CUDA continuation or physical quality.

### Offline bootstrap operator package

`learning.paused.bootstrap` prepares metadata and the existing native job graph
without credentials, Blob calls, model imports or job submission. It is an
operator-side helper, not a new training loop or a reason to rebuild the already
qualified training image. Its output explicitly says **manifest metadata checked,
payloads not verified, submission still gated**. The existing managed conversion
and training components perform full payload validation.

First pin the operator-supplied successful reference qualification, then extract
its exact profile/task identity into a **new** train-only preparation binding:

```bash
python -m learning.paused.bootstrap binding \
  --qualification "$G0_BINDING_JSON" --qualification-sha256 "$G0_BINDING_FILE_SHA256" \
  --tenant-id "$TENANT_ID" --owner-id "$OWNER_HASH" \
  --profile-sha256 "$ACCEPTED_PROFILE_SHA256" \
  --criteria-sha256 "$CRITERIA_SHA256" --frozen-plan-sha256 "$CONDITIONS_SHA256" \
  --output /approved/new-preparation-binding.json
```

This copies neither episodes nor seeds into the binding. The successful
integration seed **900002 must not enter TRAIN**, even though it supplied
profile qualification. The first bootstrap package requires exactly one
train-split episode for each seed 10001–10020, with matching owner, new profile,
task and frozen criteria/conditions. Adaptation seeds 11001–11020 and held-out
seeds are not silently included in this first cohort.

The already staged pinned vendor model/backbone bytes may be reused after exact
hash verification. A validated `physicalai.smolvla-backbone/v1` bundle is
profile-independent and reusable only with the same owner, upstream pins and
inventory. An old **parent `model.json` is not reusable** when its profile,
task or conditions differ. In an explicitly authorized managed preparation
environment running the Python 3.11 native image, run the existing command:

```bash
/opt/smolvla-venv/bin/python -m learning.paused.prepare \
  --model-source "$LOCAL_VERIFIED_VENDOR_MODEL" \
  --backbone-source "$LOCAL_VERIFIED_VENDOR_BACKBONE" \
  --binding /approved/new-preparation-binding.json \
  --output "$NEW_LOCAL_PREPARED_BUNDLE"
```

The result contains `model/model.json` with `role="pretrained", training=null`
and `model/checkpoint/*`, plus `backbone/backbone.json` and `backbone/assets/*`.
Publish to a **new** approved private prefix with manifest-last/hash readback.
Never edit an old parent manifest, change its checksum in place, or call the
six-dimensional vendor normalizers nine-dimensional training statistics.
The native `prepare_seed` derives new state/action statistics from actual TRAIN.

Native configurations deliberately use one explicit datastore. Captures currently
uploaded under the `demonstrations` container must be downloaded by the approved
private reader, validated with `learning.paused.capture.validate_dataset`, and
assembled using `assemble_dataset(..., require_live=True)` without changing any
episode/frame/image bytes. Stage the assembled cohort in the owner-scoped
**artifacts** container before registration with `learningartifacts`; a URI naming
that datastore is not an alias for the separate demonstrations container.

Managed training preflight also reads the Blob lifecycle policy. Blob Data
Contributor does not grant the management-plane
`Microsoft.Storage/storageAccounts/managementPolicies/read` action, as confirmed
by an actual worker request. The default-off
`infra/learning-worker-policy-reader.bicep` grants Reader only on the existing
account's `managementPolicies/default` resource. It changes neither policy
contents nor data access and adds no policy-write or subscription-wide permission.

Workspace networking is checked against SDK enum **values** as well as plain
strings. SDK 1.35 returns `PublicNetworkAccessType.DISABLED`, whose value is
`Disabled`; comparing its enum display string incorrectly rejected an actually
private workspace. This compatibility correction still rejects public access,
internet-wide outbound mode and missing values; it changes no Azure settings.

The actual Azure ML service adds one trailing `/` to registered `uri_folder`
paths. Preflight accepts only that exact folder-only representation difference;
the approved configuration URI remains unchanged. File inputs, nested paths,
double separators, case changes and query strings still fail the location check.

| Native input | Registered folder root | Required checksum |
|---|---|---|
| `demonstrations` | `manifest.json` and complete `episodes/<id>/...` tree | Exact assembled `manifest.json` file SHA |
| `parent_model` | New `model.json` and `checkpoint/*` | Exact new `model.json` file SHA |
| `backbone` | `backbone.json`, `assets/*`, license inventory | Exact `backbone.json` file SHA |

Supply a complete native `physicalai.smolvla-azure/v2` configuration with explicit
subscription/resource group/workspace/datastore/account/container, managed identity,
compute name/SKU/tier, digest-pinned image, run/specification/model version, scoped
output prefix/retention, full profile/task/criteria/conditions hashes and all three
named/versioned inputs. Add the reviewed checkpointing policy with `resume:null`
and keep `parameters.resume_mode="new"`. No missing SHA, image, version, deadline
or authority field is invented by this helper:

```bash
python -m learning.paused.bootstrap plan \
  --config "$REVIEWED_NATIVE_CONFIG" --binding /approved/new-preparation-binding.json \
  --raw-manifest "$ASSEMBLED_TRAIN_MANIFEST" --parent-manifest "$NEW_PARENT_MODEL_JSON" \
  --backbone-manifest "$BACKBONE_JSON" --job-name "$NEW_JOB_NAME" \
  --output "$NEW_OFFLINE_PACKAGE"
```

The package contains native `plan/job.json`, the exact code snapshot and plan SHA,
three SDK-compatible `registrations/*.json`, the binding/config, and an
`operator-sequence.json`. Registration commands include explicit subscription,
resource group and workspace but are **not executed**. Parent authorization,
private staged-payload readback, current datastore mapping/identity/compute
preflight, fresh deadline enrollment and the existing durable paid-job claim
remain mandatory. Do not bypass that path using `az ml job create`. Outputs are
the original converted dataset, candidate/checkpoint files and owner-scoped
manifest-last checkpoint publications; they are not a policy release.

A practical **proposal for operator review**, not a performance or quality
promise, is `max_steps=1000`, `checkpoint_steps=100`, `batch_size=1`,
`gradient_accumulation_steps=1`, `learning_rate=0.0001`, `seed=42`, and a
3,600-second total sequential execution ceiling. This schedules ten checkpoint
boundaries, within the default 32-checkpoint policy. The existing graph gives
conversion at most 600 seconds and training the remaining 3,000; publication
and model loading consume that budget too. Actual dataset size, conversion
runtime, training throughput, checkpoint upload cost and reviewed GPU price
must determine the final approved values. No training throughput or ability
to finish 1,000 updates within that time has been measured. A smaller 100-step
run is only an optimizer/checkpoint diagnostic, not evidence of 18/20 task
success. Select checkpoints using validation only; the frozen final twenty
cases and every physical/latency gate remain unchanged.

The simulator/native process boundary is intentional: Isaac uses its own
Python 3.12; LeRobot stays in a separately qualified Python 3.11 environment.
The native training image provides `/opt/smolvla-venv/bin/python` and `/work/learning`.
For inference use the unchanged `learning.paused.model` CLI above, with a clean
native Python environment and protected shared-kernel Unix socket/explicit peer
UIDs. Copying a venv alone into an Isaac image does not qualify it: its interpreter
symlinks, real Python 3.11 stdlib/libpython and all native libraries must work in
the final OS/driver image. Image assembly/import/ABI qualification and actual
GPU inference are runtime-owner tasks, not implied by this offline package.

### Paused evaluation evidence and release boundaries

`learning.paused.evaluation` uses paired/bootstrap **plan v2, results v3 and
report v2**. A native evaluation plan binds the actual runtime and models plus
the model-independent frozen conditions/criteria separately. Its closed
`quality_limits` are exactly 20 held-out cases, minimum success rate 0.9,
minimum absolute adaptation improvement 0.05 and zero safety violations. Seeds
30001–30020 and attempt zero are the fixed final cohort. Train conversion only
admits the separately frozen 10001–10020 and 11001–11020 cohorts; changing a
final-test split label cannot make it a training input.

`learning.paused.rollout.PausedRolloutRecorder(root, plan=..., runtime=..., grant=...)`
materializes the alternating all-attempt schedule and a distinct operator grant
v2. `record_attempt` accepts a validated raw-v3 evaluation capture, actual
per-physics-tick `TaskState` records, final camera frames and main-thread
heartbeat timestamps. It derives, rather than trusts, timing, model/reference
counts and task outcomes. Attempts must be sequential, use distinct actual
commands and stay inside the original grant. `record_failure` preserves
before-scene/camera/policy failures without inventing poses, actions or model
execution; an interrupted schedule produces `incomplete-recording.json` and
exits nonzero, never a publishable success result.

Finalization writes `results.json` last after rereading every copied capture,
PNG, grant, task trace and attempt manifest. The independent verifier checks
all source checksums and re-derives each trial, including all failures. Runtime
provenance must identify actual Azure Isaac/RTX execution, the new profile and
probe receipt; test fixtures are explicitly rejected by the live verifier.
This is still artifact consistency under the trusted runtime/operator boundary,
not proof that a self-asserted hash alone represents physical execution.

Physical success requires the existing reviewed `TaskWatchdog` semantics:
measured part lift at least 5 cm, TCP/part separation at most 9 cm and finger
gap 0.005–0.055 m sustained for 0.05 simulation seconds; then placement within
4 cm on every axis, open gap at least 0.07 m, TCP/part separation at least
0.05 m and measured part speed at most 0.02 m/s sustained for 0.3 simulation
seconds. The evaluator recomputes this from the actual trace and checks
per-physics-tick TCP speed/workspace limits. It explicitly labels this
`measured_lift_proximity_finger_gap_no_contact_sensor`, not force/contact sensing.
The independent reference implementation is source-pinned to the reviewed
runtime predicate; an actual 417-state CPU golden comparison matched its state
and outcomes without changing the runtime's control module or fingerprint.

Each report has canonical `execution_timing` / `real_time_admission=false`,
`comparison_kind`, scope/profile/runtime/criteria/frozen-plan SHAs,
`evaluation_plan_sha256`, `results_sha256`, `quality_gate_passed`, `conclusion`,
per-role `counts` / `success_rates`, `absolute_success_rate_improvement`,
`latency_wall_ms` (`samples`, `p50`, `p95`, `max`), total wall/simulation duration,
resource/safety violation counts and every trial. Trials preserve both
`wall_duration_ms` and `simulation_duration_ms`, all policy/observation/hold/
whole-interval/heartbeat wall samples and recomputed physical predicate evidence.
Paired reports identify before/after models; bootstrap reports instead identify
the real candidate and separate reference-controller hash. Bootstrap is not a
fictional pretrained before-policy.

`evaluate_pair` / `evaluate_bootstrap` validate actual trained v2 model lineage,
exclude held-out IDs/seeds from all model training ancestry, verify the complete
recording, and only then enable quality metrics. Pure `compare_trials` is a
test/analysis projection, not admission evidence. Any individual 2/2/2/5-second,
episode or heartbeat limit violation fails the gate; percentiles cannot hide it.
An adaptation is `improved` only with verified evidence, at least 18/20 candidate
successes and at least one additional success out of the same 20 cases.
Missing evidence yields an error/nonzero result, and a completed-but-failing
quality assessment writes its false report before exiting nonzero.
No new-mode report can satisfy the old real-time release catalogue.

Preflight reads **the actual separate outbound-rule endpoint** through
`client.workspace_outbound_rules.list(workspace_name=...)`. The default workspace
GET projection is insufficient. Inactive/missing approved private endpoints
block jobs. The operator must explicitly provision the managed network; preflight
never mutates networking, grants roles, enables storage keys or Internet egress.

The ordinary comparison has two genuinely trained policies P0/P1 and
`improved | not_improved | inconclusive`. First-P0 bootstrapping is a separate
privileged workflow: real teacher data -> real initial optimization -> held-out
`reference_bootstrap` evaluation -> operator publication. The scripted reference
has no model SHA and is never a fictional P0. No fabricated release/evaluation
ID is needed to authorize candidate evaluation.

Native report schemas are `physicalai.smolvla-paired-report/v1` and
`physicalai.smolvla-bootstrap-report/v1`; they contain all trials including
failures, final pose/destination, exact model IDs, counts, nearest-rank p95,
safety evidence and actual application/prediction/reference-route counters.
Frozen cases include the reviewed scene-builder hash and initial pose;
observed start positions must match within 1 mm. The held-out distribution must
span at least 2 cm, so changing defect-only seeds is not placement generalization.
Twenty complete pairs and the physical/safety/latency thresholds are required.
Learned trials require zero scripted-route calls, actual applied physics ticks
and fresh model predictions. Production report validation checks scope,
trained artifact lineage, no train/test seed/episode leakage, live Azure/Isaac
provenance and every final camera checksum. No CPU double can satisfy that path.

### Evidence boundaries and commands

`learning/smolvla/Dockerfile` builds the isolated CUDA image without vendor
weights. The parent reported actual ACR build `ch15` and digest
`sha256:914b74501c75fb8aaa60c0c5b092d640d57d1c3bef15881d1c6f1a17ca99462d`;
that is an image-build result, not GPU allocation or model accuracy.
`python -m learning.checks.smolvla_api_check` actually imports the installed
policy and runs its state/image preparation helpers without weights.
`smolvla_export_check` actually converts/reloads v3 with explicitly synthetic
input, and `smolvla_aml_check` loads all three real SDK job schemas offline.
These checks always report zero optimizer steps / no quality verification.

The model G0 command is `python -m learning.smolvla.probe --dataset ... --parent
... --backbone ... --binding ... --model-sha256 ... --conversion-sha256 ...
--output ...`, executed only by the authorized Azure operator in a bounded job.
It requires real v2 data, runs six genuine model calls with no ground-truth
action input, records driver/CUDA/memory/latency, never actuates and never
promotes the untrained initialization. Warm latency above 80 ms exits nonzero.
Actual GPU optimization, bootstrap/paired Isaac accuracy, concurrent A10
inference timing and publication remain live acceptance requirements.

#### Separately authorized vendor compatibility diagnostic

`learning/checks/smolvla_vendor_diagnostic.py` is deliberately separate from the
production probe, asset bundles and candidate validator. It may be run only as
an explicitly authorized, single existing-AML-GPU diagnostic, with at most
600 seconds of execution and a separately enforced 1,800-second allocation
deadline. It does not submit or retry itself.

The diagnostic verifies the immutable private vendor inventory and every pinned
model/backbone file, then loads the actual published Smol weights on CUDA.
It retains the vendor's **six-dimensional** state/action configuration and
**three camera slots**, native normalization and 50-step output horizon.
Inputs are explicitly synthetic fixtures, not robot observations. Only device,
disabled publishing, and verified local backbone/tokenizer paths are overridden.
Three native forward calls report shape, finiteness, latency, tensor digests and
CUDA peak memory. No optimizer or actuator is constructed or called, no
Franka-nine-dimensional adaptation is claimed, and no control-profile binding is
fabricated. Hub networking remains disabled.

The private proof always labels `observation_source: fixture`, `test_only: true`,
`optimizer_steps: 0`, `actuator_calls: 0`, and false values for
`franka_adaptation_verified`, `latency_admission_verified`,
`learning_quality_verified`, and `ready_for_live_execution`. `passed` means
only that the exact vendor weights and native API ran successfully in the
measured runtime. Import, cache, download-integrity or forward failures retain
an explicit failure proof and exit nonzero. Even a fast successful diagnostic
cannot authorize model training, physical execution or latency admission; the
real-profile/data and complete control-cycle gates are unchanged.

The single authorized run `policy-smol-vendor-compat-20260923-01` completed on
the existing A100 compute. Its independently retrieved **5,973-byte** private
proof has SHA-256
`87a4f617cc1f4ef3f31f2eef94d41feb6dc5cd702b6df1e8021b988794f6c92f`.
The dependency/code image digest is `8a8cd51f...`, while the separately supplied,
checksum-verified diagnostic harness is
`bd13b2f25508dede8c8063bce0e00669e2382fc9fa28495fc79b4de4401d5f0e`
from commit `c45693d`; the proof records both identities rather than claiming the
harness was baked into the image.

| Actual vendor-only measurement | Result |
| --- | --- |
| Device / runtime | A100 80GB PCIe, driver 580.95.05, PyTorch 2.7.1+cu126, CUDA 12.6 |
| Real parameters loaded | 450,046,176 |
| Model load | 6.335 seconds |
| Peak allocated / reserved CUDA bytes | 972,632,576 / 1,004,535,808 |
| Three finite native output shapes | `[1, 50, 6]` |
| Cold / two warm end-to-end native calls | 544.000 / 247.963 / 247.832 ms |
| Optimizer / actuator calls | 0 / 0 |

This is a **vendor-weight/runtime compatibility pass and a prospective synchronous
performance blocker**, not an 80 ms admission pass. Three timing samples do not
establish a latency distribution. The measured interval includes preprocessing,
native forward, unnormalization and synchronization, but excludes real cameras,
PNG decoding, IPC, Isaac physics and common watchdogs. It cannot be divided by
the 50 predicted actions to claim a valid one-action-per-fresh-observation
controller rate.

#### Pinned-code latency analysis; no optimization is enabled

The actual published configuration uses ten denoising steps, a 50-action horizon,
three input cameras resized internally to 512 x 512, prefix caching enabled,
`use_amp=false`, and compile/RTC disabled. In the pinned
[`VLAFlowMatching.sample_actions`](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/policies/smolvla/modeling_smolvla.py),
vision/language/state prefix processing and its KV cache occur once per call,
followed by ten sequential action-expert denoising passes.
[`SmolVLMWithExpertModel`](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
selects custom eager attention and explicitly computes its Q/K attention and
softmax in float32. Merely enabling a generic SDPA/FlashAttention setting does
not replace that custom attention implementation.

The constructor loads the VLM with bfloat16, while other modules and explicit
casts may use other dtypes. The diagnostic proof does not contain a per-module
dtype histogram, stage timings or a CUDA kernel timeline: it would be incorrect
to assert that all weights executed in one dtype, that preprocessing dominated,
or that kernel-launch overhead explains the measured 248 ms.

The first optimization candidate is a separately approved **same-precision**
`torch.compile(..., mode="reduce-overhead")` experiment on the native sampling
call, with the exact same weights, inputs, denoising count, horizon, camera
count/resolution and precision flags. PyTorch documents potential Python/CUDA
launch-overhead reductions via CUDA graphs, but not guaranteed speedups;
static-shape/memory and graph-break constraints must be measured. The upstream
`compile_model=True` convenience path also calls
`torch.set_float32_matmul_precision("high")`, so it is **not** a pure unchanged-
precision switch. Do not silently enable it and attribute any change only to
compilation. BF16 autocast, TF32/high matmul precision and alternative attention
kernels are separately declared numerical changes requiring fixed-noise output
comparisons and eventual real policy/safety evaluation.

The coordinator subsequently authorized exactly one follow-up diagnostic,
`policy-smol-vendor-profile-20260923-02`, implemented by
`learning/checks/smolvla_vendor_profile.py`. It keeps one existing GPU, at most
600 execution seconds, a 1,800-second allocation deadline and no retry. A
non-CUDA supervisor uses **spawn**, not a forked CUDA context, for separate
baseline and compile processes. The compile process has a hard 240-second limit;
its owned descendants are terminated if it overruns.

The baseline measures three warmups and 20 uninstrumented calls, then a separate
bounded CPU/CUDA profiler call. Regions distinguish preprocessing, each camera's
vision encoder, prefix KV caching, all ten expert denoising steps and
postprocessing. Parameter/activation dtypes, CPU wall spans, CUDA elapsed spans,
operator/kernel counts and launch gaps are retained. Nested spans are not falsely
summed as exclusive kernel time. The trace is limited to 8 MiB compressed and
logs to 2 MiB per process; oversize artifacts fail explicitly rather than creating
unbounded output or a successful incomplete result.

The two processes share **identical explicit safetensors input and noise
tensors**, with checksums, not merely a seed. The compile-only candidate preserves
the actual precision/matmul/TF32 flags and uses `reduce-overhead` on the native
sampling method. Native compilation's extra `high`-precision switch is not
enabled. Numerical comparisons require finite `[1,50,6]` outputs and fixed
`rtol=1e-3`, `atol=1e-4`; failures cannot relax those values. Actual graph
counters, graph breaks, eager regions, compiler logs, compile/warmup time,
20 steady latencies and CUDA memory are reported. No captured graph means
failure, not an eager fallback disguised as optimized success. Timeout,
numerical mismatch and import/cache errors retain partial diagnostic proof and
exit nonzero. The harness's local spawn/timeout tests are not GPU profiling
evidence; actual outcomes must come from the separately pinned job proof.
Even a measured speedup would not admit the real nine-DOF/two-camera/Isaac
controller.

The `...profile-20260923-02` attempt actually failed with AML
`UserTrainingCommandFailed / Bad Request`; reported start/end times were equal.
The parent confirmed the private profile proof was absent, and the available
identity could not read the default AML diagnostic files. Exact submitted
Python/shell syntax and the runpy/spawn launcher passed local self-tests, but
those results do **not** establish the cloud failure's cause. No baseline profile
or compiled speedup was obtained, and the failed job was not resubmitted.

The parent explicitly authorized one distinct packaging alternative,
`policy-smol-vendor-profile-20260923-03`: the **same** model workload, parameters,
precision, tolerances and budgets, with an immutable code-layer image and a
short `python -m learning.checks.smolvla_profile_entry` command instead of the
21 KiB inline source. `learning/checks/Dockerfile.smol-profile` includes only
the reviewed diagnostic files and a hash-pinned configuration template.
The entry point verifies the template and final configuration hashes; the sole
metadata substitution is the new immutable image digest, which cannot be
embedded into its own image. Direct-module spawn and timeout cleanup are checked
locally and during image build without loading weights or CUDA. This alternative
is a new bounded operator decision, not an automatic retry or evidence that the
original `Bad Request` was definitely caused by command length.

The actual `...profile-20260923-03` alternative was still `Queued` when resumed.
Its original allocation deadline was `2026-09-23T14:24:30.310547Z`; the attached
local monitor was absent, and the explicit cancellation request at
`14:27:28.286266Z` was approximately **178 seconds late**. Both ARM and RunHistory
subsequently confirmed `Canceled`, not a successful profiling run. The parent
confirmed the exact private proof path was absent. Read-only reconciliation saw
compute current/target nodes `0/0` with state `Resizing`, and no active recent
owned diagnostics. No stage profile, compiled forward, numerical comparison,
or original fixture SHA was obtained; therefore no fixture reconstruction was
performed. The first compatibility diagnostic's approximately 248 ms A100
measurements remain separate from this canceled attempt. This observed
monitor-lifetime gap motivated the native expiry contract above.

A subsequent source-only audit found that the original uploader had incorrectly
applied its 2 MiB **log** limit to the explicit shared fixture safetensors file.
Actual CPU serialization of the unchanged two-fixture workload is **4,732,400
bytes**. The correction leaves logs at 2 MiB and the compressed trace at 8 MiB,
caps JSON/config artifacts separately at 256 KiB, and derives the fixture limit
from two float32 sets of three `3x256x256` images, six state values and `1x50x32`
noise values, plus a 64 KiB metadata allowance: **4,796,976 bytes**, not an
unrestricted size increase. Unknown artifact classes and model-weight files
cannot use this allowance. A secondary artifact-limit failure never replaces a
primary compile/numerical error and still prevents overall success.

That fix does **not** patch an already submitted immutable image or authorize a
new job. If a failed run contains real partial stage/timing proofs, they must be
reported as partial. Missing synthetic fixture bytes may only be reconstructed
with the unchanged generator and labeled as reconstructed **after exact SHA
equality to the recorded original**; no forward metrics, stage timings or
successful result may be generated from reconstruction.

The source-only profiling entry now requires
`physicalai.smolvla-vendor-profile/v2` with an explicit, immutable
`start_deadline_utc`. It verifies the actual entry, profiling harness and
deadline-helper code hashes, and rejects expiry before identity/private asset
access and again in each spawned phase before loading weights. The v2 proof
records the exact UTC cutoff, its canonical-object SHA, each actual admission
check and all code/config/image pins. Historical v1 specifications remain
inspectable, not executable as new authority. This is a start-admission guard,
not a claim that a local observer is a durable Azure queue-cancellation service.
The proposed H100 `...profile-h100-20260923-04` build/submission authorization was
**suspended before any build or job** when the coordinator selected a separate
paused-simulation workstream. No H100 profiling or compiler success is claimed.

Official [asynchronous inference](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/docs/source/async.mdx)
and [real-time chunking](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/docs/source/rtc.mdx)
do exist. They are **different control semantics**, not an escape from the
current one-action freshness/deadline rules. Adopting them requires a new
versioned control profile and operator authority covering queue horizon,
observation age, predicted/consumed step alignment, cancellation/scene/owner
invalidation, underflow/late-result handling, per-tick safety and any RTC guidance.
It also requires explicit data/profile compatibility review and new complete
held-out physical evaluation. Existing manifests, timing attestations or releases
cannot be relabeled; the current 80 ms inference and 100 ms cycle guards remain.

References: [PyTorch 2.7 compile modes](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html),
[CUDA graph constraints](https://docs.pytorch.org/docs/2.7/torch.compiler_cudagraph_trees.html),
[profiler measurement overhead](https://docs.pytorch.org/docs/2.7/profiler.html),
and [float32 matmul precision](https://docs.pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html).

### Physical evidence producer: runtime execution, not an AML gate job

`learning.smolvla.components compare/bootstrap` **does not execute physics**.
Its input must now be a recorded
`physicalai.smolvla-paired-results/v2` or
`physicalai.smolvla-bootstrap-results/v2` collection. A legacy hand-assembled
`results.json` cannot satisfy that production verification path. The outward
paired/bootstrap **report** schemas stay v1 so the worker's checked projection
does not need invented API fields.

The actual operator-owned runtime must run every frozen case through its
approved reference or real model controller. The runtime owner supplies the
bounded physics/model episode loop (`run_evaluation_episode(grant, case, sink)`);
the coordinator supplies the frozen plan, explicit evaluation-only operator
grant, schedule and resource orchestration. Neither role may pretend the
vendor initialization is a released P0. Candidate evaluation authorization uses
an `evaluation_run_id`, never a fabricated production `policy_release_id`.
No new HTTP actuator endpoint is implied by this Python integration.

`learning.smolvla.rollout.PhysicalRolloutRecorder` is a **pure evidence sink**.
It never grants authority, loads model weights, chooses a model path, calls
Isaac, moves a robot, or synthesizes missing measurements:

```python
recorder = PhysicalRolloutRecorder(
    new_owner_scoped_output_directory,
    plan=frozen_native_plan,
    runtime=actual_pinned_runtime_identity,
    grant=operator_evaluation_grant,
    expected_plan_sha256=approved_native_plan_sha256,
    expected_grant_sha256=independently_approved_grant_sha256,
)

# Called by the actual runtime; not by a JSON report generator.
recorder.start_case(binding, initial_measured_state, initial_camera_frames)
recorder.append_control(actual_control_trace)  # One or more real counter/timing snapshots.
recorder.finish_case(
    actual_terminal_outcome, final_measured_state, final_camera_frames,
    actual_uploaded_capture_manifest_bytes,
)
results_path = recorder.finalize()  # Only after every planned attempt is complete.
```

The grant is `physicalai.operator-rollout-grant/v1`, with
`purpose: paired_policy_eval | reference_bootstrap`, `evaluation_run_id`,
tenant/opaque-owner `scope`, `operator_principal_sha256`, exact
`plan_sha256`, `runtime_sha256`, `control_profile_sha256`, approved
`task: {task_id, instruction, goal_id}`, UTC `issued_at_utc` / `expires_at_utc`,
`max_episode_seconds <= 30` and `max_total_seconds <= 3600`. The actual
authorization check happens in the runtime's trusted operator/provider boundary;
hashing a self-written grant is not permission to actuate. All per-episode
budgets are equal, bounded by the same grant window and cross-checked against
monotonic deadlines. Source/runtime identity includes the actual pinned image,
source commit, GPU/Isaac and robot asset provenance.

The exact stdlib dataclasses are:

| Type | Actual runtime fields |
| --- | --- |
| `CaseBinding` | `evaluation_run_id`, purpose, scope, role (`policy`), model SHA or null for reference, frozen episode/attempt/environment/revision/seed/builder/goal, actual command ID/epoch, profile/runtime SHAs, UTC/monotonic deadline, explicit `policy_type="smolvla"` |
| `MeasuredState` | UTC timestamp, monotonic nanoseconds, physics step, measured object position |
| `ControlTrace` | Command/epoch, actual monotonic/physics step, cumulative prediction/application/reference counters, actually applied model SHA, **new** latency and whole-cycle measurements, last actual application timestamp, actual safety violations |
| `TerminalOutcome` | Actual terminal status and reason, selected destination, capture ID, exact private capture-manifest URI/SHA and frame count |

`camera_frames` is exactly `dict[str, CameraSample]` for inspection/overview.
Initial/final images must belong to their actual measured physics step, have
real monotonic capture times and advancing native renderer identities. The
actual uploaded manifest must bind the same owner, command/episode, environment,
revision, held-out seed, source runtime and controller. Learned evidence requires
v2 capture with the exact model/task/profile; reference v1 capture remains
separate from the learned profile.

The schedule is fixed before execution and alternates role order per case:
before/after then after/before (or reference/candidate then candidate/reference).
The read-only schedule contains every planned `(role, episode_id, attempt)`.
No automatic retry, omission of failed trials or post-hoc successful subset is
allowed. Actual completed failures/cancellations/timeouts with full measurements
remain in the trial counts. A scene/camera/capture failure without measurements
calls `abort_case(phase, error)`: its incomplete record is preserved, and
`finalize` emits a failed `collection.json` but **no** publishable `results.json`.
It does not fill missing initial/final poses or cameras with zeros.

Use one persistence thread (the existing bounded runtime worker is appropriate)
and pass immutable snapshots from the simulator thread. Persist the initial
record before motion; do not make PNG/fsync work part of the timed physics loop
or silently discard trace events to meet timing. Finish with the measured stop
state after capture publication confirms the real manifest. Files include
`start.json`, hash-chained `control.jsonl`, `terminal.json`, the exact
`capture-manifest.json`, real initial/final PNGs and a final inventory. The native
gate rechecks every checksum, trace chain, authority/case/deadline binding,
application count and summary against those source artifacts.

A 20 ms model call inside a 110 ms total control cycle records a timing
violation and cannot pass quality admission. The recorder never subtracts a
single mechanics sample from 100 ms to invent a future inference budget.
Model identity becomes applied only after actual actuation; cumulative counters
cannot go backwards, exceed actual physics ticks or cross commands/epochs.
Cryptographic consistency establishes artifact integrity, not physical truth:
the trusted runtime and independent Azure execution receipt remain necessary.
Unit tests use explicitly labeled synthetic observations and test attestations;
they do not demonstrate that this new physical rollout producer has executed
on Azure or that any candidate has passed it.

This is the learning slice of the existing customer inspection cell, not a second
simulator or a replacement controller. It supports pinned Isaac Sim **5.1.0/6.0.0 Franka**
articulation, the existing customer environment revision and reviewed scene
builder, both real cameras, and all **nine** joints. `learning/` is deliberately
independent of the API, UI, and Isaac SDK. Only its capture/contract modules need
to be imported in the simulator.

**Production runs on Azure. WSL/CPU fixtures are development checks only.**
Neither a successful converter nor a decreasing training loss demonstrates
sorting accuracy. A checkpoint remains a candidate until actual paired,
held-out Azure GPU Isaac episodes pass the separate learning gate. There is no
automatic baseline fallback, Hugging Face publication, model promotion, Azure
resource provisioning, or license acceptance.

The pinned CPU runtime has also been built and executed in **Azure Container
Registry Tasks**. `learning/Dockerfile.smoke` layers the explicitly test-only
CPU check onto a digest-pinned `LEARNING_CPU_IMAGE`, restores the declared Azure
SDK extra, and runs conversion, one genuine ACT optimizer step, checkpoint reload
and inference. This cloud check submits no Azure ML job, uses no GPU and cannot
pass the held-out physical/learning-quality gate. Its stdout report labels the
fixtures and records those limitations; retain the ACR run ID and image digest
alongside it.

## Capture contract and simulator integration

The GR00T teaching path uses **`physicalai.demonstrations/v2`**, not a relabeled
v1 baseline recording. `EpisodeWriter(..., control_profile=ControlProfile(
servo_profile_sha256=...), demonstration=DemonstrationSource(kind, task_id,
instruction, goal_id, source_policy_sha256=None))` selects v2. Omitting both
arguments preserves v1 exactly. The v2 profile `franka-position-hold-10hz-v1`
requires 10 Hz control / 60 Hz physics, six held absolute nine-joint targets,
zero actual target velocities and arm-only measured PhysX gravity compensation.
Its SHA is `digest(canonical(asdict(profile)))`; the servo digest identifies the
reviewed shared runtime implementation, not an assertion of measured timing.

`FrameSample.applied_controls` must contain six `AppliedControl(physics_step,
monotonic_ns, commanded_joint_targets, commanded_joint_velocities, gravity_efforts)`
values from actual following physics ticks. Sample observation/images first,
apply the command, collect each real tick, then send the completed immutable
batch to the writer's owning thread. Changed intervening targets, missing ticks,
nonzero target velocities or finger gravity efforts invalidate the recording.
Partial/cancelled intervals are not padded into training examples. Runtime
timing approval and physical guards remain outside the writer. The unchanged
60 Hz reference controller remains v1; the GR00T exporter rejects it.

`physicalai.demonstrations/v1` is a closed, versioned JSON contract, implemented
by `learning.contract`. Unknown fields are rejected, including rewards, defect
labels, success labels, or hidden evaluator files.

| Artifact | Required evidence |
| --- | --- |
| `manifest.json` | Dataset ID, tenant UUID, opaque owner key, FPS, physics rate, exact joint order and units, nonempty episode inventory |
| Episode entry | Episode ID, customer environment ID and content-hash revision, scene seed, split, frame count, JSONL checksum, immutable capture provenance |
| `episodes/<id>/frames.jsonl` | Sequential index, UTC `Z` timestamp, strictly increasing monotonic nanoseconds and physics steps, measured positions, effective issued joint targets, two images, terminated/truncated flags |
| `episodes/<id>/<camera>/<index>.png` | Actual noninterlaced RGB simulator PNG, SHA-256, dimensions, rendering-frame index, camera physics step and monotonic timestamp |

Joint order is `panda_joint1` through `panda_joint7`, then
`panda_finger_joint1`, `panda_finger_joint2`. The first seven entries are radians;
the last two are individual finger positions in metres, **not** a combined
gripper width. Capture verifies the reference joint limits as well as finite
values and dimensions.

Provenance includes the Isaac version, digest-pinned simulator image, robot
asset checksum, reviewed scene-builder ID/checksum, source commit, actual GPU
model, Azure capture host, and RTX renderer. An environment JSON file is data;
it cannot name an arbitrary Python module to execute. Python scene extensions
continue to enter through the existing reviewed `SceneRegistry`, not through a
dataset. Record the installed builder's reviewed package/source digest, not a
hash of a caller-supplied module name.

The simulator integration must:

1. Obtain scope from the authenticated lease: `Scope(tenant_id, owner_key)`.
   `owner_key` is the existing opaque 64-hex SHA-256 of tenant/object identity;
   no raw user object ID is needed. Never query observations owned by another
   lease. Verify environment, epoch, approval and `core.begin_motion` first.
2. On the Isaac main thread, preserve the **actual previously issued** complete
   targets and resolve sparse `ArticulationAction` updates using
   `resolve_joint_targets(previous, positions, indices)`. Pass Python tuples or
   `.tolist()` values. Until every target is known, this raises; do not fill
   unknown targets with measured joint positions or zeroes.
3. Capture the observation immediately before the issued command's control
   interval. Joint state and images belong to that observation; targets are
   the command actually applied for the following interval. Do not record a
   post-action observation against a preceding action. Hold recorded targets
   for `60 / fps` physics steps. Do not downsample away intermediate commands
   and then claim the remaining target was held.
4. Call the writer with actual RGB PNG bytes and actual camera frame identity,
   not UI screenshots, placeholders, object-state drawings, or synthetic
   observations. Images must advance at the capture rate and be less than one
   control interval old in simulation time, at most one second old in wall
   time. Both cameras must have the same dimensions.
5. Record a final sample with exactly one of `terminated` or `truncated`, then
   finalize. `terminated` means the controller ended, **not** that the success
   predicate passed. Cancellation, timeout, scene failure and safety stops are
   truncations, with the detailed reason in evaluator-only evidence.

The default 10 Hz capture rate divides the reference 60 Hz physics rate.
**Do not change the normal baseline controller cadence silently.** A deliberate
capture profile must align controller holding and camera frequency; alternatively,
60 Hz capture requires genuinely fresh 60 Hz camera frames. Missing frames,
unknown targets or cadence violations invalidate the episode rather than being
interpolated. The API's current preview publishing rate is not a demonstration
clock.

```python
from pathlib import Path
from learning.capture import EpisodeWriter, resolve_joint_targets
from learning.contract import CameraSample, EpisodeSpec, FrameSample, Provenance, Scope

# All identifiers/digests below come from the active, reviewed simulator deployment.
writer = EpisodeWriter(
    Path(per_run_directory),  # Must be a new, owner-scoped local Azure VM folder.
    dataset_id=dataset_id,
    scope=Scope(tenant_id, owner_key),
    episode=EpisodeSpec(episode_id, environment_id, revision, seed, split),
    provenance=Provenance(
        source_kind="isaac_sim",
        simulator_version="5.1.0",
        simulator_image_digest=simulator_image_digest,
        robot_asset_sha256=robot_asset_sha256,
        scene_builder_id=template_id,
        scene_builder_sha256=reviewed_builder_sha256,
        code_revision=source_commit,
        capture_host="azure_gpu",
        gpu_model=actual_gpu_name,
    ),
    fps=10,
    physics_hz=60,
    max_frames=3600,
    max_bytes=512 * 1024 * 1024,
)

targets = resolve_joint_targets(previous_issued_targets, action_positions, action_indices)
writer.append(FrameSample(
    captured_at_utc=utc_timestamp_ending_z,
    monotonic_ns=observation_monotonic_ns,
    physics_step=observation_physics_step,
    joint_positions=tuple(float(value) for value in actual_joint_positions),
    commanded_joint_targets=targets,
    images={
        "inspection": CameraSample(inspection_png, inspection_render_frame,
                                   inspection_physics_step, inspection_monotonic_ns),
        "overview": CameraSample(overview_png, overview_render_frame,
                                 overview_physics_step, overview_monotonic_ns),
    },
    terminated=controller_ended,
    truncated=controller_was_interrupted,
))
manifest_path = writer.finalize()  # Only after a real terminal sample.
```

The writer is single-threaded and bounded; budget for the terminal frame.
Errors propagate and leave the folder unapproved. The parent upload adapter
must independently run
`validate_dataset(root, expected_scope=scope, require_live=True)` before any
upload, then use its existing Azure managed identity and private Blob routing.
Upload authorization is outside this module. A folder's mere existence is not
permission to upload it.

`assemble_dataset(episode_roots, destination, dataset_id=..., expected_scope=...,
require_live=True)` combines finalized runs. Choose train/validation/test
assignment **before** collecting data. Duplicate episode IDs and any scene seed
appearing across splits are rejected, even across different environment
revisions. A converted training dataset contains only the train split.
Symlinks, path traversal, drive/UNC paths, extra files, empty sets, corrupted
PNGs, missing/nonfinite values and mismatched checksums are rejected.

## Pinned real LeRobot path

### GR00T N1.5 compatibility implementation (customer use blocked)

The initial compatibility implementation uses **GR00T N1.5**, pinned to NVIDIA source commit
`4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` and model
`nvidia/GR00T-N1.5-3B` revision `869830fc749c35f34771aa5209f923ac57e4564e`.
No N1.6/N1.7 API or SO-101 six-axis mapping is substituted. The isolated
`learning/gr00t/pyproject.toml` / `uv.lock` uses Python 3.11, NumPy 1.26.4,
PyArrow 14.0.1, PyAV 12.3.0 and an optional pinned upstream runtime; neither
the root nor ACT dependency lock is changed.

`learning.gr00t.dataset.export_dataset(raw, output, expected_scope=...,
expected_manifest_sha256=..., allow_test_fixture=False)` validates v2 raw
capture and exports **LeRobot v2.1**, with episode Parquet files, real H264
camera videos, `meta/info.json`, `modality.json`, `stats.json`, episodes/tasks
JSONL, and a checksum-bound `export.json`. No resampling is performed: every
10 Hz sample survives. Videos are decoded again to verify exact frame counts
and timestamps. The train split alone enters optimization. Seed, scene revision,
demonstrator kind, termination and source counts remain provenance, not policy
features. The legitimate approved task instruction is a model input; evaluator
defect/success labels and object poses are not.

The fixed `FrankaDataConfig` maps `state.arm/action.arm` to seven radians and
`state.fingers/action.fingers` to two individual metre-valued finger positions,
with `video.inspection`, `video.overview` and the approved task instruction.
GR00T's sixteen-action horizon is padded internally to 32 dimensions by the
upstream transform; the returned actuator contract is strictly **16 x 9**.
Only one predicted action is executed per fresh 10 Hz observation. This is
not a claim that a 3B policy already meets the 80 ms inference budget.

`learning.gr00t.train.run_training` calls the pinned upstream
`LeRobotSingleDataset`, `GR00T_N1_5.from_pretrained`, and `TrainRunner.train`;
there is no local replacement optimizer loop. It freezes the LLM/vision tower
and tunes the projector/diffusion action model, uses bounded BF16 single-GPU
training, checks the actual running Azure ML job through the injected SDK client,
and rejects CPU or fixture inputs. It does not synthesize `AZUREML_RUN_ID`.
Actual optimizer steps, changed trainable-parameter samples, weight inventories,
dataset/source/config/parent hashes, GPU identity and Azure job ID are persisted.
Checkpoints are exported incrementally as safe candidate bundles. On Spot,
checkpoint interval must be less than the step budget and at most 100 steps;
the output must be `rw_mount`, not upload-on-success. Continuation is explicitly
**weights-only in a new authorized job**, with parent job/model/step lineage and
reset optimizer/scheduler; arbitrary optimizer pickle is never loaded.

`physicalai.gr00t-checkpoint/v1` separates a pinned official pretrained artifact
from an actually Franka-trained candidate. The public pretrained model cannot
execute as P0: it lacks customer/new-embodiment training evidence. P0 and P1
must each be real trained policy artifacts. Model loading checks scope, profile,
task, exact upstream revision, complete safetensors inventory, dimensions,
normalization metadata, observed weight update and actual job provenance.
Checkpoint-provided Python, pickle, `auto_map` or arbitrary processor code is
rejected. N1.5's own Eagle loader internally uses Transformers' dynamic loader
for its **bundled local** code; the entire upstream checkout must match the
fixed clean commit before any GR00T import, and a fresh private module cache is
used. No remote code URL or caller-specified class is permitted.

The runtime imports only `learning.gr00t.ipc.SocketChunkPolicy` and
`RemoteGuardedPolicyAdapter`; no Torch/GR00T package is loaded in Isaac.
Configure the socket path, scope, model manifest SHA, `ControlProfile`, and
approved task from a deployment-owned catalog, never an HTTP-supplied path.
The adapter implements the existing `reset(context)`, `step(observation,
context)`, `stop()` interface; `policy.metadata` exposes the deployment-bound
scope/model/task/control-profile information.

The separate GPU process is started by the authorized deployment with:

```bash
python -m learning.gr00t.inference \
  --model-root APPROVED_LOCAL_MODEL --source-root PINNED_UPSTREAM_CHECKOUT \
  --socket-path /run/physicalai-policy/policy.sock \
  --tenant-id APPROVED_TENANT --owner-id OPAQUE_OWNER_SHA256 \
  --model-sha256 APPROVED_MODEL_MANIFEST_SHA256 \
  --control-profile-sha256 APPROVED_PROFILE_SHA256
```

IPC is same-host Linux Unix-domain sockets, not a remote HTTP model endpoint or
pickle/ZMQ object stream. A four-byte network-order length prefixes bounded
strict JSON (maximum 16 MiB). Requests bind sequence/request ID, tenant/owner,
environment/revision, episode, epoch, command, goal, approval, deadline, model,
profile, task and exact observation. PNGs carry base64 bytes plus checksums and
real frame/timestamp evidence. Responses echo all binding hashes and return
only denormalized finite bounded 16 x 9 targets plus measured inference latency.
Replay, wrong goal/model/profile, stale input, nonfinite/bounds/slew violations,
socket timeout, or total inference latency above 80 ms stops the request.
No reference route, clipping, ACT substitution or anonymous socket access is
provided. Runtime must still verify current approval and actually applied model
SHA immediately before actuation, with its independent safety watchdogs.
The absolute deadline starts at adapter step entry and is shared by connect,
send, every partial receive, JSON decoding and response validation; a dribbling
peer cannot renew the budget. Linux `SO_PEERCRED` verifies both ends before any
packet is accepted. Defaults require the same effective UID; differing container
UIDs require deployment-only `SocketChunkPolicy(expected_peer_uid=...)` and
server `--allowed-client-uid` pins. `policy.predict_calls` is a read-only count
of IPC prediction **attempts**, not successful server inference; failed
connections still fault and cannot count as applied policy actions.

The isolated check `python -m learning.checks.groot_export_smoke --output NEW_DIR`
performs actual Parquet/MP4 export on explicitly synthetic CPU fixtures; it is
not GR00T training or Isaac accuracy evidence. `learning.checks.groot_source_check`
checks real pinned-source API signatures without importing model code.
GPU dependency/image execution, official-weight acquisition with license review,
actual optimizer runs, measured peak memory/latency and paired physical rollouts
remain separate required checks. Start capacity planning with an explicitly
approved 48 GB-or-larger training GPU, not a guarantee of minimum memory.
The A10 renderer's available memory/latency for concurrent inference is not
assumed. No model weights were downloaded or license accepted by these checks.

The **held, N1.5-only** GPU build recipe is `learning/gr00t/Dockerfile`: digest-pinned
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, frozen isolated runtime/Azure extras,
FlashAttention 2.7.4.post1 compiled from its locked source distribution, and the
exact upstream checkout at `/opt/isaac-gr00t`. It targets Ampere A100/A10
(architectures 8.0/8.6); do not assume Blackwell compatibility. The image contains
no model weights and accepts no license. An authorized Azure builder may build
it with `-f learning/gr00t/Dockerfile` and then pin the resulting ACR digest.
Local dependency resolution is not a completed GPU image build or GPU proof.

The following weight/probe commands document the implementation but are
**currently rejected by the license gate**, including when the acknowledgement
flag is supplied. Do not build a customer workflow around the held model.
Once an unambiguous compatible model revision is separately approved, the
operator must acquire its
exact three shards plus metadata into approved private Azure storage,
`python -m learning.gr00t.prepare --source LOCAL_VENDOR_FILES --output NEW_BUNDLE
--binding APPROVED_BINDING_JSON --acknowledge-license-review` checks the
published immutable shard SHA-256s and metadata Git blob IDs before copying them
into a scope/task/profile-bound **train-only** artifact. Binding JSON contains
`scope`, `control_profile` and `task`. This importer performs no download,
upload, token lookup or license acceptance itself.

On the explicitly allocated Azure GPU, the bounded G0 command is:

```bash
python -m learning.gr00t.probe \
  --dataset APPROVED_V21_EXPORT --export-sha256 EXPORT_MANIFEST_SHA256 \
  --model-root APPROVED_VENDOR_BUNDLE --model-sha256 MODEL_MANIFEST_SHA256 \
  --source-root /opt/isaac-gr00t --binding APPROVED_BINDING_JSON --output NEW_REPORT_DIR
```

Wrap this command in an approved one-node AML command job with a finite timeout
(for example 600 seconds) and explicit private mounted inputs/output. It requires
actual Azure-origin v2 observations, not generated model inputs; no valid capture
means the full probe remains blocked. It loads the real pinned vendor weights,
records driver/CUDA/GPU/peak memory, and makes six actual model calls on real
camera/joint observations (one cold, five warm). The probe temporarily uses
the real dataset normalization to measure the **unadapted** vendor model; it is
not a trained Franka P0. `probe.json` always says zero optimizer steps,
`ready_for_live_execution: false` and `learning_quality_verified: false`.
Latency excludes IPC and is only a necessary sub-budget measurement; exceeding
80 ms exits nonzero. No actuator is called, even when predicted joints happen
to fall inside reference bounds.

### Auxiliary ACT path

The isolated `learning/pyproject.toml` and `learning/uv.lock` pin
**LeRobot 0.4.4**, **Python 3.11**, **PyTorch 2.7.1**,
**torchvision 0.22.1**, and **torchcodec 0.5.0**. CPU and CUDA 12.6 wheels are
mutually exclusive extras. The root API environment and its lock are unchanged.
This version was selected against the published 0.4.4 API and its Python
requirements, rather than assuming newer LeRobot releases support Isaac's
Python version or the same dataset/processor interfaces.

Conversion uses `LeRobotDataset.create`, `add_frame`, `save_episode` and
`finalize`, producing a real v3 image/Parquet dataset. It preserves episode
boundaries and a checksummed `conversion.json` sidecar with source episode IDs,
seeds, end flags, revisions and provenance. Policy inputs are only
`observation.state` and `observation.images.inspection/overview`; the output is
nine-dimensional absolute `action`. Both cameras use the same RGB, bilinear
square-resize transform. No success/defect label, target label, seed or revision
is a model input. A constant task description contains no per-episode answer.

Training invokes the installed, pinned
`python -m lerobot.scripts.lerobot_train --policy.type=act` entry point.
It does **not** implement an imitation train loop. ACT starts from scratch:
`pretrained_backbone_weights=null`, no external pretrained checkpoint,
`push_to_hub=false`, W&B disabled, offline Hugging Face/datasets mode, and
telemetry/implicit token use disabled. HF/W&B credential environment variables
are rejected. The training wrapper requires actual CUDA and an Azure ML run ID
unless the explicit bounded CPU smoke option is used.

Each candidate contains `model.json`, the final real `checkpoint/` with
safetensors/config/saved normalization processors, and a training log.
Its immutable provenance binds the raw/converted data hashes, exact training
episodes/seeds, model name/version, steps, seed, source snapshot, device and
Azure ML run. Inference verifies these checksums and never downloads a missing
checkpoint. No pickle or arbitrary checkpoint processor class is loaded.
Pinned code and seeds aid reproducibility; upstream training uses CUDA kernels
and does not guarantee bit-identical GPU runs.

## Guarded inference and paired evaluation

`LocalACTPolicy(model_root, expected_scope=..., expected_model_sha256=...,
device="cuda")` loads actual weights and the saved pre/postprocessors. It maps
RGB images to float CHW tensors, normalizes measured nine-joint observations,
predicts ACT action chunks and denormalizes actions back to radians/metres.

The simulator calls `GuardedPolicyAdapter.reset(ControlContext(...))` only
after its existing approval/controller guard, then
`step(PolicyObservation(...), current_context)` at the configured control rate.
It receives a `JointCommand` with nine targets, physics-step binding, hold
duration, expiry, checkpoint digest and measured inference latency.

The adapter does not actuate anything or authorize a controller. The parent
must still verify the current guard/deadline immediately before applying the
command, enforce the Cartesian-speed/payload/workspace watchdogs, hold targets
for the declared steps, and stop on cancellation or error. Joint limits alone
are not a collision or Cartesian-speed guarantee. Never expose an unguarded
HTTP endpoint accepting joint arrays.

The default approved horizon is one action per observation. Longer chunks need
explicit bounded `SafetyLimits` and matching checkpoint metadata. Scene,
epoch, command, destination or owner changes invalidate buffered actions.
Skipped/repeated control ticks, repeated/stale cameras, invalid dimensions,
nonfinite/out-of-bounds actions, excessive slew/tracking error, command expiry,
or latency overrun latch a fault until an approved reset. Outputs are rejected,
not clipped or silently replaced with baseline motion. The baseline remains the
existing guarded Isaac `PickPlaceController`; it is never labeled a learned
policy.

`build_evaluation_plan` binds the whole test split, checkpoint SHA, baseline
source commit, expected Azure simulator resource/runtime and thresholds.
`EvaluationCase` holds the evaluator-only defect label, expected destination
and destination pose. Keep this plan away from inference observations.
Validation seeds are for model selection; test cases are not training inputs.

`EvaluationRecorder` records both baseline and learned runs on exactly the
same held-out episode IDs, revisions and seeds. Every outcome includes final
object pose, chosen destination, real terminal/truncation state, explicit
failure/safety evidence, elapsed time, exact control/physics counts, one latency
measurement per control step, and checksummed final camera frames. An incomplete
paired run cannot publish a complete results manifest.

`evaluate_results` recomputes success: normal termination, no failure or safety
violation, correct destination, and final object pose within the planned station
volume (at most 4 cm on each axis). It emits counts, all failures, nearest-rank
p95 latency, paired baseline comparison and explicit rejection reasons.
Defaults require at least **20** held-out episodes per controller, 90% learned
and baseline success, no more than 5 percentage points regression, p95 at most
80 ms, and **zero safety violations**. CPU/test fixtures never pass the live
learning gate, even with 100% apparent fixture success. Missing evidence causes
a nonzero command exit, not a skip.

Hashes verify identity/integrity, not physical truth. Runtime attestation must
come from the parent's trusted live Azure acceptance harness and actual device
queries. A handwritten JSON file claiming `isaac_sim` is not independent proof
that Isaac ran. The release process must bind the approved plan/runtime and
check the live acceptance evidence separately; this slice never self-promotes
a checkpoint or grants runtime controller approval.

## Azure ML v2 plans, submission and retention

`infra/learning.bicep` first provisions the private workspace/endpoint with
`provisionCompute=false`. After the deploying owner verifies narrowly scoped
workspace self-private-endpoint approval permissions, an explicit second apply
with `provisionCompute=true` creates the selected `Dedicated` or `LowPriority`
single-node compute (minimum zero, maximum one, no public node IP, SSH/local
auth disabled). It does **not** submit a job or grant permissions.
Supply `workspaceName`, `location`, existing
`storageAccountId`, `keyVaultId`, `containerRegistryId`, `applicationInsightsId`, `workspaceIdentityId`,
`computeIdentityId`, `privateEndpointSubnetId`, the two prelinked
`privateDnsZoneIds` (`privatelink.api.azureml.ms` and
`privatelink.notebooks.azure.net`), `computeName`, `computeSize`, and
`computeTier`. The deploying owner must authorize the identities, private
endpoint connections, DNS links and approved-outbound managed network
provisioning first; quota does not prove allocation capacity. Optional
`compute_tier` in legacy ACT plans is verified against actual compute when
present; GR00T plans require it. Do not run the legacy final-checkpoint-only
ACT training path on Spot as a substitute for resumable GR00T training.
The workspace uses API **2025-06-01**, whose published schema supports
`systemDatastoresAuthMode: Identity`; the original 2024-04-01 schema does not
expose that setting. Existing Application Insights is bound by resource ID,
not silently created. Registry Private Link requires an approved supporting
ACR SKU (Premium); this template does not upgrade it or relax any firewall.
Associated Blob/file/Key Vault/ACR outbound private-endpoint rules are generated
by the workspace provider. The template deliberately does not duplicate them as
user-defined rules: the real service rejects duplicate destinations even when
ARM validation passes. After managed-network provisioning, the deployment/worker
must call `validate_managed_network_dependencies(workspace)` and verify the
actual bound dependency destinations are active/approved. A missing endpoint is
an explicit blocker, never permission to switch to Internet outbound.

`learning.azure` generates real Azure ML v2 pipeline job JSON (also valid YAML)
and a deterministic source snapshot. The training graph is **convert -> train**.
A separate **validate_evidence** graph checks the actual held-out rollout
artifacts. The gate job does not pretend to run Isaac inside the training image.
Run physical rollouts in the existing approved Isaac GPU bridge, publish their
evidence to the approved private datastore through the parent adapter, then
submit the evidence gate. Register/promote a model only after both learning and
live release gates pass.

`learning/Dockerfile` pins the official Python base by digest and installs the
frozen learning lock. Build the CUDA image only after applicable license review,
push it to the approved ACR through the authorized deployment workflow, and use
its **resulting ACR digest** in the plan. The sample T4 SKU is a placeholder,
not a quota, price, hardware-compatibility or deployment claim. CUDA 12.6 /
PyTorch 2.7.1 has not been validated here on RTX PRO/Blackwell SKUs; do not assume
new GPU families work without a separately reviewed image and GPU check.

Start from `learning/examples/azure-train.json`; replace every placeholder.
The planner has no default subscription or anonymous/default identity. It
requires explicit subscription, tenant, resource group, AML workspace, GPU
compute/SKU, user-assigned managed identity resource/client IDs, identity-based
datastore, Blob account/container, owner-scoped paths, immutable data/model
versions, ACR digest, steps/timeouts and retention days. No secrets or SAS URLs
belong in the configuration.

Inside WSL for development/administration, or an approved Azure build runner:

```bash
source scripts/dev-env.sh
export UV_PROJECT_ENVIRONMENT="$PHYSICALAI_WSL_CACHE/learning-cpu"
uv sync --project learning --extra cpu --extra azure --frozen

# Offline by default: creates local JSON/code artifacts, no Azure SDK call/upload/job.
uv run --project learning --extra cpu --extra azure --frozen \
  python -m learning.azure --config approved-learning.json --plan-dir reviewed-plan
```

Review `plan.json`, `job.json`, and `code/snapshot.json`. The fixed code allowlist
contains only the learning runtime, Dockerfile and dependency lock/manifest:
no `.git`, credentials, customer data, fixtures, tests, or the rest of the repo.
The approval hash binds the configuration, every snapshot byte and the generated
job. Editing any of them invalidates approval.

**Only after explicit authorization for this exact billable job and code upload:**

```bash
uv run --project learning --extra cpu --extra azure --frozen \
  python -m learning.azure --plan-dir reviewed-plan --submit \
  --approve-plan-sha256 REVIEWED_PLAN_SHA256 \
  --confirm-billable --confirm-code-upload --confirm-licenses-reviewed
```

There is no account creation, permission grant, resource deployment or
subscription switching in this CLI. `AzureCliCredential` is bound to the
specified tenant; `MLClient` receives all three resource-scope identifiers.
Before the single submit/upload write it verifies:

- The named subscription is enabled and belongs to the named tenant.
- AML uses approved-outbound-only managed networking; compute is the approved
  GPU SKU with zero minimum and one maximum instance, no public node IP, and the
  exact attached MSI.
- The named datastore is identity-based Azure Blob in the approved container.
  Every registered input version resolves to the approved owner-scoped URI.
- An existing, enabled Blob lifecycle rule targets the exact output prefix and
  configured retention period. Retention tags alone are **not** sufficient.

Private endpoints/DNS, compute identity data-plane roles, ACR pull permission,
approved outbound storage/registry rules and reachable private data mounts must
be provisioned by the authorized infrastructure workflow. Do not re-enable
public access to get a failed job through. A nonexistent workspace, unavailable
GPU quota, missing network path, denied identity, unresolved data version or
missing lifecycle policy is a blocker, not a reason to use local production
data or another tenant's datastore.

Each component has a finite AML timeout, training also has a process timeout,
and the job requests one GPU node. The worst-case sequential training graph is
the sum of conversion and training timeout limits; this is not a dollar-cost
guarantee. Quota, pricing and the customer's overall budget require approval
before submission. Outputs use explicit paths below
`tenants/<tenant>/owners/<opaque-owner>/.../<run-id>/`. Candidate contents live
under the model output's `model/` subfolder; converted data under `dataset/`.
Keep promoted releases outside an expiring candidate prefix under separately
approved retention. This code never deletes Blob data or changes lifecycle
policies.

The pinned SDK job parser accepts the managed identity's `client_id`. Its full
resource ID remains mandatory in the reviewed plan and is verified against the
SDK compute identity list before submission. Do not add `msi_resource_id` to
job YAML: the 1.35.0 schema exposes that field but its constructor rejects it.
Identity-based Blob datastores return `NoneCredentialConfiguration`, not Python
`None`; preflight checks the actual pinned SDK representation.

For an evidence-gate plan, use `kind: "gate"`, `parameters: {"timeout_seconds":
300}`, and three versioned inputs: `model` (`uri_folder` containing `model.json`
and `checkpoint/`), `evidence` (`uri_folder` containing `results.json` and final
images), and `plan` (`uri_file`). Their hashes respectively bind the exact
`model.json` bytes, `results.json` bytes, and **canonical JSON** of the held-out
plan (`digest(canonical(plan))`). The gate emits `gate.json` and exits nonzero
when rejected. It does not register or deploy the candidate.

## Checks and licensing boundaries

The root test suite discovers only lightweight tests:

```bash
source scripts/dev-env.sh
uv run --locked pytest tests/learning -q
uv run --locked ruff check --config pyproject.toml learning tests/learning
```

Optional ML imports are lazy. Root tests use explicit doubles for converter and
checkpoint-independent controller behavior; they do not claim Torch, AML, Isaac
or Azure execution. They cover tenant isolation, unsafe/corrupt datasets,
episode/seed leakage, inference timing/bounds/approval, submission/retention
guards and exact fail-closed evaluation counts.

The separate heavyweight **non-release** check has no skip-to-green path:

```bash
source scripts/dev-env.sh
export UV_PROJECT_ENVIRONMENT="$PHYSICALAI_WSL_CACHE/learning-cpu"
uv sync --project learning --extra cpu --extra azure --frozen
uv run --project learning --extra cpu --extra azure --frozen \
  python -m learning.checks.cpu_smoke --output "$PHYSICALAI_WSL_CACHE/cpu-learning-check"
```

Use a new output folder per run. This checks installed API signatures, real
LeRobot v3 conversion/reload, one actual CPU ACT optimizer step, saved
checkpoint/normalizer reload, real inference dimensions, offline AML SDK schema
loading, and rejection of fixture-only learning evidence. Its persisted
`report.json` says `learning_quality_verified: false`. It runs neither Isaac nor
Azure jobs. Linux needs normal native build prerequisites for the upstream
`pynput`/`evdev` dependency; the image installs them. Do not install the heavy
environment into the root API venv or a Windows-mounted `.venv`.

LeRobot 0.4.4 source is Apache-2.0; dependencies have their own licenses.
Training from scratch avoids gated/pretrained model downloads, but does not
grant rights to customer data, images, robot assets or NVIDIA software.
Isaac Sim/asset/CUDA license review and any required NVIDIA assent remain the
operator's responsibility. No license-acceptance flag, automated click-through,
Hugging Face token, or model-hub upload is supplied here.

The September 20 ACT handoff had no available live GPU. By September 23 the
parent separately reported actual Isaac 6/A10 operation and an AML A100 80 GB
CUDA/private-Blob hardware probe. Those hardware results do not demonstrate
SmolVLA weight updates, 80 ms concurrent inference or held-out physical task
quality. This learning slice has not performed cloud mutations, vendor-weight
downloads, live model optimization or policy promotion; the parent owns those
explicit live gates. GR00T's separate license restrictions remain blocking.

### Verified upstream interfaces

- [LeRobot 0.4.4 package requirements](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)
- [LeRobot dataset create/add/save/finalize API](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py)
- [Actual LeRobot training entry point](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/scripts/lerobot_train.py)
- [ACT configuration and action chunks](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/configuration_act.py)
- [ACT saved normalization processors](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/processor_act.py)
- [Azure ML v2 command schema](https://learn.microsoft.com/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
- [Azure ML v2 pipeline schema](https://learn.microsoft.com/azure/machine-learning/reference-yaml-job-pipeline?view=azureml-api-2)
- [Pinned Azure ML SDK managed-identity fields](https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-ml_1.35.0/sdk/ml/azure-ai-ml/azure/ai/ml/_schema/job/identity.py)
