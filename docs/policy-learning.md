# Azure policy learning

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

### GR00T N1.5 teaching path (separate from ACT)

The target motor policy is **GR00T N1.5**, pinned to NVIDIA source commit
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

The GPU build recipe is `learning/gr00t/Dockerfile`: digest-pinned
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, frozen isolated runtime/Azure extras,
FlashAttention 2.7.4.post1 compiled from its locked source distribution, and the
exact upstream checkout at `/opt/isaac-gr00t`. It targets Ampere A100/A10
(architectures 8.0/8.6); do not assume Blackwell compatibility. The image contains
no model weights and accepts no license. An authorized Azure builder may build
it with `-f learning/gr00t/Dockerfile` and then pin the resulting ACR digest.
Local dependency resolution is not a completed GPU image build or GPU proof.

After the operator reviews the pinned NVIDIA model license and acquires its
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

`infra/learning.bicep` provisions a private AML workspace and explicitly selected
`Dedicated` or `LowPriority` single-node compute (minimum zero, maximum one,
no public node IP, SSH/local auth disabled). It does **not** submit a job or
grant permissions. Supply `workspaceName`, `location`, existing
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

As of this implementation handoff, actual Azure GPU capture, billable AML
training, held-out Isaac rollout accuracy and model promotion are **unrun**.
The deployment workstream reported unavailable A10/RTX family quota and no
captured NVIDIA assent. Those are external live-release blockers; CPU checks
cannot waive them.

### Verified upstream interfaces

- [LeRobot 0.4.4 package requirements](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)
- [LeRobot dataset create/add/save/finalize API](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py)
- [Actual LeRobot training entry point](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/scripts/lerobot_train.py)
- [ACT configuration and action chunks](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/configuration_act.py)
- [ACT saved normalization processors](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/processor_act.py)
- [Azure ML v2 command schema](https://learn.microsoft.com/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
- [Azure ML v2 pipeline schema](https://learn.microsoft.com/azure/machine-learning/reference-yaml-job-pipeline?view=azureml-api-2)
- [Pinned Azure ML SDK managed-identity fields](https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-ml_1.35.0/sdk/ml/azure-ai-ml/azure/ai/ml/_schema/job/identity.py)
