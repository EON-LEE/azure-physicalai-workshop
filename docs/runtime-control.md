# Isaac runtime control and capture

The authenticated simulator bridge remains private. `POST /v1/commands` still
selects the existing reference controller; recording does not make that
controller a learned policy. CPU protocol and physics doubles are not Azure GPU
evidence.

## Nonblocking capture

`SimulatorRuntime.tick()` owns physics, cameras, command completion and the
actual engine heartbeat. It never finalizes or uploads a dataset. An approved
command snapshots its owner, environment revision, epoch and capture ID before
creating a bounded `CaptureWorker`. Writer construction, all appends,
validation and managed-identity uploads execute on that same worker thread.
Motion waits for writer preparation without blocking the main loop.

The queue holds at most 64 pending frames and 64 MiB including a metadata
allowance. Overflow invalidates capture and stops the command; it never drops or
interpolates a sample. At most two persistence workers exist, including
finalization/upload (128 MiB aggregate queue budget). This permits episode B to
record while A uploads; a third overlapping capture fails explicitly. Physics
and the engine heartbeat continue during upload.

`GET /v1/commands/{command_id}/capture` uses the existing authenticated
`X-Environment-Owner` lease and returns:

```json
{
  "capture_id": "<UUID>",
  "command_id": "<UUID>",
  "epoch": "<UUID>",
  "status": "recording",
  "receipt": null,
  "message": null
}
```

Capture phases are `recording`, `finalizing`, `uploading`, `ready` and `invalid`.
A ready receipt has the existing `DemonstrationResult` shape: `status=uploaded`,
private manifest URI, manifest SHA-256, episode UUID and frame count. Files are
validated before upload, and the manifest is uploaded last. Cancellation of an
invalid worker is checked again before each upload, including the manifest.

Physical `Execution.status`, completion time and final measured position are
committed independently, without waiting for storage. Only the matching
owner/epoch/command/capture can attach a receipt to that historical execution.
A stale callback cannot finish another command, restore readiness, play physics
or mutate a new scene. A valid A completion can update A's historical record
after B activates a new epoch or another owner acquires the cell. It cannot
change B. Terminal capture states are atomically journalled by the worker under
`/data/demonstrations/.capture-status`, outside validated episode folders;
owner-scoped capture GETs can recover those receipts after process restart. A failed
upload does not retroactively change physical success, and a successful upload
cannot turn a cancelled or timed-out command into success.

The original reference recording cadence remains 60 Hz with two actual
synchronized camera frames and effective issued nine-joint position targets.
It is not re-labelled as 10 Hz held-position training data.

## Explicit teaching and learned control

`simulation.runtime_contracts` contains CPU-safe closed DTOs. Existing reference
commands remain at most 30 seconds. A **separate** `TeachingStart` permits an
approved session lease up to 300 seconds; that lease alone never authorizes
movement. All new motion paths keep the published 0.2 m/s measured TCP ceiling.

| Bridge endpoint | Contract |
| --- | --- |
| `POST /v1/teaching` | Flat scene/observation binding; `command_id`, `session_id`, `lease_id`, `session_expires_at`, `control_profile_id`, approved `task`, explicit `demonstrator_kind` and `split` |
| `GET /v1/teaching/{session_id}` | `TeachingState`, physical `Execution`, and independent capture state |
| `POST /v1/teaching/{session_id}/input` | Lease/epoch, sequence, deadman, Cartesian `delta_xyz_m`, gripper intent, server expiry and control grant |
| `POST /v1/teaching/{session_id}/finish` | Lease/epoch; request a measured terminal check at an actual held boundary |
| `POST /v1/teaching/{session_id}/cancel` | Lease/epoch; stop immediately, even mid-interval |
| `POST /v1/policy/commands` | `command: MotionCommand`, explicit `policy_type`, `policy_release_id`, `model_sha256`, profile ID and approved task |

The task is `{task_id, instruction, goal_id}` with a bounded single-line
instruction; its goal must match the selected station. An input moves at most
one centimetre in Euclidean distance and is valid for at most 250 ms. The API
issues a short server grant (at most one second), bound to the exact owner,
session, epoch, sequence and intent; browsers cannot supply authoritative clock
values. The bridge requires `grant_id` and `grant_expires_at` for positive
deadman inputs and intersects motion expiry with that original grant. Replays
cannot refresh authority. A neutral deadman release needs no grant and may skip
forward sequences to fence a delayed, never-seen earlier motion request.

The trusted teaching request must explicitly state `split` (`train`,
`validation` or operator-only integration `test`) and match the split already
encoded in the saved environment revision. It is an assertion, never a way to
relabel an existing case. Raw episodes still derive their split and seed from
that approved immutable scene; there is no default-to-training behavior.

No input accepts joint arrays, arbitrary model paths, URLs or Python modules.
Missing/failing providers stop explicitly; there is no baseline or animation
fallback. `smolvla` is explicitly selected, not relabelled GR00T. Historical DTOs
also recognize `gr00t_n1_5` and `gr00t_n1_7`, but those families are not registered
for commercial execution. An unavailable SmolVLA module/socket/model is an
error, not permission to select another model family.

`franka-position-hold-10hz-v1` has a deliberately different low-level contract
from the reference controller:

The control server and isolated probe both explicitly select the same real
RTX `RaytracedLighting` sensor-rendering configuration, a 320 by 320 application
surface, and disabled **unused viewport** updates. Both actual camera render
products remain enabled at 320 by 320. The normal reference-only launch keeps
its existing defaults. These settings are source/profile hashed and require
new measured timing evidence; the probe receipt records the selected profile.
They are not a synthetic image path or a change to motion/freshness thresholds.

Control-only launches additionally use the installed Isaac 6
`isaacsim.exp.base.zero_delay.kit` experience, verified against SHA-256
`776a905289b9029d760fdc0d9b9d6e6cb96b20a4a5f00763f8f120ea9f7e0b88`.
Its supported Hydra wait-idle/render-completion ordering prevents the default
multi-frame-in-flight pipeline from pairing old camera metadata with current
physics. Both production control and the isolated probe select this exact
experience; missing/mismatched files fail instead of reverting to pipelined
rendering. Reference-only launch remains unchanged. The experience name/hash
are included in the reviewed profile source and actual probe receipt. See
[Isaac 6 rendering-frame delay](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/python_scripting/util_snippets.html#rendering-frame-delay).

The reviewed control profile selects a 60 Hz physics clock and zero automatic
rendering timestep so each SDK call takes the documented explicit physics-step
path, not an unrestricted Kit application update. Every recorded tick requests
`update_fabric=True` and verifies the real physics index advances exactly one
and simulation time advances exactly 1/60 s (floating-point tolerance only).
The sixth tick publishes the resulting sensor state. Reference mode restores
its original timestep settings.

Before any command admission, the same production/probe scene load performs
exactly 60 unarmed neutral-hold render ticks, bounded to ten seconds. The full
warm-up timings are retained separately; they are not control intervals,
demonstrations or training evidence. Warm-up never repeats until passing and
does not discard a later armed budget violation. The initial part pose is
rechecked against the unchanged 1 mm pinned scene tolerance afterward.

* One synchronized observation and nine absolute targets per six actual 60 Hz
  physics ticks; no intermediate targets are dropped or resampled.
* The identical targets and explicit zero velocity targets are passed to
  `ArticulationAction` on every tick. Arm-only measured PhysX gravity efforts
  are recomputed and recorded on every tick.
* New intervals are paced against a monotonic 100 ms clock without blocking
  heartbeat handling or advancing extra physics. Camera capture, prediction
  and all six physics ticks must complete inside that budget.
* Teacher proposals and learned outputs use the same tracking/slew checks:
  0.05 rad for each arm joint and 0.004 m for each individual finger per
  interval. Teacher contact pressure is bounded relative to measured fingers;
  neutral holds retain issued finger targets rather than dropping grip force.

At each aligned observation boundary, the barrier first checks both already
published camera frames against current physics time and their previous native
identities. Fresh matching frames retain their actual publication timestamp;
they are not redrawn or restamped merely because an observation is requested.
Only stale/missing frames trigger a bounded `World.render()` refresh, which must
not advance simulation time or the physics index.
The sixth subsequent physics tick publishes its resulting state to the sensor
pipeline with rendering enabled; no seventh physics tick is inserted. The next
boundary verifies that exact state without advancing it. Profile cameras acquire every scheduled
render rather than applying a second independent frequency decimator.
The original `dt / 2` camera/physics-time check is unchanged. Barrier work is
inside, not added to, the 100 ms budget.

Isaac 6 camera `rendering_frame` is a Fabric rational identity, unlike the
legacy integer. The recorder retains the actual integral numerator (no float
round trip or invented counter), pins the positive denominator per camera and
scene, and rejects rewinds or timebase changes. The original numerator and
denominator, both camera times, world time/physics index, warmup state and
control boundary are retained in the separate operator receipt/diagnostics.
They are not fabricated as current timestamps or added to the closed raw schema.

The v2 recorder stages an observation plus its six **actual following**
`AppliedControl` records. It keeps one complete interval pending so graceful
finish can mark a genuine terminal interval without synthesizing future ticks.
Mid-interval emergency stops invalidate capture. These records include the
issued velocities and gravity evidence; v1 remains unchanged and is not
eligible for this profile.

Learned execution constructs no RMPflow controller or reference route. Every
application rechecks owner, epoch, command, original deadline, model and physics
tick immediately before the actual actuator call. Counters count real
prediction attempts and successfully submitted physics-tick actions;
`applied_model_sha` stays null until submission succeeds. `policy_runtime`
includes the immutable family/release, counts and `reference_route_calls=0`.
Stopping physics precedes model cleanup, so a reset failure cannot defer stop.

Teacher and learned TCP measurements use measured articulation joints and pure
Lula forward kinematics at the reviewed `right_gripper` frame, matching the
reference RMPflow coordinate frame. They do not substitute an arbitrary finger
prim pose; learned execution invokes neither inverse kinematics nor RMPflow
action generation. This follows the pinned
[Isaac 6 kinematics interface](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/manipulators/manipulators_lula_kinematics.html).

Success requires measured lift/grasp evidence, open fingers, part position
inside the unchanged 4 cm goal tolerance, wrist separation, and a settled part.
No model-supplied success flag is trusted.

## Reviewed initial layouts

`inspection-cell-learning-v1` uses seed-hashed initial part XY offsets within
two centimetres of the same source platform, with no defect stripe. It does not
change `inspection-cell-v1`. Placement occurs only during stopped scene reset,
never during an action. The learning scene must settle within 1 mm of its
pinned initial pose or loading fails.

`SceneSpec.part_position`, `scene_builder_sha256` and
`IsaacWorkcell.initial_state_evidence()` support frozen before/after cases.
Different seeds require separately saved environment revisions; release
catalogues allow only explicitly authorized `(environment_id, revision)` pairs.
Seed, evaluator pose and task success are never policy observation features.

## Deployment gates

The default deployment enables neither teaching nor learned execution. Enabling
the profile requires `CONTROL_TIMING_FILE` and `CONTROL_TIMING_SHA256`: a
deployment-controlled, checksum-pinned `physicalai.control-timing/v1` document.
It binds the simulator image/version, robot asset, source revision, reviewed
servo source digest, profile and real GPU trace. At least 100 complete aligned
camera/control intervals, a maximum cycle below 100 ms, inference at most
80 ms, and heartbeat gaps at most two seconds are required. CPU fixtures cannot
qualify. A model-specific proof must also name the exact family and model SHA.

Optional `POLICY_CATALOG_FILE` and `POLICY_CATALOG_SHA256` select a pinned
`physicalai.policy-catalog/v1` deployment catalogue. Each release binds
`policy_type`, model SHA, tenant/owner, control-profile SHA, approved task and
environment cases, a local Unix socket, expected peer UID, and model-specific
timing evidence. The transport checks peer credentials and one absolute
deadline across connect, send, every receive and validation. It loads no ML
framework in Isaac; the real SmolVLA process has its separate environment.

Creating a reviewed API/Cosmos `PolicyRelease` **does not install or load it**.
An approved deployment operator must install/verify the artifact, start the
separate inference process, complete timing validation, install the pinned
catalogue and perform the controlled simulator rollout. There is no browser
SSH path, automatic release activation or hot model swap under an active command.

The simulator Docker recipes copy the complete learning Python package for its
CPU import closure, but install no LeRobot, Torch or Transformers in Isaac.
The existing launcher already passes the deployment environment and `/data`
bind mount; it needs no automatic restart or network change for these options.

## Parent-operated GPU G0, not a learning gate

Only an explicitly approved **isolated** Isaac process should run this probe.
It does not stop an existing simulator or provision anything. Use the parent's
already verified Isaac 6 base image:

```bash
docker build -f simulation/Dockerfile.code \
  --build-arg SIMULATOR_BASE_IMAGE=<verified-isaac6-image@sha256:digest> \
  -t <new-runtime-image> .
```

Pin the resulting image and source revision. The container needs the existing
asset paths/checksum and managed identity plus `CAPTURE_ENABLED=true`,
`SIMULATOR_IMAGE`, `ISAAC_SIM_VERSION=6.0.0`, `SOURCE_REVISION`,
`ENTRA_TENANT_ID`, `AZURE_CLIENT_ID` and `STORAGE_ACCOUNT_URL`. Prepare a private,
writable output directory. Supply an actual saved `EnvironmentRecord` with
capture explicitly enabled and `demonstration_split=test`; do not mutate its
revision/seed inside the probe.

```bash
/isaac-sim/python.sh -m simulation.probe_control \
  --environment-record /data/probes/approved-test-environment.json \
  --owner <opaque-owner-key> --tenant-id <tenant-uuid> \
  --mode teaching --intervals 100 \
  --output /data/probes/teaching-probe.json --confirm-isolated-simulator
```

Repeat with `--mode policy-fixture` and a new output path. The first mode issues
one bounded 1 mm Cartesian jog; the second exercises the learned actuator
branch through a tiny deterministic port fixture. Both use real Isaac cameras,
physics, capture workers and private Blob upload. Both are explicitly scripted
`reference_controller` data, never claimed to be customer/human input.
The fixture is **not** a deployed SmolVLA/GR00T model and is never available
through HTTP or the production catalogue.

Reports deliberately say no model weights were loaded, no task/learning success
was established, and production is not ready. They are not automatically
accepted timing attestations. The parent must separately verify real
grip/hold/place behavior, licensed checkpoint inference, and frozen paired
held-out trials before accepting teaching data or promoting any policy.

Probe receipts are written to a same-directory temporary file, flushed and
fsynced, then published atomically without overwriting another attempt. The
containing directory and diagnostic streams are flushed **before** Isaac
teardown. A failure receipt preserves its phase, exception type and observed
state even for `Problem`, SDK initialization errors or an unexpected
`SystemExit`. A shutdown-only `SystemExit(0)` cannot replace the original error;
an unexpected zero exit during the probe is treated as incomplete/failing.
The host must still require the correct receipt schema, `probe_completed` and
ready capture evidence, never just Docker's exit status. No interval count or
physical outcome is inferred from startup/shutdown timing.

The operator host must additionally run `python -m simulation.probe_acceptance`
with the report, unchanged saved environment, actual local dataset root, expected
tenant/owner, image digest, source commit and probe mode. It rejects incomplete
receipts, missing/invalid uploads, noncontiguous six-tick intervals, widened
latency/heartbeat budgets and missing fixture actuator counters. It validates
the real v2 dataset/checksums, two 320-pixel cameras and immutable source/image/
builder/case provenance. Docker exit zero or report-file presence cannot satisfy
this gate. Upload failure logs as well as reports to the private evidence
container, and restore the previously running reference container in `finally`.

## Explicit bootstrap reference collection

After real GPU grip/hold/place validation, an approved operator can collect one
complete scripted expert attempt, not merely a jog:

```bash
/isaac-sim/python.sh -m simulation.probe_control \
  --environment-record /data/cases/approved-train-case-001.json \
  --owner <opaque-owner-key> --tenant-id <tenant-uuid> \
  --mode collect-reference \
  --task-id <approved-task-id> --instruction "<approved-single-line-instruction>" \
  --goal-id <approved-station-id> \
  --output /data/receipts/train-case-001-attempt-001.json \
  --confirm-isolated-simulator
```

The record must already select `inspection-cell-learning-v1`, capture, the
intended train/validation/test split, and the prechosen seed/revision. The tool
does not modify these, resample poses, retry a failure, or label automation as
human input. Each invocation has one immutable receipt and a deadline bounded
by both **30 seconds** and the environment's original step budget.

`ReferenceTeacher` progresses reviewed pick/inspection/place waypoints only
from measured TCP, finger opening and part state. It rejects lifting without
the actual part, then requests graceful finish for the shared measured
grasp/release/goal/settling checks. It produces only granted, one-centimetre
Cartesian/gripper intents through the teaching lease. The explicit scripted
source can request at most 0.1 m/s task-space goals (human jog goals remain
0.05 m/s); the same position-hold servo, joint tracking/slew limits and 0.2 m/s
physical watchdog remain mandatory. No expert joint command bypass exists.

The actual interval trace separates observation rendering, camera/joint reads,
teacher/model work, actuator submission, the six physics ticks including sensor
publication, capture queue work and measurement guards. The whole-cycle budget
remains below 100 ms; a standalone model latency below 80 ms is not sufficient
when rendering, physics and observation work consume the remaining time.

Receipts use `physicalai.reference-teaching-receipt/v1` and keep physical outcome,
capture status, actual initial pose, immutable environment revision, task,
split, profile hash, complete interval timings and explicit errors separate.
Failed, cancelled and timed-out attempts are retained and exit nonzero.
Interrupted partial holds are invalid rather than padded. A setup failure has
an explicit `physicalai.operator-attempt-failure/v1` receipt, not an invented
episode. Selecting accepted demonstrations for a frozen dataset remains a
separate reviewed operation; a published capture alone is not a quality label.
