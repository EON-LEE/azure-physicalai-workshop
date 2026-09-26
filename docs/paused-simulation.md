# Paused simulation learning (source work in progress)

For bounded, noninteractive Azure Batch execution without a live bridge, see
[Managed simulator jobs](managed-simulation.md). Managed scheduling does not change
the runtime criteria or establish GPU/model quality.

`NON_REALTIME_SIMULATION` is a separate simulation-only mode. It does not pass
or replace the existing real-time 100 ms control / 80 ms inference gate.
Real-time reference behavior, raw v1/v2 semantics and real-time policy
admission remain unchanged. No GPU execution or dataset is authorized by
these source changes.

The original profile identifier is `franka-position-hold-10hz-paused-v1`.
Every new command/artifact must explicitly declare
`execution_timing: "paused_simulation"` and `real_time_admission: false`.
Authorization also requires a newly saved, versioned
`learning_execution` environment field for the reviewed learning scene;
absence is denial, not an implicit fallback. The old environment's
`execution.max_step_seconds` continues to mean a **wall-clock** limit.

The optional root environment object is closed and has no implicit defaults:
`schema="physicalai.paused-simulation/v1"`, `execution_timing="paused_simulation"`,
`profile_id="franka-position-hold-10hz-paused-v1"`,
`max_simulation_seconds` (integer 1..30) and `max_wall_seconds` (integer 1..600).
Only `inspection-cell-learning-v1` may declare it. `SceneRegistry` decodes
these validated values into immutable `PausedSceneAuthority`; requested wall
and six-tick-aligned simulation budgets cannot exceed the lower saved limits.
A budget change creates a new environment revision and scene binding.

The separately approved v2 pair is `schema="physicalai.paused-simulation/v2"`
with `profile_id="franka-position-hold-10hz-paused-v2"` and an explicit
`max_simulation_seconds` of 1..60. The wall ceiling stays 600 seconds. A v1
schema with the v2 profile, the inverse combination, or an unknown version is
rejected even if a small requested budget would fit both. Legacy real-time
commands and v1 keep their original 30-second/1,800-tick ceiling.

Native `PausedControlProfile` requires the closed ID/ceiling pairs v1/1,800 or
v2/3,600; omission still selects v1. The pure driver receives the trusted
installed profile rather than a raised global maximum. Commands, resolved
grants, metrics, saved scenes, model manifests, capture and acceptance must
agree on the version and whole-profile hash. `ResolvedSimulationAuthorization`
omits the default v1 `profile_id` when serializing historical wire shapes,
but includes explicit v2. This never changes the SHA of an original grant file.
New v2 criteria and saved case revisions are required before execution; an old
v1 failure or capture cannot be renamed, extended or promoted into v2.

## Physics and wall time are independent

Physics advances only through six explicit 1/60-second ticks for each accepted
action: 10 Hz in **simulation time**, not a promise of 10 Hz wall time.
The runtime withholds physics while observation or policy inference is pending.
It must verify unchanged epoch, physics step, simulation time, joint
positions/velocities and object pose/velocities throughout that wait. It does
not reset poses or repeatedly restart the timeline to imitate a freeze.
Ground-truth freeze state stays in the runtime/evaluator, never model input.

Frozen upper bounds, intersected with lower approved environment/command limits:

| Budget | Limit |
| --- | --- |
| Observation pending, original absolute deadline | 2,000 ms wall time |
| Policy pending, original absolute deadline | 2,000 ms wall time |
| Held-six-tick subwindow, intersected with the original whole interval | 2,000 ms wall time |
| Whole observation through all six applied ticks | 5,000 ms wall time |
| Automated episode | 600 seconds wall; v1 at most 1,800 ticks, explicit v2 at most 3,600 |
| Main-thread heartbeat / blocking SDK work | 2,000 ms wall time |
| Human session and positive jog authority | Existing 300 s / 250 ms wall limits |

Stage transitions, polls and retries cannot renew an original deadline.
Cancel, changed authority or elapsed wall time wins even when simulation time
has not moved. A partial or over-budget interval remains a failed attempt;
no recorded ticks, images or issued target commands are padded, interpolated or dropped.
The unchanged joint tracking/slew, zero velocity targets, arm-only measured
gravity, simulated TCP speed and measured grasp/release/goal guards still apply.

## Initial source slice

`simulation.paused_control.PausedEpisode` implements pure frozen-state,
deadline and exact-tick accounting. It records actual wall observation,
inference, hold and interval durations independently of simulated elapsed
time. `simulation.paused_worker.PausedPolicyWorker` admits at most one
off-thread policy request. A cancelled worker retains its occupied slot until
it exits; stale freeze replies cannot become new commands. Errors propagate
without scripted or clipped substitutes.

`simulation.paused_teacher.PausedReferenceTeacher` is an explicitly scripted
expert, not a learned policy or customer demonstration. Repeated calls while
the same state is frozen return the same target. Grasp/release dwell and
waypoint progression advance only after six actual completed physics ticks
and their measured conditions, not because camera or model work took wall time.
Its automated episode authority is separate from short human jog grants.

Reported simulation duration is derived from the **verified completed physics
tick count at 60 Hz**, both per interval and for the episode. At 1,800 verified
ticks it is exactly 30 seconds, not the deprecated World clock's accumulated
float32 delta. The original World delta remains in private metrics as
`raw_world_elapsed_seconds`, with `world_clock_drift_seconds` retaining its
difference from tick-derived time. Tick-by-tick clock/epoch/actuation checks
remain unchanged; a large clock jump or an unapproved extra tick is still an
error. This fixes reporting of ordinary accumulation drift without extending
the task, operation or heartbeat deadlines. At the last allowed tick the
measured goal may succeed; otherwise the strict cap ends the episode and any
valid completed capture remains truncated, with no 1,801st step.
The same rule applies at v2's 3,600-tick/60-second boundary without a 3,601st step.

### Reference-expert target generation

The paused expert is called once per six actual ticks with an actual RMPflow
integration horizon of 0.1 seconds. Isaac's `ArticulationMotionPolicy` reads
measured joint positions **and velocities** before RMPflow internally integrates
its position/velocity proposal. That endpoint is not, by itself, a valid next
setpoint for the different zero-velocity position-hold servo. The adapter checks
the actual horizon and measured-state-feedback setting and records the native
maximum substep size; it does not shorten the horizon to pretend to run at 60 Hz.

`simulation.reference_targets.plan_reference_targets` constructs a uniform arm
waypoint toward that proposal using the existing `move_toward` helper. It
intersects the measured-position tracking and previous-issued-target slew
envelopes with fixed 90% planning limits (0.045 rad per arm joint and 0.0036 m
per finger). It evaluates at most eight FK candidates, after one origin FK read,
to bound predicted TCP displacement to `min(0.1, requested_speed) * 0.1` metres.
The independent pressure-producing gripper target is **not** rescaled when the
arm path is slowed. Like the legacy `GripperRamp`, the paused reference integrates
from its previous **issued** finger targets, not a fresh offset from measured
fingers at every interval. Each finger's feasible interval is the intersection
of its hard joint bounds, measured tracking envelope and previous-target slew
envelope. The teacher's closed/open destination is projected into those intervals,
then approached from the previous issued target by at most 0.0025 m in two-finger
vector distance per interval. This does not require a ray from the measured
position toward open/closed to enter the feasible region: at the first v2
release failure, one measured finger was 20.5 nanometres outside the previous
90% planning edge despite a nonempty intersection and a valid hard tracking gap.
Empty intersections and a bounded step that cannot reach the intersection still
fail; no tolerance, planning margin or hard limit is widened. Small measured
contact jitter can move the feasible destination without dropping accumulated pressure.
Saturation is a legitimate pressure hold, not proof of contact or grasp; targets
cannot jump to zero while the measured fingers remain about 0.025 m open.
Start/reset discards old targets and cancellation still fences actual application.
Legitimate stationary arm dwell/grasp holds remain valid.
Malformed/out-of-range proposals, infeasible paths and blocked requested arm
motion fail explicitly instead of being reported as a successful hold.

All nine generated targets still pass the unchanged shared 0.05 rad / 0.004 m
hard tracking/slew guard. Each actual tick retains zero velocity targets,
measured arm-only gravity, the measured 0.2 m/s TCP watchdog and the original
goal/grasp checks and episode deadlines. The recorded label is the **actual
issued bounded teacher target**, never the raw RMP endpoint. No learned output
uses this planner, and the legacy 60 Hz reference path is unchanged.

The private operator receipt includes `reference_target_evidence`: an independent
latest-attempt snapshot and at most 300 v1 or 600 v2 completed-interval snapshots. They retain
phase/tick, measured q/qdot, previous issued targets, native RMP positions and
velocities, Cartesian/orientation goals, unchanged limits, limiting joints,
arm-path fractions, FK displacement and actual issued targets/held ticks.
Completed snapshots also retain measured TCP speed, grasp and goal evidence.
The bounded trace never drops an earlier interval to make room; it is neither a
policy feature nor an added raw-manifest/public-DTO field. Failure diagnostics
are flushed before rethrow and receipts are persisted before simulator teardown.

`PausedEpisode.metrics()` preserves the **first** failure and its original
`failure_phase`. A later stop/cancel/cleanup is recorded separately as
`stop_reason`; it cannot replace a tracking/physics failure with cancellation.
Capture publication and physical failure remain separate outcomes.

The first actual reference attempt on source `2835191` stopped after 12 physics
ticks and two complete v3 frames, with the episode marked truncated. At step 86, measured joint 6 was
3.0285627842 rad and the second issued target was 2.9875681400 rad (a
0.0409946442 rad tracking gap). The rejected third proposal was not recorded,
so its exact offending joint is unknown. CPU regressions preserve both observed
frames and label their additional lag/proposal scenarios as synthetic; they do
not establish full-task GPU success or learned-policy quality.

### Exact manufacturing pick/place route and grasp evidence

Only `manufacturing-part-placement-v1` with the exact approved instruction
`Pick up the synthetic part from the source platform and place it in the quarantine tray.`
and the loaded quarantine goal selects `PickPlaceRoute`. Other paused tasks and
the legacy live reference retain `InspectionRoute`. A conflicting instruction
using that task identifier is rejected rather than silently selecting a route.
The task-specific route approaches directly above the source, descends, grasps,
lifts, transfers to the frozen destination, lowers, releases and retreats. It
does not visit the unrequested inspection station or change any saved pose.

Transit tool height is the maximum of the frozen source/destination centre plus
the unchanged 0.05 m grasp lift and 0.025 m planning clearance, and the highest
platform top plus the 0.025 m carried-part half-height and 0.025 m clearance.
Platform tops follow the unchanged workcell cuboids: station height minus
0.045 m centre offset plus half their 0.04 m height. This gives 0.275 m in the
current cell, rather than the inspection route's 0.38 m. A low initial tool
first moves vertically to clearance before lateral approach. Transfers stay
at clearance; descent/ascent occur only in the source/drop corridors. These
tool/payload bounds are **not** whole-arm collision or pad-contact certification.
The lift still needs actual measured part height, TCP proximity and finger gap
before transport. Neither speed limits nor the selected profile's simulation
deadline are increased by the route; a nominal CPU duration is not actual actuator admission.

For this exact task, `lower-to-destination` arrives only when the commanded
route target has reached the unchanged goal, grasp was previously verified,
the current finger gap remains within 0.005..0.055 m, measured part X/Y are each
within the existing 0.04 m goal tolerance, and part Z is within the existing
strict 0.012 m waypoint tolerance. The existing waypoint dwell still applies.
This avoids demanding unnecessary tool X/Y precision after the object reaches
the placement volume. It is a measured near-placement condition, not a contact
sensor or support-force claim. Legacy `InspectionRoute` arrival is unchanged.
Advancing to release is not success: actual opening, TCP separation/retreat,
part settling and the final physical goal checks remain required.

The actual source `687fabd` attempt reached 279 intervals/1,674 physics ticks
before failing grasp verification. The part finished at its source height
(about 0.20 m), while the wrist reached about 0.372 m and the fingers closed
nearly to zero. At lift start the issued finger targets were only about
0.00177 m inward of each measured finger; the old measured-relative ramp did
not accumulate pressure. That identifies a command-generation defect, **not**
an actual force measurement or proof that more pressure will fix grasp.

Private `reference_target_evidence.gripper_asset` reads loaded drive gains and
effort caps and bounded composed collision extents in finger-link local
coordinates. No gain, force limit, mass, friction, pose, or physics-view setup
is changed for diagnostics. Per-interval records add measured part positions
and finger world poses from the documented active **Fabric hierarchy**, not a
stale USD-world-transform fallback. Unsupported/nonfinite getters are explicitly
`unavailable`; no default gains or invented contacts are reported. Static USD
extents are labelled asset geometry, not current world poses. The recorded
phase describes the waypoint whose target was issued, even when that call
advances the route to its next waypoint.

Drive diagnostics preserve stiffness, damping and maximum-effort readbacks
independently. One unavailable getter or a nonpositive/nonfinite effort entry
cannot erase otherwise valid gains. Invalid fields retain their bounded original
readback and per-joint classification; nonfinite numbers are explicitly encoded
as strings, never zero or a made-up force limit. A nonpositive value is not
interpreted as measured contact force or proof of an independently driven DOF.
Collision traversal explicitly includes USD instance proxies, with the same
64-prim/eight-collider per-finger bounds and no instanceability or geometry edits.

An offline read of the actual `8f34c347` image's baked reference archive
`72956d2a7f0313d7effcff46c6b43ec616af8d2e1dd2055e4a152ad09767308e`
confirmed the default asset has a force drive on `panda_finger_joint1` and a
coupled `panda_finger_joint2` (`PhysxMimicJointAPI:rotX`, gearing -1, referencing
joint 1), not two authored independent finger drives. The authored joint-1
values are stiffness 400, damping 80 and maximum force 7.2. These are **asset
parameters**, not verified effective runtime gains or measured forces. The
existing nine-target action contract is unchanged.

### Asset-calibrated teacher contact frame

The canonical paused teacher uses `franka-default-inner-pad-centroid/v1` only
for `lower-to-part`, `grasp`, `lower-to-destination` and `release`. Its desired
point is the contact-pad centre/object centre, not a renamed measured TCP.
The verified inward source-triangle area centroids and authored joint origin
give a pad-centre offset from `right_gripper` of
`(0.000002636002657491832, 0, 0.002904602840903575)` metres in TCP coordinates.
The command subtracts that offset rotated by the **desired** TCP orientation.
At the down-facing orientation the TCP goal is about 2.9046 mm above the
unchanged object-centre goal; a rotated orientation rotates the correction
instead of applying a blind world-Z offset.

Arrival/dwell compares against the inverse contact point computed from the
**actual measured** TCP orientation. Grasp proof and the measured speed/workspace
watchdogs continue to use the original `right_gripper` TCP. Approach, carry/lift
clearance, transit and retreat keep their existing TCP targets. The release
contact target stays in the calibrated frame until retreat so the object goal
does not shift when opening begins. All resulting joint targets still pass the
shared tracking/slew/FK planner, and the exact issued nine-joint vectors remain
the raw action labels. No learned output or legacy inspection route is mapped
through this teacher calibration.

Before using the calibration, the runtime checks the exact approved archive
identity and completed-cache marker, the root USD and both finger geometry
checksums, the robot-schema layer checksum, and the loaded `Mesh=Performance` /
`Gripper=Default` variants. Missing or changed identity fails explicitly instead
of falling back to uncalibrated motion. Private diagnostics identify both
command/reference frames, measured TCP/reference points, calibration version
and verified asset identity. The measured frame, physical gains, damping,
force caps, material, object mass and all admission criteria are unchanged.

Offline analysis of actual attempt 04 used its recorded cube quaternion, not
an upright assumption: at physics step 1,190 the cube was tilted about 30.64
degrees. Clipping the verified inward source triangles against that oriented
cube gave about 1.50% left-pad and 20.71% right-pad source-area intersection.
During closure the pad centres were within about 1.5--2.1 mm of the part centre;
by the failed lift they were about 34.6 mm above it. This is consistent with
slip/roll during lifting, not a large static TCP error. These are source-mesh
geometry calculations, **not** cooked contact manifolds or force measurements.
The small frame correction is verified calibration, but is not claimed to cure
the slip or to establish successful grasp, placement or 30-second completion.

### Explicit paused gripper stiffness candidate

`franka-paused-finger1-stiffness/v1` intentionally changes only the live
`panda_finger_joint1` stiffness from 400 to 2,000 in the new paused source profile.
It preserves damping 80, the authored force cap 7.2 (the actual float readback
7.199999809265137 is compared within 0.000001), all seven arm gain/cap values,
and the passive mimic finger's zero gains/cap. No material, mass, collision,
joint target limit, velocity target, gravity mode, or task criterion changes.
At the conservative 0.0036 m target error this can request the existing force
cap through the spring term; it is **not** a measured normal/contact force.
Actual attempt 05 still failed grasp after the small frame correction, so this
is a separately reviewed engineering candidate, not a validated grasp result.

The same helper is called by reference and learned paused SDK preparation.
It first verifies the approved asset identity, actual joint order, authored
force drive and mimic coupling, and the original live finger values. Only
the current command's known override is idempotent; an unowned already-2,000
drive is rejected. The SDK call uses the existing initialized articulation
view, `indices=[0]`, `joint_indices=[7]`, `kps=[[2000]]` and
`save_to_usd=False`, never a force-limit or damping setter. A non-stopped
timeline, live SimulationManager physics view and valid articulation handle
are mandatory because the SDK otherwise falls back to USD authoring even
when `save_to_usd=False`.

Application is command-owned, after the original unarmed camera publication
but before recorded actuation. Complete frozen joint/object state, epoch,
physics index and simulation time must remain identical through the change.
The original publication is not retimestamped or claimed to have used 2,000
during warm-up. Every recorded tick verifies the owned full drive readback.
Stop/failure, reset/view replacement, and transitions to legacy or real-time
control restore only the known original baseline while the physics view is
still valid. Unexpected values fail closed without repairing unrelated state.
The private `gripper_servo` receipt retains authored metadata, complete
before/after/restored arrays and the frozen-state comparison. The separate
acceptance gate requires that evidence; a success flag or ignored SDK setter
cannot substitute for the actual readback and restoration.

## Reference capture vertical slice

`SimulatorRuntime` dispatches only a separately authorized
`SimulationEpisodeCommand` through the new `/v1/simulation-episodes` bridge
surface; it does not widen `MotionCommand`. The reference controller runs
through the actual articulation, zero velocity targets, measured arm gravity
and shared physical guards. The operator reference entry remains reference-only.
A separately installed learned catalogue can select the learned driver; without
that provider and its real model socket, learned commands fail explicitly.
No learned execution is replaced with a scripted route.

An opted-in scene remains physically frozen while idle, while capture I/O is
pending, and after terminal completion. Scene preparation retains the existing
settling work plus exactly 60 declared unarmed warm-up ticks. The final tick
does not render automatically: the runtime records actual private frozen state
before a non-advancing render publishes genuinely new native camera identities.
Both pixel arrays and metadata come from the camera's same `rgb` publication
callback. A previous frame is not restamped or inferred to have been frozen.

The reference-only operator entry explicitly prepares its cold Lula solver,
articulation FK wrapper, RMPflow controller and static finger collision geometry
**before** the unchanged 60-tick warm-up and its final publication. Preparation
does not call controller `forward`/`reset`, issue articulation actions, change
gains or advance physics; the complete physical snapshot must remain unchanged.
The inactive component cache is bound to the tenant/owner, environment/revision,
epoch, profile, task, exact robot and World instances. It is consumed once by
the freshly authorized reference command; stop, load/reset or changed binding
cannot reuse it. The ordinary/learned runtime never implicitly precreates RMPflow.

Command admission still checks actual asset identity and applies/verifies the
owned gripper calibration. Live gain evidence is reread after application; a
preparation-time 400 stiffness is never reported as an actual 2,000 readback.
Only static geometry is reused. Capture backend creation remains on its bounded
worker without SDK calls. Private `reference_preparation` and
`paused_startup_timings` record component construction boundaries, capture wait,
command setup and the original first-publication/joint/camera sample ages, also
on failure. If any remaining admission work exhausts the unchanged two-second
sample age, the command still fails instead of restamping pixels or adding
physics ticks. Cold setup is not retried until a warm cache happens to pass.

Only the first request can consume this known publication once, within two
seconds of its original camera/joint timestamps and only if owner, environment,
revision, epoch, native physics index/time and private physical state still
match. `InitialFrozenPublication` carries opaque record identity and sensor-only
hashes, not ground-truth pose features. Subsequent observations render genuinely
new frames only after their six actual held ticks. Any stale/repeated native
frame, unmatched physical state or phase deadline fails without hidden physics
steps or relaxed freshness.

The paused camera barrier drains at most **eight non-advancing render calls**,
all charged to the same original observation deadline (at most two seconds,
including clock reads, state checks and RGB encoding). The existing real-time
barrier remains limited to two calls under its original budget. Extra callback
opportunities cannot extend a deadline, move physics or make an old frame fresh.
Both camera-native rational times must match each other and the frozen physical
time within the unchanged half-tick tolerance; a frame one full tick behind is
still rejected. RGB and native metadata come from the same camera callback.

Before and after every paused render, the runtime rechecks the complete frozen
joint/object state and reads `SimulationManager.get_simulation_time()`,
`get_num_physics_steps()` and Fabric `/ExternalSimulationTime.omni:time`.
Available native clocks must agree with the frozen World time/index and remain
unchanged throughout the barrier. Optional getter failures are explicitly marked
`unavailable` with their reasons, never replaced with inferred values; they do
not mask the original camera failure. The private `camera_publication_evidence`
receipt retains one latest barrier, its original camera metadata and at most
eight before/after attempts. It is saved before teardown, including failures
during scene preparation before an episode exists.

This ordering follows the [Isaac 6 native simulation manager](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.core.simulation_manager/plugins/isaacsim.core.simulation_manager/PluginInterface.cpp):
the post-physics callback derives time from the integer physics count, writes
the external Fabric clock before Hydra submission, then stores its time sample.
The [camera callback](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/deprecated/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py)
evaluates the sensor graph and reads `ReferenceTime` and RGB when a new render
event is delivered. World and SimulationManager use the same underlying manual
physics API; this change does not migrate clocks or offset timestamps.
The actual `b9cb32e` attempt failed at preparation with camera time 1.316666666 s
and World step 80/time 1.3333334028720856 s. Native manager/Fabric clocks were
not recorded in that attempt, so delayed publication is a hypothesis, not a
confirmed GPU root cause. Its teacher planner was never exercised.

The bounded capture worker creates, appends and finalizes
`learning.paused.capture.PausedEpisodeWriter` on one persistence thread.
Producer mappings, byte buffers and action vectors are copied into immutable
typed snapshots before enqueue. The v3 manifest binds the original wall/sim
episode budget, exact paused profile, explicit source/purpose, frozen criteria
and case plan. Existing live managed-identity upload validates all files and
uploads the manifest last. Integration/test captures can never silently become
training data. Physical completion and capture publication remain distinct.

## Separately installed paused learned runtime

`PausedLearnedRuntime` shares frozen observation, exact-six-tick application,
capture, deadline and physical-success accounting with the reference driver,
but never constructs `PausedReferenceTeacher`, RMPflow or a contact-target
calibration. Its nine predicted joint targets go directly through the same
hard guards and SDK position/zero-velocity/arm-gravity servo, including the
explicit paused finger stiffness calibration. Invalid outputs fail without
clipping, interpolation or reference motion. The model process is separate;
Isaac imports neither Torch nor a local neural model.

Enable this source path only through **both** `PAUSED_POLICY_CATALOG_FILE`
and `PAUSED_POLICY_CATALOG_SHA256`. Absence keeps it disabled. The closed
`physicalai.paused-policy-catalog/v1` file has a bounded `policies` list, each
containing an existing `PausedOperatorGrant` as `grant`, an absolute
`model_root`, a protected same-host Unix `socket_path`, and
`expected_peer_uid`. The file hash covers its exact bytes. This is an installed
operator record, not a UI-generated grant or a model-path request parameter.
Its `policy_release` or `evaluation_grant` must bind the exact owner/tenant,
source/image, task, case revision, model/profile, criteria/conditions and
original expiry (at most 600 seconds). Evaluation grants require evaluation
purpose; learned recordings cannot silently become demonstrations for training.

Before admitting an entry, the provider validates the actual v2 candidate
manifest, checkpoint/processor inventory and simulation-time training lineage.
Pretrained vendor weights, clock relabeling, changed files or mismatched
profile/task/criteria are rejected. It creates the real
`learning.paused.ipc.SocketChunkPolicy` and `PausedGuardedPolicyAdapter`;
missing or unprotected sockets fail and the native connection checks its
peer UID. The separate reviewed model-server entry is
`python -m learning.paused.model`. These source switches do not authorize a
model job, deployment, GPU allocation or production admission.

One bounded worker performs reset and prediction after the first actual
observation exists. The main thread keeps checking the frozen physical state
and advancing heartbeats. Cancellation invalidates the guard generation
without blocking on the model, and a still-running cancelled call keeps the
worker slot occupied. First-publication checks on the worker use only the
thread-safe trusted record and protocol authority, never the Isaac SDK.
Before application and each held tick the runtime rechecks current
owner/epoch/command/model/profile/authority and the returned context, freeze,
observation and original expiry. Dequeuing cannot renew that expiry.

Capture stores actual prediction start/finish times and exact applied targets;
metrics identify actual applied model SHA, attempted prediction count and
applied action count, with `reference_route_calls=0`. A failed or cancelled
prediction reports no applied-model claim before an actual actuator call.
Shared workers, native IPC/model code and admission validators are included
in the combined profile fingerprint. Freeze and physically revalidate that
combined reference/learned profile **before** collecting genuine TRAIN data;
earlier integration captures cannot be relabelled to it.

## Operator entry (requires a later explicit GPU authorization)

No cloud resources or GPU processes are started by importing these modules.
After an authorized immutable-image build, the operator must provide a pinned
`physicalai.paused-operator-grant/v1` JSON file:

* Exact timing mode/non-real-time flag, tenant, actual source commit and image
  digest, and issued/expires UTC timestamps bounded to 600 seconds.
* `authorization` containing the immutable `ResolvedSimulationAuthorization`:
  ID/kind, owner, saved environment ID/revision, reference controller, approved
  task, final paused profile hash, wall/sim caps, purpose and the canonical
  criteria/scene-plan hashes.

The grant file's checksum is a separate argument. No arbitrary UUID grants
motion; the resolver validates its complete record against the actual source,
image, profile, task, frozen case and saved environment before admission.
There is no browser-provided Python/model path or automatic catalogue fallback.

```bash
/isaac-sim/python.sh -m simulation.paused_probe \
  --environment-record /data/paused/saved/nrt-integration-test-900002.json \
  --operator-grant /data/paused/approved-grant.json \
  --grant-sha256 <actual-grant-file-sha256> \
  --criteria /data/paused/paused-simulation-frozen-criteria-v2.json \
  --criteria-sha256 588fef92a96ace2dc5ffb5e606195a47f19e1e546b1ebf911220634708b3b1ef \
  --conditions /data/paused/paused-simulation-frozen-scenes-v2.json \
  --conditions-sha256 c28dc6a4892f9a90c9215d746fe331c221d0fb737ad765f6230f24d01ef08854 \
  --mode reference-task \
  --output /data/paused/receipts/integration-attempt-001.json \
  --confirm-isolated-simulator
```

Use `mechanics` only for a separate explicitly authorized 100-interval
integration attempt; it is not a full-task/quality claim. Every invocation has
one immutable receipt and no retry loop. Failure receipts precede simulator
teardown. All wall durations, simulation steps, actual actions, first-publication
metadata, warm-up evidence and physical/capture outcomes are retained. A live
result cannot be inferred from these CPU integration tests.

The host must run the separate CPU-only gate after the isolated process exits:

```bash
/isaac-sim/python.sh -m simulation.paused_acceptance \
  --report /data/paused/receipts/integration-attempt-001.json \
  --environment-record /data/paused/saved/nrt-integration-test-900002.json \
  --dataset-root /data/demonstrations/<owner>/<actual-command-id> \
  --tenant-id <actual-tenant> --owner <actual-owner> \
  --source-revision <actual-image-source-commit> --image-digest sha256:<actual-digest> \
  --profile-sha256 <actual-paused-profile-sha256> \
  --criteria-sha256 588fef92a96ace2dc5ffb5e606195a47f19e1e546b1ebf911220634708b3b1ef \
  --conditions-sha256 c28dc6a4892f9a90c9215d746fe331c221d0fb737ad765f6230f24d01ef08854 \
  --mode reference-task
```

It validates the actual v3 manifest/checksums, intervals and source/profile/case
bindings, plus the unchanged physical limits. `accepted` here is explicitly
non-real-time reference-capture evidence, not a learned-model or real-time gate.
Host orchestration must also retain all failure logs/receipts, check the
independent shutdown deadline and restore/deallocate resources as authorized.

The private attempt reader selects its limit from the **trusted saved scene's**
profile: the original 4 MiB ceiling for v1, and an explicit 8 MiB ceiling for v2's
complete 600-interval proof. The decoded profile and hash must still match.
This is not a global JSON-reader increase and does not truncate bad trials or
drop applied controls. Raw v3/model v2/IPC v2 schemas remain profile-bound and
otherwise unchanged.

`paused_servo_sha256()` separately fingerprints the paused driver, worker,
teacher, observation proof, contracts, capture, probe/authority code and shared
actuator/camera/safety code, plus native paused contract files, protocol versions
and all fixed resource caps. Legacy real-time fingerprinting/admission is not
used for this mode. Any code, cap or version change invalidates earlier bindings.

CPU tests of these invariants are not GPU proof, learned-quality evidence or
permission to relabel any failed real-time capture.
