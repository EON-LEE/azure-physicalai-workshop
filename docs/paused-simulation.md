# Paused simulation learning (source work in progress)

`NON_REALTIME_SIMULATION` is a separate simulation-only mode. It does not pass
or replace the existing real-time 100 ms control / 80 ms inference gate.
Real-time reference behavior, raw v1/v2 semantics and real-time policy
admission remain unchanged. No GPU execution or dataset is authorized by
these source changes.

The frozen profile identifier is `franka-position-hold-10hz-paused-v1`.
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
| Automated episode | 600 seconds wall time and 1,800 actual physics ticks |
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
arm path is slowed. Legitimate stationary arm dwell/grasp holds remain valid.
Malformed/out-of-range proposals, infeasible paths and blocked requested arm
motion fail explicitly instead of being reported as a successful hold.

All nine generated targets still pass the unchanged shared 0.05 rad / 0.004 m
hard tracking/slew guard. Each actual tick retains zero velocity targets,
measured arm-only gravity, the measured 0.2 m/s TCP watchdog and the original
goal/grasp checks and episode deadlines. The recorded label is the **actual
issued bounded teacher target**, never the raw RMP endpoint. No learned output
uses this planner, and the legacy 60 Hz reference path is unchanged.

The private operator receipt includes `reference_target_evidence`: an independent
latest-attempt snapshot and at most 300 completed-interval snapshots. They retain
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

## Reference capture vertical slice

`SimulatorRuntime` dispatches only a separately authorized
`SimulationEpisodeCommand` through the new `/v1/simulation-episodes` bridge
surface; it does not widen `MotionCommand`. The reference controller runs
through the actual articulation, zero velocity targets, measured arm gravity
and shared physical guards. Learned paused commands currently fail explicitly
because this operator entry is reference-only; no learned execution is replaced
with a scripted route.

An opted-in scene remains physically frozen while idle, while capture I/O is
pending, and after terminal completion. Scene preparation retains the existing
settling work plus exactly 60 declared unarmed warm-up ticks. The final tick
does not render automatically: the runtime records actual private frozen state
before a non-advancing render publishes genuinely new native camera identities.
Both pixel arrays and metadata come from the camera's same `rgb` publication
callback. A previous frame is not restamped or inferred to have been frozen.

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

`paused_servo_sha256()` separately fingerprints the paused driver, worker,
teacher, observation proof, contracts, capture, probe/authority code and shared
actuator/camera/safety code, plus native paused contract files, protocol versions
and all fixed resource caps. Legacy real-time fingerprinting/admission is not
used for this mode. Any code, cap or version change invalidates earlier bindings.

CPU tests of these invariants are not GPU proof, learned-quality evidence or
permission to relabel any failed real-time capture.
