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
no ticks, images or target commands are padded, interpolated or dropped.
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

**Not yet wired in this initial slice:** actual Isaac frozen-phase orchestration,
new closed raw/model/IPC/proof contracts, typed bridge admission, or deployment
authorization. The learning and API owners provide those versioned boundaries
separately. A new, complete paused source fingerprint must include all driver,
worker, capture, freeze, executor and protocol/budget dependencies before any
new immutable image, saved case or model binding is admitted.

CPU tests of these invariants are not GPU proof, learned-quality evidence or
permission to relabel any failed real-time capture.
