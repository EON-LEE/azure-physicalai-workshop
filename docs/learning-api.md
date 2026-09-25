# Owner-scoped teaching and policy learning API (v1)

This is an implementation contract, **not evidence of a trained or released
motor policy**. The feature defaults off, with an empty production model
allowlist. SmolVLA is the explicitly selected commercial candidate; it is not
a renamed GR00T model or an automatic fallback for a blocked GR00T request.
GR00T N1.5 is noncommercial and N1.7's pinned primary license/card conflict
remains unresolved. Model recognition is not license or hardware approval.
Existing inspection, reference motion,
operator authentication, public camera and recorded-reference-case routes keep
their meanings. A changed environment layout is not policy learning.

## Authentication and persistence

Every route below except `GET /api/demo/learning` requires the existing delegated
Entra token. Tenant and actor come from the validated token. Cosmos partition
`owner_key` is derived from that principal; clients cannot set owner, tenant,
worker identity, arbitrary model/data URLs, code, or raw joint arrays.

Record bodies are immutable snapshots. A transition creates a new persisted
snapshot under conditional Cosmos ETag replacement; identifiers, actor, original
request fingerprint, task, environment revision, control profile, dataset/model
pins and approved budget cannot silently change. All mutations carry a UUID
`request_id`. Reusing it with changed input is HTTP 409. Resource responses are
`{"item": <record>, "etag": "<opaque-etag>"}` and include an `ETag` header.
Listings are bounded, owner-scoped `{"items": [<resource response>]}`.

Existing-context mutations require `If-Match` from that context: project for
teaching/dataset/train/evaluate, teaching session for jog/finish/cancel, job for
job cancellation, evaluation for release. Missing precondition is 428, stale
precondition 409. Exact retries return the persisted operation, not a new job or
motion. A different owner receives 404. Errors use the existing structured error
envelope. Missing integration/capacity/artifacts returns explicit 503, not a
successful placeholder, fixture checkpoint or ACT fallback.

## Frozen surfaces

| Method / path | Body / response |
|---|---|
| `GET /api/learning/capabilities` | `enabled`, readiness/block reason, approved profiles, exact admitted `policy_types`, `model_admission`, `bootstrap_allowed`; configuration is not a successful training test |
| `POST /api/learning/projects` | `CreateProject`; returns immutable `LearningProject` |
| `GET /api/learning/projects` | Owner project list |
| `GET /api/learning/projects/{id}` | Project with ETag |
| `POST /api/learning/projects/{id}/teaching-sessions` | `request_id`, `case_id`, `source: human_teleop|reference_controller`, `motion_approved: true`; only a project-approved case may be selected |
| `GET /api/teaching-sessions/{id}` | Reconciled `TeachingSession`; physical completion and capture readiness remain separate |
| `POST /api/teaching-sessions/{id}/arm` | `request_id`, `lease_id`, `epoch`, `sequence`, `deadman: true`, `delta_xyz_m`, `gripper`; returns a server control grant valid for at most one second; never moves |
| `POST /api/teaching-sessions/{id}/jog` | Same intent plus `grant_id`; server stamps expiry on first admission. Stop-only `deadman:false` requires zero delta/hold and no grant |
| `POST /api/teaching-sessions/{id}/finish` | `request_id`, `lease_id`, `epoch`; does not assert capture readiness |
| `POST /api/teaching-sessions/{id}/cancel` | Same binding; does not assert cancellation until confirmed |
| `POST /api/learning/projects/{id}/datasets` | `request_id`, unique `teaching_session_ids`; only verified uploaded eligible captures |
| `GET /api/learning/datasets/{id}` | Frozen `DatasetVersion` |
| `POST /api/learning/projects/{id}/train` | `request_id`, `dataset_id`, `parent_release_id`, optional bootstrap `pretrained_artifact_id`, exact `policy_type`, `optimizer_steps`, `paid_approved: true`, `maximum_cost_usd` |
| `POST /api/learning/projects/{id}/evaluate` | `request_id`, `candidate_id`, `baseline_release_id`, `evaluation_plan_sha256`, `motion_approved: true`, `paid_approved: true`, `maximum_cost_usd` |
| `GET /api/learning/jobs/{id}` | Reconciles the actual named job only; never submits or restarts it |
| `POST /api/learning/jobs/{id}/cancel` | `request_id`; cancellation state from the actual backend |
| `GET /api/learning/candidates/{id}` | Verified candidate; not a released skill |
| `POST /api/policy-releases` | `request_id`, `candidate_id`, `evaluation_run_id`, `release_approved: true`; exact evaluation ETag and passing gates required |
| `GET /api/policy-releases/{id}` | Immutable reviewed release |
| `POST /api/learning/projects/{id}/coach` | `request_id`, instruction, optional dataset/evaluation IDs; typed Foundry advice and actual response ID, no cost/motion/release authority |
| `GET /api/demo/learning` | Explicitly curated publication only; no owner enumeration, dataset/model paths, identities or private histories |

`CreateProject` contains `request_id`, `display_name`, `task_id`, `instruction`
(one line, at most 512 characters), exact `policy_type`,
`goal_station_id`, `environment_id`, `revision`, `baseline_release_id`,
`control_profile_id`, `teaching_cases`, `evaluation_plan` and `budget`. `project_kind` is
`adaptation` by default. Its P0 release must exist and its model family must
match; no GR00T/SmolVLA/ACT substitutions are allowed.

The initial reviewed profile is `franka-position-hold-10hz-v1`, not the existing
60 Hz reference profile. The task goal must be a station in the pinned saved
environment. Baseline P0 must be a real approved learned policy, not a scripted
reference route relabeled as a policy. Prepared P0 provenance remains visible.

### Separate non-real-time simulation project contract

`NON_REALTIME_SIMULATION` is an explicit new mode, not a relaxation or a passing
result for the original 100 ms interval / 80 ms policy gate. Its fixed profile is
`franka-position-hold-10hz-paused-v1`: 60 Hz physics, six held ticks per 10 Hz
simulation-time action, with physics frozen while observation/inference is pending.
It cannot authorize a real robot or inherit a real-time release.

A new project must explicitly supply all of `execution_timing: paused_simulation`,
`real_time_admission: false`, the new `control_profile_id`,
`control_profile_sha256`, `criteria_sha256`, and `frozen_plan_sha256`.
The last hash pins the operator's **model-independent frozen scene conditions**,
not a later model-pair-specific native plan hash. Those remain separate bindings.
Mode, profile, all hashes and the evaluation-plan branch must agree; aliases,
missing pins, implicit paused defaults and real-time qualification are rejected.
Legacy serialized projects and request fingerprints omit these new fields and
retain their original evaluation-plan semantics.

The paused evaluation-plan branch requires exactly twenty unique held-out cases,
`minimum_success_rate >= 0.9`, `minimum_absolute_improvement >= 0.05`,
`maximum_axis_error_m <= 0.04`, and `max_cartesian_speed_m_s <= 0.2`.
It uses explicit `max_simulation_seconds` (1..30) and `max_wall_seconds` (1..600),
not legacy `max_step_seconds` or `maximum_inference_p95_ms`. Frozen wall-time
caps are `max_observation_wall_ms=2000`, `max_policy_wall_ms=2000`,
`max_hold_wall_ms=2000`, `max_interval_wall_ms=5000`, and
`max_heartbeat_wall_ms=2000`. Saved-case opt-in and any lower environment limits
remain authoritative; these request fields are not standalone motion authority.

Total project `budget.evaluation_seconds` is a separate explicit wall budget
(up to 21600 seconds); the new-mode UI suggests 7200 seconds for review rather
than changing any existing project. Per-episode 30 SIM / 600 WALL limits and
the immutable native AML absolute deadline remain distinct. An incomplete batch
cannot be reported as all twenty cases evaluated or as successful improvement.

**Current admission boundary:** the API exposes `simulation_learning` capability
metadata with `supported: true`, `enabled: false`,
`status: producer_verifier_unavailable`. Closed types and UI draft fields are not
proof that the actual runtime, v3 data, v2 model/report producers or verifiers are
admitted. Paused project/motion/training/evaluation/release mutations return
explicit 503 before falling through any old real-time path. The UI shows the
new-mode bounds and blocked status; no environment flag alone enables missing
adapters. New model/report outputs must be verified through their separate
committed native producers before this boundary can be opened.

The private bridge has separate `SimulationEpisodeCommand` /
`SimulationEpisodeExecution` API DTOs for `/v1/simulation-episodes` and its
owner-scoped read/cancel routes. A command uses `wall_expires_at`,
`max_simulation_steps`, canonical `execution_timing`, `real_time_admission:false`,
`profile_id`, and `controller` independent of the run's
`inspection|released_skill` mode. It is not a legacy `MotionCommand` with a longer
deadline. Resolution requires a trusted immutable
`ResolvedSimulationAuthorization`, including owner/scene/task/profile, purpose,
wall/SIM limits and original criteria/scene-plan hashes; a UUID is not authority.
Responses preserve required actual `simulation_runtime` wall duration, simulation
duration/steps, phase, model/action/reference counts, and false real-time admission.
Missing telemetry is rejected, not replaced with zeroes. These bridge methods
alone do not enable a public motion endpoint or runtime admission.

The same complete mode/profile/criteria/scene-plan binding is retained on
paused capture receipts, sealed datasets, teaching/job records, train-only
parents and candidates. Legacy records omit unused new fields, retaining old
serialization. A paused receipt must bind its original episode command as well
as its approved case. The worker selects the separate real v3 raw validator,
requires live demonstration-purpose evidence, and rejects integration or
evaluation captures for training. Paused v2 checkpoint validation is explicit;
the legacy model path still refuses it. Verified metadata does not convert a
train-only parent/candidate into a release or make the supported-but-disabled
paused execution capability active.

### Varied teaching cases and split isolation

New projects **must explicitly freeze** `teaching_cases`:

```json
[
  {
    "case_id": "train-10001",
    "environment_id": "the-owner-saved-training-scene",
    "revision": "exact-64-hex-revision-from-the-saved-record",
    "seed": 10001,
    "split": "train"
  },
  {
    "case_id": "validation-20001",
    "environment_id": "the-owner-saved-validation-scene",
    "revision": "exact-64-hex-revision-from-the-saved-record",
    "seed": 20001,
    "split": "validation"
  }
]
```

The revision strings above illustrate fields, not ready records to submit.
Use the actual saved-record hashes. Each case ID, seed and environment/revision
pair is unique across the teaching allowlist. Teaching seeds/scene revisions
must be disjoint from the frozen held-out test cases. Integration-only G0 seed
`900002` is rejected for both teaching and final held-out evaluation.

At project creation the API verifies **every** case's owner-scoped saved record,
content hash, valid LIVE document, reviewed `inspection-cell-learning-v1`
builder, robot profile, actual scene seed and task goal. Train/validation
records must explicitly set `execution.record_demonstration: true` and matching
`execution.demonstration_split`. Held-out saved records must have split `test`.
Changing the saved document or its revision cannot silently rebind a case.

`StartTeaching.case_id` selects only an allowlisted case. Its environment,
revision, seed and split cannot be overridden in that request. Omitting the
selector is backward-compatible **only** when the anchor environment/revision is
already explicitly listed once in the allowlist. Older projects with no
`teaching_cases` remain readable, but cannot authorize new teaching; create a
new approved project rather than retroactively editing its immutable pins.

Activate the selected saved scene explicitly before starting. Selection alone
does not activate a scene, change the seed or dispatch motion. The API observes
and authorizes the **selected** environment/revision, not the project anchor.
The stored `TeachingSession.teaching_case` and response retain the selected
case. Private `TeachingStartSpec` sends its exact environment/revision and
required `split`; the runtime derives the actual seed from the saved scene and
rejects a different split. There is no seed override or fallback to anchor.

The worker checks the original native episode's environment, revision, seed,
split, task instruction/goal, source, profile and frame count against that
case before accepting a capture. `CaptureReceipt` retains `case_id`,
`environment_id`, `revision`, `seed`, and `split`. `DatasetVersion.captures`
retains these exact per-episode receipts in the same order as `episode_ids` and
`seeds`. Legacy receipts/datasets without case provenance cannot become new
optimizer inputs. Metadata is checked again after private download and before
native dataset assembly.

A sealed bundle may contain distinct train and validation cases without
relabeling either. Native SmolVLA conversion/statistics/optimizer input selects
**only `train`** episodes; validation-only bundles cannot start optimizer work.
A mislabeled split, duplicate episode, train/validation seed overlap, unapproved
case, held-out test episode/seed or integration seed is rejected before training.
Physical completion and async historical capture reconciliation remain separate.

For the current manufacturing task, the deployment operator freezes P0 training
seeds `10001..10020`, P1 additional training seeds `11001..11020`, validation
`20001..20010`, and final test `30001..30020` as separate real saved records.
The task is `manufacturing-part-placement-v1`, goal `rejected`, instruction
`Pick up the synthetic part from the source platform and place it in the quarantine tray.`
These are explicit case partitions, not a constant-seed capture renamed varied
data or handwritten ready dataset records.

`budget` fixes `teaching_seconds` (5..300), `training_seconds` (1..86400),
`evaluation_seconds` (1..21600), `optimizer_steps` (1..100000), and
`maximum_cost_usd` (positive, decimal USD). Job-specific approval cannot exceed
it. The amount is an authorization ceiling, **not a claim that Azure provides
an automatic spending stop**: the production admission adapter must verify
approved SKU, duration, price and quota before dispatch; otherwise block.

`evaluation_plan` fixes UUID `id`, 20..100 unique held-out `seeds`,
`held_out_episode_ids`, and corresponding unique `cases`
`[{seed,environment_id,revision}]`. The API checks the owner's saved
`inspection-cell-learning-v1` scene and exact seed/revision for each case;
defect-only seed changes are not presented as spatial variation.
The plan also fixes `minimum_success_rate` (at least 0.9),
`maximum_axis_error_m` (at most 0.04), `maximum_inference_p95_ms` (at most 80),
`max_step_seconds` (at most 30), and `max_cartesian_speed_m_s` (at most 0.2).
Its canonical SHA256 is fixed before dataset sealing or training. Any held-out
seed/episode in the dataset blocks training. Evaluation retains every
before/after trial and retry, including failures; no post-hoc favorable subset.

## State and evidence

- Teaching: `starting -> recording -> finishing/finalizing -> uploading -> ready`.
  Cancellation goes through `cancelling -> cancelled`; errors become `invalid`
  or `blocked`. A physical terminal command is **not** a ready dataset.
  `ready` requires the validated manifest uploaded last. Source counts distinguish
  human teleoperation, reference controller and generated policy demonstrations.
- Paid jobs: `submitting -> submitted -> running -> terminal`, with
  `submission_unknown` after an ambiguous submission and `cancelling` before
  confirmed cancellation. A durable owner/fingerprint claim precedes the only
  submit call. The backend job name is deterministic. Ambiguous retries reconcile
  by name/tags; they never blindly issue another paid submission. A process crash
  before/after submission cannot create a second job. Terminal snapshots cannot
  be revived. A queued/submit ACK is not training success.
  `deadline` remains the original server approval bound. Additive
  `job_deadline_utc` records the earlier immutable operator/native bound;
  `backend_status` and raw `azure_status` preserve the actual provider receipt.
  Expired active receipts, not just missing receipts, trigger a durable
  per-job ETag cancellation reservation before one worker request.
  `cancellation` contains `request_id`, `reason: user|deadline`, `requested_at`,
  `state: claimed|acknowledged|uncertain|forbidden` and nullable `error_code`.
  This is a request/outcome record, never proof of termination. A queued/running
  receipt after a cancellation ACK keeps the API run `cancelling` while exposing
  the actual provider state; 403 and uncertain transport remain explicit.
  Missing receipts after expiry remain unconfirmed/reconcilable, not a terminal
  local timeout that could hide a late-queued paid job.
  The separately deployed exact-target worker tick shares a durable Blob cancel
  fence with the API. New paid admission requires its fresh, immutable-bound
  enrollment heartbeat; no UI polling/session can substitute for that scheduler.
  All scheduler switches/owner/target lists default off/empty, and the source
  template is not evidence of a deployed timer. See the
  [private worker enrollment contract](../apps/learning_worker/README.md#independent-default-off-deadline-reconciliation).
- Training success requires an actual Azure ML job ID, verified owner/dataset/
  parent/profile/config provenance, optimizer steps greater than zero, and
  changed model weights. Loss/steps are nullable until actually reported.
  `act_auxiliary` remains visibly separate; no GR00T-to-ACT fallback.
- Paired evaluation distinguishes `improved`, `not_improved`, `inconclusive`.
  P0/P1 share the frozen trial conditions and bounds. Each trial identifies
  actual model SHA, measured outcome/error/time, safety violations, inference
  latency, predict/action counts and zero reference-route calls. A failed or
  incomplete report cannot publish a candidate. A passed physical gate without
  improvement is not a claim of learning benefit.
  A native quality gate can write a complete verified report and then exit
  nonzero. The worker/API retain that report alongside Azure `failed`; private
  and explicitly curated public views show the failures without relabeling the
  job as successful. Missing or invalid failed-job evidence stays explicitly
  unverified, never replaced by an example comparison.
- Release is a separate, explicit human review after artifact and paired-gate
  verification. Foundry can propose/select an already released skill but cannot
  publish its own model, increase limits or approve motion/cost.
- The learning coach also claims a request durably before its paid Foundry
  call. Exact retries return the original recorded proposal/response ID.
  Interrupted responses expire as unconfirmed instead of silently repeating
  model inference. Its stored proposal remains advice, never approval.
  Its registered strict function schema uses the same minimal JSON Schema
  subset as inspection: every property is required, unused references/counts
  are explicit `null`, and extra fields are forbidden. Defaults, UUID `format`,
  text length/pattern and numeric bounds are not sent as unsupported service
  keywords. The unchanged Pydantic parser and scoped proposal checks enforce
  all UUID, text, optimizer, project, dataset and release constraints after
  inference. SDK serialization tests do not replace actual Foundry registration
  or imply successful policy learning.

## Empty-store first-policy bootstrap

`project_kind: bootstrap` is separately authorized by
`LEARNING_BOOTSTRAP_PRINCIPAL_IDS`, empty by default. Ordinary operators receive
403 even if the main learning capability is enabled. Bootstrap sets
`baseline_release_id: null` and a registered `pretrained_artifact_id`.
`TrainingParent` has role `pretrained_train_only` and never resolves through
`/api/runs` as an executable `PolicyRelease`.

Use the trusted, explicit `apps.learning_worker.register_parent` CLI only after
the license is approved and the actual pinned weights are already present.
It validates the native model manifest, scope, full inventory and pinned source
before registration. It does not download weights or accept a model license.
There is no API that accepts caller-supplied "ready" model metadata.

Bootstrap still needs real teaching data, actual optimizer updates and changed
weights. It evaluates the resulting Franka candidate under
`comparison_kind: reference_bootstrap`: a distinct `BootstrapReport` records
the reference-controller code SHA, candidate model SHA, complete reference and
candidate trials, and physical/safety quality gate. It does **not** invent a P0
model SHA or an improvement conclusion. Only the allowed bootstrap operator can
explicitly review a passing report and create the first actual P0 release.
Normal customer P1 adaptation then uses that P0 and the separate paired-policy
improvement gate.

Releases pin their approved task/instruction, model family/digests and explicit
`environment_cases` allowlist. Review/record creation is not proof the model is
already installed on a GPU host. GPU artifact verification, model-specific
timing attestation and runtime catalog activation remain required.

## Runtime and learner ports

CPU-safe DTOs live in `apps.api.learning_models`; protocols and trusted job
receipts in `apps.api.learning_ports`. No Isaac, PyTorch or training SDK import
is required to validate a client request.

Runtime keeps reference-only `/v1/commands` unchanged. New private bridge routes:
`POST /v1/teaching`, `GET /v1/teaching/{id}`,
`POST /v1/teaching/{id}/input|finish|cancel`,
`POST /v1/policy/commands`, `GET /v1/commands/{id}/capture`.
Owner is the authenticated bridge header only. Teaching start has separate
flat scene/observation fields, session/lease/command UUIDs,
`session_expires_at` (at most 300 seconds), profile, task
`{task_id,instruction,goal_id}`, and explicit `demonstrator_kind`.
It does not repurpose the reference/learned `MotionCommand`'s 30-second deadline.
The required teaching `split` must match the selected immutable saved scene;
only the private runtime may accept an integration-test capture outside this
train/validation audience API, and that data is not approved for training.
The public operator jog is an intent, not a trusted browser timestamp. The API
first issues a persistent grant on `/arm`, bound to the original owner/session/
lease/epoch/next sequence/exact delta and gripper with at most a one-second TTL.
`/jog` must echo this fresh, single-use grant: even a first request delayed for
seconds is rejected, rather than acquiring new authority on arrival. The API
conditionally persists the request fingerprint and sequence, consumes the grant,
and sets runtime expiry to `min(original grant expiry, server now + 250 ms)`.
Retries cannot renew either deadline or blindly repeat an uncertain motion POST.
Released-deadman zero-motion/hold inputs can advance sequence gaps so a delayed
lower-sequence motion cannot execute after release; stop-only inputs do not
require a grant or a fresh browser ETag (the server still performs its own CAS).
The UI discards an arm response if the input is no longer held or the tab is
hidden. Jog is at most 1 cm total displacement. Automated G0/scripted data uses
`reference_controller`; only actual operator teaching uses `human_teleop`.

`LearningJobs` exposes `preflight(actor,specification)`, `submit(actor,specification)`,
`status(actor,run)` and `cancel(actor,run)`. `JobSpecification` binds owner,
project, run, baseline, and dataset/candidate. `BackendJob` must carry actual
Azure job ID/name, owner and immutable specification hash, actual state,
optional real metrics and verified candidate/report. No in-memory production
job manager or fabricated receipt is acceptable.

The separate policy-learning worker adapter wraps
the model-family-specific, learner-owned SDK. The closed registry selects
`learning.smolvla.azure.PolicyJobs` for explicit `smolvla`; legacy GR00T types
are distinct and blocked for production model use. The worker binds actual
Azure parent pipeline ID and child component ID, does not infer optimizer
metrics from `Completed`, and verifies native model/report outputs.

`native_plan_sha256` preserves the native plan artifact hash separately from
the API authorization's `evaluation_plan_sha256`. Both are bound to the reviewed
job specification; every trial's seed/environment/revision, measured initial
pose, builder digest, failures/retries and actual action counters are retained.
The API does not need Torch, Isaac or the Azure ML SDK in its own runtime.
The separate worker lock and deployment procedure are in
[`apps/learning_worker/README.md`](../apps/learning_worker/README.md).
Until real train-actuate-held-out gates pass, production capability stays off.

## Protected operator sequence

First use `GET /api/learning/capabilities`; do not force-enable a blocked model.
Each successful resource response supplies the next `ETag`.
Authenticated operators then:

1. Create the immutable project using actual registered P0 (or the privileged
   train-only bootstrap path), reviewed cases and bounded budgets.
2. Explicitly activate an approved train/validation case, then start a
   `human_teleop` or explicit `reference_controller` teaching session using
   its `case_id`, the project ETag and `motion_approved: true`. Automated G0 must use
   `reference_controller`, not claim a customer's human demonstration.
3. For held controls, request `/arm`, then echo its grant on `/jog` only while
   still held/visible. Never compute authority from browser wall time.
4. `/finish` then poll the original session. Physical completion can precede
   upload. Old capture A is reconciled via its original command's capture route,
   even after scene B begins; it is never rebound to B.
5. Seal only validated `ready` captures, then submit an explicitly approved
   paid train request. Preserve its request ID through uncertain responses.
6. Read the actual named job and verified candidate; submit the fixed evaluation.
   Review all failures and exact model family/digests before policy release.
7. Select the immutable `policy_release_id` on a new
   `execution_mode: released_skill` run, omit the old freeform inspection
   instruction, and review the canonical task/goal/model and original PNG.
   This creates a distinct `SkillPlan` with no CV classification or Foundry
   response ID; it does not run the inspector. Approve separately with
   `{"skill_plan_id": "<exact-displayed-id>"}`, never `plan_response_id`.
   The API rechecks the immutable release and fresh scene before dispatch.
   A normal part intentionally placed in quarantine is not labeled defective.
   Legacy inspection remains a separate `inspection` mode using actual Foundry
   classification and its own approval reference; see
   [`http-api.md`](http-api.md#explicitly-selected-released-motor-skills).
   Learned dispatch goes only to `/v1/policy/commands`,
   carrying required `policy_type`, task, model SHA, release and control profile.
   Actual execution must report matching type/SHA and applied actions with zero
   reference-route calls. No missing-model fallback exists.
