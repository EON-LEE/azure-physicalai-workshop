# Owner-scoped teaching and policy learning API (v1)

This is an implementation contract, **not evidence of a trained or released
GR00T model**. The feature defaults off. Existing inspection, reference motion,
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
| `GET /api/learning/capabilities` | `enabled`, readiness/block reason, approved profiles, supported policy types; configuration is not a successful training test |
| `POST /api/learning/projects` | `CreateProject`; returns immutable `LearningProject` |
| `GET /api/learning/projects` | Owner project list |
| `GET /api/learning/projects/{id}` | Project with ETag |
| `POST /api/learning/projects/{id}/teaching-sessions` | `request_id`, `source: human_teleop|reference_controller`, `motion_approved: true` |
| `GET /api/teaching-sessions/{id}` | Reconciled `TeachingSession`; physical completion and capture readiness remain separate |
| `POST /api/teaching-sessions/{id}/arm` | `request_id`, `lease_id`, `epoch`, `sequence`, `deadman: true`, `delta_xyz_m`, `gripper`; returns a server control grant valid for at most one second; never moves |
| `POST /api/teaching-sessions/{id}/jog` | Same intent plus `grant_id`; server stamps expiry on first admission. Stop-only `deadman:false` requires zero delta/hold and no grant |
| `POST /api/teaching-sessions/{id}/finish` | `request_id`, `lease_id`, `epoch`; does not assert capture readiness |
| `POST /api/teaching-sessions/{id}/cancel` | Same binding; does not assert cancellation until confirmed |
| `POST /api/learning/projects/{id}/datasets` | `request_id`, unique `teaching_session_ids`; only verified uploaded eligible captures |
| `GET /api/learning/datasets/{id}` | Frozen `DatasetVersion` |
| `POST /api/learning/projects/{id}/train` | `request_id`, `dataset_id`, `parent_release_id`, `policy_type: gr00t_n1_5`, `optimizer_steps`, `paid_approved: true`, `maximum_cost_usd` |
| `POST /api/learning/projects/{id}/evaluate` | `request_id`, `candidate_id`, `baseline_release_id`, `evaluation_plan_sha256`, `motion_approved: true`, `paid_approved: true`, `maximum_cost_usd` |
| `GET /api/learning/jobs/{id}` | Reconciles the actual named job only; never submits or restarts it |
| `POST /api/learning/jobs/{id}/cancel` | `request_id`; cancellation state from the actual backend |
| `GET /api/learning/candidates/{id}` | Verified candidate; not a released skill |
| `POST /api/policy-releases` | `request_id`, `candidate_id`, `evaluation_run_id`, `release_approved: true`; exact evaluation ETag and passing gates required |
| `GET /api/policy-releases/{id}` | Immutable reviewed release |
| `GET /api/demo/learning` | Explicitly curated publication only; no owner enumeration, dataset/model paths, identities or private histories |

`CreateProject` contains `request_id`, `display_name`, `task_id`, `instruction`,
`goal_station_id`, `environment_id`, `revision`, `baseline_release_id`,
`control_profile_id`, `evaluation_plan` and `budget`.

The initial reviewed profile is `franka-position-hold-10hz-v1`, not the existing
60 Hz reference profile. The task goal must be a station in the pinned saved
environment. Baseline P0 must be a real approved learned policy, not a scripted
reference route relabeled as a policy. Prepared P0 provenance remains visible.

`budget` fixes `teaching_seconds` (5..300), `training_seconds` (1..86400),
`evaluation_seconds` (1..21600), `optimizer_steps` (1..100000), and
`maximum_cost_usd` (positive, decimal USD). Job-specific approval cannot exceed
it. The amount is an authorization ceiling, **not a claim that Azure provides
an automatic spending stop**: the production admission adapter must verify
approved SKU, duration, price and quota before dispatch; otherwise block.

`evaluation_plan` fixes UUID `id`, 20..100 unique held-out `seeds`,
`held_out_episode_ids`, `minimum_success_rate` (at least 0.9),
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
- Release is a separate, explicit human review after artifact and paired-gate
  verification. Foundry can propose/select an already released skill but cannot
  publish its own model, increase limits or approve motion/cost.

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

The separate GR00T worker adapter is expected to wrap
`learning.gr00t.azure.create_plan` and injected `Gr00tJobs` without changing
legacy ACT tooling. The API does not require training dependencies in its
runtime. Until real adapters and train-actuate-held-out gates are verified,
the default learning capability remains disabled.
