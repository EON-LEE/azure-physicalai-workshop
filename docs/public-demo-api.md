# Public demonstration viewer

The audience experience opens at `/` without a login, redirect, token, or
Microsoft authentication request. `/operator` retains the existing authenticated
console for editing, approval, motion, private history and evidence.
Do not remove authentication from existing `/api/environments`, `/api/runs`,
`/api/runtime`, schema or operator endpoints.

## Public snapshot

`GET /api/demo` is anonymous, read-only and bounded. It does not invoke Foundry,
start a GPU, activate a scene, create a run, read private run history, or grant an
anonymous identity access to an operator's data.

Response:

```json
{
  "api_version": "public-demo-v1",
  "access": "public_read_only",
  "deployment": "azure",
  "mode": "reference",
  "observed_at": "2026-09-21T00:00:00+00:00",
  "scene": {
    "id": "inspection-cell-v1",
    "name": "Inspection and sorting cell",
    "length_unit": "m",
    "stations": [
      {"id": "supply", "role": "source", "position_m": [0.35, 0.25, 0.2]},
      {"id": "inspection", "role": "inspection", "position_m": [0.5, 0.1, 0.2]},
      {"id": "accepted", "role": "accepted", "position_m": [0.42, -0.22, 0.2]},
      {"id": "rejected", "role": "rejected", "position_m": [0.22, -0.38, 0.2]}
    ],
    "robot": "Franka reference arm",
    "data_origin": "synthetic_reference_configuration"
  },
  "simulation": {
    "status": "not_published",
    "live_available": false,
    "message_code": "live_not_published",
    "frame_url": null
  },
  "presentation": null,
  "agent": {
    "provider": "microsoft_foundry",
    "connectivity": "verified",
    "verified_at": "2026-09-20T15:34:00+00:00",
    "verification_scope": "connectivity_only"
  },
  "learning": {
    "status": "cpu_smoke_verified",
    "execution_location": "azure_acr",
    "data_kind": "test_fixture",
    "optimizer_steps": 1,
    "quality_verified": false
  },
  "capabilities": {
    "anonymous_control": false,
    "anonymous_editing": false,
    "public_live_video": false
  }
}
```

`mode` is `reference` or `live`. Only `live` permits the actual camera feed.
Simulation status is `not_published`, `loading`, `unavailable`, or `ready`.
Agent connectivity is `configured`, `verified`, or `unavailable`;
`verified_at` may be null. This is connectivity evidence, not inspection accuracy.
Learning status is `not_published` or `cpu_smoke_verified`, with the other fields
shown above (optimizer_steps may be 0). The pinned CPU smoke uses test fixtures,
not actual robot demonstrations.

Reference mode shows an explicit unavailable/not-published live camera, not a
schematic, stock footage or replay substitute. Customer-use explanations, local
configuration authoring and explicitly requested recorded cases are separate from
the live camera and never claim that a new task has run.

## Customer task and local configuration authoring

The page explains the inspection-to-sorting customer workflow and distinguishes
watching a published run from designing a new experiment. The public planner
changes only local state and a downloadable reference-derived JSON document.
`anonymous_editing=false` continues to mean no anonymous **server** edits; choosing
a local experiment is not a request to activate it. The bounded operator link
hands off an unsaved draft, with no automatic save, inference or motion.
See [customer environments](customer-environments.md) for the exact actions and limits.

## Recorded customer outcome comparison

`GET /api/demo/cases` returns `public-demo-cases-v1`, with
`source=recorded_reference_runs`, a nullable `presentation_id`, and up to three
`cases`. The page requests this comparison explicitly, not as a camera fallback.
By default it uses the current presentation. An operator can pin
`PUBLIC_DEMO_CASES_PRESENTATION_ID` (Bicep `publicDemoCasesPresentationId`) to an
earlier presentation with the **same approved owner and immutable reference pair**.
The snapshot's additive `recorded_cases_presentation_id` identifies this source
separately from the live presentation. This preserves a genuine withheld example
when a fresh live window starts, without letting visitors select arbitrary history.
The curated kinds are:

- `normal_route`: the first physically successful, correctly inspected normal part.
- `defect_route`: the first physically successful, correctly inspected defect part.
- `withheld`: the first reference inspection mismatch with no approval or execution.

No case is invented when its evidence is absent. These are illustrative cases,
not a representative accuracy sample. The current presentation's actual counters
remain separate and are not replaced by a success-only comparison.

Each case carries its cycle, scenario, actual classification/reason, planned
station/position, original observation ID/time, `motion_authorized`, measured
`result`, and nullable `physical_duration_seconds` from the recorded approval to
physical completion. The fixed image path is `/api/demo/cases/evidence`.
Withheld cases have no executed duration or measured final position.

`GET /api/demo/cases/evidence?presentation_id=<id>&observation_id=<UUID>` returns
only one of those selected runs' checksum-verified original input PNGs, with
`X-Presentation-Id`, `X-Frame-Id` and `X-Captured-At`. It is **recorded input, not
LIVE**; do not fabricate physics steps or apply a live-frame age rule. An expired
presentation may still expose its selected recorded evidence until the operator
changes/revokes the publication.

Selection uses only the pinned presentation record's deterministic outcome IDs;
it never lists or scans an operator's run history. Every selected run is revalidated
against the approved owner, exact environment document/revision, original generic
instruction, request fingerprint, authorization window, image/plan linkage and
measured outcome. Image responses revalidate the selected publication/case after
Blob retrieval. Unknown/private observations or changed publication IDs are 409;
missing or inconsistent durable evidence fails closed as 503. All responses are
`Cache-Control: no-store`; POSTs remain unsupported. This read path never contacts
the live simulator or initiates Foundry/motion.

## Learning publication training progress (checkpoint timeline, not a live loss curve)

`GET /api/demo/learning`'s `publication.training` object (see
[learning-api.md](learning-api.md)) carries an additive `checkpoints` array:

```json
"training": {
  "optimizer_steps": 200,
  "model_sha256": "...",
  "parent_model_sha256": "...",
  "dataset_sha256": "...",
  "created_at": "2026-09-21T00:00:00+00:00",
  "loss": null,
  "checkpoints": [
    {
      "optimizer_steps": 100,
      "loss": null,
      "measured_at": "2026-09-20T18:04:11+00:00",
      "checkpoint_sha256": "..."
    },
    {
      "optimizer_steps": 200,
      "loss": null,
      "measured_at": "2026-09-20T18:41:52+00:00",
      "checkpoint_sha256": "..."
    }
  ]
}
```

Each entry is a verified, hash-checked training checkpoint manifest read from the
job's own output, not a resampled/animated or interpolated point: `optimizer_steps`
is the checkpoint's own cumulative step count, `measured_at` is the storage
`last_modified` time of the checkpoint manifest (its only authoritative timestamp),
and `checkpoint_sha256` is the manifest's verified content hash
(`learning/smolvla/checkpoints.py::list_checkpoint_manifests`). Entries are
strictly ordered by step and never exceed the final reconciled
`training.optimizer_steps`; the backend (`apps/api/learning_models.py`'s
`TrainingMetrics.history`) and the public contract
(`apps/web/src/public/learning-contract.ts`) both reject out-of-order or
over-the-final-count data rather than publish it.

`loss` is always `null` today: the current training loop
(`learning/smolvla/train.py`) and checkpoint manifest schema
(`learning/smolvla/checkpoints.py::_manifest`) do not persist a per-step loss
value, only step/state/origin/file metadata. The public viewer
(`apps/web/src/public/LearningPublication.tsx`) shows this truthfully as "loss
미게시" per point and never invents or interpolates a loss curve. An empty
`checkpoints` array (for example while `feat-p0-retrain-resume` has no surviving
run) renders as an explicit "학습 진행 곡선 공개 기록 없음" message, not a flat
or placeholder line.

**Connection point for `feat-p0-retrain-resume` / `feat-learned-policy-eval`:**
this feature requires no training-loop code changes to start showing real data.
It already reads whatever checkpoint manifests a training job publishes at
`<output_prefix>/<run_id>/checkpoints/step-<N>/checkpoint.json` (the existing
full-state checkpoint publisher's own layout; see
[policy-learning.md](policy-learning.md)). A manifest is only surfaced here if
its `origin.specification_sha256` matches the exact approved job specification
being reconciled (`apps/learning_worker/artifacts.py::VerifiedArtifacts.training_progress`),
so resuming/retrying the P0 job under its real specification is sufficient: once
checkpoints exist again, the next successful job status poll
(`apps/learning_worker/backend.py::_receipt`) picks them up automatically, and
`/api/demo/learning` starts returning a non-empty `checkpoints` array once that
training run is published. If a future training-loop change adds a genuine
per-step loss value to the checkpoint manifest, only `_manifest()`'s schema and
`list_checkpoint_manifests`'s return mapping need to start forwarding it — the
rest of this pipeline (`TrainingSample.loss`, the public schema, and the UI's
"loss 미게시" fallback) already supports a non-null value without further
changes.

## Actual live frames, only after deliberate publication

`GET /api/demo/frame?camera=overview|inspection&epoch=<UUID>` is anonymous but works only when
the operator has explicitly enabled a public live publication with a fixed
owner, environment ID and immutable revision. The backend rejects new/private
revisions rather than automatically republishing them.

A successful response is a real `image/png`, with `X-Frame-Id`,
`X-Captured-At`, `X-Physics-Steps`, `X-Scene-Epoch`, and a precise UTC `X-Server-Time`.
Fetch without an Authorization header.
The API rechecks the 2000 ms camera-age bound after publication fencing. Public
viewers compare capture time with server time, include the full monotonic
request/body duration conservatively, and expire LIVE at the unchanged five-second
display deadline. A skewed viewer clock must not relabel a fresh image as future
or keep a frozen image live. `/healthz` also returns `X-Server-Time` for bounded
validation-clock calibration; it does not change captured timestamps.
Use bounded polling only while visible and in live mode; revoke object URLs and
abort requests on navigation/unmount. Missing publication or live failure returns
503 with the normal structured error envelope, never a substitute frame.
An epoch mismatch is 409. Omitting `epoch` supports legacy clients, but paired
presentation clients must send the current `presentation.scene_epoch`. The API
checks the pinned scene before and after capture, including same-revision resets;
it never returns an old cached image across epochs. All image responses are
`Cache-Control: no-store`.
For a paired presentation, non-null motion telemetry naming a different command
also revokes live viewing, even in the same epoch after the presentation completed.
The command binding is checked both before and after capture: mismatch returns
409, and the snapshot suppresses current decision/motion while retaining labeled
historical outcomes. Idle telemetry with no command and matching completed-command
telemetry remain valid. Recorded original inspection evidence remains owner- and
presentation-bound; it is not recaptured from another command.
If a snapshot's scene/command check races a persisted cycle/epoch/run binding
change, it validates the current publication again and retries capture once using
only that new scope. It never combines the prior decision/result with new telemetry
or projects an old moving cycle as stopped solely because the next cycle began.
Unchanged-binding mismatches retain their guards; an unresolved retry mismatch
returns 503 rather than looping. Preparing/loading still publishes no live frame,
and rereading does not renew heartbeat or authorization timestamps.
If completion invalidates a camera between the status and frame reads, the API
revalidates the same scene and retries that read-only capture once. A still-refreshing
camera remains unavailable/loading, not a fabricated robot stop; recovered frames
must pass the unchanged epoch, command, publication and freshness checks.
The presenter's completion wait likewise retries a recovered-camera race only
within its original five-second deadline, without changing the authorized scene.
The HTTP bridge preserves `camera_not_ready` only for a structured 503 from
`GET /v1/observation`; otherwise those recovery paths would mistake refresh for
a dependency outage. Other 503 responses and malformed errors still fail closed.

The public snapshot never exposes tenant IDs, object IDs, auth scopes, keys,
private run IDs, private instructions, manifests, arbitrary artifact paths, or internal
simulator exceptions. The scene in this release is the approved synthetic
reference scene, not a public projection of all Cosmos environments.

## Bounded alternating reference presentation

The optional additive `presentation` field uses the frozen public-demo-v1 shape
below. `null` means no presentation is configured or the dedicated record has
not been created yet. It does **not** hide storage, authorization, or validation
errors: those return a generic structured 503. `/api/demo` uses `no-store` when
a presentation is configured. Legacy single-scene publication still works when
none of the three new paired settings is supplied.

This is an illustrative payload, **not a claim of live completion**:

```json
{
  "id": "approved-reference-window",
  "status": "awaiting_motion",
  "cycle": 2,
  "total_cycles": 20,
  "scenario": "surface_defect",
  "instruction": "Inspect the synthetic part and sort it using the configured accepted or rejected station.",
  "updated_at": "2026-09-21T12:00:15+00:00",
  "expires_at": "2026-09-21T12:30:00+00:00",
  "scene_epoch": "12345678-1234-4234-8234-123456789abc",
  "run_id": "23456789-1234-4234-8234-123456789abc",
  "decision": {
    "classification": "rejected",
    "summary": "Illustrative visible-surface inspection summary.",
    "target_station_id": "rejected",
    "observation_id": "34567890-1234-4234-8234-123456789abc",
    "captured_at": "2026-09-21T12:00:12+00:00",
    "image_url": "/api/demo/evidence"
  },
  "motion": null,
  "result": null,
  "counts": {
    "attempted": 2,
    "succeeded": 1,
    "failed": 0,
    "inspected_correctly": 2,
    "physically_completed": 1
  }
}
```

| Field | Contract |
|---|---|
| `status` | `preparing`, `inspecting`, `awaiting_motion`, `moving`, `completed`, `stopped`, or `failed` |
| `scenario` | `normal` on odd cycles, `surface_defect` on even cycles |
| `cycle`, `total_cycles` | Integers, 1 through 1000; current cycle cannot exceed total |
| `scene_epoch`, `run_id` | UUID or null; cleared at the start of each new cycle |
| `decision` | Null or the original Foundry classification, summary, target, observation UUID, original capture timestamp, and literal `/api/demo/evidence` |
| `motion` | Null or `{status, phase, part_position_m, target_position_m}` |
| `motion.status` | `queued`, `running`, `succeeded`, `failed`, `cancelled`, `timed_out`, or `cancelling` |
| `motion.phase` | Null or `idle`, `approaching`, `grasping`, `lifting`, `inspection_station`, `transporting`, `releasing`, `returning`, `complete`, `stopped` |
| `motion.part_position_m` | Three finite numbers or null; only current matching real telemetry |
| `motion.target_position_m` | Three finite numbers from the actual plan's canonical target |
| `result` | Null or `{status, physical_success, inspection_correct, final_position_m, completed_at, message}` |
| `result.status` | `succeeded`, `failed`, `cancelled`, or `timed_out` |
| `result.inspection_correct` | Boolean, or null if no inspection decision exists |
| `result.final_position_m` | Three finite numbers or null; never inferred from a desired destination |
| `counts` | Integer `attempted`, `succeeded`, `failed`, `inspected_correctly`, `physically_completed` |

All timestamps are timezone-aware ISO 8601. `completed` means the bounded
presentation finished its authorized cycles; use the independent result and
counts to evaluate quality, not this lifecycle label. `succeeded` requires both
correct inspection and physically verified sorting. An acknowledged command or
model text is never physical completion.

`GET /api/demo/evidence?observation_id=<UUID>` returns **only the current
authorized presentation run's original input PNG**, from Azure Blob with checksum
verification. It returns `X-Frame-Id` equal to the requested observation UUID and
`X-Captured-At` equal to the original capture time. This is recorded inspection
input, not a current camera stream, so do not apply live-frame freshness rules.
The query is required; an old observation returns 409 (or generic unavailable).
Between cycles, the runner holds the verified result for up to five seconds
within the remaining authorized window, so visitors can read the outcome before
the next scene is activated. This hold is after physical completion and does not
extend the task's 30-second deadline.
There is no owner/run/blob-path selector. The API rereads the presentation binding
after fetching pixels to detect cycle changes in flight.

Clients must clear previous camera/evidence images when cycle, epoch, or
observation changes. Decision, motion and result become null on a new cycle;
historical counters persist but an old outcome is never relabeled as the new
scenario. A frame/backend failure sets base `simulation.live_available=false`;
the last recorded terminal outcome can remain visible as history. Expired or
stale active state projects `stopped`, with no invented cancellation/completion.
Moving/awaiting-motion heartbeats expire after 10 seconds; bounded synchronous
Foundry planning allows 150 seconds. An unconfirmed queued/running/cancelling
motion is hidden when stopped, rather than shown moving forever.
Same-owner/environment/revision/epoch `loading` during camera refresh is not a
stop event: the snapshot reports camera `loading`, `live_available=false`, and
no live frame, while retaining the bounded presentation lifecycle. Heartbeat,
authorization expiry, foreign-command and unavailable/changed-scene guards still
apply. After reconciling genuine terminal execution the runner waits at most
five seconds, also capped by its remaining authorized window, for fresh
post-completion overview and inspection frames before advancing. This rendering
wait does not extend the 30-second physical command deadline or reuse invalidated
frames; timeout stops the presentation while retaining already-verified outcomes.

Motion phase comes only from optional private `SimulationStatus.motion`:

```json
{
  "command_id": "23456789-1234-4234-8234-123456789abc",
  "phase": "grasping",
  "object_position": [0.35, 0.25, 0.2],
  "target_station_id": "rejected"
}
```

`command_id` and `target_station_id` may be null for idle telemetry. The public
phase/position are null if absent, disconnected, or not matched to the current
epoch, command and planned target. They are never computed from elapsed time.

### Exact deployment settings

Configure these identically on the web API and explicitly authorized runner:

| Environment variable | Meaning |
|---|---|
| `PUBLIC_DEMO_PUBLISH_LIVE=true` | Explicit public reference publication |
| `PUBLIC_DEMO_OWNER_ID` | Approved operator object UUID in `ENTRA_TENANT_ID` |
| `PUBLIC_DEMO_ENVIRONMENT_ID` | Approved normal environment ID |
| `PUBLIC_DEMO_REVISION` | SHA-256 revision of the exact approved normal document |
| `PUBLIC_DEMO_DEFECT_ENVIRONMENT_ID` | Distinct approved surface-defect environment ID |
| `PUBLIC_DEMO_DEFECT_REVISION` | SHA-256 revision of the exact approved defect document |
| `PUBLIC_DEMO_PRESENTATION_ID` | Single-use, lowercase identifier for this approved window, at most 64 characters |

The new three settings are all-or-nothing. Both documents must already be saved
under the configured owner. They must exactly match
`examples/inspection-cell.json`, apart from environment ID/display name and the
explicit normal seed **42** versus defect seed **43**. Geometry, workflow,
payload/speed limits, freshness and live 30-second deadlines cannot vary. Pinning
a custom document hash alone does not authorize its public exposure.

The runner additionally requires `ALLOW_REFERENCE_PRESENTATION=true` and
`PRESENTATION_MAX_SECONDS` (30 through **21600**, explicitly supplied).
`PRESENTATION_CYCLES` defaults to **20**, maximum **1000**. Optional legacy
`PRESENTATION_ID`, if supplied, must equal `PUBLIC_DEMO_PRESENTATION_ID`.
All normal managed-identity API/Cosmos/Blob/Foundry/simulator settings still apply;
no new credentials or anonymous identity are introduced.

Launch only in an approved Azure job using the root locked environment:

```bash
uv run --locked python -m scripts.run_reference_demo
```

The runner creates one dedicated Cosmos `presentation:<id>` document in the
approved owner's partition. It uses create-if-absent and ETag conditional writes,
not an in-memory production store and not `list_runs`. The authorization window,
total cycles, owner, both revisions and runner claim are persisted before any
activation/inference/motion. Deterministic UUIDv5 run IDs bind owner,
presentation, both environment IDs/revisions and the 1-based cycle. Changing
duration/cycles cannot reuse the ID. A competing or crashed runner cannot renew
or take over the claim; a completed/stopped/failed invocation only returns its
recorded outcome without more paid calls or motion. Reconcile any unconfirmed
command and deliberately configure a **new approved presentation ID** before
another window; automatic restart is not reauthorization.

For each cycle, the runner activates the pinned revision and requires a newly
ready epoch, then sends actual observation image plus the same generic task
through the existing Foundry planner. Seed, expected label and evaluation result
are **never sent to Foundry**. The original input PNG, response decision and
model-response ID remain in the existing owner-scoped run/artifact records.
The dedicated presentation references only its deterministic current run, and
the public reader validates document, request fingerprint, instruction,
authorization time, epoch and artifact binding before projecting a safe DTO.
An original capture may precede run creation by at most the configured observation
age capped at 2000 ms, inclusive, because the live bridge returns fresh cached
camera frames. It cannot be later than the persisted run update. Actual capture
timestamps are retained unchanged; epoch, observation, object and artifact-path
checks still apply. On failure the runner marks its owned claim terminal before
cleanup; rejected evidence can never recursively strand it in `inspecting`.
Cancellation requires the same validated run/request/command identity even when
observation evidence is rejected, and never records an unverified outcome.
No arbitrary operator history, prompt, account identifier or artifact path is
published.

Synthetic expected labels are used only for evaluation **after** the real model
decision. A mismatch preserves the decision, cancels before motion, publishes a
failed inspection and counts no physical success. The operator window and ETag
claim are checked after inference and immediately before approval. The existing
30-second simulator task deadline/final target tolerance remain unchanged.
The runner reconciles actual terminal events, stops after three unsuccessful
cycles (inspection or physical), and never claims a cancellation ACK is terminal.
On expiry/error it may use at most a bounded extra cancellation-confirmation
window, but cannot authorize another command. Storage errors propagate; no
successful emptiness or private exception text is substituted.

This feature does not relax `live_acceptance`'s 20-episode/18-success physical
suite, imply balanced evaluation accuracy, or turn a public demo into all G0-G7
release evidence. A Spot GPU may disappear; a publication is not an always-on
availability guarantee. No live execution is implied by backend unit tests.

## UI priorities

Korean-first SE audience page, immediate workshop/demo content, not a login card
or another marketing-only landing page. Show the workcell, inspect/sort storyline,
Azure/Foundry/NVIDIA roles and honestly labeled verified versus pending capability.
Use an operator link as a secondary navigation item, not an entry requirement.
When GPU is absent, the audience can still explore the explained scenario without
login, but the page must state that live simulation is not available.

Follow keyboard/focus, reduced-motion, semantic navigation, image dimensions,
responsive layout, long-text and error-state requirements. No autoplay robot
motion. No external fonts/CDNs. Keep the existing strict CSP.
