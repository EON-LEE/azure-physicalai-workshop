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

Reference mode must be a clearly labeled **scene schematic / scenario explainer**,
not a fake moving robot, stock footage, invented defect score or recorded/live
simulation. Interactive normal/rejected-path selection explains what the system
does; it must not claim execution or send a write to any API.

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
