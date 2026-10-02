# Azure runtime HTTP contract (v1)

This is the implementation contract, not a claim that the service is deployed.
Production runs entirely on Azure. WSL is for authoring, builds, and automated
tests only. Test doubles must not be bundled as a production demo backend.

The React application and API share one origin. API paths begin with `/api`.
Authenticated requests use `Authorization: Bearer <Entra access token>`.
The server validates the configured tenant, audience and delegated scope.
The browser must never contain Foundry, Storage, Cosmos, or simulator secrets.

## Public bootstrap

`GET /api/config`

```json
{
  "api_version": "v1",
  "deployment": "azure",
  "auth": {
    "tenant_id": "configured-tenant-uuid",
    "client_id": "configured-spa-client-uuid",
    "scope": "api://configured-api-client-uuid/access_as_user"
  }
}
```

Use MSAL redirect login and silent token acquisition. The application must not
silently switch to an anonymous or mocked mode if configuration or login fails.
`GET /healthz` is process liveness only, not proof of working Azure dependencies.

## Errors

Non-2xx errors use:

```json
{"error": {"code": "invalid_environment", "message": "Actionable error", "details": []}}
```

401 means authentication is required, 403 forbidden, 409 conflicting state,
422 invalid input, and 503 a required dependency is unavailable.
No error means "use a default successful simulation instead".

## Runtime and live frames

`GET /api/runtime` requires authentication:

```json
{
  "deployment": "azure",
  "simulation": {
    "backend": "isaac_sim",
    "status": "ready",
    "environment_id": "reference-cell",
    "revision": "sha256hex",
    "epoch": "simulation-world-uuid",
    "physics_steps": 100,
    "message": null
  },
  "agent": {"provider": "microsoft_foundry", "configured": true},
  "storage": {"provider": "azure_cosmos_blob"},
  "release_ready": false
}
```

Simulation status is `ready`, `loading`, `unavailable`, or `occupied`.
Environment, revision, epoch and steps may be null when not ready. Another
owner's scene details are never returned. `configured` is not a live model test.

`GET /api/environments/{id}/frame?revision={revision}&camera=overview`
returns a real PNG, `Cache-Control: no-store`, `X-Frame-Id`, `X-Captured-At`,
and `X-Physics-Steps`. The browser fetches it with its bearer token and renders
a revocable object URL. `camera` is `overview` or `inspection`.
Poll at a bounded rate (one frame per second per visible view initially).
Stop polling hidden views and abort in-flight requests on navigation.
Display the actual capture timestamp and a stale/disconnected state.
This is an authenticated live camera feed, not WebRTC or an interactive 3D mesh.
Never substitute a stock image or animate robot movement in the browser.

## Customer environments

`GET /api/environment-schema`: the existing customer-environment JSON Schema.

`GET /api/environment-templates`:

```json
{"items": [{"name": "Reference inspection cell", "document": {}}]}
```

The template documents use the full existing customer-environment schema.
`GET /api/environments` returns `{"items": [EnvironmentRecord]}`.

Opt into bounded paging with `GET /api/environments?page_size=50`.
`page_size` is an integer from 1 to 50; each response contains at most that
many records and `next_cursor: "<opaque-server-UUID>" | null`. Pass that cursor
to `GET /api/environments?cursor=...`; a supplied page size must still match
the original page. The no-query legacy route retains its items-only shape
and existing newest-first limit. Paged listings instead use unique environment
document ID descending, with no new composite index or increased legacy limit.

Cursor state is server-issued metadata in the authenticated owner's partition,
bound to the page size, last scanned ID, and initial creation cutoff. Malformed,
missing, foreign and expired cursors return 422 before any environment query.
Only cursor documents set top-level `ttl: 600`, with a separate ten-minute
logical expiry. Pagination is unavailable until container metadata confirms
`defaultTtl: -1` (TTL enabled, no expiry for other records); deployment/readback
of that setting remains operator-owned. Existing environments/runs get no TTL.

Each query scans at most page size plus one record. Records created after the
initial cutoff are excluded; an empty filtered page can still have a next cursor.
This is **not an immutable content snapshot**: an existing record's content and
revision can change between reads. Clients must retain the exact reviewed
environment/case revision, seed and split, deduplicate IDs, flag changed
versions, and require explicit reselection rather than adopting a new revision
under an old checkbox. The server still authoritatively checks revisions at
save/project/teaching admission.

An `EnvironmentRecord` contains:

```json
{
  "environment_id": "reference-cell",
  "display_name": "Reference inspection cell",
  "revision": "sha256hex",
  "document": {},
  "created_at": "2026-09-15T00:00:00+00:00",
  "updated_at": "2026-09-15T00:00:00+00:00"
}
```

`POST /api/environments` accepts:

```json
{"document_json": "the customer's original JSON text", "expected_revision": null}
```

Do not parse and reserialize customer JSON before submission: that could discard
duplicate keys before the server validates them. A new environment uses null;
an update must supply the revision that was loaded. The response is the saved
`EnvironmentRecord`. A concurrent change returns 409. Credentials, executable
Python, and unsupported fields are rejected.

`POST /api/environments/{id}/activate` accepts `{"revision": "sha256hex"}`.
It returns 202:

```json
{"activation_id": "uuid", "environment_id": "reference-cell", "revision": "sha256hex", "status": "loading"}
```

Poll runtime status until the requested revision is ready. A shared simulator
has one active owner/scene at a time; busy or foreign-owned instances return 409.
Saving JSON is not the same as loading it into the physical simulator.
Replay configurations cannot be activated by this live-runtime API.

### Explicit non-real-time learning configuration

Only the reviewed `inspection-cell-learning-v1` template may include the
optional root `learning_execution` object:

```json
{
  "schema": "physicalai.paused-simulation/v1",
  "execution_timing": "paused_simulation",
  "profile_id": "franka-position-hold-10hz-paused-v1",
  "max_simulation_seconds": 30,
  "max_wall_seconds": 600
}
```

All five fields are required when this closed object is present. V1 simulation
seconds remain an integer from 1 through 30 and wall seconds from 1 through
600; lower explicit customer limits remain binding. No aliases, extra fields,
implicit defaults, timing-gate overrides or real-time qualification are accepted.
An absent object preserves the legacy contract and **never** authorizes paused
execution. `validate_paused_learning_environment` additionally requires an
explicit opt-in and live physics rather than replay.

This is configuration-only authorization for a separately implemented
`NON_REALTIME_SIMULATION` profile, not evidence of an activated scene, a trained
model, successful motion or real-time capability. Existing
`execution.max_step_seconds` retains its original **wall-time** meaning and
does not become a 600-second limit. Existing reference/inspection commands and
their server ceilings are unchanged. New cases use new IDs and content hashes;
existing saved scenes and frozen evaluation plans are not silently rewritten.

The separately declared v2 task budget is:

```json
{
  "schema": "physicalai.paused-simulation/v2",
  "execution_timing": "paused_simulation",
  "profile_id": "franka-position-hold-10hz-paused-v2",
  "max_simulation_seconds": 60,
  "max_wall_seconds": 600
}
```

V2 permits integer simulation seconds from 1 through 60 **only** with both
v2 schema and v2 profile. Mixed versions are rejected even at a lower budget
that would fit both. V1 remains capped at 30 seconds/1800 physics ticks; a new
v2 profile permits 60 seconds/3600 ticks. Wall, observation/inference/hold/interval,
heartbeat, physical safety and evaluation quality limits are not increased.
New IDs/revisions and operator-frozen criteria precede v2 data collection;
this version does not retroactively make a failed or incomplete v1 trial pass.
Missing opt-in still denies paused execution, and neither version confers
real-time admission or activates a runtime, model or paid job.

## Inspection and sorting runs

`POST /api/runs` accepts:

```json
{
  "request_id": "browser-generated-uuid-for-this-request",
  "environment_id": "reference-cell",
  "revision": "sha256hex",
  "instruction": "Inspect this part and quarantine it if defective."
}
```

Use the same request ID for retries of the same request, never for changed input.
`execution_mode` defaults to `inspection`; that mode requires an instruction
and forbids a `policy_release_id`. Explicitly specifying the default mode does
not change legacy inspection request fingerprints.
The server captures a real observation, stores its evidence in Azure Blob,
and obtains an inspection decision through a deployed Foundry agent.
This bounded planning request returns a `RunRecord`. Planning does not move
the robot. Missing live dependencies produce an error, not a fake plan.

```json
{
  "id": "run-uuid",
  "environment_id": "reference-cell",
  "revision": "sha256hex",
  "instruction": "Inspect this part and quarantine it if defective.",
  "execution_mode": "inspection",
  "status": "awaiting_approval",
  "created_at": "2026-09-15T00:00:00+00:00",
  "updated_at": "2026-09-15T00:00:01+00:00",
  "plan": {
    "kind": "inspection",
    "classification": "rejected",
    "target_station_id": "rejected",
    "object_id": "part-001",
    "summary": "Visible surface defect.",
    "observation_id": "frame-uuid",
    "epoch": "simulation-world-uuid",
    "state_revision": 1,
    "model_response_id": "actual-foundry-response-id"
  },
  "execution": null,
  "error": null,
  "events": [
    {"kind": "planning", "message": "Observation captured.", "at": "2026-09-15T00:00:00+00:00"}
  ]
}
```

Statuses: `planning`, `awaiting_approval`, `running`, `cancelling`, `succeeded`,
`failed`, `cancelled`, `timed_out`. `plan`, `execution`, and `error` may be null.
The inspection plan classification is `accepted` or `rejected`; target station must match
the saved workflow. Summary is a concise result explanation, not hidden reasoning.
`execution`, when present, contains `command_id`, `status`, and optional
`final_position` and `completed_at`. `error` has `code`, `message`, `retryable`.

`GET /api/runs` returns `{"items": [RunRecord]}` for the current owner.
`GET /api/runs/{id}` returns a record and reconciles an in-progress simulator
command without dispatching another motion.
`GET /api/runs/{id}/observation` returns the captured evidence PNG, authorized
against run ownership; the UI must fetch it with a token.

`POST /api/runs/{id}/approve` accepts:

```json
{"plan_response_id": "the exact model_response_id displayed to the user"}
```

It reobserves the scene, checks epoch/revision/object state, reserves one command,
and returns the updated run. Duplicate approval must not dispatch twice.
The current API also returns `plan.expires_at`. An expired approval requires a
new inspection; it never silently retries motion. This is an additive field.
The UI polls `GET /api/runs/{id}` until terminal; dispatch ACK is not success.
If the scene changed, require a new plan rather than approving a stale decision.

`POST /api/runs/{id}/cancel` has no body and returns the updated run.
Cancellation of running motion is not shown as completed until confirmed by
the simulator. The control path must not wait for an LLM response.

## Explicitly selected released motor skills

An approved task such as moving a normal synthetic part into quarantine is
**not** a defect inspection. It must not call the CV inspector, invent a
classification, override the saved inspection workflow, or generate a fake
Foundry response ID. Use the distinct planning mode:

```json
{
  "request_id": "a-new-browser-generated-uuid",
  "environment_id": "the-owner-saved-approved-case",
  "revision": "the-exact-approved-saved-revision",
  "execution_mode": "released_skill",
  "policy_release_id": "the-owner-reviewed-release-uuid"
}
```

The server resolves the owner-scoped immutable release, checks its allowed
scene/case and goal, and uses its exact task, instruction, model family/SHA
and control profile. Omit `instruction`; if supplied, it must equal the
release instruction exactly. No caller task/goal/model overrides are accepted.
The UI does not send an old freeform inspection instruction with this mode.

Planning still captures and stores a real fresh PNG and returns
`awaiting_approval` without moving. It does not require or call the Foundry
inspection agent. The `RunRecord.execution_mode` is `released_skill`,
`instruction` is the canonical release instruction, and `policy` contains
the full immutable binding. Its distinct `plan` has:

```json
{
  "kind": "released_skill",
  "skill_plan_id": "server-generated-plan-uuid",
  "policy_release_id": "the-selected-reviewed-release-uuid",
  "policy_type": "smolvla",
  "model_sha256": "exact-reviewed-model-sha256",
  "task_id": "manufacturing-part-placement-v1",
  "instruction": "Pick up the synthetic part from the source platform and place it in the quarantine tray.",
  "target_station_id": "rejected",
  "object_id": "the-actually-observed-part-id",
  "summary": "Reviewed task and original observation; no CV classification or motion.",
  "observation_id": "actual-captured-observation-uuid",
  "epoch": "actual-simulation-world-uuid",
  "state_revision": 1,
  "expires_at": "server-issued-approval-expiry-timestamp"
}
```

There is no `classification` or `model_response_id` in this plan.
After deliberate human review, approve with **only**
`{"skill_plan_id": "the-exact-displayed-server-plan-uuid"}` at the existing
approval route. Exactly one approval reference is required: this skill ID
or the legacy inspection `plan_response_id`, never both or the wrong kind.
Approval rechecks expiry, every immutable release pin, original observation,
current scene/epoch/object state and reserves one command before dispatch.
Only the selected learned policy may execute; there is no reference fallback.

Polling/cancellation retain the existing owner-scoped behavior. A successful
learned result requires the matching applied model/family/profile, actual
prediction/action counts with zero reference-route calls, and measured
completion at the approved goal within deadline. Queued ACKs are not success.
Release creation is not model installation or GPU timing proof.

The public reference-inspection projection remains inspection-only and rejects
skill-bound records. Existing inspection records lacking the new mode/kind
fields remain readable with their original semantics. Earlier mixed
inspection-plus-policy plans cannot be newly approved; create an explicit
released-skill plan instead.

## UI implementation boundary

Build the real console, JSON editor, approval workflow, live frame views, and run
history against this contract. Do not add hard-coded live metrics, fake 3D
movement, anonymous authentication bypass, or a switch to a bundled fixture.
Authentication and HTTP clients can be injected in tests through component
interfaces; the production entry point always uses MSAL and the real same-origin API.

Owner-scoped teaching/training endpoints are defined separately in
[`learning-api.md`](learning-api.md) and remain capability/admission gated.
A full Python extension editor/runner is not represented by these endpoints.
Do not present unavailable services as working features.
