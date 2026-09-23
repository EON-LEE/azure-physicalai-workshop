# Factory Console

Korean-first React/TypeScript demonstration viewer and protected operator console.
`/` opens immediately without login and uses only the curated anonymous
[`public demo API`](../../docs/public-demo-api.md). `/operator` retains the
authenticated [`operator API`](../../docs/http-api.md).

The public page shows an authorized, bounded automatic presentation: actual
Isaac Sim camera, same-cycle Foundry image input and decision, actual motion
telemetry, and independently measured inspection/physical results. The story is
to keep a surface-defective part out of the downstream process and isolate it.
There are no local scenario selectors, fabricated progress, schematic fallback,
replay clips or replacement camera images.
Actual camera PNGs are shown only after a pinned synthetic reference publication.
Private environments, histories, editing, approvals and motion remain protected.

The web/API deployment and GPU/physical acceptance are owned by the parent
workstream. Frontend CPU/browser tests do not verify deployed Azure, Foundry
classification, robot motion or real hardware. The viewer reports the current
public API state instead of hard-coding a deployment or readiness claim.

## Build and serve

Use Node.js 24 and the committed npm lockfile, from a full repository checkout:

```bash
cd apps/web
npm ci --no-fund --no-audit
npm run typecheck
npm test
npm run build
```

`dist/` is the production output and is intentionally ignored by Git. The parent
API container should build and copy this directory, serve it on the API's origin,
and route `/api/*` to the actual API, not to an SPA fallback. Static
assets and fonts do not depend on a CDN. The CI workflow retains `dist` as an
artifact; it does not deploy anything or use cloud credentials.

Build-time schema generation reads
`../../contracts/customer-environment.schema.json`. Preserve this path in a
container build context, or generate `dist` before packaging. AJV is **build/test
tooling only**: the generated standalone validator does not compile code in the
browser and works with `script-src 'self'` without `unsafe-eval`. At runtime,
`/api/environment-schema` must match the frozen schema structurally; a mismatch
blocks validation and save rather than silently using an old schema.

`npm run dev` binds Vite to `127.0.0.1:5173` and forwards `/api` to
`http://127.0.0.1:8000`. The development server still requires the real public API; operator access
also requires real Entra configuration. `npm run preview` serves the production assets for
inspection; it is not a production managed runtime.

### Windows + WSL

Run npm/Node under Ubuntu WSL, keeping source and Git in the app-managed
Windows worktree:

```powershell
wsl.exe --distribution Ubuntu --cd 'C:\path\to\your\worktree' --exec bash -lc 'cd apps/web; npm ci --no-fund --no-audit'
wsl.exe --distribution Ubuntu --cd 'C:\path\to\your\worktree' --exec bash -lc 'cd apps/web; npm run typecheck; npm test; npm run build'
```

WSL is for authoring, CPU tests and builds, not production hosting. NTFS-mounted
`node_modules` can be unusually slow. A **session-specific**, WSL-native npm
dependency directory may be linked at the ignored `apps/web/node_modules` path;
keep manifests/lockfile synchronized, do not copy source out of the worktree,
and never rewrite app-managed `.git` metadata. This is a local performance
workaround, not a CI requirement.

## Authentication and HTTP

`src/main.tsx` selects the public audience entry by default. It requests only
`GET /api/demo` without cookies or bearer tokens; it does not load MSAL or call
private APIs. `/` and `/?view=demo` are the same public experience. The public
client permits only three same-origin read-only routes:

- `GET /api/demo`: additive optional/nullable `presentation` metadata.
- `GET /api/demo/frame?camera=overview&epoch=<UUID>`: only when base
  `mode`, `simulation.live_available`, `simulation.status`, `frame_url` and
  publication capabilities allow live imagery, and the current presentation
  has an unexpired matching scene epoch.
- `GET /api/demo/evidence?observation_id=<UUID>`: the original image for the
  currently published decision; the URL is allowlisted, never arbitrary.

Every public request uses `credentials: omit`, `cache: no-store`,
`redirect: error`, no bearer token and a 15-second timeout covering the body.
No public interaction starts/stops a server task, requests a model invocation,
edits a scene, or dispatches robot motion.

Only `/operator` lazy-loads the existing authentication entry, which fetches
`GET /api/config` and initializes MSAL. Tenant ID, SPA client ID and delegated
scope come exclusively from that response. No credentials or tenant defaults
are bundled.

MSAL initializes and handles redirect results, selects only an unambiguous
account in the configured tenant, and uses `loginRedirect` plus
`acquireTokenSilent`. The login redirect URI is the current origin followed by `/operator`.
Logout returns to the public root. Register both URLs on the dedicated SPA.
The actual Entra SPA registration and API permission must be configured by an
authorized deployment operator; this frontend does not create them.

Configuration, redirect, token and HTTP failures are visible. Expired sessions
require an explicit reauthentication gesture; protected requests never proceed
with an empty token. The API client uses same-origin `/api` paths, bearer
headers, `credentials: omit`, `cache: no-store` and `redirect: error`.
Timeouts cover response bodies as well as headers: 15 seconds for reads,
120 seconds for writes. A timeout does **not** mean the server rolled back.

Production CSP can allow scripts/styles/assets from `self`, images from
`self blob: data:`, and Entra connections/frames to
`https://login.microsoftonline.com`. No `unsafe-eval`, inline scripts, remote
font host or third-party analytics is required.

## Customer workflow

### Public presentation

The default view is the actual current server scenario, not a visitor-selected
expected result. The approved presentation alternates normal and surface-defect
synthetic parts. Missing/null presentation means **no published automatic
presentation**, even when older base camera-ready flags are present.

Public snapshots poll every two seconds and camera PNGs at most once per second
while visible. PNG signature, content type, bounded size, capture timestamp,
frame ID and physics-step headers are checked. `X-Scene-Epoch` must match the
snapshot's epoch. Public live frames also require a valid UTC `X-Server-Time`.
Their capture age is measured against that server stamp, not the browser wall
clock, and must be within 2,000 ms old / 500 ms future. The **entire** monotonic
request plus body-read/validation duration is charged against the unchanged
five-second display budget. The remaining display time expires using
`performance.now()`, including between polling ticks; client clock skew or clock
adjustments cannot extend it. Original `X-Captured-At` remains unchanged.
The protected operator image hook and its clock behavior are not modified.
Evidence requires `X-Frame-Id` equal to `observation_id` and the
original `X-Captured-At` matching the decision. Historical evidence is clearly
labeled **not LIVE** and is not subject to the live five-second freshness limit.
Camera and evidence mismatches return conflict semantics, hide the mismatched
images/decision and refresh the snapshot. A new cycle/epoch/run clears previous
images before reading matching data. No frame with an error, decode failure,
stale/future timestamp or hidden-view state is labeled LIVE.

Progress labels derive only from actual presentation/motion fields, never
elapsed time. Missing `motion.phase` is shown as unavailable. A command ACK,
`motion.status: succeeded` or `presentation.status: completed` alone is not
task success. **A successful task requires a succeeded result with both
`physical_success: true` and `inspection_correct: true`.** The two outcomes
are shown separately, alongside measured part/target positions and the
server's literal result message. No confidence score, accuracy percentage or
throughput is invented. Last recorded outcomes may remain visible when the
camera is unavailable, explicitly labeled as historical rather than live.

**Pause/resume viewing** stops browser reads and image display, not the robot.
It is keyboard-accessible and uses `?viewing=paused` URL state. Reduced-motion
visitors start paused with a one-time metadata read. Hidden tabs and unmounted
views abort reads and revoke object URLs. Stopped, failed, expired, unavailable
and preparing states are prominent; retry buttons issue GETs only.

Secondary disclosure distinguishes Foundry connectivity-only evidence,
test-fixture Azure CPU training smoke, and current inspection/physics outcomes.
The demo is **Foundry inspection plus bounded conventional robotics/PhysX**,
not a trained VLA/policy controlling the robot.

### Protected operator console

Audience members need no login to view the public inspection/sorting presentation.
The controls below belong to the separately protected operator area.

- **Environment Studio:** Select an API reference template, explicitly load a
  saved environment, or import a `.json` file up to 1 MiB. Reference templates
  are labeled as reference data, not real customer-factory measurements.
  JSON syntax, duplicate keys and the frozen structural schema are checked
  locally. The server remains authoritative for semantic and safety validation.
- **Revision-aware save:** The exact editor/imported string is submitted as
  `document_json`, including original formatting. Imported files retain the
  loaded revision when updating that same environment ID. Changing the ID or
  starting from a template creates a new-environment request with
  `expected_revision: null`. A 409 leaves the draft intact. Loading the latest
  server version requires explicit confirmation before discarding edits.
- **Activation:** A save is not simulator readiness. Activation sends the saved
  revision and displays the 202 receipt as pending. Only a fresh runtime
  response matching the requested environment/revision with a real epoch and
  physics-step count confirms readiness. A 120-second observation delay is
  shown as unconfirmed, never as success. REPLAY documents cannot activate.
- **Factory Live:** Authenticated PNGs are checked for PNG type/signature and
  required frame metadata, then displayed through revocable object URLs.
  Capture times and physics steps come from server headers. Images are labeled
  as Isaac Sim synthetic camera data, not real-world sensors. Frames older than
  five seconds, disconnected feeds and clock discrepancies are marked stale.
- **Plan and execute:** Each planning request has a UUID retained across
  unchanged retries. Planning does not move a robot. Approval requires a
  checkbox and a deliberate button press and submits the exact displayed
  `model_response_id`. Runtime epoch/revision and saved workflow are checked
  before enabling approval; the server rechecks authoritative state. Approval
  conflicts block that stale decision until a new plan is obtained.
- **Results and history:** Only API runs/events/IDs are shown. A command ACK is
  not success. Cancellation is available without waiting for an in-flight
  approval response and is not complete until the API confirms it. Evidence
  images are authenticated and explicitly labeled historical, not LIVE.

Visible camera views poll at most once per second. Runtime status polls every
three seconds, nonterminal run details every two seconds, and visible history
every ten seconds. Polls never overlap; retryable reads back off to at most
30-second intervals. Authorization/input errors stop automatic retries.
Hidden tabs/views stop polling, abort in-flight reads and revoke image URLs;
superseded URLs and unmounted views are cleaned up. Mutating requests are never
automatically replayed.

Draft JSON stays in memory across console navigation. It is not persisted to
browser storage; preserve it before refresh/logout. Navigating away from a
planning request aborts the browser request, **not** a server job already
accepted; its request UUID remains available for an unchanged retry.

## Tests and production boundary

Vitest + Testing Library exercise configuration/auth failures, structured API
errors, raw JSON/CRLF preservation, duplicate keys, revision conflicts, pending
approval, stale plans, actual run transitions, cancellation races, bounded
polling, aborts and object-URL revocation.

```bash
npm test
npx playwright install chromium --only-shell
npm run test:e2e
```

`npm run test:e2e` runs both the preserved operator harness and public production
entry browser tests. `npm run test:public-e2e` runs only public browser tests.

The operator browser harness lives entirely in `tests/browser/`. It explicitly injects
a test API into `ConsoleApp`, runs under a strict CSP and never authenticates
to or calls Azure. Its static PNG and every screenshot are labeled
**TEST-ONLY FIXTURES — not Azure/GPU verification**. Browser checks cover
keyboard approval, delayed terminal confirmation, raw JSON/conflict handling,
navigation cleanup and narrow-screen layout. The public tests in
`tests/public-browser/` run the actual built entry with test-only intercepted
GET responses and an external same-origin **TEST-ONLY** banner. Static test
PNGs and screenshot filenames explicitly say they are not Azure/GPU/Foundry
verification. They cover current-cycle evidence, phase/result transitions,
missing publication, image conflicts, pause/resume, 390px long-text layout,
reduced motion, strict CSP, and the still-authenticated `/operator`.

Production always builds `index.html` → `src/main.tsx`; there is no environment
variable, URL parameter or fixture service that changes this. Vite rejects any
`tests`/`fixtures` module in the production module graph. The post-build check
requires one public production entry with lazy operator/MSAL chunks, rejects
private API/auth modules in the public static import graph, and scans built
assets for test-only markers. The preview CSP allows only self-hosted scripts
and styles, without `unsafe-eval` or inline-style allowances.
Test outputs and the separate `.fixture-dist/` are ignored and must never be
packaged into the API image.

## Teaching Studio and versioned policy learning

The protected `/operator?view=learning` view uses the owner-scoped
[`learning API`](../../docs/learning-api.md). It starts with a capability read;
disabled, missing license/hardware approval or unconfigured worker dependencies
produce visible blocked states, not example projects, fake loss or ready models.
SmolVLA is an explicit model choice. Unapproved GR00T requests are not redirected
to it; ACT remains separately labeled auxiliary tooling.

The workflow captures actual human versus reference-controller provenance,
waits for async upload verification, seals immutable datasets, requests
human-approved paid jobs and shows real Azure job IDs/nullable measured metrics.
The 20+ case comparison retains every failure and retry, pins environment
revisions/poses, and distinguishes improvement from completed optimizer work.
First-P0 bootstrap is visible only to configured bootstrap operators and is a
reference-controller quality/safety gate, not a fake learned P0/P1 comparison.
Prepared model dates, model family/source/revision and recorded outcomes remain
distinct from today's training and LIVE observations.

Teaching controls obtain a short-lived server grant before a held input. They
do not send browser wall-clock expiry or raw joints. Release/blur/hidden-tab/
unmount issues a zero-motion hold with a forward sequence, and a late arm
response cannot issue a movement after release. One held press is one bounded
5 mm input, not a browser-operated servo loop.

Policy review requires the exact evaluation ETag and passing evidence. Selecting
the resulting release on Factory Live creates a new plan; it does not
auto-approve, change a model URL or fall back to reference control. Run details
show the immutable expected policy and actual applied model/action counters.

The optional public learning disclosure fetches only `GET /api/demo/learning`
when expanded. Only a deployment-pinned project/evaluation is projected; model
paths, tenant/owner identity and arbitrary private job history are not published.
It labels all comparison data as recorded, not LIVE. No trained-policy
production-readiness claim is made by CPU or fixture browser tests.

### Remaining limits

The HTTP contract does not specify `execution.final_position`'s representation,
so the console renders that optional value as literal JSON instead of inventing
units or coordinates. There are no pagination, Python-extension review or
policy-learning APIs in v1, so this UI does not imply those features work.
Schema limits are requested limits, not authority to override a robot
controller. Hardware safety and Azure/Isaac/Foundry integration still require
separate authorized end-to-end verification.
