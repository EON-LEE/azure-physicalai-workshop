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
      {"id": "supply", "role": "source", "position_m": [-0.5, 0, 0.2]},
      {"id": "inspection", "role": "inspection", "position_m": [0, 0.4, 0.2]},
      {"id": "accepted", "role": "accepted", "position_m": [0.5, 0.4, 0.2]},
      {"id": "rejected", "role": "rejected", "position_m": [0.5, -0.4, 0.2]}
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

`GET /api/demo/frame?camera=overview|inspection` is anonymous but works only when
the operator has explicitly enabled a public live publication with a fixed
owner, environment ID and immutable revision. The backend rejects new/private
revisions rather than automatically republishing them.

A successful response is a real `image/png`, with `X-Frame-Id`,
`X-Captured-At`, `X-Physics-Steps`. Fetch without an Authorization header.
Use bounded polling only while visible and in live mode; revoke object URLs and
abort requests on navigation/unmount. Missing publication or live failure returns
503 with the normal structured error envelope, never a substitute frame.

The public snapshot never exposes tenant IDs, object IDs, auth scopes, keys,
private run IDs, instructions, manifests, arbitrary artifact paths, or internal
simulator exceptions. The scene in this release is the approved synthetic
reference scene, not a public projection of all Cosmos environments.

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
