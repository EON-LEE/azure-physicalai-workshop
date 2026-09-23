# Customer environment contract

The current contract describes **one inspection-and-sorting cell**, not an
arbitrary factory. JSON validation, the browser editor, and reviewed Python
scene-builder interfaces are implemented. Actual Azure/GPU execution is not
implied by a valid configuration: the deployed reference cell has actual
Azure/Isaac/Foundry evidence, but a new customer layout still needs its own
physical run.

## Guided customer experiment

The public page's **내 공정 실험** section authors a local, downloadable configuration,
without calling an operator API. It has two bounded experiments:

- **Inspection and sorting:** use the reference layout and select a uniform green
  part or the explicit surface-defect fixture.
- **Relocate quarantine:** change only the rejected station's X coordinate from
  0.22 m to 0.32 m. Other station coordinates, workflow roles, 0.2 m/s maximum
  requested speed, 30-second task deadline and 2000 ms observation-age bound remain
  unchanged.

The sample selector configures the synthetic scene's seed (42 or 43), not the
model's answer. Foundry still receives the actual image and generic inspection
task, never the expected label or seed.

**운영자에게 이 실험 전달** navigates to
`/operator?view=studio&experiment=relocate-quarantine&sample=surface_defect`.
Only these known experiment/sample values survive the MSAL login round trip.
The authenticated console opens the same unsaved JSON draft and supplies an
editable inspection task. It does not save, activate, plan or approve on navigation.
An unknown experiment is an explicit error, not permission to execute another task.

The operator then:

1. Reviews the JSON, checks or changes its customer environment ID, and saves it.
2. Activates that exact saved revision after coordinating the shared simulator.
3. Requests an actual image-based plan and reviews its original image, reason and
   target. A wrong plan must not be approved.
4. Explicitly approves the plan, then checks the measured final position and
   execution result against the requested target.

Generated IDs are separate from the published reference anchors. Reusing an
existing ID encounters the normal optimistic-concurrency guard; it does not
overwrite a previous configuration unconditionally. The planner inside
Environment Studio also asks before replacing unsaved JSON.

This is a way to test a line-layout assumption before moving a real tray, not a
claim that any customer geometry is reachable. The demo has no automatic ROI,
throughput improvement, new-product inspection certification or real PLC/MES
connection. Assess those with representative customer inputs and explicit
baseline measurements.

## Author a configuration

Start with `examples/inspection-cell.json`. `examples/customer-cell.json` shows
a different layout, station identifiers, robot/template identifiers, limits,
seed, and Korean display name without changing the validation code.

| Section | Contract |
|---|---|
| `schema_version` | Exactly `1.0`; unknown versions fail rather than migrate silently |
| `environment_id` | Lowercase identifier; not a path or executable import |
| `display_name` | Nonblank Unicode text, up to 120 characters |
| `length_unit` | `m`; no implicit unit conversion |
| `scene` | Template ID, robot profile ID, and nonnegative integer seed |
| `workspace` | Three lower and three upper bounds in metres; each lower is strictly less |
| `stations` | Unique IDs, declared roles, and finite 3D positions within the workspace |
| `workflow` | Four distinct station references with source/inspection/accepted/rejected roles |
| `execution` | Explicit `live` or `replay`, step deadline, and maximum observation age |
| `limits` | Positive requested speed/payload values; not authorization to control a robot |

Missing fields, unknown fields, numeric strings, Boolean numbers, duplicate
JSON keys, non-finite values, unresolved station references, and invalid
workspace positions are rejected. No values are coerced or defaulted.
The validator does not mutate the input or import the named scene.

These checks establish configuration consistency only. They **do not verify**:

- Existence, license, version compatibility, or installation of a template or robot.
- Collision-free movement, inverse kinematics, actual payload capability, or functional safety.
- Foundry model support, Azure identity, GPU quota, streaming, or connectivity.
- Whether a policy was trained or will complete the declared task.

The future controller must intersect customer-requested limits with its trusted
robot profile and process policy. A customer file must never enlarge that
trusted envelope. This controller is not implemented by the JSON validator.

## CLI

From the repository root in WSL:

```bash
uv run --locked python -m contracts.validate_environment examples/customer-cell.json --json
```

Exit 0 means the configuration passed. Exit 2 means invalid input, including
unreadable files. JSON output has `valid`, `scope`, `runtime_verified`, and
`issues`. Every issue has a JSON Pointer `path`, a machine-readable `code`,
and a human-readable `message`. Runtime verification is always false here.
Without `--json`, errors go to stderr and success text warns that the runtime
was not verified.

## Code customization

`simulation/extensions.py` provides `SceneBuilder` and `SceneSpec`.
The reference registry contains `inspection-cell-v1` and
`inspection-cell-custom-v1`. Customer packages can register a reviewed builder
through the `physicalai.scenes` Python entry-point group in the simulator image.
The registry rejects duplicates and unsupported API versions. Configuration JSON
is not a Python module path and cannot install or execute a package.

The current trusted controller supports only `reference-arm` (Franka) and a
bounded inspection cell. Supporting a different robot requires a reviewed
controller/asset integration and new real-GPU tests, not just changing an ID.
Separate observation/inspection/skill extension interfaces remain follow-up work.

## Optional real demonstration capture

Set `execution.record_demonstration` to `true` and provide
`execution.demonstration_split` (`train`, `validation`, or `test`) to request a
recording. Omission leaves normal control unchanged. The deployment must separately
enable capture, pin the simulator/source/builder/assets and grant its identity
write access only to the private demonstrations container.

The existing controller keeps its actual 60 Hz cadence. Recording uses fresh,
synchronized 60 Hz cameras at a bounded 320x320 resolution, measured nine-joint
state and the actual held command for each interval. Preview polling is not the
recording clock. If the camera, cadence, provenance or data budget cannot be
verified, the episode is not published. A physical outcome and capture/upload
failure are reported separately in execution evidence and run events.

Only validated owner-scoped files are uploaded; `manifest.json` is published last.
No arbitrary file path, executable code or pretrained model is accepted in this
JSON. See [policy learning](policy-learning.md) for conversion, training, guarded
inference and held-out evaluation. Real GPU capture and learned control still
require their live acceptance gates.

The environment UI supports validation and version-aware saves. Scene activation
uses the authenticated API and rejects busy/foreign-owned simulator state.
It does not run arbitrary uploaded Python.
Custom plugins must pass contract tests and a real second-environment
Isaac Sim/Foundry end-to-end test before being included in a live demo release.
