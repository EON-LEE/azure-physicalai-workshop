# Customer environment contract

The current contract describes **one inspection-and-sorting cell**, not an
arbitrary factory. JSON validation, the browser editor, and reviewed Python
scene-builder interfaces are implemented. Actual Azure/GPU execution is not
verified yet.

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
Separate observation/inspection/skill extension interfaces and policy training
remain follow-up work.

The environment UI supports validation and version-aware saves. Scene activation
uses the authenticated API and rejects busy/foreign-owned simulator state.
It does not run arbitrary uploaded Python.
Custom plugins must pass contract tests and a real second-environment
Isaac Sim/Foundry end-to-end test before being included in a live demo release.
