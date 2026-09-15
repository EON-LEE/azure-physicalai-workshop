# Customer environment contract

The current contract describes **one inspection-and-sorting cell**, not an
arbitrary factory or a functioning simulation. JSON authoring and offline
validation work today. Scene loading, browser editing, and reviewed Python
extensions are implementation milestones.

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

## Planned code customization

Customers will add versioned `SceneBuilder`, `ObservationAdapter`,
`InspectionRule`, and `SkillProvider` implementations through reviewed Python
packages and built container images. An approved registry maps configuration IDs
to implementations. Configuration JSON is not a Python module path.

The environment UI will support validation, comparison, and versioned application
of JSON while a scene is stopped. It will not run arbitrary uploaded Python.
Custom plugins must pass contract tests and a real second-environment
Isaac Sim/Foundry end-to-end test before being included in a live demo release.
