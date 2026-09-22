# Azure Physical AI manufacturing reference solution

**Implemented in source:** a real React/TypeScript web console, Entra-authenticated
FastAPI API, Foundry inspection adapter, Azure Cosmos/Blob persistence, a guarded
Isaac Sim bridge, customer JSON/Python scene extension interfaces, and staged
Azure deployment templates.

**Actual Azure deployment verified:** the protected web/API, Entra delegated
authentication, private Cosmos configuration writes/reads, and a real Foundry
agent connectivity response from managed-identity bootstrap. Images are built
and executed in ACR, not just checked as source. Blob, Cosmos and Key Vault use
Private Link; tenant restrictions are not bypassed.

**Actual GPU execution observed:** Isaac Sim 6.0.0 first ran on an Azure
NC72 RTX PRO Blackwell 48 GB slice with GRID 595.91.07. After repeated Spot
evictions, a private NV36ads_A10_v5 / A10-24Q host with Microsoft's GRID 570.237
also produced real camera observations and completed normal/defect
image-planned physical grasp/sort/retreat cycles. The controller uses
PhysX gravity feed-forward, contact-based grasping and a measured speed watchdog;
it does not attach or teleport the part to manufacture a successful outcome.

**Public demonstration:** an explicitly authorized, bounded presentation compares
normal and surface-defect synthetic parts. The audience page connects the original
inspection image, actual Foundry decision, measured robot phase and physical result
from the same scene epoch. Opening the public URL does not authorize motion or paid
inference. A stopped/unpublished simulator is shown as unavailable, not replaced
by a schematic or replay. Availability depends on the active Azure presentation.

**Remaining boundaries:** regular supported GPU quota is not established for the
development subscription. Its Spot VM was actually evicted, so Spot cannot be sold
as uninterrupted presentation capacity. LeRobot/ACT and Azure ML components are
CPU-smoke-tested, not validated learned robot policies. The running reference
controller is not a trained VLA, and this is not an industrial safety certification
or a blanket pass of every production-release gate.

All production workloads run on Azure. WSL is for authoring and automated tests,
not a required production server. The default web page is a login-free,
read-only public demonstration. `/operator` retains authenticated editing,
approval, motion and private data. This is not an anonymous authentication bypass.
There is no memory-store fallback, stock camera frame, or automatic replay mode.

## What exists

```text
apps/web     Public audience viewer plus protected MSAL operator console
apps/api     Authenticated API, durable revisions/runs, approval and result verification
agents       Versioned Foundry agent definition and typed inspection function
simulation   Reviewed scene builders, command guards, Isaac adapter and HTTPS bridge
learning     Scoped demonstrations, LeRobot/ACT conversion, training, inference and AML jobs
contracts    Customer JSON Schema and offline validation CLI
examples     Reference, custom compact, and configuration-only customer examples
infra        Azure foundation, bootstrap job and private GPU/web templates
scripts      WSL setup, automated checks and explicit staged deployment
tests        CPU/HTTP/security-boundary checks and the remaining live-test catalog
docs         Configuration, HTTP contract, test strategy and deployment boundaries
```

`inspection-cell.json` and `compact-cell.json` target the implemented reference
Franka scene builders; changed layouts still require actual GPU acceptance. `customer-cell.json`
is a configuration-only example: its robot/template is intentionally not installed.
Unknown live templates or profiles fail explicitly.

## Develop and test in WSL

Run from the repository root **inside Ubuntu WSL**, not Windows Python/Node.
The setup script puts dependencies in per-worktree Linux storage; it avoids the
very slow Windows-mounted `node_modules`/SDK installation path. Python 3.13 is
the API development default. The pinned Isaac image has its own Python runtime.

```bash
bash scripts/setup-dev.sh
source scripts/dev-env.sh
uv run --locked python -m contracts.validate_environment examples/inspection-cell.json --json
uv run --locked python -m contracts.validate_environment examples/customer-cell.json --json
bash scripts/check.sh
uv run --locked python -m scripts.validate_infra
```

These checks need no Azure credentials, GPU, Docker engine, or MCP server.
Infrastructure compilation requires the local Azure CLI/Bicep compiler but does
not log in or create resources. Dependency installation uses package registries.
Browser tests run with `npm run test:e2e` from `apps/web`; their fixture harness is
explicitly test-only and is excluded from the production build.

Some app-managed Windows worktrees contain Windows-only paths in `.git`.
WSL can execute Python in such a worktree even when Linux Git cannot resolve
that metadata. Do not rewrite the app's `.git` pointer or edit the main checkout
to work around it. Confirm the workspace/Git arrangement before native Linux
Git development; use one source worktree, not independent Windows and WSL copies.

## Customize the customer configuration

Copy one example, edit its workspace and stations, and run the validator.
See [the environment contract](docs/customer-environments.md).

A successful report contains:

```json
{
  "valid": true,
  "scope": "configuration",
  "runtime_verified": false,
  "issues": []
}
```

This is intentionally **not** a simulator-readiness or physical-safety claim.
Invalid input exits with status 2; it does not select a default demo environment.
Credentials and executable code are not configuration fields.

## Deploy all runtime components to Azure

See [the Azure deployment guide](docs/azure-deployment.md) before deploying.
The staged flow uses Azure Bicep, ACR cloud builds, an Azure bootstrap job, and a
private GPU VM. It creates billable resources only with explicit `--apply` and
valid non-placeholder deployment inputs. No default subscription is selected.

The backend uses managed identities, the browser uses real MSAL/Entra, and the
simulator independently verifies its allowlisted controller identity over TLS.
Robot assets must be licensed and self-contained, supplied through the approved
Blob archive or an explicitly enabled, validated private image bundle. There is
no default NVIDIA asset-server fallback.

The [HTTP contract](docs/http-api.md) distinguishes configuration, live data,
inspection proposals, approval, command ACK, and verified physical completion.
Never describe an ACK, a replay, or a test-fixture image as successful robot motion.
The [public viewer contract](docs/public-demo-api.md) separately permits anonymous
read-only viewing of a curated synthetic reference scene. It never exposes all
Cosmos environments or invokes paid agents/robot actions on behalf of visitors.

The [learning guide](docs/policy-learning.md) explains the isolated Python 3.11
training environment and the actual CPU smoke. Optional customer configuration
can request demonstration capture; it additionally requires trusted deployment
enablement, real synchronized 60 Hz cameras/commands and complete provenance.
Only finalized, validated owner-scoped manifests are published to Azure Blob.

The [live acceptance harness](docs/live-acceptance.md) tests the real deployed
endpoint. Its read-only smoke does not imply a physical gate pass. The strict
release aggregator rejects missing/planned/skipped, stale, wrong-commit or
fixture-only evidence for actual Azure/GPU gates.

## Automated test gates

[tests/cases.json](tests/cases.json) is the machine-readable test catalog.
Implemented cases drive pytest directly. Each case names its requirement,
gate, input or mutations, and expected result. Planned cases describe their
runner, prerequisites, procedure, and assertions without claiming execution.

CPU, frontend/browser and infrastructure workflows run on pull requests and
pushes. They do not deploy Azure. Additional backend suites exercise real HTTP
handlers, JWT validation, optimistic concurrency, approval expiry, cancellation,
archive extraction and bootstrap guards using explicitly injected test adapters.

See [the automated test strategy](docs/automated-testing.md) for the distinction
between configuration checks and required browser, GPU, Foundry, learning,
and deployment gates. A green CPU workflow is not permission to call v1 complete.