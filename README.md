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
as uninterrupted presentation capacity. The running reference controller is not a
trained VLA. The separate teaching/policy-learning implementation described below
has not passed its real physical learning gates. This is not an industrial safety
certification or a blanket pass of every production-release gate.

All production workloads run on Azure. WSL is for authoring and automated tests,
not a required production server. The default web page is a login-free,
read-only public demonstration. `/operator` retains authenticated editing,
approval, motion and private data. This is not an anonymous authentication bypass.
There is no memory-store fallback, stock camera frame, or automatic replay mode.

## What a customer can do with this demo

The customer problem is **inspection followed by physical sorting**, not simply
"make a robot move" or "ask an image model whether a part looks defective."
The reference connects an inspection decision to a bounded robot action and then
checks whether the part actually reached the configured destination.

| Customer question | What to do | What constitutes evidence |
|---|---|---|
| Can visual inspection change the physical handling of a part? | Watch the normal and marked-defect cases; compare their original images and actual Foundry reasons. | Different planned trays and measured final positions, not a generated explanation or dispatch acknowledgment. |
| What changes when my quarantine location moves? | In **내 공정 실험**, select the 10 cm quarantine-location experiment and a synthetic sample; download the resulting JSON or hand it to the operator. | After authenticated save, activation, planning and approval, compare the new target with the measured destination. Draft generation alone is not a successful experiment. |
| What happens when the model's decision is wrong? | Open **실제 실행 기록 3가지 비교** and inspect a withheld case, when one exists. | The original image, mismatched classification and absence of approved motion. The reference uses known synthetic ground truth; it is not a universal production fault detector. |

Visitors can prepare configurations without login; they cannot save server data,
call Foundry or dispatch motion. The operator link opens the exact unsaved draft
in Environment Studio after Entra login. Existing drafts require confirmation
before replacement, and saving never automatically activates or approves a run.
The public presentation and operator experiments share a GPU and must be scheduled
by the operator; no parallel simulator capacity is implied.

The comparison cards read at most three curated outcomes from the explicitly
published presentation. They remain labeled **recorded evidence, not LIVE** when
the live run has ended; a missing case stays missing. Real customer defect
distributions, camera/lighting variation, cycle-time baselines, throughput,
escape/false-reject rates, MES/PLC integration and physical safety require a
separate customer PoC. No ROI, trained-policy performance, arbitrary robot task
support or always-on availability is claimed.

## Teaching and policy learning: implemented, not yet admitted

The protected Teaching Studio, owner-scoped demonstrations, immutable datasets,
private managed-identity learning worker, actual Azure ML job adapters, separate
Foundry learning coach, explicit released-skill plans, and before/after evidence
verification are implemented. Released-skill execution is distinct from visual
inspection: intentionally placing a normal part into a reviewed quarantine goal
does not invent a defect classification or a Foundry inspection response.

SmolVLA is an **explicit model selection**, not a fallback labeled as GR00T.
Its pinned model and backbone cards declare Apache-2.0. The reviewed GR00T
revisions remain outside production admission because of noncommercial or
conflicting model-license artifacts. No restricted weights were used to produce
the SmolVLA results.

Actual Azure checks on 2026-09-23 established private networking/identity access,
an A100 80 GB CUDA workload, a typed Foundry coach response, and hash-verified
model assets. A separate diagnostic loaded the original SmolVLA vendor weights
and returned three finite CUDA predictions using **six-dimensional synthetic
observations**, with zero optimizer and actuator calls. That is compatibility
evidence, not a Franka-trained policy: its approximately 248 ms warm inference
time did not satisfy the proposed 80 ms inference gate.

Actual Isaac capture attempts also failed the full 100-interval, 100 ms
control-cycle gate; partial and truncated captures were retained as failures.
Consequently, model/policy and bootstrap allowlists remain empty and learning
admission remains off. A valid full-task dataset, changed weights from actual
training, model-bound actuator execution, and all frozen held-out physical
trials are still required. An Azure `Completed` status, passing component tests,
or handwritten results JSON cannot supply those missing facts.

A separate **non-real-time simulation** path is being implemented rather than
relaxing those failed real-time gates. Its saved environment must explicitly
opt in through the closed `learning_execution` contract. Physics remains frozen
while observations or policy predictions are pending; each accepted action then
holds for exactly six actual 60 Hz physics steps. Wall-clock budgets and
simulation-time limits are recorded separately, and its versioned data, models
and results cannot qualify a real-time controller. Actual Azure runs have
exercised this path's camera, frozen-state, actuation and manifest-last storage.
Early attempts lifted the part approximately 29 mm and failed the unchanged
50 mm grasp criterion. Later, an explicitly verified single-finger servo
calibration retained the authored force cap and achieved measured grasp,
lift and transport into the destination volume. Release and full-task
acceptance were still not established; failed and truncated captures remain
integration/test evidence, not training data.

The separate, explicitly selected paused v2 profile permits 60 simulation
seconds; the original paused v1 remains limited to 30 seconds, and neither
qualifies real-time control. Old recordings are not relabeled for the new
budget. The latest physical-verification attempt was interrupted by an
external administrative VM deallocation before a result could be verified.
Its physical outcome is unknown, not a pass. Further GPU execution requires
a bounded, approved managed-job attempt rather than repeated manual VM starts.
A successful full-task demonstration, actual policy training and paired physical
acceptance remain unverified, and learning admission remains disabled.

**Managed execution transition:** a private, Entra-only Azure Batch account,
both service private endpoints, explicit NAT-backed node subnet and
account-scoped submitter/reader permissions have now been deployed. The private
worker has actually read the Batch image catalog through its managed identity.
This creates no GPU pool or node and does not establish renderer compatibility.
The intended split is Batch for Isaac rendering/physics and Azure ML for
training; managed jobs still require supported GPU hardware underneath.

**Intermediate weights:** the native training path can publish periodic
checkpoints directly to private Blob, verify every file's bytes and ETag, then
write the complete manifest last. Partial newer checkpoints cannot supersede a
verified complete checkpoint. The pinned LeRobot CPU check has actually saved
intermediate model/optimizer/scheduler/RNG files and restored identical model
weights on a fresh local path. A separate **full-state** path now preserves the
optimizer, scheduler, exact random state and consumed data position. Actual
native CPU runs interrupted mid-epoch and at an epoch boundary matched
uninterrupted training, including the native resume CLI and next batch.
Weights-only recovery remains explicitly distinct. CPU equivalence is not
Azure Blob durability, cross-device GPU determinism or physical quality.
Either option must be included in a newly reviewed training plan; old plans
are not silently relabeled as interruption-safe.

See [policy learning](docs/policy-learning.md),
[the learning API](docs/learning-api.md), and
[runtime control and evidence](docs/runtime-control.md), including
[paused simulation](docs/paused-simulation.md), for the implemented
contracts, operator responsibilities, and remaining live gates.

## What exists

```text
apps/web     Public audience viewer plus protected MSAL operator console
apps/api     Authenticated API, durable revisions/runs, approval and result verification
apps/learning_worker  Private identity-only job, artifact and policy registry boundary
agents       Separate versioned Foundry inspection and proposal-only learning coach
simulation   Reviewed scene builders, command guards, Isaac adapter and HTTPS bridge
learning     Scoped demonstrations, explicit ACT/SmolVLA paths, AML jobs and physical evidence
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

CI additionally runs the frozen Python 3.11 CPU policy environment independently
of the API environment: installed SmolVLA preprocessing interfaces, real LeRobot
conversion of labeled synthetic observations, and all Azure ML pipeline schemas.
Those jobs have no Azure credentials or vendor-weight downloads; their artifacts
are labeled test-only and cannot pass the physical learning gate.

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