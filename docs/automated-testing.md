# Automated test strategy

See [current status](status.md) for actual live execution results; this document
defines test obligations, not a statement that every live gate passed.
`python -m scripts.check_docs` checks repository-local Markdown file targets
(not remote URLs, heading anchors or executable code). It runs in the CPU CI and
`scripts/check.sh`; `tests/test_docs.py` also guards the navigation structure.

## Implemented now

`tests/cases.json` drives three pytest suites:

- `environment`: apply declared mutations to a customer fixture and assert
  validity, required error codes, and no input mutation.
- `finite`: inject NaN and infinities into in-memory configurations and assert
  the exact failing field.
- `cli`: execute the real CLI in a separate process and assert exit status,
  stdout/stderr, structured error shape, and `runtime_verified: false`.

Additional tests check the JSON Schema, exact error paths, unique case IDs,
requirement coverage, supported runners, and the separation of implemented
from planned cases.

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q --junitxml=test-results/contracts.xml
```

The CPU workflow runs automatically on pushes and pull requests. It uses
read-only repository permissions and no Azure credentials. Failed checks
produce a nonzero job status; no `continue-on-error` is used.

## Required future gates

| Gate | What it must prove | Runner and evidence |
|---|---|---|
| G0 | Configuration schema, relations, numeric bounds, CLI errors | WSL/Linux pytest, JUnit |
| G1 | Command state, permissions, approval, idempotency, plugin contracts | pytest, controlled clocks, dispatch counters |
| G2 | Actual web controls and API interaction | Playwright traces and API logs; fixtures labeled |
| G3 | Actual Isaac Sim startup, rendering, movement and faults | Approved GPU runner, pose/frame/physics evidence |
| G4 | Actual Azure identity, Foundry execution and persisted data | Azure test environment, request/run/storage IDs |
| G5 | Browser-to-Foundry-to-simulator closed loop, two customer environments | Playwright plus physical-state assertions |
| G6 | Disjoint training/evaluation data, policy and inspection quality | Azure ML and simulator raw evaluation results |
| G7 | Reproducible workspace/deployment, budget, isolation and cleanup | WSL preflight, IaC and strict release evidence |

Backend and web components now have additional automated suites for G1/G2-style
behavior: real HTTP routes with injected adapters, JWT validation, approval
expiry/idempotency, cancellation, scene guards, raw JSON preservation, strict CSP,
and camera/request cleanup. The separate test harness labels fixtures explicitly.
`scripts/check.sh` also checks actual API response shapes against the production
TypeScript decoders.

`scripts/validate_infra.py` compiles all three Bicep templates and checks baseline
security invariants without Azure login. This is static validation, not G7 live
deployment/cleanup proof.

The learning suite additionally validates scoped demonstrations, real LeRobot
conversion/training interfaces, checkpoint checksums, bounded inference and
Azure ML submission guards. An isolated real CPU smoke executes a genuine ACT
optimizer step and inference; it is not evidence of Azure GPU task quality.

The catalog's planned acceptance cases remain the full release obligations.
Local unit coverage of a clause does not automatically complete that case or
its Azure/GPU gate. Actual cloud/GPU components have since been exercised, but the complete G3-G7
release obligations have not passed. See [current status](status.md) for the
partial successes, failures and remaining learned-quality blocker.

## Release must fail closed

The future release aggregator must require every applicable case ID, the same
candidate commit/config/model/image versions, and a nonempty result report.
Missing GPU or credentials, skipped/cancelled cases, stale evidence, no tests
collected, and missing reports must block the release.

Mocks can support local API/UI development but cannot pass actual Azure or GPU
gates. A replay video cannot satisfy a live-frame or physical-motion assertion.
An agent's text saying "done" cannot satisfy the object's final-pose predicate.

Do not run untrusted fork pull-request code on credentialed GPU runners.
Live jobs require an explicitly approved subscription, resource scope, budget
and lifetime. Cleanup can target only resources recorded as belonging to that
test run, never wildcard deletion of a subscription or unrelated environment.

## Proposed reference-scene acceptance targets

These are targets to confirm and freeze during the hardware/model spike, not
measured performance, customer SLAs, or industrial safety claims:

- At least 18 of 20 fixed reference episodes finish inside the correct tray
  volume with a simulator completion event.
- A 30-second task deadline is enforced. Timeout is not success.
- Observations older than 2000 ms cannot authorize new motion.
- Unauthorized, forbidden, expired, cancelled, or duplicate requests cause
  no additional physical dispatch.
- At least 40 held-out inspection examples, at least 90% recall and at most
  10% false rejection on that frozen set; retain the confusion matrix and failures.
- Training/evaluation episode and seed overlap is rejected automatically.
- Both the reference environment and a reviewed, code-extended customer
  environment must pass live G3-G5.

Record simulation time separately from wall-clock latency, record per-episode
results rather than only averages, and keep evaluator-only ground truth out of
the agent's observation input. Synthetic anomaly thresholds must not be
advertised as validated predictive-maintenance models.
