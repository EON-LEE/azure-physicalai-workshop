# Automated test strategy

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

All G1-G7 cases are currently **planned, not implemented or run**. Describing an
assertion in the catalog does not execute it.

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
