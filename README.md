# Azure Physical AI manufacturing reference solution

**Current scope: executable customer environment validation and automated tests.**
The web console, Foundry agents, Isaac Sim integration, Python scene-extension
runtime, training pipeline, and Azure deployment are planned, not implemented.
Passing the current tests does not establish a working live demo.

The intended solution runs a real inspection-and-sorting cell in Isaac Sim on
Azure, with Foundry handling high-level tasks and a separate controller enforcing
allowed actions. Customers will customize the same application through versioned
environment JSON and reviewed Python scene extensions.

## What exists

```text
contracts    JSON Schema and configuration validation CLI
examples     Reference and second-customer environment descriptions
tests        Executable CPU cases and the planned live-test catalog
docs         Customer configuration and test-gate documentation
```

The example template and robot IDs describe the future simulation interface.
Their corresponding scenes and robot controllers do not exist yet.

## Develop and test in WSL

Run these commands from the repository root **inside Ubuntu WSL**, not with
Windows Python. Python 3.13 is the project default; dependency versions are
recorded in `uv.lock`.

```bash
uv sync --locked --group dev
uv run --locked python -m contracts.validate_environment examples/inspection-cell.json --json
uv run --locked python -m contracts.validate_environment examples/customer-cell.json --json
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q --junitxml=test-results/contracts.xml
```

No Azure credentials, GPU, Docker engine, or MCP server is needed for these CPU
tests. Dependency installation may require access to the package registry.
On a Windows-mounted worktree, `UV_LINK_MODE=copy` avoids cross-filesystem
hard-link warnings.

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

## Automated test gates

[tests/cases.json](tests/cases.json) is the machine-readable test catalog.
Implemented cases drive pytest directly. Each case names its requirement,
gate, input or mutations, and expected result. Planned cases describe their
runner, prerequisites, procedure, and assertions without claiming execution.

The [CPU CI workflow](.github/workflows/contracts.yml) runs lint, formatting,
and these tests on pull requests and pushes, and retains JUnit reports.
It does not deploy Azure or validate the future live application.

See [the automated test strategy](docs/automated-testing.md) for the distinction
between configuration checks and required browser, GPU, Foundry, learning,
and deployment gates. A green CPU workflow is not permission to call v1 complete.