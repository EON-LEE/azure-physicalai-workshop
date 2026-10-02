# Live acceptance evidence and release gates

`scripts.live_acceptance` is an HTTPS Azure Container Apps CLI harness, not an
offline demonstration and not a replacement for `tests/cases.json`. Its default
is a **read-only smoke**. It never turns that smoke into G3/G4/G5 or a product
release pass. Local harness/aggregator tests use explicit transport, credential,
clock, camera and motion doubles; they are not live Azure results.

## Identity and deployment snapshot

Use the root locked Python environment in Ubuntu WSL:

```bash
source scripts/dev-env.sh
uv run --locked python -m scripts.live_acceptance --help
```

This sets `UV_PROJECT_ENVIRONMENT` to the native Linux cache, not an NTFS venv.
Authenticate Azure CLI separately in the authorized tenant. The harness uses
`azure.identity.AzureCliCredential`, bound to `--tenant-id`, and requests
`api://<api-client-id>/.default` (the SDK maps this to the API resource for Azure
CLI). Bootstrap must advertise `api://<api-client-id>/access_as_user`, and the
server requires that delegated permission in the acquired token. The Azure CLI
application must have the tenant's required delegated consent/preauthorization;
an app-only identity without this scope is not a substitute. Authentication,
consent and quota failures block acceptance; this tool does not grant permissions.

Only an HTTPS `*.azurecontainerapps.io` origin is accepted. Credentials in URLs,
paths, query strings, fragments and nonstandard ports are rejected. TLS
verification stays enabled, redirects are not followed, and ambient HTTP proxy
credentials are not used. Bootstrap tenant/scope must match the requested
identity **before a bearer token is sent**. Tokens remain in memory; reports and
console messages never contain request headers, SDK exception text or arbitrary
HTTP error bodies.

Supply `--provenance` with a JSON snapshot obtained from the actual deployment:

```json
{
  "candidate_commit": "<full 40-character lowercase Git SHA>",
  "endpoint": "https://<app>.<environment>.<region>.azurecontainerapps.io",
  "image": "<registry>.azurecr.io/<repository>@sha256:<64-character digest>",
  "model": {
    "deployment": "<actual model deployment>",
    "name": "<actual model name>",
    "version": "<actual model version>"
  },
  "deployment_config_sha256": "<SHA-256 of the reviewed deployment configuration>",
  "observed_at": "<timezone-aware ISO 8601 timestamp>",
  "source": "actual"
}
```

Placeholders above are documentation only and must be replaced. The snapshot
must be no older than 24 hours, match the requested endpoint and candidate
commit, and identify a digest-pinned image. Use a **nonsecret** reviewed deployment
configuration for the configuration hash; never include tokens, private keys or
secrets in this file.

The snapshot is explicitly labeled `provenance_origin:
operator_deployment_snapshot`: it is supplied deployment evidence, not a
server-side attestation. The HTTP API currently exposes neither its running
image digest nor model version. The operator must correlate the snapshot with
read-only Azure deployment/image/model records. The harness separately records
what it actually observes at the API. In particular, `agent.configured` is
**not** proof of a live model call. A direct non-motion Foundry bootstrap probe
can be retained separately but cannot prove physical closed-loop execution.

## Default smoke

```bash
uv run --locked python -m scripts.live_acceptance \
  --endpoint "https://<app-fqdn>.azurecontainerapps.io" \
  --tenant-id "<tenant UUID>" --api-client-id "<API application UUID>" \
  --provenance "/path/to/deployment-snapshot.json" \
  --candidate-commit "<host git rev-parse HEAD>" \
  --output "/path/to/live-smoke.json"
```

`--candidate-commit` is optional when `git rev-parse HEAD` works. App-managed
Windows worktrees may have a `.git` pointer that native WSL Git cannot resolve:
obtain the full SHA with **host Git in this worktree** and pass it explicitly.
No fallback invents a commit or uses a branch name. The explicit SHA is an
operator assertion; it must be the current candidate, not an older deployed
image's source revision.

The named cases check public bootstrap, unauthenticated runtime **401** and
Bearer challenge, authenticated exact schema, valid nonempty templates,
environment listing/record hashes, and the runtime contract. An empty saved
environment list is valid smoke data, but not proof of a Cosmos write.
Unavailable/loading/occupied GPU status is recorded as **blocked**, not pass.
Configured Foundry is reported with `foundry_execution_verified: false`.
Smoke does not write customer configuration, activate a scene, create, approve
or cancel tasks. It is not a browser test, a paid model probe, or a full release.

## Explicit physical exercise

Only `--exercise` authorizes state changes. It additionally requires a customer
environment file; there is no automatic template selection or silent replay.

```bash
uv run --locked python -m scripts.live_acceptance \
  --endpoint "https://<app-fqdn>.azurecontainerapps.io" \
  --tenant-id "<tenant UUID>" --api-client-id "<API application UUID>" \
  --provenance "/path/to/deployment-snapshot.json" \
  --candidate-commit "<current host Git SHA>" \
  --exercise --environment "/path/to/reviewed-environment.json" --episodes 20 \
  --output "/path/to/physical-evidence.json"
```

This can incur cloud/model/GPU charges and move the simulated robot. Run only
inside an explicitly authorized deployment, budget and lifetime. Each episode
saves the **original JSON text**, uses the loaded expected revision for updates,
activates that revision, waits for its ready epoch, captures a fresh PNG, creates
a new inspection run, approves the exact returned model response, and polls for
physical completion. Activation is repeated to reset each episode according to
the configured scene seed; these are repeatability episodes, **not** a balanced
normal/defect evaluation dataset or a browser acceptance suite.

Passing an episode requires a unique command/observation, actual model response
ID, matching environment/revision/epoch, a simulator `succeeded` completion with
timestamp, and a finite final position within 0.04 m on each axis of the planned
workflow station (the current service's target predicate). It requires a valid
fresh PNG before and after, distinct frame IDs, and increasing physics steps and
capture times. Camera freshness is bounded by the customer setting and **2000 ms**,
whichever is stricter. Completion must fall within the customer deadline and
**30 seconds**, whichever is stricter. Dispatch ACK or agent text is never success.
Reports store frame hashes/metadata, not bearer-protected image URLs or pixels.

The physical threshold requires **at least 20 attempted episodes, at least 18
successes and at least a 90% success rate**. One successful sample does not pass
this threshold. Individual failures remain visible and block an overall
all-passed report; the threshold alone is not a release. Missing GPU stops
exercise rather than generating twenty fake results.

An outstanding run may have been created even when its HTTP response is lost.
On failure the harness attempts cancellation under the same `--exercise`
authorization and polls to confirm terminal state. Cancellation cleanup is
explicitly labeled and never counted as motion success. If confirmation fails,
the report remains blocked/failed; the operator must inspect the recorded run ID
before another exercise. Customer environment records and saved evidence are
**not deleted** by cleanup.

Requests have a configurable timeout (default 30 seconds, maximum 120); polls
have a wall-clock deadline (default 60 seconds, maximum 300), at most one request
per second, and no unbounded retry. Azure CLI credential acquisition also has a
process timeout. An in-flight request/authentication can consume its bounded
timeout before a poll deadline is reported. Episode count is limited to 100.

## Versioned reports

The `1.0` JSON envelope contains `scope` (`smoke`, `physical`, or independently
produced `release_gate`), candidate commit, actual/fixture/local `source`,
timezone-aware start/end times, overall status, deployment provenance, and
nonempty named `cases`. Every case has:

- `id`, `name`, `status`, `source`, `started_at`, `finished_at`;
- `evidence` with actual observations/IDs/hashes, or partial episode evidence;
- `failure` with a bounded diagnostic code, never a raw credential-bearing error.

Runtime failures still produce a report and a nonzero exit code. Invalid input,
unresolved Git or unwritable output fails closed before producing a valid report.
Output is atomically replaced. Exit 0 means only the selected report's checks
passed; exit 2 means blocked/failed. Harness envelopes always have
`release_ready: false` and `gates: []`.

## Strict release aggregation

```bash
uv run --locked python -m scripts.release_gate \
  /path/to/g0.json /path/to/g1.json /path/to/g2.json /path/to/g3.json \
  /path/to/g4.json /path/to/g5.json /path/to/g6.json /path/to/g7.json \
  --expected-commit "<current host Git SHA>" \
  --output "/path/to/release-decision.json"
```

Each input must be a `1.0` `release_gate` envelope with the fields above plus
`gates: [{"id": "G0", "status": "passed", "case_ids": ["<catalog case ID>", "..."]}]`.
Each gate is supplied **exactly once**, possibly grouping several gates in one
report. Every case ID in `tests/cases.json` must occur exactly once, assigned to
its declared gate. Every gate, report and case must be passed with nonempty
evidence and no failure. Unknown IDs, duplicate cases/gates/assignments, empty
or unnamed cases, skipped/cancelled/planned/blocked/failed states, missing files,
bad JSON/version, and incomplete gate lists block the release.

The required set is exactly **G0 through G7**, not a configurable subset. The
catalog must itself mark all required cases implemented; local component test
results never rewrite or implicitly complete planned acceptance cases. Reports
must match the current candidate (`--expected-commit` has the same host-Git use
as above), endpoint, model version, image digest and configuration hash. Report
and deployment timestamps must be recent (default and maximum 24 hours), not
future or inverted; case timestamps must fall inside their report and the
deployment snapshot must precede execution. G3-G7 require
`actual` at report, provenance and case levels; fixtures cannot satisfy real
Azure/GPU/model gates. G0-G2 may use properly labeled local/fixture evidence
where the catalog allows component tests, without promoting it to a live gate.

The aggregator is a strict **evidence-contract checker**, not a signature
verifier or an implementation of all future gate-specific evaluators. A producer
must actually implement and execute each catalog assertion, retain its raw
artifacts and never relabel a fixture as actual. It cannot infer real execution
from a self-authored status string. In particular, current planned G3-G7 cases
remain release blockers regardless of harness unit-test success.
