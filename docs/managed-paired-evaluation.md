# Verified pairing of distinct managed physical episodes

`python -m simulation.paired_evaluation` closes the identity boundary between
the [one-episode managed adapter](managed-learned-evaluation.md) and the native
twenty-case before/after scorer. It is an **offline artifact verifier**, not a
job submitter, simulator, model server or alternate quality policy.

The native plan, `_trial`, task predicate, model lineage/leakage checks and
quality thresholds are reused unchanged. The old native all-attempt recording
validator remains strict. This adapter emits a new versioned report; it does not
forge that recorder's `results.json`.

## Freeze the mapping before any evaluated attempt

Create and privately preserve `physicalai.managed-paired-plan/v1` before the first
learned held-out job. Record its exact **file SHA256** as an independently
approved input. It contains:

| Field | Contract |
| --- | --- |
| `schema` | `physicalai.managed-paired-plan/v1` |
| `issued_at_utc` | Original UTC time ending `Z`, before every task grant/claim |
| `evaluation_plan` | Unchanged native `physicalai.smolvla-paired-plan/v2` |
| `runtime` | Actual `physicalai.paused-evaluation-runtime/v1`, matching the native plan hash |
| `model_runtime_sha256` | Exact approved Python 3.11 model-runtime descriptor file hash |
| `assignments` | Exactly forty records with `logical_case_id`, `role`, `physical_attempt_id` |

The native plan has exactly twenty seeds **30001..30020**, attempt zero, frozen
four-centimetre tolerance and the original placement-span requirement. It locks
the real before/after model manifest hashes, owner, profile, criteria, frozen
conditions, runtime and unmodified quality limits. Both model roots must pass
native inference validation, task/lineage checks and training/held-out exclusion.
There is no `reference`/vendor-base substitution for a before model.

Each native plan case's `episode_id` is its **logical case ID**. For each case,
assign distinct physical UUIDs to `before` and `after`; no physical UUID may occur
twice. Preserve the native alternating order: before/after for case zero,
after/before for case one, and so on. The array order is enforced.

Each new `BatchLearnedSpec` sets:

```json
{
  "attempt_id": "<this predeclared physical UUID>",
  "role": "before",
  "pairing_plan_sha256": "<exact frozen mapping file SHA256>"
}
```

These are additional bindings on the existing full spec, not a complete task
specification. The physical command ID, raw episode ID and private capture prefix
all remain that physical UUID. `pairing_plan_sha256` is included in the hashed
spec and pre-execution claim. Standalone specs omit the optional field and retain
their original serialization. The mapping does not contain short-lived grant
hashes, so each original <=600-second grant can still be issued after actual
readiness without a hash cycle or renewal.

A retry is not a replacement for a frozen slot. `previous_attempt_id` is forbidden
for scored assignments. Preserve failed/preempted attempts; do not silently mint
new UUIDs or select a better outcome. A new study requires a separately approved
mapping, not an update to this one.

## Export exact private artifacts, without rewriting them

The operator exports original private artifacts into:

```text
evidence-root/
  attempts/<physical UUID>/
    inputs/spec.json
    inputs/environment.json
    inputs/grant.json
    inputs/criteria.json
    inputs/conditions.json
    claim.json
    completion.json
    preflight.json
    probe.json
    probe.log
    acceptance.json
    acceptance.log
    raw-manifest.json
    capture/manifest.json
    capture/episodes/<same physical UUID>/frames.jsonl
    capture/episodes/<same physical UUID>/inspection/*.png
    capture/episodes/<same physical UUID>/overview/*.png
```

Only export files that actually exist; a queued/preempted slot may have a spec
and claim but no completion. Do not generate a failure capture, heartbeat or
terminal manifest to fill that gap. The original spec may be copied from its
approved private input Blob when execution never published it. Keep provider
errors separately; they cannot substitute for a native physical report.

`capture/` is a local directory holding the original owner/physical-episode Blob
contents. Its relative paths, raw manifest, frame records, image bytes and
episode IDs are unchanged. The report receipt's original private URI must match
the spec's storage account/container/owner/physical UUID.

Create a separately hash-pinned `physicalai.managed-paired-evidence/v1` snapshot:

```json
{
  "schema": "physicalai.managed-paired-evidence/v1",
  "mapping_sha256": "<frozen mapping file SHA256>",
  "snapshot_at_utc": "<actual UTC snapshot time ending Z>",
  "attempts": [
    {
      "physical_attempt_id": "<original physical UUID>",
      "files": {
        "inputs/spec.json": {"sha256": "<file SHA256>", "bytes": 1234},
        "claim.json": {"sha256": "<file SHA256>", "bytes": 1234}
      }
    }
  ]
}
```

The example is abbreviated and shows an incomplete attempt. Include **every**
exported file and its exact byte length/hash for each observed attempt, including
all capture images. The snapshot supports up to forty unique records, 4,096
files and two GiB per attempt; existing smaller proof/JSON limits still apply.
Paths are bounded, relative and nonsymlink. Unlisted attempt directories/files,
extra/retried IDs, duplicate assignments and omitted failed directories are
rejected. Slots not received remain in the frozen mapping and the output's
missing/incomplete list.

## Independent verification and outcomes

Run from the qualified source environment, with both actual trained model
directories already present:

```bash
python -m simulation.paired_evaluation \
  --root /private/evidence-root \
  --mapping /private/mapping.json --mapping-sha256 <exact-file-SHA256> \
  --evidence /private/evidence.json --evidence-sha256 <exact-file-SHA256> \
  --before-root /private/models/before --after-root /private/models/after \
  --output /private/reports/mapped-report.json
```

There are no cloud calls or GPU imports. Output is create-only and cannot be
written inside attempt/model directories. A partial set produces
`complete=false`, `quality_gate_passed=false`, `conclusion=inconclusive`, all
forty assignments, and the actual received failure/proof inventories. It does
not emit invented trials, partial success rates or a passing quality result.

The live `SimulatorRuntime` subclass is loaded only when a native probe starts.
Importing the offline verifier does not initialize the simulator or the Azure
identity SDK's platform-probing subprocess; the production-locked worker smoke
check exercises this complete cold-import path.

For every scored attempt the verifier checks the exact spec/claim/completion
binding, locked model/runtime/scope/profile/criteria, original grant chronology,
frozen logical case's seed/environment/revision/poses, every private file hash,
the native raw-v3 dataset and actual runtime provenance. It then reruns
`derive_trial` from the original raw observations/actions, per-tick `TaskState`,
actual final cameras and original main-thread heartbeat timestamps, and compares
the full derived trial against the saved native trial. Grasp/release/settling,
servo proof, applied model, zero reference-route calls and limits are rescored.
Rehashed summary booleans cannot waive those checks.

A complete, independently verified **failed** physical trial remains a failure
in its role's twenty-case denominator. Node loss, pre-control failure, partial
hold or missing terminal/native data remain **incomplete**, never fabricated
complete failures. Actual task order is checked using original UTC timestamps;
monotonic clocks are checked within each episode, not compared across different
managed-node boots.

Only forty verified trials allow the unchanged native `compare_trials` to run
with live-artifact verification. A temporary numeric projection uses logical
case IDs to match the native plan; this does not alter any original physical
trial or raw file. The `physicalai.managed-paired-report/v1` output retains both
IDs and each original `physical_trial`, plus the native numeric result. It is
not the old recorder schema and must not be submitted as that recorder's files.

## New heartbeat evidence requires a new image

The new learned probe persists its actual `heartbeat_ns` list and original
`episode_budget`; rescore checks their clocks and derives the saved gaps.
The earlier one-trial image that omitted those fields cannot be retroactively
qualified for paired aggregation. Preserve it and its evidence without
backfilling timestamps. The report-only wrapper change requires a new source/
image proof, but changes no frozen physics, controller, training or inference
semantics. `/work` and the separately qualified Python 3.11 descriptor can remain
unchanged if the operator's overlay leaves those bytes untouched.

CPU tests use explicitly labeled metadata/pixel doubles to exercise native
codecs, all-slot accounting and arithmetic. They do not demonstrate GPU policy
learning, physical success or completed production quality.
