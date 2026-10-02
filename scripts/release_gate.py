"""Fail-closed aggregation of versioned, complete release-gate evidence."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from contracts.validate_environment import parse_document
from scripts.live_acceptance import (
    ROOT,
    CheckError,
    Report,
    fresh,
    now,
    resolve_commit,
    write_report,
)

REQUIRED_GATES = frozenset(f"G{index}" for index in range(8))
REAL_GATES = frozenset({"G3", "G4", "G5", "G6", "G7"})


def aggregate(
    reports: Sequence[Report],
    catalog: dict,
    candidate_commit: str,
    *,
    reference: datetime,
    max_age_seconds: int = 86400,
) -> dict:
    failures: list[str] = []
    if not 0 < max_age_seconds <= 86400:
        failures.append("invalid_freshness_window")
    if not reports:
        failures.append("no_reports")
    catalog_gates = catalog.get("required_release_gates", [])
    if len(catalog_gates) != 8 or set(catalog_gates) != REQUIRED_GATES:
        failures.append("catalog_requires_exact_G0_through_G7")
    expected: dict[str, str] = {}
    for case in catalog.get("cases", []):
        if not case.get("id") or case.get("gate") not in REQUIRED_GATES:
            failures.append("invalid_catalog_case")
            continue
        if case["id"] in expected:
            failures.append(f"duplicate_catalog_case:{case['id']}")
        expected[case["id"]] = case["gate"]
        if case.get("status") != "implemented":
            failures.append(f"catalog_case_not_implemented:{case['id']}")
    if not expected:
        failures.append("empty_catalog")
    seen_cases: set[str] = set()
    seen_gates: set[str] = set()
    assignment: dict[str, str] = {}
    baseline = None
    for report in reports:
        if report.scope != "release_gate":
            failures.append("smoke_or_physical_sample_is_not_release_gate_evidence")
        if report.candidate_commit != candidate_commit:
            failures.append("candidate_commit_mismatch")
        if report.status != "passed":
            failures.append("report_not_passed")
        if not (
            report.started_at <= report.finished_at
            and fresh(report.started_at, reference, max_age_seconds)
            and fresh(report.finished_at, reference, max_age_seconds)
        ):
            failures.append("invalid_or_stale_report_timestamps")
        provenance = report.provenance
        if provenance.candidate_commit != candidate_commit:
            failures.append("provenance_commit_mismatch")
        if not fresh(provenance.observed_at, reference, max_age_seconds):
            failures.append("stale_provenance")
        if provenance.observed_at > report.started_at:
            failures.append("provenance_observed_after_execution")
        identity = provenance.model_dump(exclude={"observed_at", "source"})
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            failures.append("deployment_model_image_config_mismatch")
        case_ids = {case.id for case in report.cases}
        local_assignments: set[str] = set()
        for gate in report.gates:
            if gate.id in seen_gates:
                failures.append(f"duplicate_gate:{gate.id}")
            seen_gates.add(gate.id)
            if gate.status != "passed":
                failures.append(f"gate_not_passed:{gate.id}")
            if len(gate.case_ids) != len(set(gate.case_ids)):
                failures.append(f"duplicate_gate_case:{gate.id}")
            if gate.id in REAL_GATES and (
                report.source != "actual" or provenance.source != "actual"
            ):
                failures.append(f"nonactual_gate:{gate.id}")
            for case_id in gate.case_ids:
                if case_id in assignment:
                    failures.append(f"duplicate_case_assignment:{case_id}")
                assignment[case_id] = gate.id
                local_assignments.add(case_id)
                if case_id not in case_ids or expected.get(case_id) != gate.id:
                    failures.append(f"incorrect_gate_case:{case_id}")
        if local_assignments != case_ids:
            failures.append("report_has_unassigned_or_missing_cases")
        for case in report.cases:
            if case.id in seen_cases:
                failures.append(f"duplicate_case:{case.id}")
            seen_cases.add(case.id)
            if case.status != "passed" or case.failure is not None:
                failures.append(f"case_not_passed:{case.id}")
            if not case.evidence:
                failures.append(f"empty_case_evidence:{case.id}")
            if not (report.started_at <= case.started_at <= case.finished_at <= report.finished_at):
                failures.append(f"case_timestamp_outside_report:{case.id}")
            if case.source != report.source:
                failures.append(f"case_source_mismatch:{case.id}")
            if expected.get(case.id) in REAL_GATES and case.source != "actual":
                failures.append(f"nonactual_case:{case.id}")
    if seen_gates != REQUIRED_GATES:
        failures.append("missing_required_gates:" + ",".join(sorted(REQUIRED_GATES - seen_gates)))
    for case_id in sorted(set(expected) - seen_cases):
        failures.append(f"missing_required_case:{case_id}")
    for case_id in sorted(seen_cases - set(expected)):
        failures.append(f"unknown_case:{case_id}")
    return {
        "schema_version": "1.0",
        "scope": "release_decision",
        "candidate_commit": candidate_commit,
        "evaluated_at": reference.isoformat(),
        "release_ready": not failures,
        "status": "failed" if failures else "passed",
        "required_gates": sorted(REQUIRED_GATES),
        "received_gates": sorted(seen_gates),
        "case_count": len(seen_cases),
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", help="Full host Git SHA when WSL cannot resolve HEAD.")
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    args = parser.parse_args(argv)
    reports = []
    failures = []
    for index, path in enumerate(args.reports):
        try:
            reports.append(Report.model_validate(parse_document(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, ValidationError):
            failures.append(f"invalid_or_missing_report:{index + 1}")
    try:
        catalog = parse_document((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
        result = aggregate(
            reports,
            catalog,
            resolve_commit(args.expected_commit),
            reference=now(),
            max_age_seconds=args.max_age_seconds,
        )
        result["failures"].extend(failures)
        if failures:
            result.update(status="failed", release_ready=False)
        write_report(args.output, result)
    except (OSError, ValueError, CheckError, subprocess.SubprocessError):
        print("Release blocked: invalid catalog, unresolved candidate commit, or evidence output.")
        return 2
    print(f"Release {result['status']}: {len(result['failures'])} blocking findings.")
    return 0 if result["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
