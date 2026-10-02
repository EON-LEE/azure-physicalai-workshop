"""Synthetic report validation only; these tests do not produce Azure release evidence."""

import copy
import json
import subprocess
from datetime import timedelta

import pytest
from pydantic import ValidationError
from test_live_acceptance import COMMIT, START, harness, provenance

from scripts import live_acceptance as live
from scripts import release_gate as gate


def fixture():
    """An invented tiny catalog/report to isolate aggregator logic, never the real catalog."""
    catalog = {
        "required_release_gates": [f"G{i}" for i in range(8)],
        "cases": [{"id": f"TEST-{i}", "gate": f"G{i}", "status": "implemented"} for i in range(8)],
    }
    snapshot = provenance().model_copy(update={"source": "actual"})
    reports = []
    for index in range(8):
        reports.append(
            live.Report(
                scope="release_gate",
                candidate_commit=COMMIT,
                source="actual",
                started_at=START,
                finished_at=START,
                status="passed",
                provenance=snapshot,
                cases=[
                    live.Case(
                        id=f"TEST-{index}",
                        name=f"Synthetic aggregator case {index}",
                        status="passed",
                        source="actual",
                        started_at=START,
                        finished_at=START,
                        evidence={"test_double_artifact": f"synthetic-{index}"},
                    )
                ],
                gates=[live.Gate(id=f"G{index}", status="passed", case_ids=[f"TEST-{index}"])],
            )
        )
    return reports, catalog


def evaluate(reports, catalog, **kwargs):
    return gate.aggregate(
        reports, catalog, COMMIT, reference=START + timedelta(seconds=1), **kwargs
    )


def test_complete_synthetic_contract_is_accepted():
    reports, catalog = fixture()
    result = evaluate(reports, catalog)
    assert result["release_ready"] and result["case_count"] == 8
    assert not result["failures"]


def test_zero_reports_never_passes():
    _, catalog = fixture()
    result = evaluate([], catalog)
    assert not result["release_ready"]
    assert "no_reports" in result["failures"]
    assert result["case_count"] == 0


@pytest.mark.parametrize("status", ["failed", "blocked", "skipped", "cancelled", "planned"])
@pytest.mark.parametrize("where", ["report", "gate", "case"])
def test_every_nonpassing_status_blocks(status, where):
    reports, catalog = fixture()
    if where == "report":
        reports[3].status = status
    elif where == "gate":
        reports[3].gates[0].status = status
    else:
        reports[3].cases[0].status = status
    assert not evaluate(reports, catalog)["release_ready"]


@pytest.mark.parametrize("index", range(8))
def test_every_required_gate_is_mandatory(index):
    reports, catalog = fixture()
    reports.pop(index)
    result = evaluate(reports, catalog)
    assert not result["release_ready"]
    assert f"missing_required_gates:G{index}" in result["failures"]


@pytest.mark.parametrize(
    "gates", [[], ["G0"], [f"G{i}" for i in range(7)], [f"G{i}" for i in range(8)] + ["G7"]]
)
def test_catalog_cannot_lower_or_duplicate_required_gates(gates):
    reports, catalog = fixture()
    catalog["required_release_gates"] = gates
    assert "catalog_requires_exact_G0_through_G7" in evaluate(reports, catalog)["failures"]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("duplicate_report", "duplicate_gate:G0"),
        ("duplicate_case", "duplicate_case:TEST-0"),
        ("duplicate_assignment", "duplicate_gate_case:G0"),
        ("misassigned", "incorrect_gate_case:TEST-1"),
        ("unassigned", "report_has_unassigned_or_missing_cases"),
        ("empty", "empty_case_evidence:TEST-0"),
        ("failure", "case_not_passed:TEST-0"),
        ("unknown", "unknown_case:NOT-IN-CATALOG"),
    ],
)
def test_case_and_gate_coverage_is_exact(mutation, expected):
    reports, catalog = fixture()
    report = reports[0]
    if mutation == "duplicate_report":
        reports.append(report.model_copy(deep=True))
    elif mutation == "duplicate_case":
        report.cases.append(report.cases[0].model_copy(deep=True))
    elif mutation == "duplicate_assignment":
        report.gates[0].case_ids.append("TEST-0")
    elif mutation == "misassigned":
        report.gates[0].case_ids = ["TEST-1"]
    elif mutation == "unassigned":
        report.gates = []
    elif mutation == "empty":
        report.cases[0].evidence = {}
    elif mutation == "failure":
        report.cases[0].failure = "previous failure"
    elif mutation == "unknown":
        report.cases[0].id = "NOT-IN-CATALOG"
    assert expected in evaluate(reports, catalog)["failures"]


@pytest.mark.parametrize("where", ["report", "provenance"])
def test_current_commit_required(where):
    reports, catalog = fixture()
    if where == "report":
        reports[0].candidate_commit = "b" * 40
    else:
        reports[0].provenance = reports[0].provenance.model_copy(
            update={"candidate_commit": "b" * 40}
        )
    assert not evaluate(reports, catalog)["release_ready"]


@pytest.mark.parametrize("field", ["image", "model", "endpoint", "deployment_config_sha256"])
def test_all_reports_must_bind_same_actual_deployment(field):
    reports, catalog = fixture()
    snapshot = reports[0].provenance.model_copy(deep=True)
    if field == "model":
        snapshot.model.version = "changed"
    elif field == "endpoint":
        snapshot.endpoint = "https://different.region.azurecontainerapps.io"
    elif field == "image":
        snapshot.image = f"unit.azurecr.io/api@sha256:{'f' * 64}"
    else:
        snapshot.deployment_config_sha256 = "f" * 64
    reports[0].provenance = snapshot
    assert "deployment_model_image_config_mismatch" in evaluate(reports, catalog)["failures"]


@pytest.mark.parametrize(
    "timestamp",
    [
        START - timedelta(days=2),
        START + timedelta(seconds=2),
    ],
)
def test_stale_and_future_reports_rejected(timestamp):
    reports, catalog = fixture()
    reports[0].started_at = reports[0].finished_at = timestamp
    assert "invalid_or_stale_report_timestamps" in evaluate(reports, catalog)["failures"]


def test_inverted_and_out_of_report_case_times_rejected():
    reports, catalog = fixture()
    reports[0].started_at = START + timedelta(seconds=1)
    result = evaluate(reports, catalog)
    assert "invalid_or_stale_report_timestamps" in result["failures"]
    assert "case_timestamp_outside_report:TEST-0" in result["failures"]


@pytest.mark.parametrize("age", [0, -1, 86401])
def test_freshness_window_cannot_disable_validation(age):
    reports, catalog = fixture()
    assert "invalid_freshness_window" in evaluate(reports, catalog, max_age_seconds=age)["failures"]


def test_stale_provenance_rejected():
    reports, catalog = fixture()
    reports[0].provenance = reports[0].provenance.model_copy(
        update={"observed_at": START - timedelta(days=2)}
    )
    assert "stale_provenance" in evaluate(reports, catalog)["failures"]


@pytest.mark.parametrize("gate_index", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("source", ["fixture", "local"])
def test_actual_cloud_gpu_gates_cannot_use_fixtures(gate_index, source):
    reports, catalog = fixture()
    report = reports[gate_index]
    report.source = source
    report.cases[0].source = source
    report.provenance = report.provenance.model_copy(update={"source": source})
    result = evaluate(reports, catalog)
    assert f"nonactual_gate:G{gate_index}" in result["failures"]
    assert not result["release_ready"]


def test_fixture_case_cannot_hide_inside_actual_report():
    reports, catalog = fixture()
    reports[3].cases[0].source = "fixture"
    assert "nonactual_case:TEST-3" in evaluate(reports, catalog)["failures"]


def test_fixture_provenance_cannot_hide_inside_actual_report():
    reports, catalog = fixture()
    reports[3].provenance = reports[3].provenance.model_copy(update={"source": "fixture"})
    assert "nonactual_gate:G3" in evaluate(reports, catalog)["failures"]


def test_smoke_and_physical_samples_cannot_be_promoted():
    reports, catalog = fixture()
    for scope in ("smoke", "physical"):
        reports[0].scope = scope
        assert not evaluate(reports, catalog)["release_ready"]
    runner, _, _ = harness()
    assert not evaluate([runner.run()], catalog)["release_ready"]


def test_real_catalog_remains_planned_and_incomplete():
    reports, _ = fixture()
    catalog = json.loads((live.ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
    before = copy.deepcopy(catalog)
    result = evaluate(reports, catalog)
    assert not result["release_ready"]
    assert any(f.startswith("catalog_case_not_implemented:") for f in result["failures"])
    assert any(f.startswith("missing_required_case:") for f in result["failures"])
    assert before == catalog


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), "2.0"),
        (("candidate_commit",), "main"),
        (("cases",), []),
        (("started_at",), "not-a-timestamp"),
        (("gates", 0, "id"), "G8"),
        (("gates", 0, "case_ids"), []),
        (("cases", 0, "name"), " "),
        (("cases", 0, "id"), ""),
        (("provenance", "image"), "unit.azurecr.io/api:latest"),
    ],
)
def test_malformed_versioned_reports_rejected(path, value):
    reports, _ = fixture()
    data = reports[0].model_dump(mode="json")
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        live.Report.model_validate(data)


def test_cli_missing_invalid_report_writes_failed_decision(tmp_path):
    output = tmp_path / "decision.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"cases": [], "cases": []}', encoding="utf-8")
    assert (
        gate.main(
            [
                str(tmp_path / "missing.json"),
                str(invalid),
                "--output",
                str(output),
                "--expected-commit",
                COMMIT,
            ]
        )
        == 2
    )
    report = json.loads(output.read_text())
    assert report["release_ready"] is False
    assert "invalid_or_missing_report:1" in report["failures"]
    assert "invalid_or_missing_report:2" in report["failures"]


def test_cli_unresolved_git_fails_closed(tmp_path, monkeypatch, capsys):
    def broken():
        raise subprocess.CalledProcessError(128, "git", stderr="SECRET")

    monkeypatch.setattr(live, "current_commit", broken)
    assert gate.main(["--output", str(tmp_path / "decision.json")]) == 2
    assert "SECRET" not in capsys.readouterr().out
