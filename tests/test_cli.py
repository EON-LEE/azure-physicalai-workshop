from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import ROOT, cases_for


@pytest.mark.parametrize("case", cases_for("cli"), ids=lambda case: case["id"])
def test_cli_cases(case, make_document, tmp_path):
    path = tmp_path / "environment.json"
    kind = case["input_kind"]
    if kind == "fixture":
        path.write_text(json.dumps(make_document(case), ensure_ascii=False), encoding="utf-8")
    elif kind == "text":
        path.write_text(case["raw_text"], encoding="utf-8")
    elif kind == "invalid_utf8":
        path.write_bytes(b"\xff\xfe")
    elif kind == "directory":
        path.mkdir()
    elif kind != "missing":
        raise ValueError(f"Unsupported CLI fixture: {kind}")

    command = [sys.executable, "-m", "contracts.validate_environment", str(path)]
    if case["json_output"]:
        command.append("--json")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15, check=False
    )
    assert result.returncode == case["expected_exit"], result.stderr
    if case["json_output"]:
        report = json.loads(result.stdout)
        assert set(report) == {"valid", "scope", "runtime_verified", "issues"}
        assert report["valid"] is case["expected_valid"]
        assert report["scope"] == "configuration"
        assert report["runtime_verified"] is False
        assert set(case["expected_codes"]) <= {issue["code"] for issue in report["issues"]}
        assert bool(report["issues"]) is not case["expected_valid"]
        assert result.stderr == ""
    elif case["expected_valid"]:
        assert "runtime NOT verified" in result.stdout
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert all(f"[{code}]" in result.stderr for code in case["expected_codes"])
