from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))


def cases_for(suite: str) -> list[dict]:
    return [
        case
        for case in CATALOG["cases"]
        if case["status"] == "implemented" and case["suite"] == suite
    ]


def change_document(document: dict, changes: list[dict]) -> dict:
    result = copy.deepcopy(document)
    for change in changes:
        parent = result
        for part in change["path"][:-1]:
            parent = parent[part]
        key = change["path"][-1]
        if change["op"] == "set":
            parent[key] = change["value"]
        elif change["op"] == "delete":
            del parent[key]
        else:
            raise ValueError(f"Unsupported test mutation: {change['op']}")
    return result


@pytest.fixture
def make_document():
    def make(case: dict) -> dict:
        path = ROOT / case.get("fixture", CATALOG["default_fixture"])
        original = json.loads(path.read_text(encoding="utf-8"))
        return change_document(original, case.get("changes", []))

    return make
