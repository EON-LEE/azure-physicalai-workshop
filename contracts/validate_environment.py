"""Validate configuration only; never start, import, or control a customer scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("customer-environment.schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)

STATION_ROLES = {
    "source_station": "source",
    "inspect_station": "inspection",
    "accept_station": "accepted",
    "reject_station": "rejected",
}


@dataclass(frozen=True)
class Issue:
    path: str
    code: str
    message: str


def _pointer(parts: Sequence[str | int]) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _finite_issues(value: object, path: tuple[str | int, ...] = ()) -> Iterator[Issue]:
    if isinstance(value, float) and not math.isfinite(value):
        yield Issue(_pointer(path), "non_finite", "Numbers must be finite.")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _finite_issues(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _finite_issues(child, (*path, index))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def load_document(path: Path) -> object:
    return parse_document(path.read_text(encoding="utf-8"))


def parse_document(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def validate_environment(document: object) -> list[Issue]:
    issues = list(_finite_issues(document))
    if issues:
        return issues
    issues = [
        Issue(_pointer(tuple(error.absolute_path)), "schema", error.message)
        for error in VALIDATOR.iter_errors(document)
    ]
    if issues:
        return sorted(issues, key=lambda issue: (issue.path, issue.message))

    assert isinstance(document, dict)  # Guaranteed by the schema before relational checks.
    lower = document["workspace"]["lower_m"]
    upper = document["workspace"]["upper_m"]
    ordered_bounds = all(low < high for low, high in zip(lower, upper, strict=True))
    if not ordered_bounds:
        issues.append(
            Issue(
                "/workspace", "workspace_bounds", "Each lower bound must be below its upper bound."
            )
        )

    stations: dict[str, dict[str, Any]] = {}
    for index, station in enumerate(document["stations"]):
        station_id = station["id"]
        if station_id in stations:
            issues.append(
                Issue(
                    f"/stations/{index}/id", "duplicate_id", f"Duplicate station ID: {station_id}"
                )
            )
        else:
            stations[station_id] = station
        if ordered_bounds and not all(
            low <= coordinate <= high
            for low, coordinate, high in zip(lower, station["position_m"], upper, strict=True)
        ):
            issues.append(
                Issue(
                    f"/stations/{index}/position_m",
                    "station_bounds",
                    "Station position is outside the declared workspace.",
                )
            )

    used: set[str] = set()
    for field, role in STATION_ROLES.items():
        station_id = document["workflow"][field]
        path = f"/workflow/{field}"
        if station_id in used:
            issues.append(
                Issue(path, "reused_station", "Workflow stages must use distinct stations.")
            )
        used.add(station_id)
        station = stations.get(station_id)
        if station is None:
            issues.append(Issue(path, "unknown_station", f"Unknown station ID: {station_id}"))
        elif station["role"] != role:
            issues.append(Issue(path, "station_role", f"Station must have role: {role}"))
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    try:
        document = load_document(args.environment)
    except (OSError, UnicodeError, ValueError) as exc:
        issues = [Issue("/", "input", str(exc))]
    else:
        issues = validate_environment(document)

    report = {
        "valid": not issues,
        "scope": "configuration",
        "runtime_verified": False,
        "issues": [asdict(issue) for issue in issues],
    }
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    elif issues:
        for issue in issues:
            print(f"{issue.path} [{issue.code}] {issue.message}", file=sys.stderr)
    else:
        print("Configuration valid. Scene, controller, and cloud runtime NOT verified.")
    return 2 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
