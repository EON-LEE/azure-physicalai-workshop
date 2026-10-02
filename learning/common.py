from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath


class ContractError(ValueError):
    """Invalid or unsafe learning input; callers must stop, not substitute data."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def keys(value: object, expected: set[str], label: str) -> dict:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == expected, f"{label} has missing or unexpected fields")
    return value


def integer(value: object, label: str, low: int = 0, high: int = 2**63 - 1) -> int:
    require(type(value) is int and low <= value <= high, f"{label} is out of range")
    return value


def finite(value: object, label: str) -> float:
    require(type(value) in (float, int), f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ContractError(f"{label} exceeds the numeric range") from exc
    require(math.isfinite(number), f"{label} must be a finite number")
    return number


def vector(value: object, size: int, label: str) -> tuple[float, ...]:
    require(isinstance(value, (list, tuple)) and len(value) == size, f"{label}: expected {size}")
    return tuple(finite(item, label) for item in value)


def token(value: object, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value),
        f"Invalid {label}",
    )
    return value


def sha256(value: object, label: str = "sha256") -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid {label}")
    return value


def utc(value: object) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), "Expected a UTC timestamp ending Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError("Invalid UTC timestamp") from exc
    require(parsed.tzinfo == UTC, "Timestamp is not UTC")
    return parsed


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        require(name not in result, f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def parse_json(raw: str | bytes) -> dict:
    def reject_constant(value: str):
        raise ContractError(f"Nonfinite JSON constant: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"Invalid JSON: {exc}") from exc
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def read_json(path: Path, max_bytes: int = 4 * 1024 * 1024) -> dict:
    require(path.stat().st_size <= max_bytes, f"JSON exceeds size limit: {path.name}")
    return parse_json(path.read_bytes())


def relative_path(value: object) -> PurePosixPath:
    require(isinstance(value, str) and bool(value), "Empty artifact path")
    require(
        re.fullmatch(r"[A-Za-z0-9_./-]+", value) is not None
        and not PureWindowsPath(value).drive
        and not PurePosixPath(value).is_absolute()
        and all(part not in ("", ".", "..") for part in value.split("/")),
        f"Unsafe artifact path: {value}",
    )
    return PurePosixPath(value)


def safe_path(root: Path, value: object, *, must_exist: bool = True) -> Path:
    rel = relative_path(value)
    require(not root.is_symlink(), "Artifact root must not be a symlink")
    result = root
    for part in rel.parts:
        result = result / part
        require(not result.is_symlink(), f"Symlink in artifact path: {value}")
    require(result.resolve().is_relative_to(root.resolve()), "Artifact path escapes root")
    if must_exist:
        require(result.is_file(), f"Missing artifact: {value}")
    return result


def write_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def inventory(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "Expected a nonsymlink artifact directory")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlink in artifact: {path.name}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            safe_path(root, rel)
            if rel not in (exclude or set()):
                result[rel] = file_digest(path)
    return result


def verify_inventory(root: Path, expected: object, *, exclude: set[str] | None = None) -> None:
    require(isinstance(expected, dict) and bool(expected), "Empty artifact inventory")
    for rel, checksum in expected.items():
        safe_path(root, rel)
        sha256(checksum)
    require(inventory(root, exclude=exclude) == expected, "Artifact inventory/checksum mismatch")
