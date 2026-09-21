"""Check the actual main-thread heartbeat, not an inherited image's unrelated process."""

import math
import sys
import time
from pathlib import Path

HEARTBEAT = Path("/run/physicalai/engine-heartbeat")


def check_heartbeat(path: Path = HEARTBEAT, now: float | None = None) -> None:
    try:
        observed = float(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("The simulator main-thread heartbeat is unavailable.") from exc
    age = (time.monotonic() if now is None else now) - observed
    if not math.isfinite(age) or not 0 <= age <= 30:
        raise RuntimeError("The simulator main-thread heartbeat is stale or invalid.")


def main() -> None:
    try:
        check_heartbeat()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
