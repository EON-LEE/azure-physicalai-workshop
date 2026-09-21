"""Run an explicitly selected script on the dedicated GPU, checking guest exit status."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID, uuid4


def guest_succeeded(response: dict) -> bool:
    messages = "\n".join(item.get("message", "") for item in response.get("value", []))
    return re.findall(r"(?m)^PHYSICALAI_REMOTE_EXIT=(\d+)\s*$", messages) == ["0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", required=True, type=UUID)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--vm", required=True)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    script = args.script.read_text(encoding="utf-8").replace("\r\n", "\n")
    delimiter = "PHYSICALAI_BASH_" + uuid4().hex
    wrapped = (
        f"bash <<'{delimiter}'\n"
        "trap 'code=$?; echo PHYSICALAI_REMOTE_EXIT=$code; exit $code' EXIT\n"
        + script
        + f"\n{delimiter}\n"
    )
    with tempfile.TemporaryDirectory(prefix="physicalai-run-command-") as temporary:
        path = Path(temporary) / "run.sh"
        path.write_text(wrapped, encoding="utf-8")
        result = subprocess.run(
            [
                "az",
                "vm",
                "run-command",
                "invoke",
                "--subscription",
                str(args.subscription),
                "--resource-group",
                args.resource_group,
                "--name",
                args.vm,
                "--command-id",
                "RunShellScript",
                "--scripts",
                f"@{path}",
                "--output",
                "json",
                "--only-show-errors",
            ],
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if result.returncode:
        args.output.write_text(
            json.dumps(
                {
                    "status": "azure_command_failed",
                    "error": result.stderr,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(result.stderr[-5000:])
        raise SystemExit(result.returncode)
    response = json.loads(result.stdout)
    args.output.write_text(json.dumps(response, indent=2), encoding="utf-8")
    messages = "\n".join(item.get("message", "") for item in response.get("value", []))
    print(messages[-14000:])
    if not guest_succeeded(response):
        raise SystemExit("Guest success marker is absent; inspect the saved command result.")


if __name__ == "__main__":
    main()
