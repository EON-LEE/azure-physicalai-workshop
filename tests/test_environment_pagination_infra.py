import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.validate_infra import resources

ROOT = Path(__file__).resolve().parents[1]


def test_compiled_cursor_ttl_does_not_expire_existing_state_by_default(tmp_path):
    compiler = shutil.which("bicep")
    if compiler is None:
        installed = Path.home() / ".azure" / "bin" / "bicep"
        if installed.is_file():
            compiler = str(installed)
    if compiler is None:
        pytest.skip("Offline Bicep compiler is unavailable; the infrastructure CI compiles it.")
    output = tmp_path / "foundation.json"
    subprocess.run(
        [compiler, "build", str(ROOT / "infra" / "foundation.bicep"), "--outfile", str(output)],
        check=True,
    )
    compiled = json.loads(output.read_text(encoding="utf-8"))
    containers = resources(
        compiled, "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers"
    )
    assert len(containers) == 1
    state = containers[0]["properties"]["resource"]
    assert state.get("defaultTtl") == -1, (
        "Cursor item TTL must work, while documents without an explicit TTL never expire."
    )
    assert state["partitionKey"] == {"paths": ["/owner_key"], "kind": "Hash"}
