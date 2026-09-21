import subprocess
import sys


def test_simulator_bridge_does_not_import_web_app_or_public_examples():
    command = (
        "import sys; import simulation.http; "
        "assert 'apps.api.main' not in sys.modules; "
        "assert 'apps.api.public_demo' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
