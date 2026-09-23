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


def test_runtime_contracts_core_and_entrypoint_do_not_import_gpu_libraries():
    command = (
        "import sys; import simulation.runtime_contracts; import simulation.core; "
        "import simulation.capture_worker; import simulation.run_isaac; "
        "assert not {'isaacsim', 'omni', 'pxr', 'torch', 'lerobot', 'groot'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
