import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    "recipe", ["Dockerfile", "Dockerfile.six", "Dockerfile.code", "Dockerfile.bundle"]
)
def test_container_recipe_copies_the_complete_cpu_runtime_import_closure(tmp_path, recipe):
    root = Path(__file__).resolve().parents[1]
    app = tmp_path / "app"
    app.mkdir()
    for name in ("apps", "contracts", "simulation"):
        shutil.copytree(
            root / name,
            app / name,
            ignore=shutil.ignore_patterns("web", "__pycache__", "*.pyc", ".venv"),
        )
    for line in (root / "simulation" / recipe).read_text().splitlines():
        if not line.startswith("COPY learning"):
            continue
        parts = shlex.split(line)
        destination = app / parts[-1].removeprefix("/app/").strip("/")
        destination.mkdir(parents=True, exist_ok=True)
        for source in parts[1:-1]:
            path = root / source
            if path.is_dir():
                shutil.copytree(
                    path,
                    destination,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"),
                )
            else:
                shutil.copy2(path, destination / path.name)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import simulation.run_isaac; import simulation.probe_control; "
            "import learning.smolvla.ipc; "
            "assert not {'isaacsim', 'torch', 'transformers', 'lerobot'} & set(sys.modules)",
        ],
        cwd=app,
        env=os.environ | {"PYTHONPATH": str(app)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
