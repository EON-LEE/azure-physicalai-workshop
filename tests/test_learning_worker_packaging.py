import shlex
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "apps" / "learning_worker"


def test_worker_pins_batch_sdk_in_manifest_and_frozen_lock():
    project = tomllib.loads((WORKER / "pyproject.toml").read_text())["project"]
    assert "azure-batch==15.1.0" in project["dependencies"]
    packages = {
        package["name"]: package
        for package in tomllib.loads((WORKER / "uv.lock").read_text())["package"]
    }
    assert packages["azure-batch"]["version"] == "15.1.0"
    worker = packages[project["name"]]
    assert {"name": "azure-batch"} in worker["dependencies"]
    assert {"name": "azure-batch", "specifier": "==15.1.0"} in worker["metadata"]["requires-dist"]


def test_worker_recipe_copies_batch_controller_source_without_replacing_service_entrypoint():
    recipe = (WORKER / "Dockerfile").read_text()
    copies = [shlex.split(line) for line in recipe.splitlines() if line.startswith("COPY ")]
    for source in ("apps/api", "apps/learning_worker", "learning", "contracts", "simulation"):
        assert ["COPY", source, f"/srv/{source}"] in copies
    assert "uv sync --project /srv/apps/learning_worker --locked --no-dev" in recipe
    assert "USER 10001:10001" in recipe
    assert (
        'CMD ["uvicorn", "apps.learning_worker.main:create_worker", "--factory", '
        '"--host", "0.0.0.0", "--port", "8080"]'
    ) in recipe


def test_worker_dependency_lock_does_not_install_gpu_or_model_runtimes():
    packages = tomllib.loads((WORKER / "uv.lock").read_text())["package"]
    forbidden = ("torch", "nvidia-", "cuda-", "isaac", "lerobot", "transformers", "onnxruntime")
    assert not [package["name"] for package in packages if package["name"].startswith(forbidden)]
