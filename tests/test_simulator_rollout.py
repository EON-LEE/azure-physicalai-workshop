import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


class DockerDouble:
    def __init__(self):
        self.calls = []
        self.names = {"physicalai-simulator": "old"}
        self.running = {"old": True}
        self.labels = {}
        self.fail_pull = False

    def run(self, args, **kwargs):
        self.calls.append(args)
        output = ""
        if args[:2] == ["docker", "ps"]:
            name = args[-1].removeprefix("name=^").removesuffix("$")
            output = self.names.get(name, "")
        elif args[:2] == ["docker", "inspect"]:
            output = (
                json.dumps(self.running[args[-1]])
                if args[3] == "{{.State.Running}}"
                else self.labels.get(args[-1], "")
            )
        elif args[:2] == ["docker", "pull"] and self.fail_pull:
            raise subprocess.CalledProcessError(1, args)
        elif args[:2] == ["docker", "stop"]:
            self.running[args[-1]] = False
        elif args[:2] == ["docker", "rename"]:
            previous = next(name for name, identity in self.names.items() if identity == args[2])
            del self.names[previous]
            self.names[args[3]] = args[2]
        elif args[:2] == ["docker", "run"]:
            self.names["physicalai-simulator"] = "candidate"
            self.running["candidate"] = True
            self.labels["candidate"] = args[args.index("--label") + 1].split("=", 1)[1]
            output = "candidate"
        elif args[:2] == ["docker", "rm"]:
            name = next(name for name, identity in self.names.items() if identity == args[-1])
            del self.names[name]
            del self.running[args[-1]]
        elif args[:2] == ["docker", "start"]:
            self.running[args[-1]] = True
        return SimpleNamespace(stdout=output)


@pytest.fixture
def host_script(monkeypatch, tmp_path):
    script = (ROOT / "scripts" / "start-live-simulator.sh").read_text(encoding="utf-8")
    python = script.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    namespace = {"__name__": "test_host_script"}
    exec(compile(python, "start-live-simulator.sh", "exec"), namespace)
    docker = DockerDouble()
    namespace["run"] = docker.run
    namespace["wait_ready"] = lambda identity, ca_file, port: None
    monkeypatch.setattr(namespace["os"], "chown", lambda *args: None)
    (tmp_path / "live-runtime.json").write_text('{"previous":true}')
    (tmp_path / "live-runtime.env").write_text("PREVIOUS=true\n")
    config = {
        "REGISTRY_NAME": "testregistry",
        "SIMULATOR_IMAGE": "testregistry.azurecr.io/physicalai-simulator@sha256:" + "a" * 64,
        "AZURE_CLIENT_ID": "test-only-identity",
        "STORAGE_ACCOUNT_URL": "https://test.blob.core.windows.net",
        "ISAAC_SIM_VERSION": "6.0.0",
    }
    return namespace, docker, config, tmp_path


def test_pull_failure_never_stops_the_previous_simulator(host_script):
    namespace, docker, config, root = host_script
    docker.fail_pull = True
    with pytest.raises(subprocess.CalledProcessError):
        namespace["deploy"](config, root)
    assert docker.names == {"physicalai-simulator": "old"}
    assert docker.running["old"]
    assert not any(args[:2] == ["docker", "stop"] for args in docker.calls)
    assert (root / "live-runtime.json").read_text() == '{"previous":true}'
    assert (root / "live-runtime.env").read_text() == "PREVIOUS=true\n"
    assert not list(root.glob(".live-runtime-*"))


def test_failed_candidate_restores_previous_container_and_reports_failure(host_script, capsys):
    namespace, docker, config, root = host_script
    checked = []

    def readiness(identity, ca_file, port):
        checked.append(identity)
        if identity == "candidate":
            raise RuntimeError("Test-only candidate failure")

    namespace["wait_ready"] = readiness
    with pytest.raises(RuntimeError, match="candidate failure"):
        namespace["deploy"](config, root)
    assert docker.names == {"physicalai-simulator": "old"}
    assert docker.running["old"]
    assert checked == ["candidate", "old"]
    assert (root / "live-runtime.json").read_text() == '{"previous":true}'
    assert "PHYSICALAI_ROLLBACK=previous_container_restored" in capsys.readouterr().out


def test_success_removes_old_container_only_after_candidate_readiness(host_script, capsys):
    namespace, docker, config, root = host_script
    namespace["wait_ready"] = lambda identity, ca_file, port: docker.calls.append(
        ["verified_tls_health", identity]
    )
    namespace["deploy"](config, root)
    assert docker.names == {"physicalai-simulator": "candidate"}
    ready = docker.calls.index(["verified_tls_health", "candidate"])
    retired = docker.calls.index(["docker", "rm", "old"])
    assert ready < retired
    assert json.loads((root / "live-runtime.json").read_text()) == config
    assert not list(root.glob(".live-runtime-*"))
    assert '"physical_acceptance": "not_assessed"' in capsys.readouterr().out


def test_rollback_does_not_start_a_previously_stopped_simulator(host_script):
    namespace, docker, config, root = host_script
    docker.running["old"] = False

    def readiness(*args):
        raise RuntimeError("Test-only candidate failure")

    namespace["wait_ready"] = readiness
    with pytest.raises(RuntimeError):
        namespace["deploy"](config, root)
    assert docker.names == {"physicalai-simulator": "old"}
    assert not docker.running["old"]
    assert not any(args[:2] == ["docker", "start"] for args in docker.calls)


def test_unpinned_image_fails_before_any_remote_operation(host_script):
    namespace, docker, config, root = host_script
    config["SIMULATOR_IMAGE"] = "testregistry.azurecr.io/physicalai-simulator:latest"
    with pytest.raises(ValueError, match="pinned"):
        namespace["deploy"](config, root)
    assert docker.calls == []
