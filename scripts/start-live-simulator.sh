#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
install -d -m 0755 /var/lib/physicalai
install -d -m 0755 /var/lib/physicalai/setup
install -d -m 0700 -o 1234 -g 1234 \
  /var/lib/physicalai/assets \
  /var/lib/physicalai/demonstrations \
  /var/lib/physicalai/cache \
  /var/lib/physicalai/computecache \
  /var/lib/physicalai/logs

if ! command -v az >/dev/null 2>&1; then
  curl --fail --silent --show-error --location \
    https://aka.ms/InstallAzureCLIDeb -o /var/lib/physicalai/setup/install-azure-cli.sh
  bash /var/lib/physicalai/setup/install-azure-cli.sh
fi

python3 - <<'PY'
import base64
import json
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

NAME = 'physicalai-simulator'
PREVIOUS = NAME + '-previous'
GPU_ARGUMENTS = ['--runtime', 'runc', '--device', 'nvidia.com/gpu=all']


def run(arguments, **kwargs):
    return subprocess.run(arguments, text=True, check=True, **kwargs)


def container_id(name):
    return run(['docker', 'ps', '-aq', '--filter', f'name=^{name}$'],
               capture_output=True).stdout.strip()


def wait_ready(identity, ca_file, port):
    context = ssl.create_default_context(cafile=str(ca_file))
    deadline = time.monotonic() + 240
    last_error = 'The TLS bridge has not answered.'
    while time.monotonic() < deadline:
        state = json.loads(run(['docker', 'inspect', '--format', '{{json .State}}', identity],
                               capture_output=True).stdout)
        if not state['Running'] and not state.get('Restarting'):
            raise RuntimeError(f"Candidate container stopped with exit {state.get('ExitCode')}.")
        try:
            with urllib.request.urlopen(
                f'https://sim.physicalai.internal:{port}/healthz', context=context, timeout=5
            ) as response:
                payload = json.load(response)
            if payload != {'status': 'process_running', 'runtime': 'isaac_sim'}:
                raise ValueError('The simulator returned an unexpected health response.')
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        time.sleep(2)
    raise RuntimeError(f'Simulator TLS readiness deadline expired: {last_error}')


def deploy(config, root=Path('/var/lib/physicalai')):
    for key, value in config.items():
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or '\n' in str(value) or '\r' in str(value):
            raise ValueError('Only single-line, named environment values are accepted.')
    prefix = config['REGISTRY_NAME'] + '.azurecr.io/physicalai-simulator@sha256:'
    if not config['SIMULATOR_IMAGE'].startswith(prefix) or not re.fullmatch(
        r'[a-f0-9]{64}', config['SIMULATOR_IMAGE'][len(prefix):]
    ):
        raise ValueError('A pinned simulator image in the approved registry is required.')
    if container_id(PREVIOUS):
        raise RuntimeError('A previous deployment container needs explicit reconciliation.')
    old = container_id(NAME)
    old_running = bool(old) and json.loads(
        run(['docker', 'inspect', '--format', '{{.State.Running}}', old],
            capture_output=True).stdout
    )
    deployment = uuid4().hex
    environment = root / f'.live-runtime-{deployment}.env'
    config_path = root / 'live-runtime.json'
    active_environment = root / 'live-runtime.env'
    previous_files = {
        path: path.read_bytes() if path.exists() else None
        for path in (config_path, active_environment)
    }
    ca_file = root / 'simulator-ca.pem'
    stopped = renamed = False
    candidate = None
    try:
        environment.write_text(''.join(f'{key}={value}\n' for key, value in config.items()))
        environment.chmod(0o600)
        suffix = '6' if config.get('ISAAC_SIM_VERSION', '').startswith('6.') else '5'
        cache, compute_cache = root / ('cache-' + suffix), root / ('computecache-' + suffix)
        for folder in (cache, compute_cache):
            folder.mkdir(exist_ok=True, mode=0o700)
            os.chown(folder, 1234, 1234)
        run(['az', 'login', '--identity', '--client-id', config['AZURE_CLIENT_ID'],
             '--allow-no-subscriptions', '--output', 'none'])
        run(['az', 'acr', 'login', '--name', config['REGISTRY_NAME']])
        run(['docker', 'pull', config['SIMULATOR_IMAGE']])
        run(['docker', 'run', '--rm', '--pull', 'never', *GPU_ARGUMENTS,
             '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
             '--entrypoint', 'nvidia-smi', config['SIMULATOR_IMAGE'], '-L'])
        account = config['STORAGE_ACCOUNT_URL'].split('://', 1)[1].split('.', 1)[0]
        run(['az', 'storage', 'blob', 'download', '--auth-mode', 'login',
             '--account-name', account, '--container-name', 'artifacts',
             '--name', 'bootstrap/simulator-ca.pem', '--file', str(ca_file),
             '--overwrite', 'true', '--only-show-errors', '--output', 'none'])
        if old:
            run(['docker', 'stop', '--time', '30', old])
            stopped = True
            run(['docker', 'rename', old, PREVIOUS])
            renamed = True
        candidate = run([
            'docker', 'run', '-d', '--name', NAME, '--restart', 'unless-stopped',
            '--label', 'physicalai.deployment=' + deployment,
            *GPU_ARGUMENTS, '--network', 'host', '--ipc', 'host', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--env-file', str(environment),
            '--tmpfs', '/run/physicalai:rw,nosuid,nodev,mode=0700,uid=1234,gid=1234',
            '--mount', f'type=bind,source={root},target=/data',
            '--mount', f'type=bind,source={cache},target=/isaac-sim/.cache',
            '--mount', f'type=bind,source={compute_cache},target=/isaac-sim/.nv/ComputeCache',
            '--mount', f'type=bind,source={root / "logs"},target=/isaac-sim/.nvidia-omniverse/logs',
            config['SIMULATOR_IMAGE'],
        ], capture_output=True).stdout.strip()
        wait_ready(candidate, ca_file, int(config.get('BRIDGE_PORT', '8443')))
        temporary_config = root / f'.live-runtime-{deployment}.json'
        temporary_config.write_text(json.dumps(config, indent=2))
        temporary_config.chmod(0o600)
        temporary_config.replace(config_path)
        environment.replace(active_environment)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError):
        partial = candidate or container_id(NAME)
        if partial and partial != old:
            label = run(
                ['docker', 'inspect', '--format',
                 '{{index .Config.Labels "physicalai.deployment"}}', partial],
                capture_output=True,
            ).stdout.strip()
            if label == deployment:
                run(['docker', 'rm', '-f', partial])
        if renamed:
            run(['docker', 'rename', old, NAME])
        for path, content in previous_files.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
                path.chmod(0o600)
        if stopped and old_running:
            run(['docker', 'start', old])
            wait_ready(old, ca_file, int(config.get('BRIDGE_PORT', '8443')))
            print('PHYSICALAI_ROLLBACK=previous_container_restored', flush=True)
        raise
    finally:
        environment.unlink(missing_ok=True)
        (root / f'.live-runtime-{deployment}.json').unlink(missing_ok=True)
    if old:
        run(['docker', 'rm', old])
    print('PHYSICALAI_SIMULATOR_READY ' + json.dumps({
        'image': config['SIMULATOR_IMAGE'], 'scope': 'verified_tls_bridge_health',
        'physical_acceptance': 'not_assessed',
    }), flush=True)


if __name__ == '__main__':
    deploy(json.loads(base64.b64decode('__LIVE_CONFIG_BASE64__')))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker ps --filter name=^physicalai-simulator$
