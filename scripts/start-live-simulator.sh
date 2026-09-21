#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
install -d -m 0755 /var/lib/physicalai
printf '%s' '__LIVE_CONFIG_BASE64__' | base64 --decode > /var/lib/physicalai/live-runtime.json
chmod 0600 /var/lib/physicalai/live-runtime.json
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
import json
import os
import subprocess
from pathlib import Path
config = json.loads(Path('/var/lib/physicalai/live-runtime.json').read_text())
for value in config.values():
    if '\n' in str(value) or '\r' in str(value):
        raise ValueError('Multiline environment values are not accepted.')
environment = Path('/var/lib/physicalai/live-runtime.env')
environment.write_text(''.join(f'{key}={value}\n' for key, value in config.items()))
environment.chmod(0o600)
cache_suffix = '6' if config.get('ISAAC_SIM_VERSION', '').startswith('6.') else '5'
cache = Path('/var/lib/physicalai') / ('cache-' + cache_suffix)
compute_cache = Path('/var/lib/physicalai') / ('computecache-' + cache_suffix)
for folder in (cache, compute_cache):
    folder.mkdir(exist_ok=True, mode=0o700)
    os.chown(folder, 1234, 1234)
subprocess.run(['az', 'login', '--identity', '--client-id', config['AZURE_CLIENT_ID'],
                '--allow-no-subscriptions', '--output', 'none'], check=True)
subprocess.run(['az', 'acr', 'login', '--name', config['REGISTRY_NAME']], check=True)
subprocess.run(['docker', 'pull', config['SIMULATOR_IMAGE']], check=True)
old = subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^physicalai-simulator$'],
                     capture_output=True, text=True, check=True).stdout.strip()
if old:
    subprocess.run(['docker', 'rm', '-f', old], check=True)
subprocess.run([
    'docker', 'run', '-d', '--name', 'physicalai-simulator', '--restart', 'unless-stopped',
    '--gpus', 'all', '--network', 'host', '--ipc', 'host', '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges', '--env-file', str(environment),
    '--tmpfs', '/run/physicalai:rw,nosuid,nodev,mode=0700,uid=1234,gid=1234',
    '--mount', 'type=bind,source=/var/lib/physicalai,target=/data',
    '--mount', f'type=bind,source={cache},target=/isaac-sim/.cache',
    '--mount', f'type=bind,source={compute_cache},target=/isaac-sim/.nv/ComputeCache',
    '--mount', 'type=bind,source=/var/lib/physicalai/logs,target=/isaac-sim/.nvidia-omniverse/logs',
    config['SIMULATOR_IMAGE'],
], check=True)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker ps --filter name=^physicalai-simulator$
