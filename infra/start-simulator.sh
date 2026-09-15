#!/usr/bin/env bash
set -euo pipefail
umask 077

install -d -m 0700 /var/lib/physicalai
printf '%s' '__CONFIGURATION_BASE64__' | base64 --decode > /var/lib/physicalai/runtime.json
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl ca-certificates gnupg python3
systemctl enable --now docker

curl --fail --silent --show-error --location \
  https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --fail --silent --show-error --location \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
nvidia-smi

if ! command -v az >/dev/null 2>&1; then
  curl --fail --silent --show-error --location https://aka.ms/InstallAzureCLIDeb \
    -o /var/lib/physicalai/install-azure-cli.sh
  bash /var/lib/physicalai/install-azure-cli.sh
fi

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

config = json.loads(Path('/var/lib/physicalai/runtime.json').read_text())
for key, value in config.items():
    if '\n' in str(value) or '\r' in str(value):
        raise SystemExit('Invalid multiline runtime setting.')
Path('/var/lib/physicalai/runtime.env').write_text(
    ''.join(f'{key}={value}\n' for key, value in config.items())
)
subprocess.run([
    'az', 'login', '--identity', '--client-id', config['AZURE_CLIENT_ID'],
    '--allow-no-subscriptions', '--output', 'none'
], check=True)
subprocess.run(['az', 'acr', 'login', '--name', config['REGISTRY_NAME']], check=True)
subprocess.run(['docker', 'pull', config['SIMULATOR_IMAGE']], check=True)
existing = subprocess.run(
    ['docker', 'ps', '-aq', '--filter', 'name=^physicalai-simulator$'],
    check=True, capture_output=True, text=True,
).stdout.strip()
if existing:
    subprocess.run(['docker', 'rm', '-f', existing], check=True)
subprocess.run([
    'docker', 'run', '-d', '--name', 'physicalai-simulator', '--restart', 'unless-stopped',
    '--gpus', 'all', '--network', 'host', '--ipc', 'host',
    '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
    '--env-file', '/var/lib/physicalai/runtime.env',
    '--tmpfs', '/run/physicalai:rw,nosuid,nodev,mode=0700',
    '--mount', 'type=bind,source=/var/lib/physicalai,target=/data',
    config['SIMULATOR_IMAGE'],
], check=True)
PY
