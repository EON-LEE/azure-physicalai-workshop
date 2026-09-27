#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
: "${AZ_BATCH_POOL_ID:?Batch-managed bootstrap only}"
: "${AZ_BATCH_NODE_STARTUP_WORKING_DIR:?Missing Batch startup directory}"
: "${PHYSICALAI_BOOTSTRAP_SHA256:?Missing reviewed bootstrap digest}"
[[ "$(id -u)" == 0 ]] || { printf '%s\n' 'Bootstrap requires the Batch administrative start-task user.' >&2; exit 1; }
image="${1:?Explicit immutable simulator image required}"
[[ "$image" =~ ^[a-z0-9]+\.azurecr\.io/physicalai-simulator@sha256:[a-f0-9]{64}$ ]] || exit 2
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || { printf '%s\n' 'Only the reviewed Ubuntu 22.04 host is supported.' >&2; exit 2; }

driver_sha=1a529dd4d173ba3b36f3c284bb54fce5dd39c9e757856980b8ae0b7798e7764b
driver_url=https://download.microsoft.com/download/5e213ec5-834f-4b0a-87f7-772751353b06/NVIDIA-Linux-x86_64-570.237-grid-azure.run
driver="$AZ_BATCH_NODE_STARTUP_WORKING_DIR/grid-570.237.run"
for service in nvidia-persistenced nvidia-dcgm; do
  state="$(systemctl show --property=LoadState --value "$service")"
  if [[ "$state" == loaded ]]; then
    systemctl stop "$service"
  fi
done
for module in nvidia_peermem nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  if grep -q "^${module} " /proc/modules; then
    modprobe -r "$module"
  fi
done

package_report="$(dpkg-query -W -f='${binary:Package}\t${db:Status-Status}\n')"
mapfile -t packages < <(
  printf '%s\n' "$package_report" |
    awk '$2 == "installed" && ($1 ~ /^nvidia-/ || $1 ~ /^libnvidia-/ || $1 ~ /^cuda-drivers/) &&
      $1 !~ /^nvidia-container-/ && $1 !~ /^libnvidia-container/ && $1 !~ /^nvidia-docker2(:|$)/ {print $1}'
)
if ((${#packages[@]})); then
  printf 'Removing conflicting package-managed drivers: %s\n' "${packages[*]}"
  apt-get -o DPkg::Lock::Timeout=120 purge -y "${packages[@]}"
fi
apt-get -o DPkg::Lock::Timeout=120 update -qq
apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends \
  build-essential dkms "linux-headers-$(uname -r)" pkg-config curl ca-certificates \
  libglvnd-dev libvulkan1 vulkan-tools
curl --fail --location --max-time 180 --output "$driver" "$driver_url"
printf '%s  %s\n' "$driver_sha" "$driver" | sha256sum --check --status
sh "$driver" --check
sh "$driver" --silent --accept-license --dkms
modprobe nvidia
modprobe nvidia_uvm
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v nvidia-ctk
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

proof_dir="$(mktemp -d "$AZ_BATCH_NODE_STARTUP_WORKING_DIR/gpu-proof-XXXXXXXX")"
chown 1234:1234 "$proof_dir"
chmod 0700 "$proof_dir"
docker run --rm --gpus all --user 1234:1234 --cap-drop ALL --security-opt no-new-privileges \
  --env NVIDIA_DRIVER_CAPABILITIES=all --env PYTHONPATH=/app \
  --mount "type=bind,source=$proof_dir,target=/proof" \
  --entrypoint /isaac-sim/python.sh "$image" \
  -m simulation.batch_task preflight --output /proof/preflight.json
python3 - "$proof_dir/preflight.json" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
proof = json.loads(source.read_text())
assert proof["gpu_count"] == 1
assert proof["gpu_name"] == "NVIDIA A10-24Q"
assert proof["driver_version"] == "570.237"
assert proof["simulation_app_started"] is False
proof["bootstrap_sha256"] = os.environ["PHYSICALAI_BOOTSTRAP_SHA256"]
proof["driver_installer_sha256"] = "1a529dd4d173ba3b36f3c284bb54fce5dd39c9e757856980b8ae0b7798e7764b"
destination = Path(os.environ["AZ_BATCH_NODE_STARTUP_WORKING_DIR"]) / "preflight.json"
temporary = source.parent / "verified.json"
temporary.write_text(json.dumps(proof, sort_keys=True))
temporary.replace(destination)
print("PHYSICALAI_MANAGED_DRIVER_READY " + json.dumps(proof), flush=True)
PY
