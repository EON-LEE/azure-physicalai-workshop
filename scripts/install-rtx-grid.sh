#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
case "${GRID_GPU_FAMILY:-rtxpro6000}" in
  rtxpro6000)
    driver_version=595.91.07
    driver_url=https://download.microsoft.com/download/a7cb6d36-3bbc-43d6-9e88-e0842e6f9ab9/NVIDIA-Linux-x86_64-595.91.07-grid-azure.run
    kernel_arguments=(-M open)
    ;;
  a10)
    driver_version=570.237
    driver_url=https://download.microsoft.com/download/5e213ec5-834f-4b0a-87f7-772751353b06/NVIDIA-Linux-x86_64-570.237-grid-azure.run
    kernel_arguments=()
    ;;
  *)
    printf 'Unsupported GRID_GPU_FAMILY.\n' >&2
    exit 2
    ;;
esac
install -d -m 0755 /var/lib/physicalai/setup
apt-get update -qq
apt-get install -y --no-install-recommends \
  build-essential dkms "linux-headers-$(uname -r)" pkg-config \
  curl ca-certificates gnupg pciutils libglvnd-dev libvulkan1 vulkan-tools docker.io
driver="/var/lib/physicalai/setup/NVIDIA-Linux-x86_64-${driver_version}-grid-azure.run"
curl --fail --location --retry 3 --output "$driver" \
  "$driver_url"
sha256sum "$driver" > /var/lib/physicalai/setup/grid-driver.sha256
sh "$driver" --check
sh "$driver" --silent --accept-license --dkms "${kernel_arguments[@]}"
modprobe nvidia
modprobe nvidia_uvm
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

curl --fail --silent --show-error --location \
  https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --fail --silent --show-error --location \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
install -d -m 0755 /run/physicalai
XDG_RUNTIME_DIR=/run/physicalai vulkaninfo --summary
