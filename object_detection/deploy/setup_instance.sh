#!/usr/bin/env bash
# One-time setup of a fresh Vultr GPU instance: Docker + NVIDIA container toolkit.
# Skip it if the instance was created from an image that already has both
# (`nvidia-smi` and `docker run --gpus all ...` work). Ubuntu 22.04/24.04 assumed.
#   object_detection/deploy/setup_instance.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"

ssh "${SSH_OPTS[@]}" "$SSH_USER@$VULTR_HOST" bash -s <<'REMOTE'
set -euo pipefail
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: pick a Vultr GPU image with NVIDIA drivers, or install them first." >&2
  exit 1
fi
nvidia-smi

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

apt-get install -y rsync curl >/dev/null
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
echo "Instance ready."
REMOTE
