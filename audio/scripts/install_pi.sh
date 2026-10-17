#!/usr/bin/env bash
# Fresh Raspberry Pi OS (64-bit) setup for audio/. Run from the repo root: bash audio/scripts/install_pi.sh
set -euo pipefail
AUDIO="$(cd "$(dirname "$0")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y libportaudio2 python3-venv python3-dev curl

# --system-site-packages keeps the preinstalled gpiozero and lgpio visible for GPIO push-to-talk
python3 -m venv --system-site-packages "$AUDIO/.venv"
. "$AUDIO/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$AUDIO/requirements-pi.txt"
# openwakeword's metadata requires tflite-runtime, which has no aarch64 wheel for 3.11/3.13.
# Install it without deps; its real runtime deps are in requirements-pi.txt, and we always use the onnx backend.
pip install --no-deps openwakeword==0.6.0

bash "$AUDIO/scripts/fetch_models.sh"
echo "done. Activate with: . audio/.venv/bin/activate"
