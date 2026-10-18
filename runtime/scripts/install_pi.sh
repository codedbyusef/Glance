#!/usr/bin/env bash
# Raspberry Pi 5 (64-bit Raspberry Pi OS, Bookworm or Trixie) setup for the whole client side (runtime/:
# agent, audio, camera, guidance, glove). Extends audio/scripts/install_pi.sh. Run from the repo root:
#   bash runtime/scripts/install_pi.sh
# Then: . .venv/bin/activate && python -m runtime.selftest
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

sudo apt-get update
# libportaudio2: sounddevice. libgl1/libglib2.0-0: opencv-python. python3-picamera2: the Pi camera module
# (only used with camera.source: picam; a USB webcam needs nothing extra).
sudo apt-get install -y libportaudio2 python3-venv python3-dev curl libgl1 libglib2.0-0 python3-picamera2

# --system-site-packages keeps the preinstalled gpiozero/lgpio (GPIO push-to-talk) and picamera2 visible.
python3 -m venv --system-site-packages "$REPO/.venv"
. "$REPO/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$REPO/runtime/requirements.txt"   # client role, no torch
# openwakeword's metadata requires tflite-runtime, which has no aarch64 wheel for 3.11/3.13. Install it without
# deps; its real runtime deps are in runtime/requirements.txt, and the code always uses the onnx backend.
pip install --no-deps openwakeword==0.6.0
pip install --no-deps -e "$REPO"

bash "$REPO/audio/scripts/fetch_models.sh"        # Piper voice, wake models, tiny.en fallback

# The glove's USB serial port (Step 4) needs the dialout group; takes effect at the next login.
sudo usermod -aG dialout "$USER" || true

if [[ ! -f "$REPO/runtime/glance.env" ]]; then
  cp "$REPO/runtime/glance.env.example" "$REPO/runtime/glance.env"   # fill in the tokens and GEMINI_API_KEY
fi

if [[ ! -f "$REPO/runtime/config.local.yaml" ]]; then
  cat > "$REPO/runtime/config.local.yaml" <<'YAML'
# This Pi's runtime settings (gitignored). USB webcam index 0; use "picam" for the camera module.
camera:
  source: 0
glove:
  kind: hardware       # the Feather ESP32-S3 on USB, found automatically (port: auto)
YAML
fi

echo
echo "done. Next:"
echo "  edit runtime/glance.env (tokens, GEMINI_API_KEY), then log out and back in (dialout group)"
echo "  . .venv/bin/activate && python -m runtime.glove ports && python -m runtime.glove test"
echo "  bash runtime/scripts/run_glance.sh --ptt-only  # selftest + agent + real reaches + glove;"
echo "                                                 # the live view is at http://<pi>:8090/"
echo
echo "If picamera2 fails to import with a numpy error, pip's numpy is shadowing apt's: use a USB webcam"
echo "(camera.source: 0) or 'pip install \"numpy<2\"' in the venv."
