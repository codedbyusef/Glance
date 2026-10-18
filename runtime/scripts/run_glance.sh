#!/usr/bin/env bash
# Start Glance on the Pi: selftest, then the voice agent with real reaches and the glove (runtime/run.py).
#   bash runtime/scripts/run_glance.sh              # wake word (USB headset)
#   bash runtime/scripts/run_glance.sh --ptt-only   # Enter to talk
# Server URLs, tokens and GEMINI_API_KEY come from runtime/glance.env (copy runtime/glance.env.example).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
. "$REPO/.venv/bin/activate"
exec python -m runtime.run "$@"
