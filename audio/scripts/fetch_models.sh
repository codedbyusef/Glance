#!/usr/bin/env bash
# Fetch model files the Pi needs. Run from the repo root with the audio venv active:
#   bash audio/scripts/fetch_models.sh
set -euo pipefail
python "$(dirname "$0")/fetch_models.py" "$@"
