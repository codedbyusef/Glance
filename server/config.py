"""Tiny YAML config loader shared by the server-side modules.

Resolves paths relative to the repo root regardless of the caller's own
depth under server/, so config/*.yaml stays the single source of truth for
thresholds (per CLAUDE.md "Config over constants").
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "config"


def load(name: str) -> dict:
    with (_CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)
