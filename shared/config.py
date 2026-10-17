"""Tiny YAML config loader shared by every component.

Resolves config/ from the repo root (this file sits one level under it), so
config/*.yaml stays the single source of truth for thresholds (per CLAUDE.md
"Config over constants").
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "config"


def load(name: str) -> dict:
    with (_CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)
