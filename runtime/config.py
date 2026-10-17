"""Loads runtime/config.yaml over built-in defaults (nested dicts merge key by key)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUNTIME_DIR / "config.yaml"

DEFAULTS = {
    "camera": {"source": 1, "width": 1280, "height": 720, "dshow": True, "warmup_frames": 45, "jpeg_quality": 85},
    "view": {"web_port": 8090, "width": 960},
}


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key not in base:
            raise ValueError(f"unknown runtime config key {where}{key}")
        out[key] = _merge(base[key], value, f"{where}{key}.") if isinstance(base[key], dict) else value
    return out


def load(path: str | Path | None = None) -> dict:
    path = Path(path) if path else CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return _merge(DEFAULTS, data or {})
