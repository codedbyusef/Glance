"""Loads runtime/config.yaml over built-in defaults, then runtime/config.local.yaml (gitignored) over that, so
each machine (laptop, Pi) keeps its own camera index and glove port without editing the shared file.
Nested dicts merge key by key; unknown keys are an error."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUNTIME_DIR / "config.yaml"
LOCAL_PATH = RUNTIME_DIR / "config.local.yaml"

DEFAULTS = {
    "camera": {"source": 1, "width": 1280, "height": 720, "dshow": True, "warmup_frames": 45, "jpeg_quality": 85},
    "view": {"web_port": 8090, "width": 960},
    "glove": {"kind": "fake", "port": None, "on_level": 255},
}


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key not in base:
            raise ValueError(f"unknown runtime config key {where}{key}")
        out[key] = _merge(base[key], value, f"{where}{key}.") if isinstance(base[key], dict) else value
    return out


def _read(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def load(path: str | Path | None = None, local: str | Path | None = None) -> dict:
    """path/local default to runtime/config.yaml and runtime/config.local.yaml."""
    cfg = _merge(DEFAULTS, _read(Path(path) if path else CONFIG_PATH))
    return _merge(cfg, _read(Path(local) if local else LOCAL_PATH))


def default_view() -> str:
    """window where there is a display; web on a headless Linux box (the Pi worn by the user)."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "web"
    return "window"
