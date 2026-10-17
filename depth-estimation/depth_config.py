"""Loads depth-estimation/config.yaml over built-in defaults.

Same pattern as hand-detection/hand_marker.py: a missing file or missing key
falls back to the defaults below, so the folder works with no config at all.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "frame": {"frame_width": 640, "frame_height": 480, "jpeg_quality": 70},
    "camera": {"source": 0},
    "model": {
        "kind": "relative",
        "ids": {
            "relative": "depth-anything/Depth-Anything-V2-Small-hf",
            "metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        },
        "input_size": 364,
        "device": "auto",
        "fp16": True,
        "every_n_frames": 1,
    },
    "normalize": {"low_pct": 5, "high_pct": 95},
    "sampling": {"erode_px": 3, "box_shrink": 0.2, "min_pixels": 30, "hand_region_scale": 2.0},
    "smoothing": {"median_window": 5, "ema_alpha": 0.4, "max_rejects": 3},
    "occlusion": {"freeze_overlap": 0.30, "unfreeze_overlap": 0.15, "min_visible": 0.5, "moved_frac": 0.25},
    "thresholds": {
        "metric": {"jump": 0.15, "contact": 0.04, "near": 0.15, "far": 0.40, "area_cross_check": 0.05},
        "relative": {"jump": 0.25, "contact": 0.05, "near": 0.15, "far": 0.35, "area_cross_check": 0.08},
    },
    "touching": {"hold_frames": 7},
    "contact_offset": {
        "active_object": "paper",
        "metric": {"paper": 0.0, "can": 0.0},
        "relative": {"paper": 0.0, "can": 0.0},
    },
    "band": {"area_ratio": 1.25, "window": 5, "width_m": 0.025, "focal_px": None},
    "controller": {"relative_scale": 1.0},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: Optional[Path] = None, **overrides: Any) -> dict:
    """Defaults <- config.yaml <- overrides (nested dicts merge key by key)."""
    path = path or CONFIG_PATH
    try:
        with Path(path).open() as f:
            user = yaml.safe_load(f) or {}
    except FileNotFoundError:
        user = {}
    return _merge(_merge(DEFAULTS, user), overrides)


def units(cfg: dict) -> str:
    """'metric' or 'relative' -- selects the threshold and offset set."""
    kind = cfg["model"]["kind"]
    if kind not in ("metric", "relative"):
        raise ValueError(f"model.kind must be 'metric' or 'relative', got {kind!r}")
    return kind
