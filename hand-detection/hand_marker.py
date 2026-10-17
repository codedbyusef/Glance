"""Hand point localization by a colored band around the knuckles.

The glove's motors and wiring break MediaPipe Hands, so the glove wears a
matte neon band wrapped all the way around the knuckles. From the head-worn
camera some of the band is visible whether the hand is palm down, sideways or
palm up. We find band-colored pixels, group the pieces that belong to the
hand, and return their center as the hand point.

The band often shows up as several pieces (fingers, wires and the edge of the
hand split it, and a sideways hand shows the back and palm strips separately),
so pieces close to the biggest one are merged instead of only taking the
largest blob.

Tunables live in hand-detection/config.yaml under `hand_marker`.
Use hand-detection/tune_marker.py to pick the HSV range under the demo lighting.

Coordinates are pixels of the decoded frame. The orientation of the hand is
never estimated: thumb = left in the camera's view, always (no hand-rotation
compensation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DEFAULTS = {
    # OpenCV HSV: H 0-179, S/V 0-255. Default = fluorescent green.
    # Several ranges allowed so a red band (hue wraps at 0/180) can be two ranges.
    "hsv_ranges": [[[40, 120, 80], [85, 255, 255]]],
    "min_area_px": 40,          # pieces smaller than this are noise
    "max_area_frac": 0.15,      # a piece bigger than this is a shirt/wall, not the band
    "merge_radius_frac": 0.08,  # pieces this close to the group (fraction of frame diagonal) join it
    "max_jump_frac": 0.35,      # prefer pieces near last frame's point when any exist
    "blur_ksize": 5,
    "morph_ksize": 5,
}

_cfg: Optional[dict] = None
_kernel = None
_last_point: Optional[tuple[float, float]] = None


@dataclass
class HandMarker:
    point: tuple[float, float]              # (x, y) center of the band
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 around all merged pieces
    area_px: float                          # total band area (grows as the hand nears the camera)
    pieces: int                             # how many blobs were merged


def _load_cfg() -> dict:
    global _cfg, _kernel
    if _cfg is None:
        try:
            with CONFIG_PATH.open() as f:
                user = (yaml.safe_load(f) or {}).get("hand_marker", {}) or {}
        except FileNotFoundError:
            user = {}
        _cfg = {**_DEFAULTS, **user}
        k = int(_cfg["morph_ksize"])
        _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return _cfg


def set_config(**overrides) -> None:
    """Override tunables at runtime (tests, tuning script)."""
    cfg = _load_cfg()
    cfg.update(overrides)


def marker_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of band-colored pixels."""
    cfg = _load_cfg()
    b = int(cfg["blur_ksize"])
    if b > 1:
        bgr = cv2.GaussianBlur(bgr, (b, b), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg["hsv_ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)   # remove speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)  # fill small gaps from glare
    return mask


def detect(bgr: np.ndarray) -> Optional[HandMarker]:
    """Full result for debugging/tests. Updates the last-point memory."""
    global _last_point
    cfg = _load_cfg()
    h, w = bgr.shape[:2]
    diag = float(np.hypot(w, h))
    contours, _ = cv2.findContours(marker_mask(bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = cfg["max_area_frac"] * w * h
    pieces = []  # (area, cx, cy, contour)
    for c in contours:
        area = cv2.contourArea(c)
        if cfg["min_area_px"] <= area <= max_area:
            m = cv2.moments(c)
            if m["m00"] > 0:
                pieces.append((area, m["m10"] / m["m00"], m["m01"] / m["m00"], c))
    if not pieces:
        return None

    # 1. Pick the main piece: biggest one, preferring pieces near last frame's
    #    point so a stray green object can't steal the hand for a frame.
    candidates = pieces
    if _last_point is not None:
        max_jump = cfg["max_jump_frac"] * diag
        near = [p for p in pieces
                if np.hypot(p[1] - _last_point[0], p[2] - _last_point[1]) <= max_jump]
        if near:
            candidates = near
    main = max(candidates, key=lambda p: p[0])

    # 2. Grow a group from the main piece: any piece within merge radius of a
    #    piece already in the group joins it (chained), so a wide band split
    #    into many pieces is kept whole.
    radius = cfg["merge_radius_frac"] * diag
    group, rest = [main], [p for p in pieces if p is not main]
    grew = True
    while grew and rest:
        grew = False
        for p in rest[:]:
            if any(np.hypot(p[1] - g[1], p[2] - g[2]) <= radius for g in group):
                group.append(p)
                rest.remove(p)
                grew = True

    total = sum(p[0] for p in group)
    cx = sum(p[0] * p[1] for p in group) / total   # area-weighted center
    cy = sum(p[0] * p[2] for p in group) / total
    x, y, bw, bh = cv2.boundingRect(np.vstack([p[3] for p in group]))

    _last_point = (cx, cy)
    return HandMarker(point=(cx, cy), box=(x, y, x + bw, y + bh), area_px=total, pieces=len(group))


def locate_in_image(bgr: np.ndarray) -> Optional[tuple[float, float]]:
    result = detect(bgr)
    return result.point if result else None


def locate_hand(frame: bytes) -> Optional[tuple[float, float]]:
    """frame = JPEG bytes from the WebSocket."""
    bgr = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return locate_in_image(bgr)


def reset() -> None:
    """Forget the last point. Call at the start of each reach session."""
    global _last_point
    _last_point = None
