"""Box helpers. Every box in this folder is pixel xyxy: (x1, y1, x2, y2) floats.

All format conversion goes through here so it is tested in one place
(object-detection/CLAUDE.md "Code and interfaces").
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Box = tuple[float, float, float, float]


def cxcywh_norm_to_xyxy(box: Sequence[float], width: int, height: int) -> Box:
    """Normalized (cx, cy, w, h) in 0-1, as raw Grounding DINO outputs, to pixel xyxy."""
    cx, cy, w, h = box
    return clip(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height), width, height)


def clip(box: Sequence[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        float(min(max(x1, 0), width)),
        float(min(max(y1, 0), height)),
        float(min(max(x2, 0), width)),
        float(min(max(y2, 0), height)),
    )


def area(box: Optional[Sequence[float]]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(cover: Optional[Sequence[float]], target: Optional[Sequence[float]]) -> float:
    """Fraction of `target` covered by `cover` (0-1). Used for the inferred-success rule."""
    if cover is None or target is None or area(target) == 0:
        return 0.0
    return intersection(cover, target) / area(target)


def center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def contains(box: Optional[Sequence[float]], point: Optional[Sequence[float]]) -> bool:
    if box is None or point is None:
        return False
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def distance_to_box(box: Sequence[float], point: Sequence[float]) -> float:
    """Pixels from `point` to the nearest edge of `box`; 0 when inside."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return float((dx * dx + dy * dy) ** 0.5)


def mask_to_box(mask: np.ndarray) -> Optional[Box]:
    """Tight pixel xyxy around the True pixels of an HxW mask, or None if empty.
    x2/y2 are exclusive (one past the last pixel) so a 1-pixel mask has area 1."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def region_thirds(box: Sequence[float], width: int, height: int) -> str:
    """'left|center|right-top|middle|bottom' by the box center, for `last_seen.region`."""
    cx, cy = center(box)
    horiz = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
    vert = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
    return f"{horiz}-{vert}"
