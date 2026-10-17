"""Where to read depth: masks, boxes and medians. Pure numpy/OpenCV, no model.

Boxes are pixel (x1, y1, x2, y2), the same format hand-detection's HandMarker.box
and the object tracker use. Masks are bool arrays the size of the frame.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def box_to_mask(box: Box, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = _clip_box(box, w, h)
    mask = np.zeros((h, w), bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def shrink_box(box: Box, frac: float) -> Box:
    """Shrink a box by `frac` of its size (0.2 -> 20% smaller), keeping its center."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac / 2, (y2 - y1) * frac / 2
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def scale_box(box: Box, factor: float) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def mask_box(mask: np.ndarray) -> Optional[Box]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_diag(box: Box) -> float:
    x1, y1, x2, y2 = box
    return float(np.hypot(x2 - x1, y2 - y1))


def point_in_box(point: Optional[tuple[float, float]], box: Optional[Box]) -> bool:
    if point is None or box is None:
        return False
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def median_in(dmap: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[Optional[float], int]:
    """Median of the depth map under the mask. Returns (value or None, pixel count)."""
    values = dmap[mask]
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < min_pixels:
        return None, n
    return float(np.median(values)), n


def overlap_fraction(cover: np.ndarray, region: np.ndarray) -> float:
    """Fraction of `region` covered by `cover`."""
    area = int(region.sum())
    if area == 0:
        return 0.0
    return float((cover & region).sum()) / area


def hand_region(band_mask: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    """The band only covers the knuckles, but the whole hand hides the target.
    Approximate the hand as the band's box scaled up by `scale`."""
    if band_mask is None:
        return None
    box = mask_box(band_mask)
    if box is None:
        return None
    return box_to_mask(scale_box(box, scale), band_mask.shape)


def band_mask_from_marker(full_mask: np.ndarray, band_box: Box, pad_px: int = 4) -> np.ndarray:
    """hand_marker.marker_mask() returns every band-colored pixel in the frame.
    Keep only the pixels inside the tracked band's box (its merged pieces), so a
    stray green object elsewhere doesn't get sampled as the hand."""
    keep = box_to_mask((band_box[0] - pad_px, band_box[1] - pad_px,
                        band_box[2] + pad_px, band_box[3] + pad_px), full_mask.shape)
    return full_mask.astype(bool) & keep


def band_width_px(band_mask: np.ndarray) -> Optional[float]:
    """Visible tape width in pixels: the short side of the min-area rectangle
    around the largest band piece. Pose-sensitive (sideways, tilt, blur) --
    use as a coarse check only (see README)."""
    contours, _ = cv2.findContours(band_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(largest)
    width = float(min(w, h))
    return width if width > 0 else None


def band_distance_m(width_px: Optional[float], focal_px: Optional[float], width_m: float) -> Optional[float]:
    """Pinhole: Z = f * real_width / pixel_width."""
    if not width_px or not focal_px:
        return None
    return focal_px * width_m / width_px


def focal_from_band(width_px: float, distance_m: float, width_m: float) -> float:
    """Calibration: hold the band at a measured distance, solve the pinhole formula for f."""
    return width_px * distance_m / width_m


def _clip_box(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (int(max(0, np.floor(x1))), int(max(0, np.floor(y1))),
            int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2))))
