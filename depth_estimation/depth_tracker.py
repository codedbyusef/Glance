"""Hand and target depth from the same map, per frame, for one reach.

    tracker = DepthTracker()                        # one per reach; reset() between reaches
    reading = tracker.update(dmap, frame_id,
                             band_mask=...,          # hand_detection band pixels (bool, frame size)
                             target_box=... or target_mask=...,
                             hand_point=...,         # HandMarker.point, for the 2D in-box check
                             band_area_px=...)       # HandMarker.area_px, for the cross-check

Conventions (same as depth_model.py): larger = farther. So
    dz  = target - hand          positive = target farther than the hand
    gap = dz - contact_offset    what is left to close; ~0 at real contact

Why the offset: the band is on the knuckles. At real contact the knuckles are
still a few cm above flat paper, or several cm back from an upright can, so
dz at contact is not 0. Measure it per object (README, step 5).

Table mode (camera looks down, haptics default): gap is the remaining height
above the target once the hand is over it -- it decides when "down" is done.
Wall mode: gap is the forward/back error; `depth_error` is it in the units of
haptics/controller.py's depth_error (positive = target farther than hand).

Hand and target are only ever compared within one frame's map. Head motion
shifts both together, so dz stays valid.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import numpy as np

from depth_estimation import depth_regions as regions
from depth_estimation import depth_config as settings

DepthBand = Literal["near", "mid", "far"]
Approach = Literal["touching", "near", "far"]


@dataclass
class DepthReading:
    frame_id: int
    units: str                                # "metric" (meters) or "relative" (fraction of frame range)
    hand: Optional[float] = None              # smoothed hand distance (larger = farther)
    target: Optional[float] = None            # smoothed target distance (frozen while the hand covers it)
    dz: Optional[float] = None                # target - hand
    gap: Optional[float] = None               # dz - contact offset
    depth_error: Optional[float] = None       # gap in haptics controller units (wall mode)
    hand_valid: bool = False                  # a fresh hand sample was accepted this frame
    target_valid: bool = False                # a target value exists (fresh or frozen)
    target_frozen: bool = False
    overlap: float = 0.0                      # fraction of the target covered by the hand region
    hand_in_box: bool = False                 # 2D: hand point inside the target box
    touch_streak: int = 0
    touching: bool = False                    # |gap| <= contact, in box, for hold_frames in a row
    band: Optional[DepthBand] = None          # near | mid | far from |gap| (agent's last_seen.depth)
    inconsistent: bool = False                # band area and hand depth moved in opposite directions
    band_width_px: Optional[float] = None
    band_distance_m: Optional[float] = None   # pinhole estimate (only if focal_px is calibrated)
    raw_hand: Optional[float] = None          # unsmoothed medians, for calibration and logs
    raw_target: Optional[float] = None
    map_age: int = 0                          # frames since the map was computed
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Smoother:
    """Median of the last N samples, then an EMA. A sample that jumps more than
    `jump` from the current value is dropped; after `max_rejects` drops in a row
    the new level is accepted (the value really moved)."""

    def __init__(self, window: int, alpha: float, jump: float, max_rejects: int):
        self.window, self.alpha, self.jump, self.max_rejects = window, alpha, jump, max_rejects
        self.reset()

    def reset(self) -> None:
        self.history: deque[float] = deque(maxlen=self.window)
        self.value: Optional[float] = None
        self.rejects = 0

    def update(self, sample: Optional[float]) -> tuple[Optional[float], bool]:
        if sample is None:
            return self.value, False
        if self.value is not None and abs(sample - self.value) > self.jump:
            self.rejects += 1
            if self.rejects <= self.max_rejects:
                return self.value, False
            self.history.clear()                    # persistent jump: re-seed at the new level
            self.value = None
        self.rejects = 0
        self.history.append(sample)
        med = float(np.median(self.history))
        self.value = med if self.value is None else self.alpha * med + (1 - self.alpha) * self.value
        return self.value, True


class DepthTracker:
    def __init__(self, cfg: Optional[dict] = None, object_name: Optional[str] = None):
        self.cfg = cfg or settings.load()
        self.units = settings.units(self.cfg)
        self.th = self.cfg["thresholds"][self.units]
        offsets = self.cfg["contact_offset"]
        self.object_name = object_name or offsets["active_object"]
        self.offset = float(offsets[self.units].get(self.object_name, 0.0))
        sm = self.cfg["smoothing"]
        self.hand_s = Smoother(sm["median_window"], sm["ema_alpha"], self.th["jump"], sm["max_rejects"])
        self.target_s = Smoother(sm["median_window"], sm["ema_alpha"], self.th["jump"], sm["max_rejects"])
        self.reset()

    def reset(self) -> None:
        self.hand_s.reset()
        self.target_s.reset()
        self.frozen = False
        self._freeze_center: Optional[tuple[float, float]] = None
        self._ref_pixels = 0
        self.touch_streak = 0
        self.ever_touching = False
        self.min_abs_gap: Optional[float] = None
        self.last_gap: Optional[float] = None
        self._area_hist: deque[tuple[float, float]] = deque(maxlen=int(self.cfg["band"]["window"]))

    # ---- per frame -----------------------------------------------------------
    def update(
        self,
        dmap: np.ndarray,
        frame_id: int,
        band_mask: Optional[np.ndarray] = None,
        target_box: Optional[regions.Box] = None,
        target_mask: Optional[np.ndarray] = None,
        hand_point: Optional[tuple[float, float]] = None,
        band_area_px: Optional[float] = None,
        map_age: int = 0,
    ) -> DepthReading:
        t0 = time.perf_counter()
        cfg, th = self.cfg, self.th
        samp, occ = cfg["sampling"], cfg["occlusion"]
        shape = dmap.shape[:2]
        r = DepthReading(frame_id=frame_id, units=self.units, map_age=map_age)

        # -- hand: median over the band's own pixels, eroded --------------------
        raw_hand = None
        if band_mask is not None:
            raw_hand, _ = regions.median_in(dmap, regions.erode(band_mask, samp["erode_px"]), samp["min_pixels"])
        r.raw_hand = raw_hand
        r.hand, r.hand_valid = self.hand_s.update(raw_hand)

        # -- target region: tracker mask if given, else the shrunk box ----------
        if target_mask is not None and target_mask.any():
            target_region = target_mask.astype(bool)
            target_bbox = regions.mask_box(target_region)
        elif target_box is not None:
            target_region = regions.box_to_mask(target_box, shape)
            target_bbox = target_box
        else:
            target_region, target_bbox = None, None

        hand_area = regions.hand_region(band_mask, samp["hand_region_scale"])
        if target_region is not None:
            r.overlap = regions.overlap_fraction(hand_area, target_region) if hand_area is not None else 0.0
            sample_region = target_region
            if target_mask is None:
                sample_region = regions.box_to_mask(regions.shrink_box(target_bbox, samp["box_shrink"]), shape)
            sample_region = regions.erode(sample_region, samp["erode_px"])
            if hand_area is not None:
                sample_region = sample_region & ~hand_area           # never sample the hand as the target
            visible = int(sample_region.sum())
            visible_frac = visible / self._ref_pixels if self._ref_pixels else 1.0

            # -- occlusion freeze / unfreeze ---------------------------------------
            center = regions.box_center(target_bbox)
            if not self.frozen:
                if r.overlap > occ["freeze_overlap"] or visible_frac < occ["min_visible"]:
                    self.frozen, self._freeze_center = True, center
            else:
                moved = (self._freeze_center is not None and
                         np.hypot(center[0] - self._freeze_center[0], center[1] - self._freeze_center[1])
                         > occ["moved_frac"] * regions.box_diag(target_bbox))
                cleared = r.overlap < occ["unfreeze_overlap"] and visible_frac >= occ["min_visible"]
                if moved or cleared:
                    self.frozen, self._freeze_center = False, None
                    if moved:
                        self._ref_pixels = 0                           # new place, new reference size

            if not self.frozen or self.target_s.value is None:
                raw_target, n = regions.median_in(dmap, sample_region, samp["min_pixels"])
                r.raw_target = raw_target
                _, accepted = self.target_s.update(raw_target)
                if accepted and not self.frozen and r.overlap < occ["unfreeze_overlap"]:
                    self._ref_pixels = n                               # size of the target when unobstructed
        r.target_frozen = self.frozen
        r.target = self.target_s.value
        r.target_valid = r.target is not None

        # -- the relation -----------------------------------------------------------
        if r.hand is not None and r.target is not None:
            r.dz = r.target - r.hand
            r.gap = r.dz - self.offset
            scale = 1.0 if self.units == "metric" else float(cfg["controller"]["relative_scale"])
            r.depth_error = r.gap * scale
            r.band = self._band(abs(r.gap))
            self.last_gap = r.gap
            self.min_abs_gap = abs(r.gap) if self.min_abs_gap is None else min(self.min_abs_gap, abs(r.gap))

        # -- touching: 2D in box AND depth gap closed, held for N frames ----------
        r.hand_in_box = regions.point_in_box(hand_point, target_bbox)
        close = r.gap is not None and abs(r.gap) <= th["contact"] and r.hand_valid
        self.touch_streak = self.touch_streak + 1 if (close and r.hand_in_box) else 0
        r.touch_streak = self.touch_streak
        r.touching = self.touch_streak >= int(cfg["touching"]["hold_frames"])
        self.ever_touching = self.ever_touching or r.touching

        # -- band checks: area vs depth, pinhole width ------------------------------
        r.inconsistent = self._area_check(band_area_px, r.hand)
        if band_mask is not None and band_mask.any():
            r.band_width_px = regions.band_width_px(band_mask)
            r.band_distance_m = regions.band_distance_m(
                r.band_width_px, cfg["band"]["focal_px"], cfg["band"]["width_m"])

        r.timings_ms = {"depth_sample": (time.perf_counter() - t0) * 1000}
        return r

    # ---- per reach (agent's haptic_result vocabulary) ---------------------------
    def last_seen_depth(self) -> DepthBand:
        """last_seen.depth: near | mid | far, from the last known gap."""
        return "far" if self.last_gap is None else self._band(abs(self.last_gap))

    def closest_approach(self) -> Approach:
        """Depth's view of closest_approach: touching | near | far."""
        if self.ever_touching:
            return "touching"
        if self.min_abs_gap is not None and self.min_abs_gap < self.th["near"]:
            return "near"
        return "far"

    # ---- helpers ------------------------------------------------------------------
    def _band(self, abs_gap: float) -> DepthBand:
        if abs_gap < self.th["near"]:
            return "near"
        if abs_gap > self.th["far"]:
            return "far"
        return "mid"

    def _area_check(self, area: Optional[float], hand: Optional[float]) -> bool:
        """The band looks bigger when the hand is closer. Flag frames where the
        band grew a lot but depth says the hand moved away (or the reverse)."""
        if area is None or hand is None or area <= 0:
            return False
        self._area_hist.append((float(area), float(hand)))
        if len(self._area_hist) < self._area_hist.maxlen:
            return False
        old_area, old_hand = self._area_hist[0]
        ratio = area / old_area
        moved = hand - old_hand                    # positive = farther
        limit = self.cfg["band"]["area_ratio"]
        thr = self.th["area_cross_check"]
        return (ratio >= limit and moved > thr) or (ratio <= 1 / limit and moved < -thr)
