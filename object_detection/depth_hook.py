"""Depth for the reach session: Depth Anything V2 (depth_estimation) behind session.DepthProbe.

One DepthCache per server process (the model is loaded and warmed once); one DepthTracker per reach.

    cache = load_depth_cache()                 # at server startup; GLANCE_FAKE_MODELS=1 -> fake model, no torch
    probe = ReachDepth(cache, target)          # per reach
    sample = probe.read(frame, frame_id, target_box, target_mask, hand)

The hand region is the band's own pixels: hand_marker.marker_mask() limited to the tracked band's box
(depth_regions.band_mask_from_marker). The target is the SAM 2 mask when the tracker has one, else its box.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from depth_estimation import depth_config, depth_regions
from depth_estimation.depth_model import DepthCache, DepthModel
from depth_estimation.depth_tracker import DepthTracker
from hand_detection import hand_marker
from object_detection import boxes
from object_detection.session import DepthSample, HandReading


def load_depth_cache(cfg: Optional[dict] = None) -> DepthCache:
    """Load and warm the model once per process. Respects depth's model.every_n_frames."""
    model = DepthModel(cfg)
    model.warmup()
    return DepthCache(model)


def object_name_for(target: str, cfg: dict) -> str:
    """Pick the contact offset whose object name appears in the target phrase ("red soda can" -> can);
    otherwise depth_estimation's contact_offset.active_object."""
    offsets = cfg["contact_offset"]
    names = set(offsets.get("metric", {})) | set(offsets.get("relative", {}))
    words = target.lower().split()
    for name in sorted(names):
        if name in words:
            return name
    return offsets["active_object"]


class ReachDepth:
    """session.DepthProbe for one reach."""

    def __init__(self, cache: DepthCache, target: str, cfg: Optional[dict] = None):
        self.cache = cache
        self.cfg = cfg or depth_config.load()
        self.cache.reset()  # a new reach never reuses the last reach's map
        self.tracker = DepthTracker(self.cfg, object_name=object_name_for(target, self.cfg))

    def read(self, frame: np.ndarray, frame_id: int, target_box: Optional[boxes.Box],
             target_mask: Optional[np.ndarray], hand: Optional[HandReading]) -> DepthSample:
        t0 = time.perf_counter()
        dmap, age = self.cache.get(frame, frame_id)
        t1 = time.perf_counter()
        band_mask = None
        if hand is not None and hand.box is not None:
            band_mask = depth_regions.band_mask_from_marker(hand_marker.marker_mask(frame) > 0, hand.box)
        t2 = time.perf_counter()
        r = self.tracker.update(dmap, frame_id, band_mask=band_mask, target_box=target_box, target_mask=target_mask,
                                hand_point=hand.point if hand is not None else None,
                                band_area_px=hand.area_px if hand is not None else None, map_age=age)
        timings = {
            # the model only runs on frames where the map is fresh (every_n_frames); 0 means a reused map
            "depth_model": (t1 - t0) * 1000 if age == 0 else 0.0,
            "band_mask": (t2 - t1) * 1000,
            "depth_sample": r.timings_ms.get("depth_sample", (time.perf_counter() - t2) * 1000),
        }
        return DepthSample(
            units=r.units, hand=r.hand, target=r.target, dz=r.dz, gap=r.gap, depth_error=r.depth_error,
            band=r.band, touching=r.touching, target_frozen=r.target_frozen, inconsistent=r.inconsistent,
            map_age=r.map_age, timings_ms=timings,
        )

    def last_seen_depth(self) -> str:
        return self.tracker.last_seen_depth()

    def closest_approach(self) -> str:
        return self.tracker.closest_approach()
