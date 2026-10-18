"""The server's depth hook with the fake depth model (GLANCE_FAKE_MODELS=1). No torch or weights."""
import cv2
import numpy as np
import pytest

from depth_estimation import depth_config
from hand_detection import hand_marker
from object_detection.depth_hook import ReachDepth, load_depth_cache, object_name_for
from object_detection.session import HandReading


def band_bgr():
    lo, hi = (np.array(v, np.int32) for v in hand_marker._load_cfg()["hsv_ranges"][0])
    hsv = ((lo + hi) // 2).astype(np.uint8).reshape(1, 1, 3)
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setenv("GLANCE_FAKE_MODELS", "1")
    return load_depth_cache()


def test_object_name_for_picks_the_offset_named_in_the_target():
    cfg = depth_config.load()
    assert object_name_for("red soda can", cfg) == "can"
    assert object_name_for("white sheet of paper", cfg) == "paper"
    assert object_name_for("door handle", cfg) == cfg["contact_offset"]["active_object"]


def test_reach_depth_reads_band_and_target_from_one_map(cache):
    frame = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(frame, (300, 300), (340, 320), band_bgr(), -1)          # the band, below the target
    info = hand_marker.detect(frame)
    hand = HandReading(point=info.point, box=info.box, area_px=info.area_px)
    probe = ReachDepth(cache, "red soda can")
    assert probe.tracker.object_name == "can"
    s = None
    for i in range(1, 4):
        s = probe.read(frame, i, (260.0, 120.0, 380.0, 220.0), None, hand)
    assert s.units == depth_config.units(depth_config.load())
    assert s.hand is not None and s.target is not None
    assert s.dz == pytest.approx(s.target - s.hand)
    assert s.dz > 0  # fake map: farther toward the top of the image, and the target is above the band
    assert not s.touching and s.band in ("near", "mid", "far")
    assert set(s.timings_ms) == {"depth_model", "band_mask", "depth_sample"}
    assert probe.last_seen_depth() in ("near", "mid", "far") and probe.closest_approach() in ("near", "far")


def test_reach_depth_without_a_hand_has_no_gap(cache):
    frame = np.full((480, 640, 3), 90, np.uint8)
    s = ReachDepth(cache, "paper").read(frame, 1, (260.0, 120.0, 380.0, 220.0), None, None)
    assert s.hand is None and s.gap is None and s.target is not None
