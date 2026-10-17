"""Unit tests for depth-estimation/ on synthetic depth maps. No camera, GPU,
torch or model weights needed.

    pytest depth-estimation -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_config  # noqa: E402
import depth_model  # noqa: E402
import depth_regions as regions  # noqa: E402
from depth_tracker import DepthTracker, Smoother  # noqa: E402

H, W = 480, 640
TABLE = 0.62                       # meters to the table, metric tests
TARGET_BOX = (260, 200, 380, 300)  # paper on the table
BAND_BOX = (150, 330, 210, 345)    # band, starting outside the target


def cfg(kind="metric", **overrides):
    """Defaults only (ignores the folder's config.yaml), no smoothing lag."""
    base = {"model": {"kind": kind}, "smoothing": {"median_window": 1, "ema_alpha": 1.0}}
    return depth_config.load(Path("/nonexistent.yaml"), **depth_config._merge(base, overrides))


def scene(hand_d, target_d=TABLE, band_box=BAND_BOX, hand_scale=2.0):
    """Depth map: table everywhere, the target at target_d, the whole hand
    (band box scaled up) at hand_d. Returns (dmap, band_mask)."""
    dmap = np.full((H, W), TABLE, np.float32)
    x1, y1, x2, y2 = TARGET_BOX
    dmap[y1:y2, x1:x2] = target_d
    if hand_d is not None:
        hand = regions.box_to_mask(regions.scale_box(band_box, hand_scale), (H, W))
        dmap[hand] = hand_d
        return dmap, regions.box_to_mask(band_box, (H, W))
    return dmap, None


def at(box, dx=0, dy=0):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def center(box):
    return regions.box_center(box)


# ---- config ---------------------------------------------------------------------

def test_defaults_without_config_file():
    c = depth_config.load(Path("/nonexistent.yaml"))
    assert c["model"]["kind"] == "relative"
    assert c["thresholds"]["metric"]["contact"] == 0.04


def test_nested_override_keeps_siblings():
    c = depth_config.load(Path("/nonexistent.yaml"), model={"kind": "metric"})
    assert c["model"]["kind"] == "metric" and c["model"]["input_size"] == 364


def test_folder_config_matches_defaults_keys():
    """config.yaml and the built-in defaults describe the same settings."""
    import yaml
    user = yaml.safe_load(depth_config.CONFIG_PATH.read_text())

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out
    assert keys(user) == keys(depth_config.DEFAULTS)


# ---- model helpers ----------------------------------------------------------------

def test_model_input_size_is_multiple_of_14_and_keeps_aspect():
    h, w = depth_model.model_input_size(480, 640, 364)
    assert h % 14 == 0 and w % 14 == 0
    assert w == 364 and abs(w / h - 640 / 480) < 0.05


def test_relative_normalization_flips_to_larger_is_farther():
    raw = np.array([[10.0, 5.0, 1.0]] * 3, np.float32)      # model: larger = closer
    out = depth_model.normalize_relative(raw, 5, 95)
    assert out[0, 0] < out[0, 1] < out[0, 2]                 # closest pixel now smallest


def test_relative_normalization_flat_frame():
    out = depth_model.normalize_relative(np.ones((4, 4), np.float32), 5, 95)
    assert np.all(out == 0)


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.cfg = depth_config.load(Path("/nonexistent.yaml"))

    def infer(self, bgr):
        self.calls += 1
        return np.full(bgr.shape[:2], float(self.calls), np.float32)


def test_cache_one_inference_per_frame():
    m = CountingModel()
    cache = depth_model.DepthCache(m, every_n=1)
    frame = np.zeros((H, W, 3), np.uint8)
    cache.get(frame, 1)
    cache.get(frame, 1)                       # hand and target asking for the same frame
    assert m.calls == 1
    _, age = cache.get(frame, 2)
    assert m.calls == 2 and age == 0


def test_cache_every_other_frame_reports_age():
    m = CountingModel()
    cache = depth_model.DepthCache(m, every_n=2)
    frame = np.zeros((H, W, 3), np.uint8)
    cache.get(frame, 10)
    _, age = cache.get(frame, 11)
    assert m.calls == 1 and age == 1
    cache.get(frame, 12)
    assert m.calls == 2


def test_cache_recomputes_when_frame_ids_restart():
    m = CountingModel()
    cache = depth_model.DepthCache(m, every_n=3)
    frame = np.zeros((H, W, 3), np.uint8)
    cache.get(frame, 50)
    cache.get(frame, 0)
    assert m.calls == 2


def test_fake_model_runs_without_torch(monkeypatch):
    monkeypatch.setenv("GLANCE_FAKE_MODELS", "1")
    m = depth_model.DepthModel(cfg("metric"))
    d = m.infer(np.zeros((H, W, 3), np.uint8))
    assert d.shape == (H, W) and d[0, 0] > d[-1, 0]           # top of the table is farther


# ---- regions --------------------------------------------------------------------

def test_median_needs_min_pixels():
    d = np.ones((10, 10), np.float32)
    mask = np.zeros((10, 10), bool)
    mask[0, :3] = True
    assert regions.median_in(d, mask, 5) == (None, 3)


def test_erode_drops_edge_pixels():
    d = np.full((50, 50), 1.0, np.float32)
    d[10:30, 10:30] = 0.5
    mask = regions.box_to_mask((9, 9, 31, 31), (50, 50))     # box slightly larger than the object
    raw = d[mask].mean()
    eroded = d[regions.erode(mask, 2)].mean()
    assert abs(eroded - 0.5) < abs(raw - 0.5)


def test_band_mask_from_marker_ignores_other_green():
    full = np.zeros((H, W), np.uint8)
    full[300:320, 100:180] = 255          # the band
    full[50:80, 500:560] = 255            # a green cup elsewhere
    band = regions.band_mask_from_marker(full, (100, 300, 180, 320))
    assert band[310, 140] and not band[60, 520]


def test_band_width_on_tilted_band():
    mask = np.zeros((H, W), np.uint8)
    rect = ((320, 240), (120, 24), 30)                          # 24 px wide, rotated 30 degrees
    cv2.fillPoly(mask, [cv2.boxPoints(rect).astype(np.int32)], 1)
    assert abs(regions.band_width_px(mask.astype(bool)) - 24) < 3


def test_pinhole_round_trip():
    f = regions.focal_from_band(width_px=30, distance_m=0.5, width_m=0.025)
    assert abs(regions.band_distance_m(30, f, 0.025) - 0.5) < 1e-9
    assert regions.band_distance_m(None, f, 0.025) is None


# ---- smoother -------------------------------------------------------------------

def test_smoother_drops_single_spike():
    s = Smoother(window=5, alpha=1.0, jump=0.15, max_rejects=3)
    for _ in range(5):
        s.update(0.5)
    value, accepted = s.update(0.9)
    assert not accepted and value == pytest.approx(0.5)


def test_smoother_accepts_persistent_jump():
    s = Smoother(window=5, alpha=1.0, jump=0.15, max_rejects=3)
    s.update(0.5)
    for _ in range(3):
        s.update(0.9)
    value, accepted = s.update(0.9)
    assert accepted and value == pytest.approx(0.9)


# ---- tracker: the hand-target relation --------------------------------------------

def test_dz_positive_when_target_farther_than_hand():
    t = DepthTracker(cfg("metric"))
    dmap, band = scene(hand_d=0.45)
    r = t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX, hand_point=center(BAND_BOX))
    assert r.hand == pytest.approx(0.45) and r.target == pytest.approx(TABLE)
    assert r.dz == pytest.approx(TABLE - 0.45) and r.dz > 0
    assert r.depth_error == pytest.approx(r.gap) and r.depth_error > 0   # controller: forward


def test_contact_offset_is_subtracted():
    c = cfg("metric", contact_offset={"active_object": "paper", "metric": {"paper": 0.03}})
    t = DepthTracker(c)
    dmap, band = scene(hand_d=0.50)
    r = t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    assert r.gap == pytest.approx(TABLE - 0.50 - 0.03)


def test_object_name_selects_offset():
    c = cfg("metric", contact_offset={"metric": {"paper": 0.02, "can": 0.07}})
    assert DepthTracker(c, object_name="can").offset == pytest.approx(0.07)


def test_relative_depth_error_uses_scale():
    c = cfg("relative", controller={"relative_scale": 2.0})
    t = DepthTracker(c)
    dmap, band = scene(hand_d=0.3, target_d=0.5)
    r = t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    assert r.units == "relative" and r.depth_error == pytest.approx(2.0 * r.gap)


def test_hovering_over_target_is_not_touching():
    """The 2D false positive depth exists to catch: band inside the box, hand 15 cm up."""
    c = cfg("metric", contact_offset={"metric": {"paper": 0.03}})
    t = DepthTracker(c)
    band_box = (300, 240, 340, 252)
    for i in range(12):
        dmap, band = scene(hand_d=TABLE - 0.18, band_box=band_box)
        r = t.update(dmap, i, band_mask=band, target_box=TARGET_BOX, hand_point=center(band_box))
    assert r.hand_in_box and not r.touching


def test_touching_needs_in_box_and_hold_frames():
    c = cfg("metric", contact_offset={"metric": {"paper": 0.03}}, touching={"hold_frames": 5})
    t = DepthTracker(c)
    band_box = (300, 240, 340, 252)
    # Seed the target before the hand covers it; the hand is coming down, not teleporting
    # (a jump bigger than thresholds.jump in one frame is rejected as noise).
    dmap, band = scene(hand_d=0.50)
    t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX, hand_point=center(BAND_BOX))
    results = []
    for i in range(1, 7):
        dmap, band = scene(hand_d=TABLE - 0.03, band_box=band_box)      # knuckles 3 cm up = contact
        results.append(t.update(dmap, i, band_mask=band, target_box=TARGET_BOX,
                                hand_point=center(band_box)))
    assert [r.touching for r in results] == [False, False, False, False, True, True]
    assert t.closest_approach() == "touching"


def test_touching_needs_hand_point_in_box():
    c = cfg("metric", touching={"hold_frames": 2})
    t = DepthTracker(c)
    for i in range(5):
        dmap, band = scene(hand_d=TABLE)                                 # right depth, wrong place
        r = t.update(dmap, i, band_mask=band, target_box=TARGET_BOX, hand_point=(10, 10))
    assert not r.touching


def test_occlusion_freezes_target_depth():
    t = DepthTracker(cfg("metric"))
    dmap, band = scene(hand_d=0.40)
    first = t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX)
    over = (290, 235, 350, 255)                                          # hand region covers the paper
    dmap, band = scene(hand_d=0.57, band_box=over, hand_scale=3.0)
    r = t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    assert r.target_frozen and r.overlap > 0.3
    assert r.target == pytest.approx(first.target)


def test_unfreezes_when_hand_leaves():
    t = DepthTracker(cfg("metric"))
    dmap, band = scene(hand_d=0.40)
    t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX)
    over = (290, 235, 350, 255)
    dmap, band = scene(hand_d=0.57, band_box=over, hand_scale=3.0)
    t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    dmap, band = scene(hand_d=0.40)
    r = t.update(dmap, 2, band_mask=band, target_box=TARGET_BOX)
    assert not r.target_frozen


def test_unfreezes_when_target_moves():
    t = DepthTracker(cfg("metric"))
    dmap, band = scene(hand_d=0.40)
    t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX)
    over = (290, 235, 350, 255)
    dmap, band = scene(hand_d=0.57, band_box=over, hand_scale=3.0)
    t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    moved = at(TARGET_BOX, dx=200)
    r = t.update(dmap, 2, band_mask=band, target_box=moved)
    assert not r.target_frozen


def test_target_mask_is_used_instead_of_box():
    t = DepthTracker(cfg("metric"))
    dmap = np.full((H, W), TABLE, np.float32)
    can = np.zeros((H, W), bool)
    can[200:300, 300:340] = True                       # thin can; its box would be mostly table
    dmap[can] = 0.55
    r = t.update(dmap, 0, target_mask=can, target_box=(250, 200, 390, 300))
    assert r.target == pytest.approx(0.55)


def test_labels_for_agent():
    t = DepthTracker(cfg("metric"))
    assert t.last_seen_depth() == "far" and t.closest_approach() == "far"
    dmap, band = scene(hand_d=TABLE - 0.25)
    t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX)
    assert t.last_seen_depth() == "mid"
    dmap, band = scene(hand_d=TABLE - 0.10)
    t.update(dmap, 1, band_mask=band, target_box=TARGET_BOX)
    assert t.last_seen_depth() == "near" and t.closest_approach() == "near"


def test_area_cross_check_flags_contradiction():
    t = DepthTracker(cfg("metric", band={"window": 3}))
    r = None
    for i, (hand_d, area) in enumerate([(0.40, 500), (0.45, 650), (0.50, 800)]):
        dmap, band = scene(hand_d=hand_d)
        r = t.update(dmap, i, band_mask=band, target_box=TARGET_BOX, band_area_px=area)
    assert r.inconsistent                    # band grew 60% but depth says hand moved 10 cm away


def test_area_cross_check_agreeing_frames_pass():
    t = DepthTracker(cfg("metric", band={"window": 3}))
    r = None
    for i, (hand_d, area) in enumerate([(0.50, 500), (0.45, 650), (0.40, 800)]):
        dmap, band = scene(hand_d=hand_d)
        r = t.update(dmap, i, band_mask=band, target_box=TARGET_BOX, band_area_px=area)
    assert not r.inconsistent


def test_no_inputs_gives_empty_reading():
    t = DepthTracker(cfg("metric"))
    r = t.update(np.full((H, W), TABLE, np.float32), 0)
    assert r.hand is None and r.target is None and r.dz is None and not r.touching


def test_reset_clears_reach_state():
    t = DepthTracker(cfg("metric"))
    dmap, band = scene(hand_d=0.45)
    t.update(dmap, 0, band_mask=band, target_box=TARGET_BOX)
    t.reset()
    assert t.hand_s.value is None and t.last_gap is None and not t.frozen
