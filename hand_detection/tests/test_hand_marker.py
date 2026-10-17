"""Unit tests for hand_detection/hand_marker.py using drawn images. No camera needed.

    pytest hand_detection -v
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hand_detection import hand_marker as hand

def band_bgr() -> tuple[int, int, int]:
    """BGR color in the middle of the configured band HSV range, so the tests follow retuning."""
    lo, hi = (np.array(v, np.int32) for v in hand._load_cfg()["hsv_ranges"][0])
    hsv = ((lo + hi) // 2).astype(np.uint8).reshape(1, 1, 3)
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


GREEN = band_bgr()           # BGR, inside the configured HSV range (config.yaml)
SKIN = (150, 170, 200)
TABLE = (90, 80, 70)


def blank():
    return np.full((480, 640, 3), TABLE, np.uint8)


def jpeg(img, quality=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def fresh_state():
    hand.reset()
    yield
    hand.reset()


def test_palm_down_single_band():
    img = blank()
    cv2.rectangle(img, (200, 150), (340, 330), SKIN, -1)        # hand
    cv2.rectangle(img, (200, 200), (340, 225), GREEN, -1)       # band across the knuckles
    x, y = hand.locate_hand(jpeg(img))
    assert abs(x - 270) < 5 and abs(y - 212) < 5


def test_band_split_by_fingers_is_merged():
    img = blank()
    for x1 in (200, 240, 280, 320):                              # 4 pieces with gaps
        cv2.rectangle(img, (x1, 200), (x1 + 28, 225), GREEN, -1)
    r = hand.detect(img)
    assert r is not None and r.pieces == 4
    assert abs(r.point[0] - 274) < 6                             # center of the whole band


def test_sideways_hand_two_strips():
    img = blank()
    cv2.rectangle(img, (300, 180), (318, 260), GREEN, -1)       # back of hand, edge-on
    cv2.rectangle(img, (330, 185), (345, 255), GREEN, -1)       # palm strip
    r = hand.detect(img)
    assert r is not None and r.pieces == 2


def test_palm_up_only_palm_strip_visible():
    img = blank()
    cv2.rectangle(img, (250, 300), (360, 322), GREEN, -1)
    assert hand.locate_hand(jpeg(img)) is not None


def test_tiny_green_speck_ignored():
    img = blank()
    cv2.rectangle(img, (200, 200), (300, 225), GREEN, -1)
    cv2.circle(img, (600, 40), 3, GREEN, -1)                    # far away speck
    x, _ = hand.locate_hand(jpeg(img))
    assert x < 320


def test_huge_green_background_rejected():
    img = np.full((480, 640, 3), GREEN, np.uint8)                # green shirt filling the frame
    assert hand.locate_hand(jpeg(img)) is None


def test_no_band_returns_none():
    assert hand.locate_hand(jpeg(blank())) is None


def test_stray_object_does_not_steal_tracked_hand():
    img = blank()
    cv2.rectangle(img, (100, 300), (160, 320), GREEN, -1)       # hand band
    assert hand.locate_hand(jpeg(img)) is not None
    cv2.rectangle(img, (520, 40), (620, 90), GREEN, -1)         # bigger green thing appears far away
    x, _ = hand.locate_hand(jpeg(img))
    assert x < 200                                               # still on the hand


def test_area_grows_as_hand_gets_closer():
    far, near = blank(), blank()
    cv2.rectangle(far, (300, 220), (340, 232), GREEN, -1)
    cv2.rectangle(near, (260, 200), (380, 236), GREEN, -1)
    a_far = hand.detect(far).area_px
    hand.reset()
    a_near = hand.detect(near).area_px
    assert a_near > a_far


def test_bad_bytes_return_none():
    assert hand.locate_hand(b"not a jpeg") is None
