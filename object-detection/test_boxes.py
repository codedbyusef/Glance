"""Box helper tests. Run: pytest object-detection -v"""

import numpy as np
import pytest

import boxes


def test_cxcywh_norm_to_xyxy():
    assert boxes.cxcywh_norm_to_xyxy((0.5, 0.5, 0.5, 0.5), 400, 200) == (100.0, 50.0, 300.0, 150.0)


def test_cxcywh_clips_to_frame():
    assert boxes.cxcywh_norm_to_xyxy((0.0, 0.0, 0.5, 0.5), 400, 200) == (0.0, 0.0, 100.0, 50.0)


def test_mask_to_box():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 3:8] = True
    assert boxes.mask_to_box(mask) == (3.0, 2.0, 8.0, 5.0)
    assert boxes.area(boxes.mask_to_box(mask)) == mask.sum()


def test_mask_to_box_empty():
    assert boxes.mask_to_box(np.zeros((5, 5), dtype=bool)) is None


def test_coverage():
    target = (0, 0, 10, 10)
    assert boxes.coverage((0, 0, 5, 10), target) == pytest.approx(0.5)
    assert boxes.coverage((20, 20, 30, 30), target) == 0.0
    assert boxes.coverage(None, target) == 0.0


def test_contains_and_distance():
    box = (10, 10, 20, 20)
    assert boxes.contains(box, (15, 15))
    assert not boxes.contains(box, (25, 15))
    assert boxes.distance_to_box(box, (15, 15)) == 0.0
    assert boxes.distance_to_box(box, (23, 24)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "box, expected",
    [((0, 0, 10, 10), "left-top"), ((140, 90, 160, 110), "center-middle"), ((290, 190, 300, 200), "right-bottom")],
)
def test_region_thirds(box, expected):
    assert boxes.region_thirds(box, 300, 200) == expected
