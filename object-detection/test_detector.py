"""Detector post-processing tests (no model needed). Run: pytest object-detection -v"""

from detector import Detection, drop_oversized


def test_drop_oversized_removes_whole_scene_boxes():
    # Absent target: Grounding DINO boxes the whole blanket/desk instead of returning nothing.
    scene = Detection((2, 71, 638, 478), 0.51, "sheet paper")      # ~85% of a 640x480 frame
    real = Detection((454, 364, 496, 408), 0.30, "paper")
    assert drop_oversized([scene, real], 640, 480, max_frac=0.6) == [real]


def test_drop_oversized_keeps_large_but_plausible_boxes():
    bed = Detection((0, 278, 399, 482), 0.75, "bed")               # ~26% of the frame
    assert drop_oversized([bed], 640, 483, max_frac=0.6) == [bed]
