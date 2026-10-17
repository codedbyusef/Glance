"""Depth Anything V2 (small). Relative depth only -- larger value means closer.

Normalise per frame: patch medians divided by the frame's 5th-95th percentile
range. Verify the forward/back sign empirically (hand-wave at the camera) before
trusting it in the controller.
"""

from __future__ import annotations


def estimate_depth(frame: bytes, patch=None) -> float:
    raise NotImplementedError
