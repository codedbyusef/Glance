"""Tracks the detected target across frames. SAM 2, with OpenCV CSRT as fallback.

Runs every frame; detector only runs at step start and on re-acquire.
"""

from __future__ import annotations


def track(frame: bytes, prev_box):
    raise NotImplementedError
