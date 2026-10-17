"""Hand point localization. MediaPipe Hands -- can also run on the laptop instead
of Vultr; decide after the first latency measurement.

Fallback if MediaPipe struggles with a gloved hand from a head-worn view: a
bright marker on the glove tracked by color.
"""

from __future__ import annotations


def locate_hand(frame: bytes):
    raise NotImplementedError
