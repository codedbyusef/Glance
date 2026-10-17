"""Scripted sweep over all 8 channels, so the hardware team can test the glove
without waiting on vision or the perception server.
"""

from __future__ import annotations

from haptics.link import HapticLink


def run_sweep(link: HapticLink) -> None:
    raise NotImplementedError
