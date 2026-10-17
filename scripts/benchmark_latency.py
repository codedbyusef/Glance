"""Measures hand-movement -> felt-vibration latency per stage: camera capture,
network up, detection/tracking, depth, controller, network down, motor update.
"""

from __future__ import annotations


def benchmark() -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
