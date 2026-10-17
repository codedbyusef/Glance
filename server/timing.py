"""Per-stage timers, returned to the client alongside each FrameResult."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(timings: dict[str, float], stage: str):
    start = time.perf_counter()
    yield
    timings[stage] = (time.perf_counter() - start) * 1000
