"""Canned detector and tracker: no GPU, no weights. For tests and for checking the
server/client/tunnel wiring before the real models are on Vultr (GLANCE_FAKE_MODELS=1)."""

from __future__ import annotations

from typing import Optional

import numpy as np

import boxes
from detector import Detection


class FakeDetector:
    """Returns a box in the middle third of the frame for any phrase, or the scripted results."""

    def __init__(self, script: Optional[list[list[Detection]]] = None):
        self.script = list(script) if script is not None else None
        self.calls = 0

    def detect(self, image: np.ndarray, phrase: str) -> list[Detection]:
        self.calls += 1
        if self.script is not None:
            return self.script.pop(0) if self.script else []
        h, w = image.shape[:2]
        return [Detection(box=(w / 3, h / 3, 2 * w / 3, 2 * h / 3), score=0.6, label=phrase)]

    def warmup(self) -> None:
        pass


class FakeTracker:
    """Keeps returning the seeded box, or the scripted boxes (None = lost this frame)."""

    def __init__(self, script: Optional[list[Optional[boxes.Box]]] = None):
        self.script = list(script) if script is not None else None
        self.box: Optional[boxes.Box] = None
        self.inits = 0

    def init(self, frame: np.ndarray, box: boxes.Box) -> None:
        self.box = box
        self.inits += 1

    def update(self, frame: np.ndarray) -> tuple[Optional[boxes.Box], float]:
        if self.script is not None:
            box = self.script.pop(0) if self.script else None
            return box, (0.9 if box is not None else 0.0)
        return self.box, (0.9 if self.box is not None else 0.0)

    def reset(self) -> None:
        self.box = None

    def warmup(self) -> None:
        pass
