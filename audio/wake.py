"""Wake phrase detection with openWakeWord (CLAUDE.md section 8, decision 18)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .log import EventLog, null_log

STAND_IN = "hey_jarvis"


class Detector(Protocol):
    name: str

    def score(self, block: np.ndarray) -> float: ...  # one 80 ms block of 16 kHz int16; returns 0..1

    def reset(self) -> None: ...


class OwwDetector:
    """openWakeWord, ONNX backend. model is a path to an .onnx file or a pretrained name such as hey_jarvis."""

    def __init__(self, model: str, vad_threshold: float = 0.5):
        from openwakeword.model import Model

        kwargs = {"vad_threshold": vad_threshold} if vad_threshold > 0 else {}
        self.model = Model(wakeword_models=[model], inference_framework="onnx", **kwargs)
        self.name = next(iter(self.model.models))

    def score(self, block: np.ndarray) -> float:
        scores = self.model.predict(np.asarray(block, dtype=np.int16))
        return float(max(scores.values())) if scores else 0.0

    def reset(self) -> None:
        self.model.reset()


class FakeDetector:
    """Scores come from a list (one per block), then 0. Records how many blocks it saw."""

    def __init__(self, scores: list[float] | None = None, name: str = "fake"):
        self.scores = list(scores or [])
        self.name = name
        self.blocks = 0
        self.resets = 0

    def score(self, block: np.ndarray) -> float:
        self.blocks += 1
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self) -> None:
        self.resets += 1


def load_detector(cfg, log: EventLog | None = None) -> Detector | None:
    """The configured wake model, else the hey_jarvis stand-in, else None (push-to-talk only). Never raises."""
    log = log or null_log()
    want = cfg.wake_model
    looks_like_file = want.endswith(".onnx") or "/" in want or "\\" in want
    if looks_like_file:
        path = cfg.path(want)
        if path.is_file():
            candidates = [str(path)]
        else:
            log.write("error", where="wake.load", error=f"wake model {path} not found, using stand-in {STAND_IN}")
            candidates = [STAND_IN]
    else:
        candidates = [want] if want == STAND_IN else [want, STAND_IN]
    for model in candidates:
        try:
            d = OwwDetector(model, cfg.wake_vad_threshold)
            log.write("wake_model", model=d.name, path=model if Path(model).suffix else None,
                      stand_in=d.name == STAND_IN)
            return d
        except Exception as e:
            log.error("wake.load", e, model=model)
    log.write("error", where="wake.load", error="no wake word model; push-to-talk only")
    return None
