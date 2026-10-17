"""Open-vocab detector on `target`: YOLO-World (Ultralytics).

Loaded once per server process; runs only at reach() session start and on
re-acquire -- SAM 2 (tracker.py) handles every-frame tracking after that.
See CLAUDE.md "Object detection and tracking (settled)".

Returns every candidate box sorted by confidence. The vision loop hands
several candidates to the VLM disambiguator (using `refer`) when more than
one comes back; it's free to just take the top one when there's only one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import yaml
from ultralytics import YOLOWorld

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class Candidate(TypedDict):
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, pixel coordinates
    confidence: float


_model: YOLOWorld | None = None
_confidence_threshold: float | None = None


def _get_model() -> YOLOWorld:
    global _model, _confidence_threshold
    if _model is None:
        with _CONFIG_PATH.open() as f:
            settings = yaml.safe_load(f)["detector"]
        _model = YOLOWorld(settings["model_id"])
        _confidence_threshold = settings["confidence_threshold"]
    return _model


def warmup() -> None:
    """Runs one dummy inference so the first real reach isn't the cold one."""
    model = _get_model()
    model.set_classes(["object"])
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)


def detect(frame: bytes, target: str) -> list[Candidate]:
    model = _get_model()
    model.set_classes([target])

    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image, conf=_confidence_threshold, verbose=False)[0]

    candidates: list[Candidate] = [
        {"box": tuple(box), "confidence": conf}
        for box, conf in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist())
    ]
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates
