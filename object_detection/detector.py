"""Open-vocabulary detection with Grounding DINO (Hugging Face transformers).

The only model that understands the `target` phrase. It is slow, so it runs once
at reach start and again only to re-acquire; SAM 2 (tracker.py) follows the box
on every other frame.

    det = GroundingDinoDetector()
    candidates = det.detect(bgr_image, "sheet of paper")   # sorted, best first
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

from object_detection import boxes
from object_detection.common import load_config, pick_device


@dataclass
class Detection:
    box: boxes.Box      # pixel xyxy
    score: float
    label: str


class Detector(Protocol):
    def detect(self, image: np.ndarray, phrase: str) -> list[Detection]: ...


def score_gap(candidates: list[Detection]) -> Optional[float]:
    """Top score minus second score; None with fewer than two candidates."""
    if len(candidates) < 2:
        return None
    return candidates[0].score - candidates[1].score


def drop_oversized(candidates: list[Detection], width: int, height: int, max_frac: float) -> list[Detection]:
    """Remove boxes covering more than `max_frac` of the frame. When the target is absent,
    Grounding DINO tends to box the whole table or blanket with a mid score instead of
    returning nothing; a reachable object never fills most of the head-camera view."""
    limit = max_frac * width * height
    return [d for d in candidates if boxes.area(d.box) <= limit]


def prompt_for(phrase: str) -> str:
    """Grounding DINO wants lowercase text ending in a period."""
    phrase = phrase.strip().lower().rstrip(".")
    return f"{phrase}."


class GroundingDinoDetector:
    def __init__(self, cfg: Optional[dict] = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        full = load_config()
        cfg = cfg or full["detector"]
        self.box_threshold = float(cfg["box_threshold"])
        self.text_threshold = float(cfg["text_threshold"])
        self.max_box_frac = float(cfg["max_box_frac"])
        self.device = pick_device(full["server"]["device"])
        # fp16 only on CUDA; CPU/MPS stay in float32.
        self.dtype = torch.float16 if (self.device == "cuda" and full["server"]["fp16"]) else torch.float32

        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["model_id"], dtype=self.dtype)
        self.model.to(self.device).eval()

    def detect(self, image: np.ndarray, phrase: str) -> list[Detection]:
        import torch

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=prompt_for(phrase), return_tensors="pt").to(self.device)
        if self.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # target_sizes makes the post-processor return pixel xyxy (it converts from
        # normalized cx,cy,w,h itself), so only clipping is needed here.
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        labels = result.get("text_labels", result.get("labels"))
        candidates = [
            Detection(box=boxes.clip(box, width, height), score=float(score), label=str(label))
            for box, score, label in zip(result["boxes"].float().tolist(), result["scores"].float().tolist(), labels)
        ]
        candidates = drop_oversized(candidates, width, height, self.max_box_frac)
        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates

    def warmup(self) -> None:
        self.detect(np.zeros((360, 480, 3), dtype=np.uint8), "object")
