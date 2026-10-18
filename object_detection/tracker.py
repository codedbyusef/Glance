"""Frame-to-frame tracking of the detected box. SAM 2 video (streaming), CSRT fallback.

SAM 2 knows nothing about words. It is seeded with Grounding DINO's box on one
frame, then carries a pixel mask forward using a memory of how the object looked,
which survives motion and brief occlusion. The mask is turned into a box and a
confidence for the guidance controller.

    trk = make_tracker()
    trk.init(bgr_frame, box)
    box, confidence = trk.update(next_bgr_frame)   # box is None when the mask is gone
"""

from __future__ import annotations

from typing import Optional, Protocol

import cv2
import numpy as np

from object_detection import boxes
from object_detection.common import load_config, pick_device

# SAM 2 attends to the last 7 frames plus object pointers from up to 16 frames back.
# Older frames and outputs are dropped so a 30 s reach does not fill GPU memory.
_KEEP_FRAMES = 20


class Tracker(Protocol):
    def init(self, frame: np.ndarray, box: boxes.Box) -> None: ...

    def update(self, frame: np.ndarray) -> tuple[Optional[boxes.Box], float]: ...


class Sam2Tracker:
    def __init__(self, cfg: Optional[dict] = None):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        full = load_config()
        cfg = cfg or full["tracker"]
        self.mask_threshold = float(cfg["mask_threshold"])
        self.min_mask_frac = float(cfg["min_mask_frac"])
        self.min_score = float(cfg["min_score"])
        self.device = pick_device(full["server"]["device"])
        self.dtype = torch.float16 if (self.device == "cuda" and full["server"]["fp16"]) else torch.float32

        self.processor = Sam2VideoProcessor.from_pretrained(cfg["model_id"])
        self.model = Sam2VideoModel.from_pretrained(cfg["model_id"], dtype=self.dtype)
        self.model.to(self.device).eval()

        self._session = None
        self._frame_idx = 0
        self._cond_idx = 0
        self.last_mask: Optional[np.ndarray] = None  # bool (H, W) from the last update, None when lost

    def init(self, frame: np.ndarray, box: boxes.Box) -> None:
        """Start a fresh tracking session on `frame`, prompted with `box`."""
        import torch

        # A fresh session per seed: re-acquire throws away the old object's memory.
        self._session = self.processor.init_video_session(inference_device=self.device, dtype=self.dtype)
        self._frame_idx = 0
        self._cond_idx = 0
        self.last_mask = None

        inputs = self._prepare(frame)
        self.processor.add_inputs_to_inference_session(
            inference_session=self._session,
            frame_idx=0,
            obj_ids=1,
            input_boxes=[[[float(v) for v in box]]],
            original_size=inputs.original_sizes[0],
        )
        with torch.inference_mode():
            self.model(inference_session=self._session, frame_idx=0, frame=inputs.pixel_values[0])
        self._frame_idx = 1

    def update(self, frame: np.ndarray) -> tuple[Optional[boxes.Box], float]:
        import torch

        self.last_mask = None
        if self._session is None:
            return None, 0.0

        inputs = self._prepare(frame)
        idx = self._frame_idx
        with torch.inference_mode():
            out = self.model(inference_session=self._session, frame_idx=idx, frame=inputs.pixel_values[0])
        self._frame_idx += 1
        self._prune(idx)

        masks = self.processor.post_process_masks(
            [out.pred_masks], original_sizes=inputs.original_sizes, binarize=False
        )[0]
        mask = (masks[0, 0] > self.mask_threshold).cpu().numpy()
        score = float(torch.sigmoid(out.object_score_logits.float()).flatten()[0]) if out.object_score_logits is not None else 1.0

        height, width = frame.shape[:2]
        if score < self.min_score or mask.sum() < self.min_mask_frac * width * height:
            return None, score
        self.last_mask = mask  # depth samples the target over the mask (free: already computed)
        return boxes.mask_to_box(mask), score

    def reset(self) -> None:
        self._session = None
        self.last_mask = None

    def warmup(self) -> None:
        blank = np.zeros((360, 480, 3), dtype=np.uint8)
        self.init(blank, (100.0, 100.0, 200.0, 200.0))
        self.update(blank)
        self.reset()

    def _prepare(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def _prune(self, current_idx: int) -> None:
        """Drop stored frames and non-conditioning outputs older than the memory window.
        The seed (conditioning) frame is always kept."""
        cutoff = current_idx - _KEEP_FRAMES
        if cutoff <= self._cond_idx:
            return
        session = self._session
        frames = session.processed_frames or {}
        for k in [k for k in frames if k < cutoff and k != self._cond_idx]:
            del frames[k]
        for obj_out in session.output_dict_per_obj.values():
            non_cond = obj_out["non_cond_frame_outputs"]
            for k in [k for k in non_cond if k < cutoff]:
                del non_cond[k]


class CsrtTracker:
    """OpenCV CSRT: CPU-only box tracker, no mask, no memory. Use if SAM 2 streaming is a
    time sink or for laptop-only runs. Needs opencv-contrib-python."""

    def __init__(self, cfg: Optional[dict] = None):
        self._tracker = None

    def init(self, frame: np.ndarray, box: boxes.Box) -> None:
        x1, y1, x2, y2 = box
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(frame, (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))))

    def update(self, frame: np.ndarray) -> tuple[Optional[boxes.Box], float]:
        if self._tracker is None:
            return None, 0.0
        ok, (x, y, w, h) = self._tracker.update(frame)
        if not ok or w <= 0 or h <= 0:
            return None, 0.0
        height, width = frame.shape[:2]
        return boxes.clip((x, y, x + w, y + h), width, height), 1.0

    def reset(self) -> None:
        self._tracker = None

    def warmup(self) -> None:
        pass


def make_tracker(cfg: Optional[dict] = None):
    cfg = cfg or load_config()["tracker"]
    return CsrtTracker(cfg) if cfg["backend"] == "csrt" else Sam2Tracker(cfg)
