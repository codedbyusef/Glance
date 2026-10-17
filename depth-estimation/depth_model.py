"""Depth Anything V2 Small: one BGR frame in, one depth map out.

Every map this module returns uses the same convention, whichever model runs:
    larger value = farther from the camera
    metric model   -> meters
    relative model -> fraction of the frame's depth range (5th-95th percentile),
                      flipped from the model's native "larger = closer"

One inference per frame. Hand and target are both sampled from that one map
(see depth_tracker.py), never from two different frames or two inferences.

    model = DepthModel()                  # reads config.yaml
    dmap  = model.infer(bgr)              # float32 (H, W), same size as bgr
    cache = DepthCache(model)             # adds every_n_frames + "same frame, same map"
    dmap, age = cache.get(bgr, frame_id)

GLANCE_FAKE_MODELS=1 swaps in a fake model (no torch, no weights), the same
switch object-detection uses, for wiring checks and tests.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

import depth_config as settings

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def fake_enabled() -> bool:
    return os.environ.get("GLANCE_FAKE_MODELS", "") not in ("", "0", "false", "False")


def model_input_size(height: int, width: int, longest: int, multiple: int = 14) -> tuple[int, int]:
    """(h, w) keeping aspect ratio, longest side ~= `longest`, both multiples of 14 (ViT patch)."""
    scale = longest / max(height, width)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    return h, w


def normalize_relative(raw: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Relative model output (larger = closer) -> larger = farther, scaled by the
    frame's low-high percentile range. Not clipped: the hand is often closer than
    the 95th percentile and must keep its own value."""
    lo, hi = np.percentile(raw, [low_pct, high_pct])
    span = float(hi - lo)
    if span < 1e-6:
        return np.zeros_like(raw, dtype=np.float32)
    return ((hi - raw) / span).astype(np.float32)


class DepthModel:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or settings.load()
        self.kind = settings.units(self.cfg)
        self.fake = fake_enabled()
        self.last_timings: dict[str, float] = {}
        self.device = "cpu"
        self._model = None
        self._torch = None
        self._dtype = None
        if not self.fake:
            self._load()

    # ---- loading -------------------------------------------------------------
    def _load(self) -> None:
        import torch  # imported here so tests and the fake path never need torch
        from transformers import AutoModelForDepthEstimation

        mcfg = self.cfg["model"]
        self._torch = torch
        self.device = _pick_device(torch, mcfg["device"])
        use_fp16 = bool(mcfg["fp16"]) and self.device == "cuda"
        self._dtype = torch.float16 if use_fp16 else torch.float32
        model_id = mcfg["ids"][self.kind]
        # Load in fp32, then cast: works across transformers versions (torch_dtype vs dtype kwarg).
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self._model.to(device=self.device, dtype=self._dtype).eval()

    def warmup(self, runs: int = 2) -> None:
        h, w = self.cfg["frame"]["frame_height"], self.cfg["frame"]["frame_width"]
        dummy = np.zeros((h, w, 3), np.uint8)
        for _ in range(runs):
            self.infer(dummy)

    # ---- inference -----------------------------------------------------------
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 (H, W, 3) -> float32 (H, W), larger = farther (see module doc)."""
        if self.fake:
            return self._fake_infer(bgr)

        torch = self._torch
        t0 = time.perf_counter()
        H, W = bgr.shape[:2]
        h, w = model_input_size(H, W, int(self.cfg["model"]["input_size"]))
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, self._dtype)
        t1 = time.perf_counter()

        with torch.inference_mode():
            pred = self._model(pixel_values=tensor).predicted_depth          # (1, h, w)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        raw = pred.cpu().numpy().astype(np.float32)
        if self.kind == "relative":
            n = self.cfg["normalize"]
            dmap = normalize_relative(raw, n["low_pct"], n["high_pct"])
        else:
            dmap = raw                                                     # meters, larger = farther
        t3 = time.perf_counter()
        self.last_timings = {
            "depth_pre": (t1 - t0) * 1000,
            "depth_infer": (t2 - t1) * 1000,
            "depth_post": (t3 - t2) * 1000,
        }
        return dmap

    def _fake_infer(self, bgr: np.ndarray) -> np.ndarray:
        """A tilted table: farther toward the top of the image. Brighter pixels
        read a little closer, so drawn objects show up in wiring checks."""
        t0 = time.perf_counter()
        H, W = bgr.shape[:2]
        rows = np.linspace(0.9, 0.5, H, dtype=np.float32)[:, None]
        table = np.repeat(rows, W, axis=1)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dmap = table - 0.1 * gray
        if self.kind == "relative":
            dmap = dmap / 0.5
        self.last_timings = {"depth_infer": (time.perf_counter() - t0) * 1000}
        return dmap.astype(np.float32)

    def gpu_memory_mb(self) -> Optional[float]:
        if self._torch is not None and self.device == "cuda":
            return self._torch.cuda.max_memory_allocated() / 1e6
        return None


class DepthCache:
    """Runs the model at most once per frame_id, and only every Nth frame.

    get() returns (map, age_frames). Age 0 = computed on this frame. Asking twice
    for the same frame_id never runs the model twice.
    """

    def __init__(self, model: DepthModel, every_n: Optional[int] = None):
        self.model = model
        self.every_n = max(1, int(every_n or model.cfg["model"]["every_n_frames"]))
        self._map: Optional[np.ndarray] = None
        self._map_frame: Optional[int] = None

    def reset(self) -> None:
        self._map = None
        self._map_frame = None

    def get(self, bgr: np.ndarray, frame_id: int) -> tuple[np.ndarray, int]:
        stale = (
            self._map is None
            or self._map.shape != bgr.shape[:2]
            or frame_id < self._map_frame                      # ids restarted (new clip / session)
            or frame_id - self._map_frame >= self.every_n
        )
        if stale:
            self._map = self.model.infer(bgr)
            self._map_frame = frame_id
        return self._map, frame_id - self._map_frame


def _pick_device(torch, wanted: str) -> str:
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
