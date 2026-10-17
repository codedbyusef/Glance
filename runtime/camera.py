"""One camera capture thread for the whole runtime (CLAUDE.md section 6). Newest frame only, thread-safe.

    newest_bgr()    -> the newest BGR image (ReachAPI's frame_source, the live view)
    latest_frame()  -> the agent's Frame: JPEG, long edge <= max_long_edge, not mirrored (Mic and controller)

Capture logic follows guide_agent/scripts/webcam_chat.py: CAP_DSHOW on Windows and a few warm-up frames while
auto-exposure settles.
"""
from __future__ import annotations

import collections
import sys
import threading
import time

import cv2
import numpy as np

from guide_agent.schemas import Frame


def open_capture(source, width: int, height: int, dshow: bool = True):
    """cv2.VideoCapture for a webcam index or a video path/URL."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if (dshow and sys.platform == "win32") else cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {source!r} (wrong index? try --camera 0)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def encode_frame(image: np.ndarray, max_long_edge: int, quality: int = 85) -> bytes:
    """JPEG with the long edge capped at max_long_edge."""
    h, w = image.shape[:2]
    scale = max_long_edge / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class CameraHub:
    def __init__(self, capture, max_long_edge: int = 1024, warmup_frames: int = 45, jpeg_quality: int = 85,
                 clock=time.time):
        """capture: anything with read() -> (ok, bgr) and release(), e.g. cv2.VideoCapture."""
        self.capture = capture
        self.max_long_edge = max_long_edge
        self.warmup_frames = warmup_frames
        self.jpeg_quality = jpeg_quality
        self.clock = clock
        self.ready = threading.Event()  # set after the warm-up frames
        self._lock = threading.Lock()
        self._image: np.ndarray | None = None
        self._at = 0.0
        self._count = 0
        self._frame_ids = 0
        self._stamps: collections.deque[float] = collections.deque(maxlen=30)
        self._running = False
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cam_cfg: dict, max_long_edge: int, source=None) -> "CameraHub":
        cap = open_capture(cam_cfg["source"] if source is None else source, cam_cfg["width"], cam_cfg["height"],
                           cam_cfg.get("dshow", True))
        return cls(cap, max_long_edge, cam_cfg["warmup_frames"], cam_cfg["jpeg_quality"])

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> "CameraHub":
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.capture.release()
        except Exception:
            pass

    def _run(self) -> None:
        while self._running:
            ok, image = self.capture.read()
            if not ok or image is None:
                time.sleep(0.01)
                continue
            now = self.clock()
            with self._lock:
                self._image, self._at = image, now
                self._count += 1
                self._stamps.append(time.monotonic())
                count = self._count
            if count >= self.warmup_frames:
                self.ready.set()

    # ---------------------------------------------------------------- readers (any thread)

    def newest_bgr(self) -> np.ndarray | None:
        """The newest image, or None before the warm-up ends. Treat it as read-only (copy before drawing)."""
        if not self.ready.is_set():
            return None
        with self._lock:
            return self._image

    def latest_frame(self) -> Frame | None:
        """The agent's Frame (guide_agent/CLAUDE.md section 5): JPEG, capped long edge, capture time."""
        if not self.ready.is_set():
            return None
        with self._lock:
            image, at = self._image, self._at
            self._frame_ids += 1
            frame_id = f"cam-{self._frame_ids}"
        if image is None:
            return None
        return Frame(id=frame_id, image=encode_frame(image, self.max_long_edge, self.jpeg_quality), captured_at=at)

    @property
    def fps(self) -> float:
        with self._lock:
            stamps = list(self._stamps)
        if len(stamps) < 2 or stamps[-1] == stamps[0]:
            return 0.0
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])
