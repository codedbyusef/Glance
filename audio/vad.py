"""Voice activity detection and endpointing (CLAUDE.md section 7, decision 3).

VadClassifier labels 20 ms frames voiced or not. Endpointer is a pure state machine over those labels and frame
times, so it is tested with synthetic sequences.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
FRAME_S = 0.02
FRAME_SAMPLES = 320
START_WINDOW_S = 0.3  # speech_start_ms of voiced audio must fall within this window
KEEP_RATIO = 0.5  # only voiced frames with over half of the last 300 ms voiced keep a capture going (decision 20)


class VadClassifier(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...  # 320 int16 samples


class WebRtcVad:
    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self.vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.vad.is_speech(np.ascontiguousarray(frame, dtype=np.int16).tobytes(), SAMPLE_RATE)


class FakeVad:
    """Voiced when the frame's RMS is over a threshold. Tests build speech from loud and quiet blocks."""

    def __init__(self, threshold: float = 500.0):
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) > self.threshold


class StartDetector:
    """Speech starts when speech_start_ms of voiced frames fall within the last 300 ms."""

    def __init__(self, speech_start_ms: int):
        self.window = deque(maxlen=round(START_WINDOW_S / FRAME_S))
        self.needed = max(1, round(speech_start_ms / 1000 / FRAME_S))

    def push(self, voiced: bool, t: float) -> float | None:
        """Add one frame starting at t. Returns the start time of the first voiced frame in the window on a start."""
        self.window.append((voiced, t))
        if sum(v for v, _ in self.window) >= self.needed:
            return next(ft for v, ft in self.window if v)
        return None

    def ratio(self) -> float:
        """Voiced share of the last 300 ms."""
        return sum(v for v, _ in self.window) / self.window.maxlen

    def reset(self) -> None:
        self.window.clear()


@dataclass
class EndpointResult:
    kind: str  # end | short | no_speech | max
    first_voiced: float | None = None
    last_voiced_end: float | None = None


class Endpointer:
    """Tracks one VAD-endpointed capture.

    - started: speech has begun (VAD start, or the start rule met after a wake detection).
    - An end needs end_silence_ms after the last voiced frame that had over half of the last 300 ms voiced, so
      scattered noise doesn't hold a capture open. It never ends on a voiced frame, so speech that resumes after a
      pause isn't cut while the ratio builds back up.
    - A finished burst shorter than min_utterance_ms (first voiced to last voiced) doesn't end the capture: it waits up
      to no_speech_after_wake_s for speech to resume (same utterance), else reports "short".
    - Without speech, the capture reports "no_speech" after no_speech_after_wake_s from its start.
    - max_utterance_s from the capture start always ends it ("max").
    """

    def __init__(self, cfg, start_t: float, first_voiced: float | None = None, last_voiced_end: float | None = None):
        self.cfg = cfg
        self.start_t = start_t
        self.first_voiced = first_voiced
        self.last_voiced_end = last_voiced_end
        self.detector = StartDetector(cfg.speech_start_ms)
        if first_voiced is not None:  # started by a VAD start: the last 300 ms were speech
            self.detector.window.extend([(True, first_voiced)] * self.detector.window.maxlen)
        self.waiting_until: float | None = None  # set while waiting after a too-short burst

    @property
    def started(self) -> bool:
        return self.first_voiced is not None

    def span_s(self) -> float:
        if self.first_voiced is None or self.last_voiced_end is None:
            return 0.0
        return self.last_voiced_end - self.first_voiced

    def push(self, voiced: bool, t: float) -> EndpointResult | None:
        """One frame starting at t. Returns a result when the capture is over."""
        cfg = self.cfg
        frame_end = t + FRAME_S
        start = self.detector.push(voiced, t)
        if not self.started:
            if start is not None:
                self.first_voiced = start
                self.last_voiced_end = frame_end
        elif self.waiting_until is not None:
            if start is not None:  # speech resumed: same utterance
                self.waiting_until = None
                self.last_voiced_end = frame_end
            elif frame_end >= self.waiting_until - 1e-9:
                return EndpointResult("short", self.first_voiced, self.last_voiced_end)
        elif voiced:
            if self.detector.ratio() > KEEP_RATIO:
                self.last_voiced_end = frame_end
        elif frame_end - self.last_voiced_end >= cfg.end_silence_ms / 1000 - 1e-9:
            if self.span_s() * 1000 >= cfg.min_utterance_ms - 1e-6:
                return EndpointResult("end", self.first_voiced, self.last_voiced_end)
            self.waiting_until = self.last_voiced_end + cfg.no_speech_after_wake_s
            self.detector.reset()
        if frame_end - self.start_t >= cfg.max_utterance_s - 1e-9:
            return EndpointResult("max", self.first_voiced, self.last_voiced_end)
        if not self.started and frame_end - self.start_t >= cfg.no_speech_after_wake_s - 1e-9:
            return EndpointResult("no_speech")
        return None


def frames(block: np.ndarray) -> list[np.ndarray]:
    """Split an 80 ms block (1280 samples) into four 20 ms frames."""
    return [block[i:i + FRAME_SAMPLES] for i in range(0, len(block) - FRAME_SAMPLES + 1, FRAME_SAMPLES)]
