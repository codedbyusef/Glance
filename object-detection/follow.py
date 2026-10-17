"""Detect once, then track: Grounding DINO finds the target, SAM 2 follows it.

1. acquire(): Grounding DINO runs on the newest frame with the `target` phrase.
   The top box at or above the score threshold seeds SAM 2 on that same frame.
2. step(): every later frame, SAM 2 updates the mask -> box + confidence.
3. If the mask is gone for `lost_frames` frames in a row, Grounding DINO runs once
   to re-acquire, taking only a box that overlaps where the target was last seen (a
   look-alike elsewhere in the frame is ignored). If that fails, `reacquire_failed` is set
   and the session decides between target_lost and success(inferred) (session.py).

Model-free: takes any Detector and Tracker, so tests and laptop runs can pass fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import boxes
from detector import Detection, Detector, score_gap
from tracker import Tracker


@dataclass
class Acquisition:
    box: Optional[boxes.Box]
    candidates: list[Detection] = field(default_factory=list)
    gap: Optional[float] = None       # top score minus second; None with <2 candidates
    ambiguous: bool = False           # top two too close to choose (proposed status)


@dataclass
class FollowUpdate:
    box: Optional[boxes.Box]          # None when the target is not visible this frame
    confidence: float
    reacquired: bool = False          # the detector ran this frame and found it again
    reacquire_failed: bool = False    # the detector ran this frame and did not find it
    candidates: list[Detection] = field(default_factory=list)


class TargetFollower:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        lost_frames: int = 4,
        ambiguous_gap: float = 0.0,
        reacquire_min_iou: float = 0.1,
    ):
        self.detector = detector
        self.tracker = tracker
        self.lost_frames = lost_frames
        self.ambiguous_gap = ambiguous_gap
        self.reacquire_min_iou = reacquire_min_iou
        self.phrase = ""
        self.last_box: Optional[boxes.Box] = None
        self._missing = 0
        self._retried = False   # one detector retry per loss episode

    def acquire(self, frame: np.ndarray, phrase: str) -> Acquisition:
        """Run the detector; on a hit, seed the tracker on this frame."""
        self.phrase = phrase
        candidates = self.detector.detect(frame, phrase)
        gap = score_gap(candidates)
        if not candidates:
            return Acquisition(box=None, candidates=candidates, gap=gap)

        ambiguous = gap is not None and gap < self.ambiguous_gap
        top = candidates[0].box
        if not ambiguous:
            self.tracker.init(frame, top)
            self.last_box = top
            self._missing = 0
            self._retried = False
        return Acquisition(box=None if ambiguous else top, candidates=candidates, gap=gap, ambiguous=ambiguous)

    def step(self, frame: np.ndarray) -> FollowUpdate:
        box, confidence = self.tracker.update(frame)
        if box is not None:
            self.last_box = box
            self._missing = 0
            self._retried = False
            return FollowUpdate(box=box, confidence=confidence)

        self._missing += 1
        if self._missing < self.lost_frames or self._retried:
            return FollowUpdate(box=None, confidence=confidence)

        # Mask gone for several frames: one re-acquire attempt with the detector.
        self._retried = True
        candidates = self.detector.detect(frame, self.phrase)
        top = self._near_last_box(candidates)
        if top is None:
            # Nothing near where the target was. A look-alike elsewhere is not taken; the tracker
            # keeps its memory and may still pick the target up when the hand moves away.
            return FollowUpdate(box=None, confidence=0.0, reacquire_failed=True, candidates=candidates)
        self.tracker.init(frame, top.box)
        self.last_box = top.box
        self._missing = 0
        self._retried = False
        return FollowUpdate(box=top.box, confidence=top.score, reacquired=True, candidates=candidates)

    def _near_last_box(self, candidates: list[Detection]) -> Optional[Detection]:
        """Best-scoring candidate overlapping the last box by at least reacquire_min_iou.
        Candidates are sorted best first."""
        if self.last_box is None:
            return candidates[0] if candidates else None
        for d in candidates:
            if boxes.iou(d.box, self.last_box) >= self.reacquire_min_iou:
                return d
        return None
