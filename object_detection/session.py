"""One reach session: ACQUIRING -> TRACKING -> DONE, with PAUSED on the side.

Feeds each frame through TargetFollower (detect once, then track), reads the hand,
builds the per-frame result for the guidance controller, and decides when the reach
has ended and why. End statuses and report fields match guide_agent/CLAUDE.md section 10
(haptic_result) plus `cancelled` and the proposed `ambiguous`.

The hand and depth readers are injected. hand_detection/ supplies the hand
(`hand_reader()` below); depth is optional until Depth Anything V2 is added, and
without it success is "fingertip in the box for N frames in a row".
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from object_detection import boxes
from object_detection.follow import TargetFollower


@dataclass
class HandReading:
    point: tuple[float, float]                         # fingertip / band center, pixels
    box: Optional[boxes.Box] = None                    # hand extent, for the cover rule


HandFn = Callable[[np.ndarray], Optional[HandReading]]
DepthFn = Callable[[np.ndarray, list[tuple[float, float]]], list[Optional[float]]]


@dataclass
class FrameResult:
    """Per-frame result for the guidance controller (object_detection/CLAUDE.md)."""
    session_id: str
    frame_id: int
    target_box: Optional[boxes.Box]
    confidence: float
    hand_point: Optional[tuple[float, float]]
    target_depth: Optional[float]
    hand_depth: Optional[float]
    reacquired: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class LastSeen:
    region: str          # e.g. "right-middle" (image thirds)
    depth: str           # near | mid | far
    age_s: float


@dataclass
class EndReport:
    session_id: str
    status: str          # success | target_not_found | target_lost | hand_lost | stalled | timeout
                         # | stale_link | cancelled | ambiguous
    inferred: bool
    elapsed_s: float
    target: str
    last_seen: Optional[LastSeen]
    closest_approach: str                    # touching | near | far
    frame_id: Optional[int]                  # the last frame; None on success
    candidate_count: int = 0
    top_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Phase(Enum):
    ACQUIRING = auto()
    TRACKING = auto()
    PAUSED = auto()
    DONE = auto()


class ReachSession:
    def __init__(
        self,
        target: str,
        refer: str,
        follower: TargetFollower,
        cfg: dict,
        hand: Optional[HandFn] = None,
        depth: Optional[DepthFn] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.session_id = str(uuid.uuid4())
        self.target = target
        self.refer = refer      # accepted, ignored for now (one of each object on the demo table)
        self.follower = follower
        self.cfg = cfg
        self.hand = hand
        self.depth = depth
        self.clock = clock

        self.phase = Phase.ACQUIRING
        self.end_report: Optional[EndReport] = None

        now = clock()
        self.started_at = now
        self._paused_at: Optional[float] = None
        self._paused_total = 0.0
        self._phase_before_pause = Phase.ACQUIRING
        self.target_seen_at: Optional[float] = None
        self.hand_seen_at = now
        self.best_dist: Optional[float] = None
        self.best_dist_at = now
        self.success_streak = 0
        self._cover_history: deque[bool] = deque(maxlen=max(1, int(cfg["inferred_cover_frames"]) + 2))
        self._candidates: list = []
        self._size = (0, 0)

    # --- control ---------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def is_active(self) -> bool:
        """True while frames are expected (gates the stale_link timer)."""
        return self.phase in (Phase.ACQUIRING, Phase.TRACKING)

    def pause(self) -> None:
        """Freeze the loss timers. The caller zeroes the motors."""
        if self.is_active:
            self._phase_before_pause = self.phase
            self.phase = Phase.PAUSED
            self._paused_at = self.clock()

    def resume(self) -> None:
        if self.phase is not Phase.PAUSED:
            return
        paused_for = self.clock() - (self._paused_at or self.clock())
        self._shift_timers(paused_for)
        self._paused_at = None
        too_long = paused_for > self.cfg["resume_reacquire_after_s"]
        self.phase = Phase.ACQUIRING if too_long else self._phase_before_pause

    def cancel(self) -> EndReport:
        return self._end("cancelled", frame_id=None)

    def end_as_stale_link(self) -> EndReport:
        return self._end("stale_link", frame_id=None)

    # --- per frame -------------------------------------------------------------

    def handle_frame(self, frame: np.ndarray, frame_id: int) -> Optional[FrameResult]:
        """Advance by one frame. Returns the per-frame result, or None when the session
        is not active. Sets self.end_report when this frame ends the reach."""
        if not self.is_active:
            return None
        timings: dict[str, float] = {}
        now = self.clock()
        height, width = frame.shape[:2]
        self._size = (width, height)

        box: Optional[boxes.Box] = None
        confidence = 0.0
        reacquired = False
        reacquire_failed = False

        if self.phase is Phase.ACQUIRING:
            t = time.perf_counter()
            acq = self.follower.acquire(frame, self.target)
            timings["detect"] = (time.perf_counter() - t) * 1000
            self._candidates = acq.candidates
            if acq.ambiguous:
                self._end("ambiguous", frame_id)
                return self._result(frame_id, None, 0.0, None, None, None, False, timings)
            if acq.box is None:
                if self._elapsed(now) > self.cfg["target_not_found_after_s"]:
                    self._end("target_not_found", frame_id)
                return self._result(frame_id, None, 0.0, None, None, None, False, timings)
            box, confidence = acq.box, acq.candidates[0].score
            self.target_seen_at = now
            self.phase = Phase.TRACKING
        else:
            t = time.perf_counter()
            upd = self.follower.step(frame)
            timings["track"] = (time.perf_counter() - t) * 1000
            box, confidence = upd.box, upd.confidence
            reacquired, reacquire_failed = upd.reacquired, upd.reacquire_failed
            if upd.candidates:
                self._candidates = upd.candidates
            if box is not None:
                self.target_seen_at = now

        hand = None
        if self.hand is not None:
            t = time.perf_counter()
            hand = self.hand(frame)
            timings["hand"] = (time.perf_counter() - t) * 1000
        if hand is not None:
            self.hand_seen_at = now

        target_depth = hand_depth = None
        if self.depth is not None and box is not None and hand is not None:
            t = time.perf_counter()
            target_depth, hand_depth = self.depth(frame, [boxes.center(box), hand.point])
            timings["depth"] = (time.perf_counter() - t) * 1000

        # Progress: hand distance to the (last known) target box, as a fraction of the diagonal.
        ref_box = box or self.follower.last_box
        if hand is not None and ref_box is not None:
            dist = boxes.distance_to_box(ref_box, hand.point) / float(np.hypot(width, height))
            # Only a real improvement (1% of the diagonal) resets the stalled timer.
            if self.best_dist is None or dist < self.best_dist - 0.01:
                self.best_dist, self.best_dist_at = dist, now

        # Cover history for the inferred-success rule (hand hides the target at the end).
        # The glove band's box is smaller than the hand, so a hand point inside the
        # target box also counts as covering it.
        covered = hand is not None and ref_box is not None and (
            boxes.contains(ref_box, hand.point)
            or boxes.coverage(hand.box, ref_box) > self.cfg["inferred_cover_frac"]
        )
        self._cover_history.append(covered)

        in_box = box is not None and hand is not None and boxes.contains(box, hand.point)
        depth_ok = target_depth is None or hand_depth is None or \
            abs(target_depth - hand_depth) <= self.cfg.get("depth_near_band", 0.1)
        self.success_streak = self.success_streak + 1 if (in_box and depth_ok) else 0

        result = self._result(frame_id, box, confidence, hand, target_depth, hand_depth, reacquired, timings)
        self._check_end(now, frame_id, box, reacquire_failed)
        return result

    # --- internals -------------------------------------------------------------

    def _check_end(self, now: float, frame_id: int, box, reacquire_failed: bool) -> None:
        cfg = self.cfg
        recently_covered = sum(self._cover_history) >= cfg["inferred_cover_frames"]
        target_gone_s = now - self.target_seen_at if self.target_seen_at is not None else 0.0

        if self.success_streak >= cfg["success_hold_frames"]:
            self._end("success", frame_id)
        elif box is None and recently_covered and (reacquire_failed or target_gone_s > cfg["target_lost_after_s"]):
            self._end("success", frame_id, inferred=True)
        elif box is None and target_gone_s > cfg["target_lost_after_s"]:
            self._end("target_lost", frame_id)
        elif self.hand is not None and now - self.hand_seen_at > cfg["hand_lost_after_s"]:
            self._end("hand_lost", frame_id)
        elif self.best_dist is not None and now - self.best_dist_at > cfg["stalled_after_s"]:
            self._end("stalled", frame_id)
        elif self._elapsed(now) > cfg["timeout_s"]:
            self._end("timeout", frame_id)

    def _result(self, frame_id, box, confidence, hand, target_depth, hand_depth, reacquired, timings) -> FrameResult:
        return FrameResult(
            session_id=self.session_id,
            frame_id=frame_id,
            target_box=box,
            confidence=float(confidence),
            hand_point=hand.point if hand is not None else None,
            target_depth=target_depth,
            hand_depth=hand_depth,
            reacquired=reacquired,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
        )

    def _end(self, status: str, frame_id: Optional[int], inferred: bool = False) -> EndReport:
        if self.end_report is not None:
            return self.end_report
        now = self.clock()
        success = status == "success"
        last_box = self.follower.last_box
        last_seen = None
        if not success and last_box is not None and self.target_seen_at is not None:
            width, height = self._size
            last_seen = LastSeen(
                region=boxes.region_thirds(last_box, width, height),
                depth=self._depth_band(last_box),
                age_s=round(now - self.target_seen_at, 2),
            )
        self.phase = Phase.DONE
        self.end_report = EndReport(
            session_id=self.session_id,
            status=status,
            inferred=inferred,
            elapsed_s=round(self._elapsed(now), 2),
            target=self.target,
            last_seen=last_seen,
            closest_approach="touching" if success else self._approach(),
            frame_id=None if success else frame_id,
            candidate_count=len(self._candidates),
            top_scores=[round(c.score, 3) for c in self._candidates[:3]],
        )
        return self.end_report

    def _approach(self) -> str:
        if self.best_dist is None:
            return "far"
        if self.best_dist <= self.cfg["touching_dist_frac"]:
            return "touching"
        if self.best_dist <= self.cfg["near_dist_frac"]:
            return "near"
        return "far"

    def _depth_band(self, box) -> str:
        """Without a depth model, use how much of the frame the box fills: bigger = nearer."""
        width, height = self._size
        frac = boxes.area(box) / float(max(1, width * height))
        return "near" if frac > 0.15 else "mid" if frac > 0.03 else "far"

    def _elapsed(self, now: float) -> float:
        return now - self.started_at - self._paused_total

    def _shift_timers(self, paused_for: float) -> None:
        self._paused_total += paused_for
        self.hand_seen_at += paused_for
        self.best_dist_at += paused_for
        if self.target_seen_at is not None:
            self.target_seen_at += paused_for


def hand_reader() -> Optional[HandFn]:
    """Adapter for hand_detection/hand_marker.py (the glove's neon band).
    Returns None when that package is not available."""
    try:
        from hand_detection import hand_marker
    except ImportError:
        return None

    hand_marker.reset()

    def read(frame: np.ndarray) -> Optional[HandReading]:
        info = hand_marker.detect(frame)
        if info is None:
            return None
        return HandReading(point=tuple(info.point), box=tuple(float(v) for v in info.box))

    return read
