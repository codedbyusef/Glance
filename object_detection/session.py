"""One reach session: ACQUIRING -> TRACKING -> DONE, with PAUSED on the side.

Feeds each frame through TargetFollower (detect once, then track), reads the hand,
builds the per-frame result for the guidance controller, and decides when the reach
has ended and why. End statuses and report fields match guide_agent/CLAUDE.md section 10
(haptic_result) plus `cancelled` and the proposed `ambiguous`.

The hand and depth readers are injected. hand_detection/ supplies the hand
(`hand_reader()` below). Depth is a per-reach DepthProbe (object_detection/depth_hook.py,
Depth Anything V2 + depth_estimation's DepthTracker). When a frame has a depth reading,
success is depth's `touching` (hand in the box and the depth gap closed, held); without
one it is "hand point in the box for N frames in a row".
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol

import numpy as np

from object_detection import boxes
from object_detection.follow import TargetFollower


@dataclass
class HandReading:
    point: tuple[float, float]                         # fingertip / band center, pixels
    box: Optional[boxes.Box] = None                    # hand extent, for the cover rule
    area_px: Optional[float] = None                    # band area (grows as the hand nears the camera)


HandFn = Callable[[np.ndarray], Optional[HandReading]]


@dataclass
class DepthSample:
    """One frame's depth reading, as the progress message carries it (depth_estimation's DepthReading).
    Larger = farther; dz = target - hand; gap = dz - contact offset (~0 at contact)."""
    units: str                                         # metric (meters) | relative (fraction of frame range)
    hand: Optional[float] = None
    target: Optional[float] = None
    dz: Optional[float] = None
    gap: Optional[float] = None
    depth_error: Optional[float] = None                # gap in haptics controller units (wall mode)
    band: Optional[str] = None                         # near | mid | far
    touching: bool = False
    target_frozen: bool = False
    inconsistent: bool = False
    map_age: int = 0                                   # frames since the depth map was computed
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("timings_ms")
        return d


class DepthProbe(Protocol):
    """Depth for one reach (object_detection/depth_hook.py). Called once per frame."""

    def read(self, frame: np.ndarray, frame_id: int, target_box: Optional[boxes.Box],
             target_mask: Optional[np.ndarray], hand: Optional[HandReading]) -> Optional[DepthSample]: ...

    def last_seen_depth(self) -> str: ...      # near | mid | far

    def closest_approach(self) -> str: ...     # touching | near | far


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
    frame_size: tuple[int, int] = (0, 0)               # (width, height) of the frame the pixels refer to
    hand_box: Optional[boxes.Box] = None               # the band's box
    band_area_px: Optional[float] = None
    depth: Optional[dict] = None                       # DepthSample.to_dict(), None without a reading


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
        depth: Optional[DepthProbe] = None,
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
        self._hand_ever_seen = False      # before the first sighting, hand_first_seen_s applies instead
        self.best_dist: Optional[float] = None
        self.best_dist_at = now
        self.success_streak = 0
        self._depth_touching = False       # depth decided "touching" on the latest frame
        self._depth_seen = False           # some frame had a depth gap: depth owns last_seen.depth/closest_approach
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
        target_mask: Optional[np.ndarray] = None
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
                return self._result(frame_id, None, 0.0, None, None, False, timings)
            if acq.box is None:
                if self._elapsed(now) > self.cfg["target_not_found_after_s"]:
                    self._end("target_not_found", frame_id)
                return self._result(frame_id, None, 0.0, None, None, False, timings)
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
                if not reacquired:  # the tracker's own mask for this box (SAM 2 only; None for CSRT)
                    target_mask = getattr(self.follower.tracker, "last_mask", None)

        hand = None
        if self.hand is not None:
            t = time.perf_counter()
            hand = self.hand(frame)
            timings["hand"] = (time.perf_counter() - t) * 1000
        if hand is not None:
            self.hand_seen_at = now
            self._hand_ever_seen = True

        # Progress: hand distance to the (last known) target box, as a fraction of the diagonal.
        ref_box = box or self.follower.last_box

        sample: Optional[DepthSample] = None
        if self.depth is not None and ref_box is not None:
            t = time.perf_counter()
            sample = self.depth.read(frame, frame_id, box, target_mask, hand)
            timings["depth"] = (time.perf_counter() - t) * 1000
            if sample is not None:
                timings.update({k: v for k, v in sample.timings_ms.items() if k != "depth"})
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

        # Success: depth's `touching` when this frame has a depth gap (it already needs the hand in the box,
        # held for its hold_frames); otherwise the 2D rule, hand point in the box for success_hold_frames.
        in_box = box is not None and hand is not None and boxes.contains(box, hand.point)
        if sample is not None and sample.gap is not None:
            self._depth_seen = True
            self._depth_touching = sample.touching
            self.success_streak = 0
        else:
            self._depth_touching = False
            self.success_streak = self.success_streak + 1 if in_box else 0

        result = self._result(frame_id, box, confidence, hand, sample, reacquired, timings)
        self._check_end(now, frame_id, box, reacquire_failed)
        return result

    # --- internals -------------------------------------------------------------

    def _check_end(self, now: float, frame_id: int, box, reacquire_failed: bool) -> None:
        cfg = self.cfg
        recently_covered = sum(self._cover_history) >= cfg["inferred_cover_frames"]
        target_gone_s = now - self.target_seen_at if self.target_seen_at is not None else 0.0

        if self._depth_touching or self.success_streak >= cfg["success_hold_frames"]:
            self._end("success", frame_id)
        elif box is None and recently_covered and (reacquire_failed or target_gone_s > cfg["target_lost_after_s"]):
            self._end("success", frame_id, inferred=True)
        elif box is None and target_gone_s > cfg["target_lost_after_s"]:
            self._end("target_lost", frame_id)
        elif self.hand is not None and now - self.hand_seen_at > self._hand_limit_s():
            self._end("hand_lost", frame_id)
        elif self.best_dist is not None and now - self.best_dist_at > cfg["stalled_after_s"]:
            self._end("stalled", frame_id)
        elif self._elapsed(now) > cfg["timeout_s"]:
            self._end("timeout", frame_id)

    def _hand_limit_s(self) -> float:
        """Dropout allowed once the hand was seen; a longer wait before the first sighting (the user is
        still bringing the hand into view while the target is being detected)."""
        lost = float(self.cfg["hand_lost_after_s"])
        return lost if self._hand_ever_seen else float(self.cfg.get("hand_first_seen_s", lost))

    def _result(self, frame_id, box, confidence, hand, sample, reacquired, timings) -> FrameResult:
        return FrameResult(
            session_id=self.session_id,
            frame_id=frame_id,
            target_box=box,
            confidence=float(confidence),
            hand_point=hand.point if hand is not None else None,
            target_depth=sample.target if sample is not None else None,
            hand_depth=sample.hand if sample is not None else None,
            reacquired=reacquired,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            frame_size=self._size,
            hand_box=hand.box if hand is not None else None,
            band_area_px=hand.area_px if hand is not None else None,
            depth=sample.to_dict() if sample is not None else None,
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
                depth=self.depth.last_seen_depth() if self._depth_seen else self._depth_band(last_box),
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
            closest_approach="touching" if success else (
                self.depth.closest_approach() if self._depth_seen else self._approach()),
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
        return HandReading(point=tuple(info.point), box=tuple(float(v) for v in info.box),
                           area_px=float(info.area_px))

    return read
