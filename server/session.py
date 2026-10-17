"""Per-reach session state machine: ACQUIRING -> TRACKING -> DONE.

Owns the thresholds from CLAUDE.md "Object detection and tracking (settled)":
when to call the detector vs. the tracker, and when a reach has ended and why.
Thresholds live in config/settings.yaml (reach_session) and config/controller.yaml
(the depth "near" band), not here.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum, auto
from typing import Optional

from glance.schemas import Approach, DepthBand, EndStatus, FrameResult, LastSeen, ReachReport
from server import config
from server.models import depth as depth_model
from server.models import detector as detector_model
from server.models import hand as hand_model
from server.models import tracker as tracker_model
from server.timing import timed


class _Phase(Enum):
    ACQUIRING = auto()
    TRACKING = auto()
    PAUSED = auto()
    DONE = auto()


class ReachSession:
    def __init__(self, target: str, refer: str):
        self.session_id = str(uuid.uuid4())
        self.target = target
        self.refer = refer

        cfg = config.load("settings.yaml")
        self._cfg = cfg["reach_session"]
        self._near_depth_band = config.load("controller.yaml")["deadband"]["dz"]

        self.phase = _Phase.ACQUIRING
        self.end_report: Optional[ReachReport] = None

        now = time.monotonic()
        self.started_at = now
        self._paused_at: Optional[float] = None
        self.last_target_box: Optional[tuple[float, float, float, float]] = None
        self.last_target_seen_at = now
        self.last_hand_point: Optional[tuple[float, float]] = None
        self.last_hand_seen_at = now
        self.success_streak = 0
        self.overlap_streak = 0
        self.best_error: Optional[float] = None
        self._frame_width = 0
        self._frame_height = 0

    @property
    def done(self) -> bool:
        return self.phase is _Phase.DONE

    @property
    def is_active(self) -> bool:
        """True while frames are expected (used to gate the stale_link timer)."""
        return self.phase in (_Phase.ACQUIRING, _Phase.TRACKING)

    def pause(self) -> None:
        if self.phase is not _Phase.DONE:
            self.phase = _Phase.PAUSED
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self.phase is not _Phase.PAUSED:
            return
        paused_for = time.monotonic() - (self._paused_at or time.monotonic())
        self.phase = _Phase.ACQUIRING if paused_for > self._cfg["resume_reacquire_after_s"] else _Phase.TRACKING
        self._paused_at = None

    def cancel(self) -> None:
        """Client-initiated stop. No ReachReport -- the client already knows why."""
        self.phase = _Phase.DONE

    def handle_frame(self, frame: bytes, frame_id: int, frame_width: int, frame_height: int) -> FrameResult:
        """Advances the state machine by one frame. Sets self.end_report (and
        flips self.done) when this frame ends the reach.
        """
        timings: dict[str, float] = {}
        now = time.monotonic()
        self._frame_width, self._frame_height = frame_width, frame_height

        if self.phase is _Phase.ACQUIRING:
            with timed(timings, "detect"):
                candidates = detector_model.detect(frame, self.target)
            if not candidates:
                if now - self.started_at > self._cfg["target_not_found_after_s"]:
                    self._end("target_not_found", FrameResult(frame_id=frame_id, timings_ms=timings))
                return FrameResult(frame_id=frame_id, timings_ms=timings)

            # TODO: when >1 candidate comes back, hand them + self.refer to the
            # VLM disambiguator (not built yet) instead of always taking the top
            # confidence box.
            self.last_target_box = candidates[0]["box"]
            self.last_target_seen_at = now
            self.phase = _Phase.TRACKING
            # Falls through so this same frame also gets a tracker/hand/depth read.

        with timed(timings, "track"):
            box = tracker_model.track(frame, self.last_target_box)
        with timed(timings, "hand"):
            hand_point = hand_model.locate_hand(frame)

        if box is not None:
            self.last_target_box = box
            self.last_target_seen_at = now
        target_lost = box is None and (now - self.last_target_seen_at) > self._cfg["target_lost_after_s"]

        if hand_point is not None:
            self.last_hand_point = hand_point
            self.last_hand_seen_at = now
        hand_lost = hand_point is None and (now - self.last_hand_seen_at) > self._cfg["hand_lost_after_s"]

        with timed(timings, "depth"):
            target_depth = depth_model.estimate_depth(frame, patch=box) if box is not None else None
            hand_depth = depth_model.estimate_depth(frame, patch=hand_point) if hand_point is not None else None

        # "Overlap" is hand-point-inside-box, not area coverage -- hand.py only
        # returns a point, not a box, so "covers over half the box" (CLAUDE.md)
        # is approximated as in/out until hand pose gives us a bounding region.
        reference_box = box if box is not None else self.last_target_box
        in_box = _point_in_box(hand_point, reference_box)
        self.overlap_streak = self.overlap_streak + 1 if in_box else 0

        if box is not None and in_box and _depth_near(target_depth, hand_depth, self._near_depth_band):
            self.success_streak += 1
        else:
            self.success_streak = 0

        error = _combined_error(box, hand_point, target_depth, hand_depth, frame_width, frame_height)
        if error is not None and (self.best_error is None or error < self.best_error):
            self.best_error = error
            self._best_error_at = now
        elif self.best_error is None:
            self._best_error_at = now

        result = FrameResult(
            frame_id=frame_id,
            target_box=box,
            hand_point=hand_point,
            hand_depth=hand_depth,
            target_depth=target_depth,
            timings_ms=timings,
        )

        if self.success_streak >= self._cfg["success_hold_frames"]:
            self._end("success", result)
        elif box is None and self.overlap_streak >= self._cfg["inferred_overlap_frames"]:
            self._end("success", result, inferred=True)
        elif target_lost:
            self._end("target_lost", result)
        elif hand_lost:
            self._end("hand_lost", result)
        elif self.best_error is not None and (now - self._best_error_at) > self._cfg["stalled_after_s"]:
            self._end("stalled", result)
        elif (now - self.started_at) > self._cfg["timeout_s"]:
            self._end("timeout", result)

        return result

    def end_as_stale_link(self) -> ReachReport:
        self._end("stale_link", None)
        return self.end_report

    def _end(self, status: EndStatus, last_frame: Optional[FrameResult], inferred: bool = False) -> None:
        now = time.monotonic()
        last_seen = None
        if status != "success" and self.last_target_box is not None:
            last_seen = LastSeen(
                region=_region_thirds(self.last_target_box, self._frame_width, self._frame_height),
                depth=_depth_band(self.best_error),
                age_s=now - self.last_target_seen_at,
            )
        self.phase = _Phase.DONE
        self.end_report = ReachReport(
            session_id=self.session_id,
            status=status,
            inferred=inferred,
            elapsed_s=now - self.started_at,
            last_seen=last_seen,
            closest_approach=_approach(self.best_error) if status != "success" else "touching",
            last_frame=last_frame if status != "success" else None,
        )


def _point_in_box(point, box) -> bool:
    if point is None or box is None:
        return False
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _depth_near(target_depth: Optional[float], hand_depth: Optional[float], band: float) -> bool:
    if target_depth is None or hand_depth is None:
        return False
    return abs(target_depth - hand_depth) <= band


def _combined_error(box, hand_point, target_depth, hand_depth, frame_width, frame_height) -> Optional[float]:
    """Single scalar used only to track "has this reach stopped improving"
    for the `stalled` status -- not fed to the haptic controller, which does
    its own dx/dy/dz in glance/vision/controller.py.
    """
    if box is None or hand_point is None:
        return None
    x1, y1, x2, y2 = box
    target_cx, target_cy = (x1 + x2) / 2, (y1 + y2) / 2
    hand_x, hand_y = hand_point
    dx = (target_cx - hand_x) / frame_width
    dy = (target_cy - hand_y) / frame_height
    dz = 0.0 if target_depth is None or hand_depth is None else (target_depth - hand_depth)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _approach(error: Optional[float]) -> Approach:
    if error is None:
        return "far"
    if error < 0.1:
        return "touching"
    if error < 0.3:
        return "near"
    return "far"


def _depth_band(error: Optional[float]) -> DepthBand:
    if error is None:
        return "far"
    if error < 0.1:
        return "near"
    if error < 0.3:
        return "mid"
    return "far"


def _region_thirds(box, frame_width: int, frame_height: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    horiz = "left" if cx < frame_width / 3 else "right" if cx > 2 * frame_width / 3 else "center"
    vert = "top" if cy < frame_height / 3 else "bottom" if cy > 2 * frame_height / 3 else "middle"
    return f"{horiz}-{vert}"
