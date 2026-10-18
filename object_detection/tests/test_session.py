"""Detect-then-track and reach-session status tests with fake models and a fake clock.
No GPU, weights or camera needed. Run: pytest object_detection -v"""

import numpy as np

from object_detection.detector import Detection
from object_detection.fake import FakeDetector, FakeTracker
from object_detection.follow import TargetFollower
from object_detection.session import DepthSample, HandReading, ReachSession

FRAME = np.zeros((300, 300, 3), dtype=np.uint8)
BOX = (100.0, 100.0, 200.0, 200.0)
CFG = {
    "target_not_found_after_s": 3.0,
    "target_lost_after_s": 2.0,
    "hand_lost_after_s": 1.5,
    "stalled_after_s": 7.0,
    "timeout_s": 28.0,
    "stale_link_after_s": 0.5,
    "resume_reacquire_after_s": 3.0,
    "success_hold_frames": 3,
    "inferred_cover_frac": 0.5,
    "inferred_cover_frames": 2,
    "touching_dist_frac": 0.05,
    "near_dist_frac": 0.2,
}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def hit(score=0.6, box=BOX):
    return [Detection(box=box, score=score, label="paper")]


def make(detector, tracker, hand_points, lost_frames=2, ambiguous_gap=0.0, depth=None):
    """hand_points: list consumed one per frame; None = no hand; last value repeats."""
    clock = Clock()
    points = list(hand_points)

    def hand(frame):
        p = points.pop(0) if len(points) > 1 else points[0]
        return None if p is None else HandReading(point=p, box=(p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5))

    follower = TargetFollower(detector, tracker, lost_frames=lost_frames, ambiguous_gap=ambiguous_gap)
    session = ReachSession("paper", "", follower, CFG, hand=hand, depth=depth, clock=clock)
    return session, clock


def run(session, clock, frames, dt=0.1):
    for i in range(frames):
        if session.done:
            break
        session.handle_frame(FRAME, i)
        clock.t += dt
    return session.end_report


# --- follower ----------------------------------------------------------------

def test_acquire_seeds_tracker_with_top_box():
    det = FakeDetector([[Detection(BOX, 0.7, "a"), Detection((0, 0, 10, 10), 0.3, "b")]])
    trk = FakeTracker()
    acq = TargetFollower(det, trk).acquire(FRAME, "paper")
    assert acq.box == BOX and trk.box == BOX and trk.inits == 1
    assert acq.gap == 0.7 - 0.3


def test_reacquire_runs_detector_once_after_lost_frames():
    det = FakeDetector([hit(), [], hit()])
    trk = FakeTracker([None, None, None, None])
    f = TargetFollower(det, trk, lost_frames=2)
    f.acquire(FRAME, "paper")
    assert f.step(FRAME).box is None          # missing 1
    upd = f.step(FRAME)                        # missing 2 -> detector runs, finds nothing
    assert upd.reacquire_failed and det.calls == 2
    assert not f.step(FRAME).reacquire_failed  # no second retry in the same loss episode
    assert det.calls == 2


def test_reacquire_success_reseeds_tracker():
    det = FakeDetector([hit(), hit(box=(110, 110, 210, 210))])
    trk = FakeTracker([None, None])
    f = TargetFollower(det, trk, lost_frames=2)
    f.acquire(FRAME, "paper")
    f.step(FRAME)
    upd = f.step(FRAME)
    assert upd.reacquired and upd.box == (110, 110, 210, 210) and trk.inits == 2


def test_reacquire_ignores_lookalike_far_from_last_box():
    # The hand hides the target; the detector's top box is a second, similar object across the frame.
    det = FakeDetector([hit(), hit(score=0.8, box=(0, 0, 40, 40))])
    trk = FakeTracker([None, None])
    f = TargetFollower(det, trk, lost_frames=2, reacquire_min_iou=0.1)
    f.acquire(FRAME, "paper")
    f.step(FRAME)
    upd = f.step(FRAME)
    assert upd.reacquire_failed and upd.box is None
    assert trk.inits == 1 and f.last_box == BOX   # tracker not re-seeded on the look-alike


def test_reacquire_prefers_near_candidate_over_higher_far_one():
    near = (105.0, 95.0, 205.0, 195.0)
    det = FakeDetector([hit(), [Detection((0, 0, 40, 40), 0.8, "far"), Detection(near, 0.4, "near")]])
    trk = FakeTracker([None, None])
    f = TargetFollower(det, trk, lost_frames=2, reacquire_min_iou=0.1)
    f.acquire(FRAME, "paper")
    f.step(FRAME)
    upd = f.step(FRAME)
    assert upd.reacquired and upd.box == near and trk.box == near and upd.confidence == 0.4


def test_reacquire_ignores_nearby_box_that_does_not_overlap():
    # Real-model case: a weak box just above the occluded plant, center close but no overlap.
    det = FakeDetector([hit(), hit(score=0.18, box=(130.0, 80.0, 170.0, 99.0))])
    trk = FakeTracker([None, None])
    f = TargetFollower(det, trk, lost_frames=2, reacquire_min_iou=0.1)
    f.acquire(FRAME, "paper")
    f.step(FRAME)
    assert f.step(FRAME).reacquire_failed and trk.inits == 1


# --- session statuses ----------------------------------------------------------

def test_success_when_fingertip_stays_in_box():
    s, c = make(FakeDetector([hit()]), FakeTracker(), [(150, 150)])
    r = run(s, c, 10)
    assert r.status == "success" and not r.inferred
    assert r.closest_approach == "touching" and r.frame_id is None and r.last_seen is None


def test_target_not_found():
    s, c = make(FakeDetector([]), FakeTracker(), [(10, 10)])
    r = run(s, c, 50)
    assert r.status == "target_not_found" and r.last_seen is None


def test_target_lost_without_hand_cover():
    s, c = make(FakeDetector([hit()]), FakeTracker([BOX] + [None] * 100), [(10, 10)])
    r = run(s, c, 60)
    assert r.status == "target_lost"
    assert r.last_seen.region == "center-middle" and r.frame_id is not None


def test_inferred_success_when_hand_covers_vanished_target():
    # Hand moves into the box, then the target disappears under it; re-acquire finds nothing.
    s, c = make(FakeDetector([hit(), []]), FakeTracker([None] * 100), [(150, 150)], lost_frames=2)
    s.cfg = {**CFG, "success_hold_frames": 99}
    r = run(s, c, 20)
    assert r.status == "success" and r.inferred


def test_hand_lost():
    s, c = make(FakeDetector([hit()]), FakeTracker(), [(10, 10), None])
    r = run(s, c, 40)
    assert r.status == "hand_lost"


def test_stalled_when_hand_stops_getting_closer():
    s, c = make(FakeDetector([hit()]), FakeTracker(), [(10, 10)])
    r = run(s, c, 100)
    assert r.status == "stalled" and r.closest_approach == "far"


def test_timeout():
    s, c = make(FakeDetector([hit()]), FakeTracker(), [(10, 10)])
    s.cfg = {**CFG, "stalled_after_s": 999}
    r = run(s, c, 400)
    assert r.status == "timeout"


def test_ambiguous_when_enabled():
    det = FakeDetector([[Detection(BOX, 0.50, "a"), Detection((0, 0, 9, 9), 0.49, "b")]])
    s, c = make(det, FakeTracker(), [(150, 150)], ambiguous_gap=0.03)
    r = run(s, c, 3)
    assert r.status == "ambiguous" and r.candidate_count == 2


def test_cancel_and_stale_link():
    s, c = make(FakeDetector([hit()]), FakeTracker(), [(10, 10)])
    run(s, c, 2)
    assert s.cancel().status == "cancelled" and s.done
    s2, c2 = make(FakeDetector([hit()]), FakeTracker(), [(10, 10)])
    assert s2.end_as_stale_link().status == "stale_link"


def test_pause_freezes_timers_and_long_pause_reacquires():
    det = FakeDetector([hit(), hit()])
    s, c = make(det, FakeTracker(), [(10, 10)])
    run(s, c, 2)
    s.pause()
    assert s.handle_frame(FRAME, 99) is None    # frames ignored while paused
    c.t += 10.0                                 # longer than hand_lost / stalled windows
    s.resume()
    assert det.calls == 1
    run(s, c, 1)
    assert not s.done and det.calls == 2        # > resume_reacquire_after_s -> detector again


# --- depth hook ----------------------------------------------------------------

class ScriptedDepth:
    """session.DepthProbe: one scripted (gap, touching) per frame (the last repeats); gap None = no reading."""

    def __init__(self, script, last_seen="near", closest="near"):
        self.script = list(script)
        self.calls = []
        self._last_seen, self._closest = last_seen, closest

    def read(self, frame, frame_id, target_box, target_mask, hand):
        self.calls.append((frame_id, target_box, target_mask, hand))
        gap, touching = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return DepthSample(units="relative", hand=0.5, target=0.5 + (gap or 0), dz=gap, gap=gap,
                           touching=touching, timings_ms={"depth_model": 1.0, "depth_sample": 0.5})

    def last_seen_depth(self):
        return self._last_seen

    def closest_approach(self):
        return self._closest


def test_depth_touching_decides_success():
    depth = ScriptedDepth([(0.2, False)] * 3 + [(0.01, True)])
    s, clock = make(FakeDetector([hit()]), FakeTracker(), [(150, 150)], depth=depth)
    report = run(s, clock, 4)
    assert report.status == "success" and report.closest_approach == "touching"
    assert len(depth.calls) == 4


def test_in_box_without_depth_contact_is_not_success():
    """With a depth gap the 2D rule doesn't apply: the hand hovering over the target is not a touch."""
    depth = ScriptedDepth([(0.2, False)])
    s, clock = make(FakeDetector([hit()]), FakeTracker(), [(150, 150)], depth=depth)
    run(s, clock, 10)
    assert not s.done


def test_no_depth_reading_falls_back_to_the_2d_rule():
    depth = ScriptedDepth([(None, False)])
    s, clock = make(FakeDetector([hit()]), FakeTracker(), [(150, 150)], depth=depth)
    assert run(s, clock, 5).status == "success"


def test_progress_carries_depth_and_end_report_uses_it():
    depth = ScriptedDepth([(0.3, False)], last_seen="far", closest="far")
    s, clock = make(FakeDetector([hit()]), FakeTracker(), [(150, 150), None], depth=depth)
    first = s.handle_frame(FRAME, 0)
    assert first.depth["gap"] == 0.3 and first.depth["units"] == "relative" and "timings_ms" not in first.depth
    assert first.timings_ms["depth_model"] == 1.0 and "depth" in first.timings_ms
    assert first.frame_size == (300, 300) and first.hand_box == (145, 145, 155, 155)
    report = run(s, clock, 30)
    assert report.status == "hand_lost"
    assert report.last_seen.depth == "far" and report.closest_approach == "far"


def test_depth_gets_the_tracker_mask_when_there_is_one():
    class MaskTracker(FakeTracker):
        def update(self, frame):
            self.last_mask = np.ones(frame.shape[:2], bool)
            return super().update(frame)

    depth = ScriptedDepth([(0.2, False)])
    s, clock = make(FakeDetector([hit()]), MaskTracker(), [(150, 150)], depth=depth)
    run(s, clock, 3)
    acquire_call, track_call = depth.calls[0], depth.calls[1]
    assert acquire_call[2] is None              # the acquiring frame has a box but no tracker mask yet
    assert track_call[2] is not None and track_call[2].all()
