"""Detect-then-track and reach-session status tests with fake models and a fake clock.
No GPU, weights or camera needed. Run: pytest object-detection -v"""

import numpy as np

from detector import Detection
from fake import FakeDetector, FakeTracker
from follow import TargetFollower
from session import HandReading, ReachSession

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
    "depth_near_band": 0.1,
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


def make(detector, tracker, hand_points, lost_frames=2, ambiguous_gap=0.0):
    """hand_points: list consumed one per frame; None = no hand; last value repeats."""
    clock = Clock()
    points = list(hand_points)

    def hand(frame):
        p = points.pop(0) if len(points) > 1 else points[0]
        return None if p is None else HandReading(point=p, box=(p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5))

    follower = TargetFollower(detector, tracker, lost_frames=lost_frames, ambiguous_gap=ambiguous_gap)
    session = ReachSession("paper", "", follower, CFG, hand=hand, clock=clock)
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
    det = FakeDetector([hit(), hit(box=(10, 10, 50, 50))])
    trk = FakeTracker([None, None])
    f = TargetFollower(det, trk, lost_frames=2)
    f.acquire(FRAME, "paper")
    f.step(FRAME)
    upd = f.step(FRAME)
    assert upd.reacquired and upd.box == (10, 10, 50, 50) and trk.inits == 2


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
