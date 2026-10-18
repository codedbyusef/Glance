"""ReachHaptic (the agent's HapticLoop on ReachAPI) with a fake ReachAPI: every status, pause/resume/cancel,
all-off on every exit, result conversion, and results arriving through dispatch."""
from __future__ import annotations

import pytest

from guide_agent.interfaces import HapticStartError
from guide_agent.schemas import HAPTIC_STATUSES, Frame, HapticResult, LastSeen
from haptics import protocol
from object_detection.reach_api import ReachFrame, ReachResult, ReachStartError
from object_detection.session import LastSeen as VisionLastSeen
from runtime.glove_out import OFF, FakeGlove
from runtime.guidance import Guidance
from runtime.reach_haptic import ReachHaptic, to_haptic_result


class FakeReachAPI:
    def __init__(self, on_result, on_progress):
        self.on_result, self.on_progress = on_result, on_progress
        self.calls = []
        self.n = 0
        self.fail = False

    def start(self, target, refer=""):
        self.calls.append(("start", target, refer))
        if self.fail:
            raise ReachStartError("could not start reach: connection refused")
        self.n += 1
        return f"s{self.n}"

    def pause(self, sid):
        self.calls.append(("pause", sid))

    def resume(self, sid):
        self.calls.append(("resume", sid))

    def cancel(self, sid):
        self.calls.append(("cancel", sid))

    def close(self):
        self.calls.append(("close",))


class Dispatch:
    def __init__(self):
        self.calls = []

    def __call__(self, fn, *args):
        self.calls.append((fn, args))
        fn(*args)


def result(sid, status, frame=True):
    return ReachResult(
        session_id=sid, status=status, inferred=status == "success", elapsed_s=6.1, target="red soda can",
        last_seen=None if status in ("success", "target_not_found") else VisionLastSeen("right-middle", "near", 2.3),
        closest_approach="touching" if status == "success" else "near",
        frame=None if status == "success" or not frame else ReachFrame("vis-311", b"jpeg", 1234.5))


def progress(sid, hand=(100, 240)):
    return {"type": "progress", "session_id": sid, "frame_id": 1, "frame_size": [640, 480],
            "target_box": [300, 220, 340, 260], "hand_point": list(hand), "depth": None, "timings_ms": {}}


@pytest.fixture
def rig():
    got, dispatch, glove = [], Dispatch(), FakeGlove()
    frames = []
    h = ReachHaptic(FakeReachAPI, dispatch, got.append, Guidance("table"), glove,
                    on_frame=lambda msg, st: frames.append((msg, st)))
    return h, got, dispatch, glove, frames


@pytest.mark.parametrize("status", HAPTIC_STATUSES)
def test_every_status_is_converted_and_dispatched(rig, status):
    h, got, dispatch, glove, _ = rig
    sid = h.start("red soda can", "the red can on the table")
    assert h.api.calls[0] == ("start", "red soda can", "the red can on the table")
    h.api.on_progress(progress(sid))
    assert glove.command != OFF
    h.api.on_result(result(sid, status))
    assert glove.command == OFF  # all-off on every end
    (r,) = got
    assert len(dispatch.calls) == 1 and dispatch.calls[0][0] == got.append  # through dispatch, once
    assert isinstance(r, HapticResult) and r.session_id == sid and r.status == status
    assert r.target == "red soda can" and r.elapsed_s == 6.1 and r.inferred == (status == "success")
    if status == "success":
        assert r.frame is None and r.closest_approach == "touching" and r.last_seen is None
    else:
        assert r.frame == Frame(id="vis-311", image=b"jpeg", captured_at=1234.5)
        if status != "target_not_found":
            assert r.last_seen == LastSeen(region="right-middle", depth="near", age_s=2.3)
    assert "reach: none" in h.describe()


def test_cancelled_results_are_dropped_and_ambiguous_maps_to_not_found(rig):
    h, got, _, glove, _ = rig
    sid = h.start("can", "")
    h.cancel(sid)
    assert ("cancel", sid) in h.api.calls and glove.command == OFF
    h.api.on_result(result(sid, "cancelled"))
    assert got == []
    sid2 = h.start("can", "")
    h.api.on_result(result(sid2, "ambiguous"))
    assert got[0].status == "target_not_found"


def test_start_failure_raises_haptic_start_error(rig):
    h, _, _, glove, _ = rig
    h.api.fail = True
    with pytest.raises(HapticStartError, match="connection refused"):
        h.start("can", "")
    assert h.session_id is None and glove.command == OFF and "start failed" in h.snapshot()["error"]


def test_pause_resume_and_stale_ids(rig):
    h, _, _, glove, frames = rig
    sid = h.start("can", "")
    h.api.on_progress(progress(sid))
    assert glove.command.on == frozenset({protocol.PINKY})
    h.pause(sid)
    assert ("pause", sid) in h.api.calls and glove.command == OFF and "PAUSED" in h.describe()
    h.api.on_progress(progress(sid))  # in flight when the pause landed: ignored
    assert glove.command == OFF and len(frames) == 1
    h.resume(sid)
    assert ("resume", sid) in h.api.calls and "RUNNING" in h.describe()
    h.api.on_progress(progress(sid))
    assert glove.command != OFF and len(frames) == 2
    h.pause("old-session")
    h.cancel("old-session")
    assert ("pause", "old-session") not in h.api.calls and ("cancel", "old-session") not in h.api.calls
    h.api.on_progress(progress("old-session"))  # another session's progress: ignored
    assert len(frames) == 2


def test_all_off_on_a_guidance_error_and_on_close(rig):
    h, _, _, glove, _ = rig
    sid = h.start("can", "")
    h.api.on_progress(progress(sid))
    h.guidance.update = lambda msg: 1 / 0
    h.api.on_progress(progress(sid))
    assert glove.command == OFF and "guidance" in h.snapshot()["error"]
    h.close()
    assert glove.command == OFF and ("close",) in h.api.calls


def test_start_resets_guidance_and_toggle_mode(rig):
    h, _, _, _, _ = rig
    h.start("can", "")
    assert h.toggle_mode() == "wall" and h.snapshot()["state"].mode == "wall"
    h.start("can", "")
    assert h.snapshot()["state"].mode == "wall"  # the mode survives a new reach


def test_to_haptic_result_without_frame():
    r = to_haptic_result(result("s1", "stale_link", frame=False))
    assert r.frame is None and r.status == "stale_link"
