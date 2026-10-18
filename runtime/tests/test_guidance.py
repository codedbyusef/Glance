"""Guidance mapping from synthetic progress messages, and the live loop's all-off rules. No camera or server."""
from __future__ import annotations

import json

import pytest

from haptics import protocol
from runtime.glove_out import OFF, FakeGlove
from runtime.guidance import Guidance, command_from, distance_text
from runtime.live_guidance import FrameLog, ReachLoop

SIZE = (640, 480)
BOX = (300.0, 220.0, 340.0, 260.0)  # a can in the middle of the frame


def progress(hand, box=BOX, depth=None, session="s1", frame_id=1):
    return {"type": "progress", "session_id": session, "frame_id": frame_id, "frame_size": list(SIZE),
            "target_box": list(box) if box else None, "hand_point": list(hand) if hand else None,
            "depth": depth, "timings_ms": {"track": 30.0, "hand": 2.0, "depth_model": 20.0},
            "server_ms": 55.0, "client": {"encode_ms": 3.0, "network_ms": 20.0, "round_trip_ms": 75.0}}


def settle(g, hand, n=8, **kw):
    """The controller smooths the hand point; feed it a few times so it sits where we put it."""
    st = None
    for _ in range(n):
        st = g.update(progress(hand, **kw))
    return st


# ------------------------------------------------------------------ directions

@pytest.mark.parametrize("hand,word,channel", [
    ((100, 240), "RIGHT", protocol.PINKY),   # target to the right of the hand
    ((540, 240), "LEFT", protocol.THUMB),
    ((320, 450), "FORWARD", protocol.INDEX),  # table: target higher in the image = farther across the desk
    ((320, 40), "BACK", protocol.RING),
])
def test_table_mode_directions(hand, word, channel):
    st = settle(Guidance("table"), hand)
    assert st.words == [word] and st.cmd.on == frozenset({channel})
    assert st.cmd.pulse_hz[protocol.WRIST] == st.wrist_hz > 0


@pytest.mark.parametrize("hand,depth_error,words", [
    ((320, 450), None, ["UP"]),
    ((320, 40), None, ["DOWN"]),
    ((320, 240), 0.3, ["FORWARD"]),          # wall: forward/back only from depth
    ((320, 240), -0.3, ["BACK"]),
])
def test_wall_mode_directions(hand, depth_error, words):
    depth = None if depth_error is None else {"units": "metric", "gap": depth_error, "depth_error": depth_error,
                                              "hand": 0.5, "touching": False}
    st = settle(Guidance("wall"), hand, depth=depth)
    assert st.words == words


def test_two_axes_at_once_and_toggle_mode():
    g = Guidance("table")
    assert set(settle(g, (100, 450)).words) == {"RIGHT", "FORWARD"}
    assert g.toggle_mode() == "wall"
    assert set(settle(g, (100, 450)).words) == {"RIGHT", "UP"}


def test_inside_the_box_cues_down_in_table_mode():
    st = settle(Guidance("table"), (320, 240))
    assert st.words == ["DOWN"] and st.status == "aligned" and st.dx == 0 and st.dy == 0


# ------------------------------------------------------------------ deadband and hysteresis

def test_deadband_and_hysteresis():
    g = Guidance("table")
    # ex = 0.05 of the width: between the deadband (0.04) and the on threshold (0.06)
    just_outside = (300 - 0.05 * 640, 240)
    assert settle(g, just_outside).words == []           # an off channel needs 0.06 to turn on
    settle(g, (100, 240))                                  # far: RIGHT turns on
    assert settle(g, just_outside).words == ["RIGHT"]    # an on channel stays on down to the deadband
    assert settle(g, (300 - 0.03 * 640, 240)).words == []  # under the deadband: off


# ------------------------------------------------------------------ proximity

def test_wrist_pulse_rate_rises_as_the_hand_gets_closer():
    rates = []
    for x in (20, 120, 220, 280):
        rates.append(settle(Guidance("table"), (x, 240)).wrist_hz)
    assert rates == sorted(rates) and rates[0] < rates[-1]
    assert 1.0 <= rates[0] and rates[-1] <= 6.0


# ------------------------------------------------------------------ all off

@pytest.mark.parametrize("hand,box,status", [((320, 240), None, "target_lost"), (None, BOX, "hand_lost")])
def test_lost_is_all_off(hand, box, status):
    g = Guidance("table")
    settle(g, (100, 240))
    st = g.update(progress(hand, box=box))
    assert st.cmd == OFF and st.words == [] and st.status == status and st.dx is None


def test_touching_is_all_off():
    g = Guidance("table")
    settle(g, (320, 240))
    st = g.update(progress((320, 240), depth={"units": "relative", "gap": 0.01, "touching": True}))
    assert st.cmd == OFF and st.touching and st.status == "success"


def test_command_from_levels():
    levels = [0] * 8
    levels[protocol.THUMB], levels[protocol.WRIST] = 150, 200
    cmd = command_from(levels, 3.0)
    assert cmd.on == frozenset({protocol.THUMB}) and cmd.pulse_hz == {protocol.WRIST: 3.0}
    assert command_from([0] * 8, 3.0) == OFF


# ------------------------------------------------------------------ distance

def test_distance_text():
    metric = {"units": "metric", "gap": 0.12, "hand": 0.5}
    assert "gap +12 cm" in distance_text(metric, 0.1, 0.0, SIZE, 600.0)
    assert "lateral +5 / +0 cm" in distance_text(metric, 0.1, 0.0, SIZE, 600.0)  # 64 px at 0.5 m, f=600
    assert "focal_px not calibrated" in distance_text(metric, 0.1, 0.0, SIZE, None)
    assert "relative depth units" in distance_text({"units": "relative", "gap": 0.2}, 0.1, 0.0, SIZE, None)
    assert distance_text(None, 0.1, 0.0, SIZE, None).startswith("no depth gap")


def test_dz_and_frozen_come_from_the_depth_reading():
    st = Guidance("table").update(progress((100, 240), depth={"units": "relative", "gap": 0.3, "touching": False,
                                                              "target_frozen": True}))
    assert st.dz == 0.3 and st.units == "relative" and st.frozen


# ------------------------------------------------------------------ the live loop

class FakeAPI:
    def __init__(self, on_result, on_progress):
        self.on_result, self.on_progress = on_result, on_progress
        self.calls = []
        self.n = 0

    def start(self, target, refer=""):
        self.n += 1
        self.calls.append(("start", target))
        return f"s{self.n}"

    def pause(self, sid):
        self.calls.append(("pause", sid))

    def resume(self, sid):
        self.calls.append(("resume", sid))

    def cancel(self, sid):
        self.calls.append(("cancel", sid))

    def close(self):
        self.calls.append(("close",))


class Result:
    def __init__(self, session_id, status):
        self.session_id, self.status = session_id, status


def make_loop(tmp_path):
    glove = FakeGlove()
    log = FrameLog(tmp_path / "g.jsonl")
    loop = ReachLoop(FakeAPI, Guidance("table"), glove, log, capture_age=lambda: 12.0)
    loop.start("red can")
    return loop, glove, log


def test_loop_progress_drives_the_glove_and_logs_every_stage(tmp_path):
    loop, glove, log = make_loop(tmp_path)
    for i in range(5):
        loop.api.on_progress(progress((100, 240), session="s1", frame_id=i))
    assert glove.command.on == frozenset({protocol.PINKY})
    rec = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[-1])
    assert rec["words"] == ["RIGHT"] and rec["frame_id"] == 4
    assert {"capture", "encode", "network", "server", "track", "hand", "depth_model", "guidance"} <= set(rec["ms"])
    assert "5 progress frames" in log.summary() and "network" in log.summary()


def test_loop_all_off_on_pause_end_restart_error_and_close(tmp_path):
    loop, glove, _ = make_loop(tmp_path)
    on = lambda: loop.api.on_progress(progress((100, 240), session=loop.session_id))  # noqa: E731

    on()
    loop.toggle_pause()
    assert glove.command == OFF and ("pause", "s1") in loop.api.calls
    on()  # progress while paused is ignored
    assert glove.command == OFF
    loop.toggle_pause()
    on()
    assert glove.command != OFF

    loop.api.on_result(Result("s1", "success"))
    assert glove.command == OFF and loop.snapshot()["result"].status == "success"

    loop.restart()
    assert loop.session_id == "s2" and loop.snapshot()["result"] is None
    on()
    assert glove.command != OFF
    loop.api.on_progress(progress((100, 240), session="s1"))  # an old session's progress: ignored
    loop.guidance.update = lambda msg: 1 / 0  # a guidance bug
    on()
    assert glove.command == OFF and "guidance" in loop.snapshot()["error"]

    loop.close()
    assert glove.command == OFF and ("close",) in loop.api.calls


def test_start_failure_is_shown_not_raised(tmp_path):
    class Down(FakeAPI):
        def start(self, target, refer=""):
            raise RuntimeError("connection refused")

    loop = ReachLoop(Down, Guidance("table"), FakeGlove(), FrameLog(None))
    loop.start("red can")
    assert loop.session_id is None and "start failed" in loop.snapshot()["error"]
