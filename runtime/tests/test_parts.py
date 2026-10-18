"""Unit tests for the Step 1 runtime parts. No camera, audio device, network or model."""
from __future__ import annotations

import io
import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

from guide_agent.schemas import Frame
from haptics import protocol
from runtime import config as runtime_config
from runtime.camera import CameraHub, encode_frame
from runtime.glove_out import OFF, FakeGlove, GloveCommand, HardwareGlove, channel_on
from runtime.keyboard_reach import KEY_STATUS, KeyboardReach
from runtime.keys import StdinRouter, TogglePtt, route
from runtime.latency import measure, sessions
from runtime.view import View


# ------------------------------------------------------------------ KeyboardReach

class Cam:
    def latest_frame(self):
        return Frame(id="cam-9", image=b"jpeg", captured_at=1.0)


def make_reach():
    results, glove = [], FakeGlove()
    reach = KeyboardReach(lambda fn, *a: fn(*a), results.append, Cam(), glove)
    return reach, results, glove


@pytest.mark.parametrize("key,status", sorted(KEY_STATUS.items()))
def test_every_key_ends_the_reach_with_its_status(key, status):
    reach, results, glove = make_reach()
    sid = reach.start("red soda can", "the red can on the table")
    assert "REACH: red soda can (the red can on the table) RUNNING" in reach.describe()
    glove.apply(GloveCommand(on=frozenset({protocol.THUMB})))
    assert reach.key(key)
    (r,) = results
    assert r.session_id == sid and r.status == status and r.target == "red soda can"
    if status == "success":
        assert r.frame is None and r.closest_approach == "touching" and r.last_seen is None
    else:
        assert r.frame.id == "cam-9" and r.closest_approach == "near"
        assert (r.last_seen is None) == (status == "target_not_found")
    assert glove.command == OFF  # all-off on every end
    assert reach.session_id is None and not reach.key(key)  # nothing running any more


def test_pause_resume_cancel_and_keys_only_while_running():
    reach, results, glove = make_reach()
    assert not reach.key("s")  # no reach
    sid = reach.start("can", "the can")
    reach.pause(sid)
    assert "PAUSED" in reach.describe() and glove.command == OFF
    assert not reach.key("s")  # paused: the session can't end on a key
    reach.resume(sid)
    assert "RUNNING" in reach.describe()
    reach.pause("other-session")  # stale ids are ignored
    assert "RUNNING" in reach.describe()
    reach.cancel(sid)
    assert reach.session_id is None and "cancel" in reach.describe() and results == []
    assert not reach.key("x")


def test_results_go_through_dispatch():
    hops = []
    reach = KeyboardReach(lambda fn, *a: hops.append((fn, a)), lambda r: None)
    reach.start("can", "the can")
    reach.key("l")
    assert len(hops) == 1 and hops[0][1][0].status == "target_lost"


# ------------------------------------------------------------------ keys

KEYS = set(KEY_STATUS) | {"q"}


@pytest.mark.parametrize("line,typed,expected", [
    ("\n", False, ("enter", "")),
    ("  \n", True, ("enter", "")),
    ("s\n", False, ("key", "s")),
    ("S\n", True, ("key", "s")),
    ("q\n", True, ("key", "q")),
    ("get me the can\n", True, ("line", "get me the can")),
    ("get me the can\n", False, None),
    ("x\n", False, None),
    ("x\n", True, ("line", "x")),
])
def test_route(line, typed, expected):
    assert route(line, KEYS, typed) == expected


def test_stdin_router_and_toggle_ptt():
    got = []
    ptt = TogglePtt(clock=lambda: 5.0)
    ptt.start(got.append)
    keys = []
    lines = []
    router = StdinRouter(KEYS, keys.append, on_enter=ptt.toggle, on_line=lines.append,
                         stream=io.StringIO("\nhello there\ns\n\n"))
    router.start()._thread.join(2)
    assert got == [("ptt", True, 5.0), ("ptt", False, 5.0)]
    assert keys == ["s"] and lines == ["hello there"]


# ------------------------------------------------------------------ glove

def test_channel_on_steady_and_pulsing():
    cmd = GloveCommand(on=frozenset({protocol.THUMB}), pulse_hz={protocol.WRIST: 2.0})
    assert channel_on(cmd, protocol.THUMB, 0.3)
    assert not channel_on(cmd, protocol.PINKY, 0.3)
    assert channel_on(cmd, protocol.WRIST, 0.1) and not channel_on(cmd, protocol.WRIST, 0.3)  # 2 Hz: on 0-0.25 s
    assert not channel_on(OFF, protocol.WRIST, 0.1)


def test_fake_glove_draws_idle_and_active_panels():
    t = [0.1]
    glove = FakeGlove(clock=lambda: t[0])
    idle = np.zeros((300, 300, 3), np.uint8)
    glove.draw(idle, (10, 10))
    glove.apply(GloveCommand(on=frozenset({protocol.INDEX}), pulse_hz={protocol.WRIST: 2.0}))
    active = np.zeros((300, 300, 3), np.uint8)
    glove.draw(active, (10, 10))
    assert idle.any() and (active != idle).any()
    glove.all_off()
    assert glove.command == OFF


class FakeLink:
    """haptics.link.GloveLink's surface, recording calls."""

    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append(("start",))

    def set(self, levels, wrist_hz=0.0, pulse_hz=None):
        self.calls.append(("set", list(levels), wrist_hz, dict(pulse_hz or {})))

    def off(self):
        self.calls.append(("off",))

    def stop(self):
        self.calls.append(("stop",))


def test_hardware_glove_maps_steady_and_pulsing_motors_to_the_link():
    link = FakeLink()
    glove = HardwareGlove(link=link, on_level=200)
    assert link.calls == [("start",)]
    glove.apply(GloveCommand(on=frozenset({protocol.THUMB}), pulse_hz={protocol.PALM: 3.0, protocol.INDEX: 0.0}))
    _, levels, wrist_hz, pulse = link.calls[-1]
    expected = [0] * 8
    expected[protocol.THUMB] = expected[protocol.PALM] = 200
    assert levels == expected and wrist_hz == 0.0 and pulse == {protocol.PALM: 3.0}
    glove.apply(OFF)
    assert link.calls[-1] == ("set", [0] * 8, 0.0, {})
    glove.all_off()
    assert link.calls[-1] == ("off",)
    glove.close()
    assert link.calls[-1] == ("stop",)


def test_hardware_glove_without_a_port_says_where_to_set_it():
    with pytest.raises(ValueError, match="glove.port"):
        HardwareGlove()


def test_make_glove_and_tee():
    from runtime.glove_out import TeeGlove, make_glove

    assert isinstance(make_glove("fake"), FakeGlove)
    with pytest.raises(ValueError):
        make_glove("laser")
    link, panel = FakeLink(), FakeGlove()
    tee = TeeGlove(HardwareGlove(link=link), panel)
    tee.apply(GloveCommand(on=frozenset({protocol.INDEX})))
    assert panel.command.on == frozenset({protocol.INDEX}) and link.calls[-1][0] == "set"
    tee.all_off()
    assert panel.command == OFF and link.calls[-1] == ("off",)
    img = np.zeros((220, 200, 3), np.uint8)
    tee.draw(img, (0, 0))
    assert img.any()
    tee.close()
    assert link.calls[-1] == ("stop",)


# ------------------------------------------------------------------ camera

class FakeCapture:
    def __init__(self, image):
        self.image = image
        self.released = False

    def read(self):
        time.sleep(0.002)
        return True, self.image

    def release(self):
        self.released = True


def test_camera_hub_frames_after_warmup_downscaled_not_mirrored():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, :640] = (0, 0, 255)  # left half red: must stay on the left
    hub = CameraHub(FakeCapture(image), max_long_edge=1024, warmup_frames=5, clock=lambda: 42.0)
    assert hub.latest_frame() is None and hub.newest_bgr() is None
    hub.start()
    try:
        assert hub.ready.wait(2)
        f1, f2 = hub.latest_frame(), hub.latest_frame()
        assert (f1.id, f2.id) == ("cam-1", "cam-2") and f1.captured_at == 42.0
        decoded = cv2.imdecode(np.frombuffer(f1.image, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (576, 1024)
        assert decoded[288, 100, 2] > 200 and decoded[288, 900, 2] < 50
        assert hub.newest_bgr() is image
        time.sleep(0.05)
        assert hub.fps > 0
    finally:
        hub.stop()
    assert hub.capture.released


def test_encode_frame_keeps_small_images():
    small = np.zeros((240, 320, 3), np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_frame(small, 1024), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (240, 320)


# ------------------------------------------------------------------ config

def test_runtime_config_defaults_and_unknown_keys(tmp_path):
    cfg = runtime_config.load(tmp_path / "missing.yaml")
    assert cfg["camera"]["source"] == 1 and cfg["view"]["web_port"] == 8090
    assert runtime_config.load()["camera"]["warmup_frames"] > 0  # the checked-in file loads
    bad = tmp_path / "bad.yaml"
    bad.write_text("camera:\n  sorce: 0\n")
    with pytest.raises(ValueError, match="camera.sorce"):
        runtime_config.load(bad)


# ------------------------------------------------------------------ latency

def test_latency_pairs_each_utterance_with_its_first_cue_and_speech():
    def ev(event, wall, **kw):
        return {"event": event, "wall": wall, **kw}

    events = [
        {"event": "session", "cmd": "listen", "id": "100000"},
        ev("utterance_delivered", 1.0, chars=5, latency_ms=100),
        {"event": "session", "cmd": "voice_agent", "id": "120000"},
        ev("utterance_delivered", 10.5, chars=12, latency_ms=500),   # speech ended at 10.0
        ev("cue", 10.6, name="thinking", played=True),
        ev("playback_start", 12.0, speech_id="sp-1"),
        ev("playback_start", 15.0, speech_id="sp-2"),               # not the first reply
        ev("utterance_delivered", 20.2, chars=0, latency_ms=200),   # empty: ignored
        ev("playback_start", 21.0, speech_id="sp-3"),
        ev("utterance_delivered", 30.4, chars=4, latency_ms=400),   # ended at 30.0
        ev("playback_start", 31.0, speech_id="sp-4"),
    ]
    found = sessions(events)
    assert [s for s, _ in found] == ["120000"]
    m = measure(found[0][1])
    assert m["end -> utterance"] == [500, 400]
    assert m["end -> thinking cue"] == pytest.approx([600])
    assert m["end -> first speech"] == pytest.approx([2000, 1000])


# ------------------------------------------------------------------ view

def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_web_view_streams_mjpeg():
    port = free_port()
    view = View("web", port=port, max_fps=100)
    try:
        img = np.full((120, 160, 3), 128, np.uint8)
        stop = threading.Event()

        def feed():
            while not stop.is_set():
                view.show(img)

        t = threading.Thread(target=feed, daemon=True)
        t.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as page:
            assert b"/stream" in page.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=5) as stream:
            assert "multipart/x-mixed-replace" in stream.headers["Content-Type"]
            head = stream.read(200)
            assert b"--frame" in head and b"image/jpeg" in head
        stop.set()
        t.join(2)
    finally:
        view.close()


def test_none_view_returns_no_keys():
    view = View("none", max_fps=1000)
    assert view.show(None) == []
    view.close()
