"""Reach API tests against the real server app with fake models (GLANCE_FAKE_MODELS=1).
Runs uvicorn on a free local port; no GPU, weights or camera needed. Run: pytest object_detection -v

FakeDetector always finds the middle third of the frame. A square in the band's color (the middle of
the configured hand HSV range) in the frame center puts the hand point inside that box -> success."""

import socket
import threading
import time

import cv2
import numpy as np
import pytest

from hand_detection import hand_marker
from object_detection import server
from object_detection.reach_api import ReachAPI, ReachResult, ReachStartError


def band_bgr() -> tuple[int, int, int]:
    """BGR color in the middle of the configured band HSV range, so the tests follow retuning."""
    lo, hi = (np.array(v, np.int32) for v in hand_marker._load_cfg()["hsv_ranges"][0])
    hsv = ((lo + hi) // 2).astype(np.uint8).reshape(1, 1, 3)
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


BLANK = np.zeros((360, 480, 3), dtype=np.uint8)
HAND_ON_TARGET = BLANK.copy()
cv2.rectangle(HAND_ON_TARGET, (220, 160), (260, 200), band_bgr(), -1)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(monkeypatch, depth: bool):
    """The real server app on a free port, fake models. depth=False keeps the 2D success rule (the fake
    depth map never closes the gap, so these plumbing tests could not succeed with it)."""
    import copy

    import uvicorn

    from object_detection.common import load_config

    cfg = copy.deepcopy(load_config())
    cfg["depth"]["enabled"] = depth
    cfg["reach_session"]["hand_lost_after_s"] = 1.5   # plumbing tests: keep the hand_lost paths fast
    cfg["reach_session"]["hand_first_seen_s"] = 1.5
    monkeypatch.setattr(server, "load_config", lambda: cfg)
    monkeypatch.setenv("GLANCE_FAKE_MODELS", "1")
    monkeypatch.delenv("GLANCE_TOKEN", raising=False)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    return f"ws://127.0.0.1:{port}/ws", srv, thread


@pytest.fixture
def live_server(monkeypatch):
    url, srv, thread = serve(monkeypatch, depth=False)
    yield url, srv
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def live_server_depth(monkeypatch):
    url, srv, thread = serve(monkeypatch, depth=True)
    yield url, srv
    srv.should_exit = True
    thread.join(timeout=5)


class Results:
    def __init__(self):
        self.items: list[ReachResult] = []
        self.event = threading.Event()

    def __call__(self, result: ReachResult) -> None:
        self.items.append(result)
        self.event.set()

    def wait(self, timeout: float = 5.0) -> ReachResult:
        assert self.event.wait(timeout), "no reach result"
        return self.items[-1]


def make_api(url, frames, results, progress=None):
    return ReachAPI(on_result=results, frame_source=lambda: frames[0], on_progress=progress, url=url, fps=30)


def test_success_when_hand_is_on_target(live_server):
    url, _ = live_server
    results, progress = Results(), []
    api = make_api(url, [HAND_ON_TARGET], results, progress.append)
    try:
        session_id = api.start("sheet of paper")
        r = results.wait()
        assert r.session_id == session_id and r.status == "success" and r.target == "sheet of paper"
        assert r.frame is None and r.closest_approach == "touching" and r.last_seen is None
        assert progress and progress[0]["target_box"] is not None and progress[0]["hand_point"] is not None
        c = progress[0]["client"]
        assert c["encode_ms"] > 0 and c["round_trip_ms"] >= c["network_ms"]
        assert api.active_session is None
    finally:
        api.close()


def test_hand_lost_returns_last_frame_and_last_seen(live_server):
    url, _ = live_server
    results = Results()
    api = make_api(url, [BLANK], results)
    try:
        api.start("sheet of paper")
        r = results.wait()
        assert r.status == "hand_lost"
        assert r.last_seen is not None and r.last_seen.region == "center-middle"
        assert r.frame is not None and r.frame.id.startswith("vis-")
        w, h = api.frame_size  # frames are resized to object_detection/config.yaml's frame size before sending
        assert cv2.imdecode(np.frombuffer(r.frame.image, np.uint8), cv2.IMREAD_COLOR).shape == (h, w, 3)
    finally:
        api.close()


def test_cancel_reports_cancelled(live_server):
    url, _ = live_server
    results = Results()
    api = make_api(url, [BLANK], results)
    try:
        session_id = api.start("sheet of paper")
        time.sleep(0.2)
        api.cancel(session_id)
        r = results.wait()
        assert r.status == "cancelled" and r.session_id == session_id
        assert len(results.items) == 1
    finally:
        api.close()


def test_pause_holds_the_reach_and_resume_continues(live_server):
    url, _ = live_server
    results = Results()
    api = make_api(url, [BLANK], results)
    try:
        session_id = api.start("sheet of paper")
        time.sleep(0.2)
        api.pause(session_id)
        time.sleep(2.5)   # longer than hand_lost (1.5 s), stale_link (0.5 s) and no_reply_s (2 s)
        assert not results.items and api.active_session == session_id
        api.resume(session_id)
        assert results.wait().status == "hand_lost"
    finally:
        api.close()


def test_calls_for_another_session_are_ignored(live_server):
    url, _ = live_server
    results = Results()
    api = make_api(url, [HAND_ON_TARGET], results)
    try:
        api.start("sheet of paper")
        api.cancel("not-this-one")
        assert results.wait().status == "success"
    finally:
        api.close()


def test_progress_carries_depth_hand_box_and_frame_size(live_server_depth):
    url, _ = live_server_depth
    results, progress = Results(), []
    api = make_api(url, [HAND_ON_TARGET], results, progress.append)
    try:
        session_id = api.start("red soda can")
        deadline = time.monotonic() + 5
        while not any(p.get("depth") for p in progress) and time.monotonic() < deadline:
            time.sleep(0.05)
        p = next(p for p in progress if p.get("depth"))
        assert p["frame_size"] == list(api.frame_size)
        assert p["hand_box"] is not None and p["band_area_px"] > 0
        d = p["depth"]
        assert d["units"] in ("relative", "metric") and d["gap"] is not None and d["touching"] is False
        assert {"hand", "target", "dz", "band", "target_frozen", "inconsistent"} <= set(d)
        assert "depth_model" in p["timings_ms"] and "depth_sample" in p["timings_ms"]
        api.cancel(session_id)
        assert results.wait().status == "cancelled"
    finally:
        api.close()


def test_server_gone_mid_reach_is_stale_link(live_server):
    url, srv = live_server
    results = Results()
    api = make_api(url, [BLANK], results)
    try:
        api.start("sheet of paper")
        time.sleep(0.2)
        srv.should_exit = True
        srv.force_exit = True
        r = results.wait(timeout=6.0)
        assert r.status == "stale_link" and r.frame is not None and r.closest_approach == "far"
    finally:
        api.close()


def test_start_fails_when_nothing_is_listening():
    api = ReachAPI(on_result=Results(), frame_source=lambda: BLANK, url=f"ws://127.0.0.1:{free_port()}/ws")
    try:
        with pytest.raises(ReachStartError):
            api.start("sheet of paper")
        assert api.active_session is None
    finally:
        api.close()
