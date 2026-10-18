"""Step 2: vision + depth + guidance, standalone (no agent). Point the camera at the glove and an object and
watch the commands, dx/dy/dz and the distance estimate live.

    python -m runtime.live_guidance --target "red can"                     # GLANCE_VISION_URL, GLANCE_TOKEN
    python -m runtime.live_guidance --target "sheet of paper" --view web

Wiring check without a GPU: run a local fake server first,
    $env:GLANCE_FAKE_MODELS = "1"; .venv\\Scripts\\python -m uvicorn object_detection.server:app --port 8000
    $env:GLANCE_VISION_URL = "ws://127.0.0.1:8000/ws"
(the fake detector always "finds" the middle third of the frame; the fake depth never reports touching).

Keys (window, or the letter + Enter in the terminal):
    r restart the reach    space / p pause or resume    m toggle table / wall    q quit

Logs one JSON line per progress frame (runtime/logs/guidance-*.jsonl) with per-stage timings: capture (frame
age when sent), encode, network, detect/track, hand, depth, guidance. Prints an FPS and latency summary on exit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config as runtime_config
from .glove_out import GloveOutput, glove_from_config
from .guidance import Guidance, GuidanceState
from .keys import StdinRouter
from .view import FpsCounter, Timings, View, blank, fit_width, text_panel

KEYS = {"r", "p", " ", "m", "q"}
LOG_DIR = Path(__file__).resolve().parent / "logs"

# progress timings_ms keys -> log stage names
SERVER_STAGES = {"detect": "detect", "track": "track", "hand": "hand", "depth_model": "depth_model",
                 "band_mask": "band_mask", "depth_sample": "depth_sample"}


class FrameLog:
    """One JSON line per progress frame, plus per-stage values for the exit summary."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        self.stages: dict[str, list[float]] = {}
        self.count = 0
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        now = time.monotonic()
        with self._lock:
            self.count += 1
            self.first = self.first if self.first is not None else now
            self.last = now
            for k, v in record.get("ms", {}).items():
                if v is not None:
                    self.stages.setdefault(k, []).append(float(v))
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    def summary(self) -> str:
        with self._lock:
            if not self.count:
                return "no progress frames"
            span = (self.last - self.first) if self.last and self.first and self.last > self.first else 0.0
            fps = (self.count - 1) / span if span else 0.0
            lines = [f"{self.count} progress frames, {fps:.1f} FPS"]
            for name, values in self.stages.items():
                vs = sorted(values)
                p95 = vs[min(len(vs) - 1, int(round(0.95 * (len(vs) - 1))))]
                lines.append(f"  {name:14s} p50 {statistics.median(vs):7.1f} ms   p95 {p95:7.1f} ms   n={len(vs)}")
            return "\n".join(lines)


def frame_record(msg: dict, state: GuidanceState, capture_age: Optional[float]) -> dict:
    """One JSONL line for a progress frame: per-stage ms (capture, encode, network, server stages, guidance)
    and what the guidance made of it."""
    t = msg.get("timings_ms", {})
    client = msg.get("client", {})
    ms = {"capture": capture_age, "encode": client.get("encode_ms"),
          "network": client.get("network_ms"), "server": msg.get("server_ms")}
    ms.update({name: t[key] for key, name in SERVER_STAGES.items() if key in t})
    ms["guidance"] = round(state.ms, 3)
    ms["round_trip"] = client.get("round_trip_ms")
    return {
        "t": round(time.time(), 3), "session_id": msg.get("session_id"), "frame_id": msg.get("frame_id"),
        "ms": ms, "mode": state.mode, "words": state.words, "wrist_hz": round(state.wrist_hz, 2),
        "status": state.status, "dx": state.dx, "dy": state.dy, "dz": state.dz, "units": state.units,
        "touching": state.touching, "frozen": state.frozen,
        "target_box": msg.get("target_box"), "hand_point": msg.get("hand_point"),
        "depth": msg.get("depth"),
    }


def log_frame(log: FrameLog, timings: Timings, msg: dict, state: GuidanceState,
              capture_age: Optional[float]) -> None:
    rec = frame_record(msg, state, capture_age)
    for k, v in rec["ms"].items():
        if v is not None:
            timings.set(k, v)
    log.write(rec)


class ReachLoop:
    """One reach at a time. Progress -> guidance -> glove; every end, pause, restart or error -> all-off.

    api_factory(on_result, on_progress) -> an object with ReachAPI's start/pause/resume/cancel/close.
    Callbacks arrive on the API's thread; the state the view reads is guarded by a lock."""

    def __init__(self, api_factory: Callable, guidance: Guidance, glove: GloveOutput, log: FrameLog,
                 timings: Optional[Timings] = None, capture_age: Callable[[], Optional[float]] = lambda: None):
        self.guidance = guidance
        self.glove = glove
        self.log = log
        self.timings = timings or Timings()
        self.capture_age = capture_age
        self._lock = threading.Lock()
        self.target = ""
        self.refer = ""
        self.session_id: Optional[str] = None
        self.paused = False
        self.progress: Optional[dict] = None
        self.state = GuidanceState(mode=guidance.mode)
        self.result = None
        self.error = ""
        self.api = api_factory(self.on_result, self.on_progress)

    # ---------------------------------------------------------------- controls (UI thread)

    def start(self, target: str, refer: str = "") -> None:
        self.target, self.refer = target, refer
        self.restart()

    def restart(self) -> None:
        """Start (again). ReachAPI.start drops a reach still running, without a result."""
        self.glove.all_off()
        with self._lock:
            self.session_id, self.paused, self.progress, self.result, self.error = None, False, None, None, ""
            self.guidance.reset()
            self.state = GuidanceState(mode=self.guidance.mode)
        try:
            sid = self.api.start(self.target, self.refer)
        except Exception as e:  # ReachStartError: server down, wrong URL or token
            with self._lock:
                self.error = f"start failed: {e}"
            return
        with self._lock:
            self.session_id = sid

    def toggle_pause(self) -> None:
        sid = self.session_id
        if sid is None:
            return
        if self.paused:
            self.api.resume(sid)
            self.paused = False
        else:
            self.paused = True
            self.glove.all_off()
            self.api.pause(sid)

    def toggle_mode(self) -> str:
        with self._lock:
            mode = self.guidance.toggle_mode()
            self.state = GuidanceState(mode=mode)
        return mode

    def close(self) -> None:
        self.glove.all_off()
        try:
            self.api.close()
        finally:
            self.glove.all_off()

    # ---------------------------------------------------------------- callbacks (API thread)

    def on_progress(self, msg: dict) -> None:
        try:
            with self._lock:
                if self.paused or msg.get("session_id") != self.session_id:
                    return
                state = self.guidance.update(msg)
                self.progress, self.state = msg, state
            self.glove.apply(state.cmd)
            log_frame(self.log, self.timings, msg, state, self.capture_age())
        except Exception as e:  # never leave a motor on after a bug
            self.glove.all_off()
            with self._lock:
                self.error = f"guidance: {e!r}"

    def on_result(self, result) -> None:
        self.glove.all_off()
        with self._lock:
            if result.session_id == self.session_id:
                self.result, self.session_id = result, None

    def snapshot(self) -> dict:
        with self._lock:
            return {"progress": self.progress, "state": self.state, "result": self.result, "paused": self.paused,
                    "session_id": self.session_id, "error": self.error, "target": self.target}


# ------------------------------------------------------------------ drawing

def draw_reach(canvas: np.ndarray, snap: dict, frame_size: tuple[int, int]) -> None:
    """Target box, hand point and an arrow from the hand to the box (pixels scaled to the canvas)."""
    p, st = snap["progress"], snap["state"]
    if not p:
        return
    fw, fh = tuple(p.get("frame_size") or frame_size)
    sx, sy = canvas.shape[1] / max(1, fw), canvas.shape[0] / max(1, fh)
    box, hand, hbox = p.get("target_box"), p.get("hand_point"), p.get("hand_box")
    if box:
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if st.touching else (0, 200, 255) if st.frozen else (255, 160, 0)
        cv2.rectangle(canvas, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), color, 2)
    if hbox:
        x1, y1, x2, y2 = hbox
        cv2.rectangle(canvas, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 255, 255), 1)
    if hand:
        hx, hy = int(hand[0] * sx), int(hand[1] * sy)
        cv2.circle(canvas, (hx, hy), 7, (0, 0, 255), -1)
        if box:
            tx = int(min(max(hand[0], box[0]), box[2]) * sx)
            ty = int(min(max(hand[1], box[1]), box[3]) * sy)
            if (tx, ty) != (hx, hy):
                cv2.arrowedLine(canvas, (hx, hy), (tx, ty), (0, 0, 255), 2, tipLength=0.2)


def guidance_lines(snap: dict) -> list:
    st = snap["state"]
    d = (snap["progress"] or {}).get("depth") or {}

    def f(v, spec="+.2f"):
        return "-" if v is None else format(v, spec)

    words = " ".join(st.words) or "-"
    lines = [
        (f"COMMAND: {words}   wrist {st.wrist_hz:.1f} Hz   [{st.mode}]", (0, 220, 255)),
        f"dx {f(st.dx)}  dy {f(st.dy)}  dz {f(st.dz)} {st.units or ''}   status {st.status}",
        f"distance: {st.distance}",
        f"depth: hand {f(d.get('hand'), '.2f')} target {f(d.get('target'), '.2f')} band {d.get('band') or '-'}"
        f"{'  FROZEN' if st.frozen else ''}{'  INCONSISTENT' if d.get('inconsistent') else ''}",
    ]
    if st.touching:
        lines.append(("TOUCHING", (0, 255, 0)))
    return lines


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live vision + depth + guidance, no agent.")
    parser.add_argument("--target", required=True, help='detector phrase, e.g. "red can"')
    parser.add_argument("--refer", default="", help="what tells it apart (accepted, unused by the server for now)")
    parser.add_argument("--view", choices=("window", "web", "none"), default=runtime_config.default_view())
    parser.add_argument("--camera", help="camera index or video path (default: runtime/config.yaml)")
    parser.add_argument("--mode", choices=("table", "wall"), default="table")
    parser.add_argument("--log", help="JSONL path (default runtime/logs/guidance-<time>.jsonl)")
    parser.add_argument("--glove", choices=("fake", "hardware"), help="default: glove.kind in runtime/config")
    parser.add_argument("--glove-port", help="serial port for --glove hardware (default: glove.port, auto = detect)")
    args = parser.parse_args(argv)

    from object_detection.common import load_config as od_config
    from object_detection.reach_api import ReachAPI

    from .camera import CameraHub

    runtime_config.load_env()  # runtime/glance.env: GLANCE_VISION_URL, GLANCE_TOKEN
    rt = runtime_config.load()
    camera = CameraHub.from_config(rt["camera"], 1024, source=args.camera).start()
    frame_size = (int(od_config()["frame"]["frame_width"]), int(od_config()["frame"]["frame_height"]))
    ages: list[float] = []

    def frame_source():
        image, at = camera.newest_with_time()
        if image is not None:
            ages.append((time.time() - at) * 1000)
            del ages[:-5]
        return image

    log_path = Path(args.log) if args.log else LOG_DIR / f"guidance-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log = FrameLog(log_path)
    glove = glove_from_config(rt, args.glove, args.glove_port)
    timings = Timings()
    loop = ReachLoop(lambda on_result, on_progress: ReachAPI(on_result, frame_source, on_progress),
                     Guidance.from_config(args.mode), glove, log, timings,
                     capture_age=lambda: round(ages[-1], 1) if ages else None)
    view = View(args.view, title="glance live guidance", port=rt["view"]["web_port"])
    quit_flag = threading.Event()
    fps = FpsCounter()

    def on_key(k: str) -> None:
        if k == "q":
            quit_flag.set()
        elif k == "r":
            loop.restart()
        elif k in ("p", " "):
            loop.toggle_pause()
        elif k == "m":
            print(f"mode: {loop.toggle_mode()}", flush=True)

    StdinRouter(KEYS - {" "}, on_key=on_key).start()
    try:
        print("waiting for the camera...", flush=True)
        camera.ready.wait(15)
        loop.start(args.target, args.refer)
        print(f"reach for {args.target!r}. keys: r restart, space/p pause, m table/wall, q quit. log: {log_path}",
              flush=True)
        while not quit_flag.is_set():
            canvas = None
            if args.view != "none":
                image = camera.newest_bgr()
                canvas = fit_width(image, rt["view"]["width"]) if image is not None else blank(rt["view"]["width"])
                snap = loop.snapshot()
                draw_reach(canvas, snap, frame_size)
                timing = "  ".join(f"{k} {v:.0f}" for k, v in timings.items()) or "-"
                head = [f"view {fps.tick():.0f} fps | camera {camera.fps:.0f} fps | target: {snap['target']}",
                        f"ms: {timing}"]
                if snap["result"] is not None:
                    head.append((f"END: {snap['result'].status} (r restarts)", (80, 255, 80)))
                elif snap["paused"]:
                    head.append(("PAUSED", (0, 200, 255)))
                if snap["error"]:
                    head.append((snap["error"], (80, 80, 255)))
                text_panel(canvas, head + guidance_lines(snap), (8, 8), scale=0.5)
                h = canvas.shape[0]
                if h > 230:
                    glove.draw(canvas, (8, h - 230))
                text_panel(canvas, ["keys: r restart | space/p pause | m table/wall | q quit"], (170, h - 30),
                           scale=0.42)
            for k in view.show(canvas):
                if k in KEYS:
                    on_key(k)
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        if hasattr(glove, "close"):
            glove.close()
        camera.stop()
        view.close()
        print(log.summary(), flush=True)


if __name__ == "__main__":
    main()
