"""Talk to the real agent with the real mic, wake word, STT and TTS, using live camera frames.

    python -m runtime.voice_agent                       # wake word "hey glance" (needs headphones)
    python -m runtime.voice_agent --ptt-only            # no headphones: Enter starts and stops each utterance
    python -m runtime.voice_agent --typed               # type what you say; real TTS and camera
    python -m runtime.voice_agent --view web            # MJPEG at http://<host>:8090/, keys in the terminal
    python -m runtime.voice_agent --reach real          # Step 3: real reaches on the vision server (GLANCE_VISION_URL)

--reach keys (Step 1, the default) simulates the reach with keys; --reach real drives ReachAPI, guidance and the
glove (fake by default).

Keys (window, or a letter + Enter in the terminal):
    s l h n t k   (--reach keys) end the running reach: success, target_lost, hand_lost, target_not_found,
                  timeout, stale_link
    m             (--reach real) toggle table / wall guidance
    Enter         push-to-talk (start, then stop)
    q             quit

Wiring (audio/CLAUDE.md section 15): dispatch = loop.call_soon_threadsafe; the Player is the controller's audio;
the controller (through a pass-through tap for the overlays) is the Mic's sink; task_active = controller.task is
not None; frame_source = camera.latest_frame; controller.tick() every 100 ms. The asyncio loop runs on a
background thread, the UI on the main thread.

Env: GEMINI_API_KEY (or guide_agent/.env), GLANCE_STT_URL, GLANCE_STT_TOKEN.
"""
from __future__ import annotations

import argparse
import asyncio
import threading
import time
import traceback
from typing import Callable

import cv2
import numpy as np

from guide_agent.config import AGENT_ROOT, Config as AgentConfig, load_config as load_agent_config, load_dotenv
from guide_agent.controller import Controller
from guide_agent.interfaces import RealClock
from guide_agent.prompt import build_system_prompt
from guide_agent.schemas import Frame
from guide_agent.turnlog import TurnLog

from . import config as runtime_config
from .glove_out import FakeGlove, glove_from_config
from .keyboard_reach import KEY_STATUS, KeyboardReach
from .keys import StdinRouter, TogglePtt
from .live_guidance import draw_reach, guidance_lines
from .view import FpsCounter, Timings, View, blank, fit_width, inset, text_panel

KEYS = set(KEY_STATUS) | {"q", "m"}


def _clip(text: str, n: int = 95) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


class Status:
    """What the overlays show. Written by the loop thread and the audio threads, read by the UI thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.listen = "starting"
        self.transcript = ""
        self.spoken = ""
        self.task = "none"
        self.step = ""
        self.sent_frame: Frame | None = None
        self.errors: list[str] = []
        self._utterance_end: float | None = None  # wall time of the last end of speech, until its first audio

    def get(self) -> dict:
        with self._lock:
            return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def set(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self, k, v)

    def update_from(self, ctl: Controller) -> None:
        """Copy the controller's state. Runs on the loop thread, so it never reads a half-made change."""
        task = ctl.task
        if task is None:
            desc, step = "none", ""
        else:
            desc = f"{task.goal}"
            s = task.current_step
            if s is None:
                step = "no pending step"
            elif s.type == "speak":
                step = f"{s.n} speak {s.state} expect={s.expect}: {s.text}"
            else:
                step = f"{s.n} reach {s.state}: {s.target}"
        spoken = ctl.last_spoken.text if ctl.last_spoken is not None else ""
        self.set(task=desc, step=step, spoken=spoken)

    # audio log events (any thread)
    def on_audio_event(self, rec: dict, timings: Timings) -> None:
        ev = rec.get("event")
        if ev == "state":
            self.set(listen=rec["state"])
        elif ev == "stt":
            timings.set("stt", rec.get("ms", 0))
        elif ev == "synth" and not rec.get("cached"):
            timings.set("tts synth", rec.get("ms", 0))
        elif ev == "utterance_delivered":
            timings.set("end->utterance", rec.get("latency_ms", 0))
            with self._lock:
                self._utterance_end = rec["wall"] - rec.get("latency_ms", 0) / 1000
        elif ev == "playback_start":
            with self._lock:
                end, self._utterance_end = self._utterance_end, None
            if end is not None:
                timings.set("end->audio", (rec["wall"] - end) * 1000)
        elif ev == "error":
            self.add_error(f"audio {rec.get('where')}: {rec.get('error')}")

    def add_error(self, text: str) -> None:
        with self._lock:
            self.errors = (self.errors + [_clip(text, 110)])[-3:]


class SinkTap:
    """The Mic's sink: notes the transcript and the frame for the overlays, then hands on to the controller.
    Its methods run on the loop thread (the Mic calls them through dispatch)."""

    def __init__(self, controller: Controller, status: Status):
        self.controller = controller
        self.status = status

    def on_speech_start(self, at: float | None = None) -> None:
        self.status.set(transcript="(listening...)")
        self.controller.on_speech_start(at)

    def on_utterance(self, text: str, ended_at: float | None = None, frame: Frame | None = None) -> None:
        self.status.set(transcript=text or "(nothing heard)", sent_frame=frame if frame is not None else None)
        self.controller.on_utterance(text, ended_at, frame)


class App:
    """Everything except the UI: the event loop thread, controller, audio, camera frames and the keyboard reach."""

    def __init__(self, agent_cfg: AgentConfig, model, player, camera, *, glove=None, status: Status | None = None,
                 timings: Timings | None = None, turnlog: TurnLog | None = None,
                 mic_factory: Callable | None = None, tick_s: float = 0.1,
                 loop: asyncio.AbstractEventLoop | None = None, haptic_factory: Callable | None = None):
        """loop: the agent's event loop (not running yet); the Player's dispatch must be its call_soon_threadsafe."""
        self.status = status or Status()
        self.timings = timings or Timings()
        self.player = player
        self.camera = camera
        self.glove = glove or FakeGlove()
        self.turnlog = turnlog or TurnLog()
        self.tick_s = tick_s
        self.loop = loop or asyncio.new_event_loop()
        self.dispatch = self.loop.call_soon_threadsafe
        on_result = lambda r: self.controller.on_haptic_result(r)  # noqa: E731  (runs on the loop via dispatch)
        # haptic_factory(dispatch, on_result) -> a HapticLoop (ReachHaptic for --reach real); default: keys
        self.reach = haptic_factory(self.dispatch, on_result) if haptic_factory is not None \
            else KeyboardReach(self.dispatch, on_result, camera, self.glove)
        self.controller = Controller(agent_cfg, model, player, self.reach, camera, RealClock(),
                                     build_system_prompt(agent_cfg), self.turnlog)
        self.sink = SinkTap(self.controller, self.status)
        self.mic = mic_factory(self.sink, self.dispatch, lambda: self.controller.task is not None) \
            if mic_factory is not None else None
        self._thread: threading.Thread | None = None
        self._turns_seen = 0

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> "App":
        self._thread = threading.Thread(target=self.loop.run_forever, name="agent-loop", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._tick_forever(), self.loop)
        if self.mic is not None:
            self.mic.start()
        return self

    def stop(self) -> None:
        """Every exit path: glove off, mic and player closed, loop stopped."""
        self.glove.all_off()
        for name, close in (("mic", getattr(self.mic, "close", None)), ("player", getattr(self.player, "close", None)),
                            ("reach", getattr(self.reach, "close", None))):
            if close is not None:
                try:
                    close()
                except Exception as e:
                    print(f"[runtime] {name} close failed: {e!r}")
        if self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._cancel_tasks(), self.loop).result(3)
            except Exception as e:
                print(f"[runtime] loop shutdown: {e!r}")
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)

    @staticmethod
    async def _cancel_tasks() -> None:
        """The ticker and any turn still in flight."""
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in others:
            t.cancel()
        await asyncio.gather(*others, return_exceptions=True)

    def on_loop(self, fn: Callable, *args, timeout: float = 5.0):
        """Run fn on the loop thread and return its result (tests and the UI)."""
        async def run():
            return fn(*args)
        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout)

    async def _tick_forever(self) -> None:
        while True:
            try:
                self.controller.tick()
                while self.controller.errors:
                    exc = self.controller.errors.pop(0)
                    traceback.print_exception(exc)
                    self.status.add_error(f"agent: {exc!r}")
                self.status.update_from(self.controller)
                self._note_turn_latency()
            except Exception as e:  # never let the ticker die
                traceback.print_exc()
                self.status.add_error(f"tick: {e!r}")
            await asyncio.sleep(self.tick_s)

    def _note_turn_latency(self) -> None:
        turns = self.turnlog.turns()
        if len(turns) > self._turns_seen:
            self._turns_seen = len(turns)
            last = turns[-1]
            if last.get("latency_ms") is not None:
                self.timings.set("turn", last["latency_ms"])
            calls = [m["latency_ms"] for m in last.get("model_calls", []) if "latency_ms" in m]
            if calls:
                self.timings.set("model", sum(calls))

    # ---------------------------------------------------------------- inputs (any thread)

    def type_utterance(self, text: str) -> None:
        """--typed: what the audio layer would deliver for a spoken utterance, with the current frame."""
        now = time.time()
        frame = self.camera.latest_frame() if self.camera is not None else None
        self.dispatch(self.sink.on_speech_start, now)
        self.dispatch(self.sink.on_utterance, text, now, frame)


# ------------------------------------------------------------------ the live view

SIDE_W = 430  # side panel width (layout "side")
LAYOUTS = ("side", "overlay")


def _wrap(lines: list, width: int) -> list:
    """Wrap each (text or (text, color)) line to `width` characters, keeping its color."""
    import textwrap

    out = []
    for ln in lines:
        text, color = (ln, None) if isinstance(ln, str) else ln
        parts = textwrap.wrap(" ".join(text.split()), width) or [""]
        out += [p if color is None else (p, color) for p in parts[:3]]
    return out


class Overlay:
    """The live view. layout "side": the camera on the left, untouched except for the reach markers, and a
    panel on the right with the status, the glove and the frame sent with the last utterance.
    layout "overlay": everything drawn on top of the camera image (narrower window)."""

    def __init__(self, app: App, width: int, mode_label: str, layout: str = "side"):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        self.app = app
        self.width = width
        self.mode_label = mode_label
        self.layout = layout
        self.fps = FpsCounter()
        self._inset_id: str | None = None
        self._inset_img = None

    def _lines(self, st: dict, snap, fps: float) -> list:
        app = self.app
        timing = "  ".join(f"{k} {v:.0f}" for k, v in app.timings.items()) or "-"
        reach = app.reach.describe()
        lines = [
            f"view {fps:.0f} fps | camera {app.camera.fps if app.camera else 0:.0f} fps | {self.mode_label}",
            f"ms: {timing}",
            (f"listening: {st['listen']}", (120, 255, 120)),
            f"heard: {st['transcript']}",
            f"said: {st['spoken']}",
            f"task: {st['task']}",
            f"step: {st['step']}" if st["step"] else "step: -",
            (reach, (0, 220, 255) if reach.startswith("REACH") else (235, 235, 235)),
        ]
        if snap is not None and snap["session_id"] is not None:
            lines += guidance_lines(snap)
        if snap is not None and snap["error"]:
            lines.append((snap["error"], (80, 80, 255)))
        return lines + [(e, (80, 80, 255)) for e in st["errors"]]

    def _sent_image(self, st: dict):
        frame = st["sent_frame"]
        if frame is None or not frame.image:
            return None, None
        if frame.id != self._inset_id:
            self._inset_img = cv2.imdecode(np.frombuffer(frame.image, np.uint8), cv2.IMREAD_COLOR)
            self._inset_id = frame.id
        return self._inset_img, frame.id

    def render(self):
        app = self.app
        image = app.camera.newest_bgr() if app.camera is not None else None
        video = fit_width(image, self.width) if image is not None else blank(self.width, round(self.width * 9 / 16))
        snap = app.reach.snapshot() if hasattr(app.reach, "snapshot") else None
        if snap is not None:
            draw_reach(video, snap, (640, 480))
        st = app.status.get()
        lines = self._lines(st, snap, self.fps.tick())
        sent, sent_id = self._sent_image(st)
        keys = "m = table/wall" if snap is not None else "s l h n t k = reach result"
        keys_line = f"keys: {keys} | Enter = talk | q = quit"
        if self.layout == "overlay":
            return self._render_overlay(video, lines, sent, sent_id, keys_line)
        return self._render_side(video, lines, sent, sent_id, keys_line)

    def _render_overlay(self, canvas, lines, sent, sent_id, keys_line):
        text_panel(canvas, [ln if isinstance(ln, tuple) else _clip(ln) for ln in lines], (8, 8), scale=0.5)
        if sent is not None:
            inset(canvas, sent, "br", width=220, label=f"sent: {sent_id}")
        h = canvas.shape[0]
        if h > 230:
            self.app.glove.draw(canvas, (8, h - 230))
        text_panel(canvas, [keys_line], (170, h - 30), scale=0.42)
        return canvas

    def _render_side(self, video, lines, sent, sent_id, keys_line):
        vh, vw = video.shape[:2]
        wrapped = _wrap(lines, 54)
        text_h = 22 * len(wrapped) + 16
        sent_h = round((SIDE_W - 200) * 3 / 4) + 30 if sent is not None else 0
        h = max(vh, text_h + 200 + 40)
        canvas = np.full((h, vw + SIDE_W, 3), (24, 24, 24), np.uint8)
        canvas[:vh, :vw] = video
        x = vw + 8
        y = text_panel(canvas, wrapped, (x, 8), scale=0.5, width=SIDE_W - 16) + 8
        self.app.glove.draw(canvas, (x, y))  # 150 x 190
        if sent is not None and y + sent_h <= h:
            small = fit_width(sent, SIDE_W - 200)
            sx, sy = x + 165, y + 22
            if sy + small.shape[0] <= h:
                canvas[sy:sy + small.shape[0], sx:sx + small.shape[1]] = small
                cv2.putText(canvas, f"sent: {sent_id}", (sx, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        text_panel(canvas, [keys_line], (x, h - 30), scale=0.42)
        return canvas


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Voice + agent with live camera and a keyboard-simulated reach.")
    parser.add_argument("--view", choices=("window", "web", "none"), default=runtime_config.default_view())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ptt-only", action="store_true",
                      help="no wake word and no open-state VAD: Enter starts and stops each utterance (no headphones)")
    mode.add_argument("--typed", action="store_true", help="type utterances in the terminal; real TTS and camera")
    parser.add_argument("--camera", help="camera index or video path (default: runtime/config.yaml)")
    parser.add_argument("--reach", choices=("keys", "real"), default="keys",
                        help="keys: simulated with keys (Step 1); real: ReachAPI + guidance + glove (Step 3)")
    parser.add_argument("--mode", choices=("table", "wall"), default="table", help="guidance axis mode (--reach real)")
    parser.add_argument("--layout", choices=LAYOUTS, default="side",
                        help="side: camera left, info + glove panel right (default); overlay: drawn on the camera")
    parser.add_argument("--glove", choices=("fake", "hardware"), help="default: glove.kind in runtime/config")
    parser.add_argument("--glove-port", help="serial port for --glove hardware (default: glove.port, auto = detect)")
    args = parser.parse_args(argv)

    from audio import config as audio_config
    from audio.listen import build_mic
    from audio.log import EventLog
    from audio.player import build_player
    from guide_agent.models.gemini import GeminiAdapter

    from .camera import CameraHub

    runtime_config.load_env()  # runtime/glance.env: server URLs and tokens on the Pi
    rt = runtime_config.load()
    load_dotenv()
    agent_cfg = load_agent_config()
    model = GeminiAdapter(agent_cfg.model, thinking_level=agent_cfg.thinking_level)
    camera = CameraHub.from_config(rt["camera"], agent_cfg.frame_max_long_edge_px, source=args.camera).start()

    audio_cfg = audio_config.load()
    if args.ptt_only:
        audio_cfg.ptt_only = True
    status, timings = Status(), Timings()
    log = EventLog(audio_cfg.path(audio_cfg.log_dir), echo=lambda rec: status.on_audio_event(rec, timings))
    mode_label = "typed" if args.typed else ("ptt-only" if args.ptt_only else "wake word")
    log.write("session", cmd="voice_agent", id=time.strftime("%H%M%S"), mode=mode_label)

    ptt = TogglePtt()
    loop = asyncio.new_event_loop()
    player = build_player(audio_cfg, log, dispatch=loop.call_soon_threadsafe)
    mic_factory = None
    if not args.typed:
        def mic_factory(sink, dispatch, task_active):
            return build_mic(audio_cfg, sink, player=player, dispatch=dispatch, frame_source=camera.latest_frame,
                             task_active=task_active, log=log, ptt=ptt)

    glove = glove_from_config(rt, args.glove, args.glove_port)
    haptic_factory = None
    frame_log = None
    if args.reach == "real":
        from object_detection.reach_api import ReachAPI

        from .guidance import Guidance
        from .live_guidance import LOG_DIR, FrameLog, log_frame
        from .reach_haptic import ReachHaptic

        frame_log = FrameLog(LOG_DIR / f"voice-reach-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        guidance = Guidance.from_config(args.mode)

        def haptic_factory(dispatch, on_result):
            return ReachHaptic(lambda res_cb, prog_cb: ReachAPI(res_cb, camera.newest_bgr, prog_cb), dispatch,
                               on_result, guidance, glove,
                               on_frame=lambda msg, st: log_frame(frame_log, timings, msg, st, None))

    app = App(agent_cfg, model, player, camera, glove=glove, status=status, timings=timings,
              turnlog=TurnLog.daily(AGENT_ROOT / agent_cfg.log_dir), mic_factory=mic_factory, loop=loop,
              haptic_factory=haptic_factory)
    view = View(args.view, title="glance voice agent", port=rt["view"]["web_port"])
    overlay = Overlay(app, rt["view"]["width"], mode_label, args.layout)
    quit_flag = threading.Event()

    def on_key(k: str) -> None:
        if k == "q":
            quit_flag.set()
        elif k == "\n":
            if not args.typed:
                print("[ptt] talk now, Enter to stop" if ptt.toggle() else "[ptt] released", flush=True)
        elif k == "m":
            if hasattr(app.reach, "toggle_mode"):
                print(f"guidance mode: {app.reach.toggle_mode()}", flush=True)
        elif hasattr(app.reach, "key"):
            app.reach.key(k)

    try:
        player.start()
        print("warming up the voice (the first run takes several seconds)...", flush=True)
        # The fixed phrases are cached first; a reply queued before that would wait behind them. Keep the view
        # live meanwhile.
        status.set(listen="warming up voice")
        deadline = time.monotonic() + 60
        while not player.warm.is_set() and time.monotonic() < deadline and not quit_flag.is_set():
            if "q" in view.show(overlay.render() if args.view != "none" else None):
                quit_flag.set()
        status.set(listen="typed" if args.typed else "starting")
        app.start()
        StdinRouter(KEYS, on_key=on_key, on_enter=None if args.typed else (lambda: on_key("\n")),
                    on_line=app.type_utterance if args.typed else None).start()
        how = {"typed": "Type what you say, then Enter.", "ptt-only": "Press Enter to talk, Enter again to stop.",
               "wake word": 'Say "hey glance". Enter also works as push-to-talk.'}[mode_label]
        print(f"model {agent_cfg.model}, assistant {agent_cfg.assistant_name}. {how} q (+ Enter) quits.", flush=True)
        while not quit_flag.is_set():
            canvas = overlay.render() if args.view != "none" else None
            for k in view.show(canvas):
                if k == "\n" or k in KEYS:
                    on_key(k)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        if hasattr(glove, "close"):
            glove.close()
        camera.stop()
        view.close()
        if frame_log is not None:
            print(frame_log.summary(), flush=True)
        print("stopped. Latency summary: python -m runtime.latency", flush=True)


if __name__ == "__main__":
    main()
