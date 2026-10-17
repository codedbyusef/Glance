"""Listening policy and the Mic (CLAUDE.md sections 5.2, 5.3, 6 and 7; decisions 3, 6, 11 and 13).

Listener is the pure state machine: it takes audio blocks and push-to-talk events with timestamps and returns
actions. Mic wires it to the capture source, the wake detector, STT, the frame source and the sink, on one worker
thread, and sends every sink call through dispatch.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .log import EventLog, null_log
from .stt_client import to_wav
from .vad import FRAME_S, START_WINDOW_S, Endpointer, StartDetector, VadClassifier, frames

EXPECTS = ("none", "done", "answer")
BLOCK_S = 0.08
TAIL_S = 0.2  # audio kept after the last voiced frame, so the last syllable isn't clipped


class ReplyWindow:
    """Open after say(expect=done|answer); closed by a delivered non-empty utterance or a later say(expect=none).

    It never times out on its own. A transcript that is only the wake phrase doesn't close it (decision 6).
    """

    def __init__(self, assistant_name: str = "Glance"):
        self._open = False
        self._lock = threading.Lock()
        name = re.sub(r"[^\w\s]", " ", assistant_name.lower()).split()
        self._name_only = {" ".join(name), " ".join(["hey", *name])}

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def on_say(self, expect: str) -> None:
        with self._lock:
            if expect in ("done", "answer"):
                self._open = True
            elif expect == "none":
                self._open = False

    def on_delivered(self, text: str) -> None:
        key = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        if not key or key in self._name_only:
            return
        with self._lock:
            self._open = False


# ------------------------------------------------------------------ the pure state machine

@dataclass
class SpeechStart:
    at: float
    source: str  # wake | vad | ptt


@dataclass
class Finish:
    audio: np.ndarray | None  # None: deliver "" without calling STT
    ended_at: float
    reason: str  # end | max | ptt | short | no_speech | too_short | device_lost
    duration_ms: int


@dataclass
class _Capture:
    source: str
    start_t: float
    chunks: list = field(default_factory=list)  # (frame, t)
    endpointer: Endpointer | None = None  # None for push-to-talk


class Listener:
    def __init__(self, cfg, vad: VadClassifier, detector, wake_needed: Callable[[], bool],
                 log: EventLog | None = None):
        self.cfg = cfg
        self.vad = vad
        self.detector = detector
        self.wake_needed = wake_needed
        self.log = log or null_log()
        self.ring: deque = deque(maxlen=round((cfg.preroll_ms / 1000 + START_WINDOW_S) / FRAME_S) + 4)
        self.start_detector = StartDetector(cfg.speech_start_ms)
        self.capture: _Capture | None = None
        self.last_wake = float("-inf")
        self.last_state: str | None = None
        self.score_hook: Callable[[float, float], None] | None = None  # (score, t), for tuning

    def state(self) -> str:
        if self.capture is not None:
            return "capture"
        return "wake" if self.wake_needed() else "open"

    def reset(self) -> None:
        """After a delivery: forget the audio history, which is stale after a transcription."""
        self.capture = None
        self.ring.clear()
        self.start_detector.reset()

    def _note_state(self, state: str) -> None:
        if state != self.last_state:
            self.log.write("state", state=state, previous=self.last_state)
            self.last_state = state

    # ---------------------------------------------------------------- inputs

    def feed(self, block: np.ndarray, t: float) -> list:
        """One 80 ms block whose first sample is at wall time t."""
        actions: list = []
        state = self.state()
        self._note_state(state)
        woke = False
        if self.detector is not None:
            score = self.detector.score(block)  # fed in every state, so its features stay current
            if self.score_hook is not None:
                self.score_hook(score, t)
            if (state == "wake" and score >= self.cfg.wake_threshold
                    and t - self.last_wake >= self.cfg.wake_cooldown_s):
                self.last_wake = t
                at = t + BLOCK_S
                self.log.write("wake", score=round(score, 3), model=getattr(self.detector, "name", None))
                self.log.write("speech_start", source="wake")
                # No pre-roll: capture starts after this block, so the wake phrase stays out of the clip.
                self.capture = _Capture("wake", at, endpointer=Endpointer(self.cfg, at))
                actions.append(SpeechStart(at, "wake"))
                woke = True
        for i, frame in enumerate(frames(block)):
            ft = t + i * FRAME_S
            voiced = self.vad.is_speech(frame)
            self.ring.append((frame, ft))
            start = self.start_detector.push(voiced, ft)
            if woke:
                continue
            cap = self.capture
            if cap is not None:
                cap.chunks.append((frame, ft))
                done = self._capture_frame(cap, voiced, ft)
                if done is not None:
                    actions.append(done)
                    return actions  # the rest of the block is discarded with the audio during STT
            elif state == "open" and start is not None:
                self.log.write("speech_start", source="vad")
                ep = Endpointer(self.cfg, start, first_voiced=start, last_voiced_end=ft + FRAME_S)
                chunks = [(f, ftt) for f, ftt in self.ring if ftt >= start - self.cfg.preroll_ms / 1000 - 1e-9]
                self.capture = _Capture("vad", start, chunks, ep)
                actions.append(SpeechStart(start, "vad"))
        return actions

    def ptt(self, down: bool, t: float) -> list:
        cap = self.capture
        if down:
            if cap is not None:
                if cap.source != "ptt":  # already capturing: hand it to push-to-talk, no second speech_start
                    self.log.write("ptt_takeover", source=cap.source)
                    cap.source, cap.endpointer = "ptt", None
                return []
            self._note_state("capture")
            self.log.write("speech_start", source="ptt")
            chunks = [(f, ft) for f, ft in self.ring if ft >= t - self.cfg.preroll_ms / 1000 - 1e-9]
            self.capture = _Capture("ptt", t, chunks)
            return [SpeechStart(t, "ptt")]
        if cap is None or cap.source != "ptt":
            return []
        return [self._ptt_finish(cap, t, "ptt")]

    def device_lost(self, t: float) -> list:
        cap = self.capture
        if cap is None:
            return []
        self.capture = None
        return [Finish(None, t, "device_lost", 0)]

    # ---------------------------------------------------------------- capture

    def _capture_frame(self, cap: _Capture, voiced: bool, ft: float) -> Finish | None:
        if cap.endpointer is None:  # push-to-talk: only release or the hard stop end it
            if ft + FRAME_S - cap.start_t >= self.cfg.max_utterance_s - 1e-9:
                return self._ptt_finish(cap, ft + FRAME_S, "max")
            return None
        res = cap.endpointer.push(voiced, ft)
        if res is None:
            return None
        self.capture = None
        span_ms = int(round(cap.endpointer.span_s() * 1000))
        long_enough = cap.endpointer.started and span_ms >= self.cfg.min_utterance_ms
        if res.kind in ("end", "max") and long_enough:
            end = res.last_voiced_end
            audio = np.concatenate([f for f, t in cap.chunks if t < end + TAIL_S])
            return Finish(audio, end, res.kind, span_ms)
        ended = res.last_voiced_end if res.last_voiced_end is not None else ft + FRAME_S
        reason = "short" if cap.endpointer.started else "no_speech"
        return Finish(None, ended, reason, span_ms)

    def _ptt_finish(self, cap: _Capture, t: float, reason: str) -> Finish:
        self.capture = None
        held_ms = int(round((t - cap.start_t) * 1000))
        if held_ms < self.cfg.min_utterance_ms or not cap.chunks:
            return Finish(None, t, "too_short", held_ms)
        return Finish(np.concatenate([f for f, _ in cap.chunks]), t, reason, held_ms)


# ------------------------------------------------------------------ the Mic

def direct_dispatch(fn, *args) -> None:
    fn(*args)


class Mic:
    """Runs the Listener on one worker thread and talks to the agent's sink only through dispatch."""

    def __init__(self, cfg, sink, transcriber, capture, vad: VadClassifier, detector=None, player=None,
                 window: ReplyWindow | None = None, dispatch: Callable = direct_dispatch,
                 frame_source: Callable[[], object] = lambda: None, task_active: Callable[[], bool] = lambda: False,
                 ptt=None, log: EventLog | None = None):
        self.cfg = cfg
        self.sink = sink
        self.transcriber = transcriber
        self.capture = capture
        self.player = player
        self.window = window if window is not None else (player.window if player is not None else ReplyWindow())
        self.dispatch = dispatch
        self.frame_source = frame_source
        self.task_active = task_active
        self.ptt_source = ptt
        self.log = log or null_log()
        self.listener = Listener(cfg, vad, detector, lambda: self.wake_needed(), self.log)
        self.q: queue.Queue = queue.Queue(maxsize=250)  # 20 s of blocks
        self.open_starts = 0  # speech_starts not yet answered by an utterance
        self.idle = threading.Event()  # set while the worker waits for input (tests)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0

    def wake_needed(self) -> bool:
        """Section 6: the player is idle, the reply window is closed and no task is active."""
        try:
            active = bool(self.task_active())
        except Exception as e:
            self.log.error("mic.task_active", e)
            active = False
        busy = self.player.busy if self.player is not None else False
        return not busy and not self.window.is_open and not active

    # ---------------------------------------------------------------- lifecycle

    def put(self, item: tuple) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:  # the worker is stuck; drop the oldest audio rather than block the audio callback
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            if self._dropped % 50 == 1:
                self.log.write("error", where="mic.queue", error=f"queue full, dropped {self._dropped} items")
            self.q.put_nowait(item)

    def start(self) -> "Mic":
        self._thread = threading.Thread(target=self._run, name="mic-worker", daemon=True)
        self._thread.start()
        self.capture.start(self.put)
        if self.ptt_source is not None:
            self.ptt_source.start(self.put)
        return self

    def close(self) -> None:
        self._closed.set()
        for part in (self.capture, self.ptt_source):
            if part is not None:
                try:
                    part.close()
                except Exception as e:
                    self.log.error("mic.close", e)
        if self._thread is not None:
            self._thread.join(3)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                self.idle.set()
                item = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            self.idle.clear()
            try:
                self.handle(item)
            except Exception as e:
                self.log.error("mic.worker", e)
                self._recover()

    def _recover(self) -> None:
        """Keep the one-utterance-per-speech-start promise even after a bug."""
        self.listener.reset()
        while self.open_starts > 0:
            self._deliver_text("", time.time(), None, "error")

    # ---------------------------------------------------------------- work

    def handle(self, item: tuple) -> None:
        kind, payload, t = item
        if kind == "audio":
            actions = self.listener.feed(payload, t)
        elif kind == "ptt":
            actions = self.listener.ptt(bool(payload), t)
        elif kind == "device_lost":
            actions = self.listener.device_lost(t)
        else:
            raise ValueError(f"unknown mic item {kind!r}")
        for a in actions:
            if isinstance(a, SpeechStart):
                self.open_starts += 1
                self.dispatch(self.sink.on_speech_start, a.at)
            else:
                self._finish(a)

    def _finish(self, f: Finish) -> None:
        self.log.write("speech_end", duration_ms=f.duration_ms, reason=f.reason)
        text, frame = "", None
        if f.audio is not None:
            try:
                frame = self.frame_source()  # what the user faced when they stopped talking, before STT
            except Exception as e:
                self.log.error("mic.frame_source", e)
            wav = to_wav(f.audio)
            if self.cfg.save_clips:
                self._save_clip(wav)
            try:
                text = self.transcriber.transcribe(wav).text
            except Exception as e:
                self.log.error("mic.stt", e)
        self._deliver_text(text, f.ended_at, frame, f.reason)
        self.listener.reset()
        self._flush_audio()

    def _deliver_text(self, text: str, ended_at: float, frame, reason: str) -> None:
        self.window.on_delivered(text)
        self.open_starts = max(0, self.open_starts - 1)
        self.log.write("utterance_delivered", chars=len(text), reason=reason,
                       latency_ms=int((time.time() - ended_at) * 1000))
        self.dispatch(self.sink.on_utterance, text, ended_at, frame)

    def _flush_audio(self) -> None:
        """Drop audio that arrived during transcription (decision 11); keep push-to-talk and device events."""
        keep = []
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if item[0] != "audio":
                keep.append(item)
        for item in keep:
            self.put(item)

    def _save_clip(self, wav: bytes) -> None:
        try:
            d = self.cfg.path(self.cfg.log_dir) / "clips"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03}.wav").write_bytes(wav)
        except OSError as e:
            self.log.error("mic.save_clip", e)


def build_mic(cfg, sink, player=None, window: ReplyWindow | None = None, dispatch: Callable = direct_dispatch,
              frame_source: Callable[[], object] = lambda: None, task_active: Callable[[], bool] = lambda: False,
              log: EventLog | None = None, transcriber=None, capture=None) -> Mic:
    """The real Mic from config: sounddevice capture, WebRTC VAD, openWakeWord, the STT client, push-to-talk."""
    from .capture import SoundDeviceCapture
    from .ptt import build_ptt
    from .stt_client import build_transcriber, check_health, preload_fallback
    from .vad import WebRtcVad
    from .wake import load_detector

    log = log or null_log()
    if transcriber is None:
        transcriber = build_transcriber(cfg, log)
        check_health(transcriber, log)
        if cfg.fallback_preload:
            preload_fallback(transcriber, log)
    capture = capture if capture is not None else SoundDeviceCapture(cfg.input_device, log)
    return Mic(cfg, sink, transcriber, capture, WebRtcVad(cfg.vad_aggressiveness),
               detector=load_detector(cfg, log), player=player, window=window, dispatch=dispatch,
               frame_source=frame_source, task_active=task_active, ptt=build_ptt(cfg.ptt, log), log=log)
