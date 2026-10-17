"""Reach API for the agent: start / pause / resume / cancel one reach, get one result back.

This is the object_detection side of the agent's haptic interface (guide_agent/CLAUDE.md sections 5 and 10).
The agent's LLM calls `start(target, refer)` once; frames, detection, tracking and the end status all
happen here and on the vision server. The LLM never sees individual frames.

    api = ReachAPI(on_result=handle, frame_source=camera.newest_bgr)
    session_id = api.start("sheet of paper", "the white sheet on the desk")
    api.pause(session_id) / api.resume(session_id) / api.cancel(session_id)
    ...  # later, from the API thread: handle(ReachResult(status="success", ...))

What it does:
- Opens one WebSocket per reach to the vision server (client.VisionClient, same URL rules),
  streams the newest camera frame at `reach_api.stream_fps`, and keeps the last few JPEGs so the
  server's `frame_id` comes back as a full frame (the agent needs the image on failures).
- Turns the server's `end` message into a ReachResult, field for field the agent's haptic_result.
- Synthesizes `stale_link` when the link dies or the server goes quiet for `reach_api.no_reply_s`.
- Passes `progress` messages (target box, hand point, timings) to `on_progress`, for the glove
  controller once it exists.

Callers are synchronous (like the agent's controller). The API runs its own asyncio loop in a
background thread, and `on_result` / `on_progress` are called FROM THAT THREAD: hop to your own
loop with `loop.call_soon_threadsafe` before touching your state.

Statuses are passed through as the server reports them: success, target_not_found, target_lost,
hand_lost, stalled, timeout, stale_link, cancelled (a cancel you asked for; the agent drops it),
and ambiguous (only when `detector.ambiguous_gap > 0`, off by default; the agent has no such status).

Hooking it into the agent later (a few lines, in the agent):
    ReachStartError          -> raise HapticStartError (the controller turns it into stale_link)
    ReachResult              -> HapticResult(session_id, status, inferred, elapsed_s, target,
                                 LastSeen(**...) or None, closest_approach,
                                 Frame(id, image, captured_at) or None)
    on_result                -> loop.call_soon_threadsafe(controller.on_haptic_result, converted)

Standalone check with the webcam (server must be running):
    python -m object_detection.reach_api --target "sheet of paper"
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from object_detection.client import VisionClient
from object_detection.common import encode_jpeg, load_config
from object_detection.session import LastSeen

FrameSource = Callable[[], Optional[np.ndarray]]   # newest BGR frame, or None if none yet; must be fast


class ReachStartError(Exception):
    """The reach could not be started (server unreachable, no `started` reply)."""


@dataclass
class ReachFrame:
    id: str
    image: bytes          # JPEG, as sent to the server
    captured_at: float    # time.time() when the frame was taken from the source


@dataclass
class ReachResult:
    """Matches the agent's haptic_result (guide_agent/CLAUDE.md section 10)."""
    session_id: str
    status: str
    inferred: bool
    elapsed_s: float
    target: str
    last_seen: Optional[LastSeen]
    closest_approach: str              # touching | near | far
    frame: Optional[ReachFrame]        # None on success
    candidate_count: int = 0           # for the log only
    top_scores: list[float] = field(default_factory=list)


class _Run:
    """State of the one reach in progress."""

    def __init__(self, vc: VisionClient, session_id: str, target: str, keep_frames: int):
        self.vc = vc
        self.session_id = session_id
        self.target = target
        self.started = time.monotonic()
        self.frames: collections.OrderedDict[int, ReachFrame] = collections.OrderedDict()
        self.keep_frames = keep_frames
        self.resumed = asyncio.Event()
        self.resumed.set()
        self.done = False
        self.tasks: list[asyncio.Task] = []

    def remember(self, frame_id: int, frame: ReachFrame) -> None:
        self.frames[frame_id] = frame
        while len(self.frames) > self.keep_frames:
            self.frames.popitem(last=False)

    def newest_frame(self) -> Optional[ReachFrame]:
        return next(reversed(self.frames.values()), None)


class ReachAPI:
    def __init__(
        self,
        on_result: Callable[[ReachResult], None],
        frame_source: Optional[FrameSource] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        url: Optional[str] = None,
        fps: Optional[float] = None,
    ):
        cfg = load_config()
        api_cfg = cfg["reach_api"]
        self.on_result = on_result
        self.on_progress = on_progress
        self.url = url
        self.fps = float(fps or api_cfg["stream_fps"])
        self.start_timeout_s = float(api_cfg["start_timeout_s"])
        self.no_reply_s = float(api_cfg["no_reply_s"])
        self.keep_frames = int(api_cfg["keep_frames"])
        self.frame_size = (int(cfg["frame"]["frame_width"]), int(cfg["frame"]["frame_height"]))
        self.jpeg_quality = int(cfg["frame"]["jpeg_quality"])

        self._camera: Optional[_CameraThread] = None
        if frame_source is None:
            self._camera = _CameraThread(cfg["camera"]["source"])
            self._camera.start()
            frame_source = self._camera.newest
        self.frame_source = frame_source

        self._run: Optional[_Run] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="reach-api", daemon=True)
        self._thread.start()

    # --- public, synchronous -------------------------------------------------------

    def start(self, target: str, refer: str = "") -> str:
        """Start a reach and return its session id. A reach already running is dropped
        without a result. Raises ReachStartError if the server does not accept it."""
        future = asyncio.run_coroutine_threadsafe(self._start(target, refer), self._loop)
        try:
            return future.result(timeout=self.start_timeout_s + 1.0)
        except Exception as exc:
            future.cancel()
            raise ReachStartError(f"could not start reach for {target!r}: {exc!r}") from exc

    def pause(self, session_id: str) -> None:
        """Stop vibration-driving updates now; loss timers freeze on the server."""
        self._submit(self._pause(session_id))

    def resume(self, session_id: str) -> None:
        self._submit(self._resume(session_id))

    def cancel(self, session_id: str) -> None:
        """End the reach. The result arrives through on_result with status `cancelled`."""
        self._submit(self._cancel(session_id))

    def close(self) -> None:
        """Drop any reach in progress (no result) and stop the background thread and camera."""
        asyncio.run_coroutine_threadsafe(self._drop_run(), self._loop).result(timeout=3.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3.0)
        if self._camera is not None:
            self._camera.stop()

    @property
    def active_session(self) -> Optional[str]:
        run = self._run
        return run.session_id if run is not None and not run.done else None

    def _submit(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=3.0)

    # --- inside the API loop ---------------------------------------------------------

    async def _start(self, target: str, refer: str) -> str:
        await self._drop_run()
        vc = VisionClient(self.url)
        try:
            await asyncio.wait_for(vc.connect(), timeout=self.start_timeout_s)
            session_id = await asyncio.wait_for(vc.start(target, refer), timeout=self.start_timeout_s)
        except BaseException:
            await vc.close()
            raise
        run = _Run(vc, session_id, target, self.keep_frames)
        run.tasks = [asyncio.create_task(self._send_frames(run)), asyncio.create_task(self._receive(run))]
        self._run = run
        return session_id

    def _current(self, session_id: str) -> Optional[_Run]:
        run = self._run
        return run if run is not None and not run.done and run.session_id == session_id else None

    async def _pause(self, session_id: str) -> None:
        run = self._current(session_id)
        if run is None or not run.resumed.is_set():
            return
        run.resumed.clear()
        await self._send_cmd(run, run.vc.pause())

    async def _resume(self, session_id: str) -> None:
        run = self._current(session_id)
        if run is None or run.resumed.is_set():
            return
        await self._send_cmd(run, run.vc.resume())
        run.resumed.set()

    async def _cancel(self, session_id: str) -> None:
        run = self._current(session_id)
        if run is None:
            return
        run.resumed.set()
        try:
            await run.vc.cancel()   # the server answers with an `end` (cancelled) that _receive delivers
        except Exception:
            await self._finish(run, self._synthesized(run, "cancelled"))

    async def _send_cmd(self, run: _Run, send) -> None:
        try:
            await send
        except Exception:
            await self._finish(run, self._synthesized(run, "stale_link"))

    async def _send_frames(self, run: _Run) -> None:
        period = 1.0 / self.fps
        try:
            while not run.done:
                await run.resumed.wait()
                t0 = time.monotonic()
                image = self.frame_source()
                if image is not None:
                    captured_at = time.time()
                    if (image.shape[1], image.shape[0]) != self.frame_size:
                        image = cv2.resize(image, self.frame_size, interpolation=cv2.INTER_AREA)
                    jpeg = encode_jpeg(image, self.jpeg_quality)
                    if run.done or not run.resumed.is_set():
                        continue
                    frame_id = await run.vc.send_frame(jpeg)
                    run.remember(frame_id, ReachFrame(id=f"vis-{frame_id}", image=jpeg, captured_at=captured_at))
                await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._finish(run, self._synthesized(run, "stale_link"))

    async def _receive(self, run: _Run) -> None:
        try:
            while not run.done:
                try:
                    msg = await run.vc.next_message(timeout=self.no_reply_s)
                except asyncio.TimeoutError:
                    if run.resumed.is_set():   # a paused reach is quiet on purpose
                        await self._finish(run, self._synthesized(run, "stale_link"))
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    if self.on_progress is not None:
                        self._call(self.on_progress, msg)
                elif kind == "end":
                    await self._finish(run, self._from_end(run, msg))
                elif kind == "error":
                    await self._finish(run, self._synthesized(run, "stale_link"))
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._finish(run, self._synthesized(run, "stale_link"))

    async def _finish(self, run: _Run, result: Optional[ReachResult]) -> None:
        """End `run` once: stop its tasks, close the link, deliver `result` (None = no result)."""
        if run.done:
            return
        run.done = True
        run.resumed.set()
        current = asyncio.current_task()
        for task in run.tasks:
            if task is not current:
                task.cancel()
        try:
            await run.vc.close()
        except Exception:
            pass
        if self._run is run:
            self._run = None
        if result is not None:
            self._call(self.on_result, result)

    async def _drop_run(self) -> None:
        if self._run is not None:
            run = self._run
            try:
                await asyncio.wait_for(run.vc.cancel(), timeout=0.5)
            except Exception:
                pass
            await self._finish(run, None)

    def _from_end(self, run: _Run, msg: dict) -> ReachResult:
        frame = None
        if msg.get("status") != "success":
            frame_id = msg.get("frame_id")
            frame = run.frames.get(frame_id) if frame_id is not None else None
            frame = frame or run.newest_frame()
        last_seen = msg.get("last_seen")
        return ReachResult(
            session_id=run.session_id,
            status=msg["status"],
            inferred=bool(msg.get("inferred", False)),
            elapsed_s=float(msg.get("elapsed_s", 0.0)),
            target=msg.get("target", run.target),
            last_seen=LastSeen(**last_seen) if last_seen else None,
            closest_approach=msg.get("closest_approach", "far"),
            frame=frame,
            candidate_count=int(msg.get("candidate_count", 0)),
            top_scores=list(msg.get("top_scores", [])),
        )

    def _synthesized(self, run: _Run, status: str) -> ReachResult:
        """A result made on this side when the server could not report one."""
        return ReachResult(
            session_id=run.session_id,
            status=status,
            inferred=False,
            elapsed_s=round(time.monotonic() - run.started, 2),
            target=run.target,
            last_seen=None,
            closest_approach="far",
            frame=run.newest_frame(),
        )

    @staticmethod
    def _call(callback, arg) -> None:
        try:
            callback(arg)
        except Exception as exc:   # a broken callback must not kill the reach loop
            print(f"reach_api: callback {callback!r} raised {exc!r}")


class _CameraThread(threading.Thread):
    """Reads the camera nonstop and keeps the newest frame (used when no frame_source is given)."""

    def __init__(self, source):
        super().__init__(daemon=True, name="reach-api-camera")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {source!r}")
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = True

    def run(self) -> None:
        while self._running:
            ok, image = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = image

    def newest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False
        self.join(timeout=2.0)
        self.cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one reach on the webcam through the vision server.")
    parser.add_argument("--target", default="sheet of paper")
    parser.add_argument("--refer", default="")
    args = parser.parse_args()

    finished = threading.Event()

    def on_result(result: ReachResult) -> None:
        frame = f"{result.frame.id} ({len(result.frame.image)} B)" if result.frame else None
        print(f"\nRESULT status={result.status} inferred={result.inferred} elapsed={result.elapsed_s}s "
              f"last_seen={result.last_seen} closest={result.closest_approach} frame={frame} "
              f"candidates={result.candidate_count} top={result.top_scores}")
        finished.set()

    def on_progress(msg: dict) -> None:
        box = [round(v) for v in msg["target_box"]] if msg.get("target_box") else None
        hand = [round(v) for v in msg["hand_point"]] if msg.get("hand_point") else None
        print(f"frame {msg['frame_id']:5d} box={box} conf={msg['confidence']:.2f} hand={hand} "
              f"server={msg['server_ms']:.0f}ms {msg['timings_ms']}")

    api = ReachAPI(on_result=on_result, on_progress=on_progress)
    try:
        session_id = api.start(args.target, args.refer)
        print(f"reach {session_id} started for {args.target!r}; Ctrl+C cancels")
        while not finished.wait(0.2):
            pass
    except KeyboardInterrupt:
        if api.active_session:
            api.cancel(api.active_session)
            finished.wait(2.0)
    finally:
        api.close()


if __name__ == "__main__":
    main()
