"""Vision server: FastAPI + WebSocket, run on the Vultr GPU instance (or a laptop).

Models load once at startup and get a warm-up pass. Only the newest frame is
processed; older buffered frames are dropped, never queued (a queue makes the
glove lag the hand).

Run:  uvicorn object_detection.server:app --host 0.0.0.0 --port 8000      (from the repo root)
      GLANCE_FAKE_MODELS=1 uvicorn object_detection.server:app ...        no GPU, canned detector/tracker
Auth: if GLANCE_TOKEN is set, clients must connect to /ws?token=<value>.
      Not needed behind the SSH tunnel; required if port 8000 is opened to the internet.

Wire protocol on /ws (one reach session at a time per socket):
  client -> server  text   {"cmd": "start", "target": "...", "refer": "..."}
                           {"cmd": "pause" | "resume" | "cancel"}
                    binary struct(">Id", frame_id, client_ts) + JPEG bytes
  server -> client  text   {"type": "started", "session_id": "..."}
                           {"type": "progress", "client_ts": ..., "server_ms": ..., **FrameResult}
                           {"type": "end", **EndReport}          (status cancelled after "cancel")
                           {"type": "error", "detail": "..."}
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from object_detection.common import decode_jpeg, load_config, load_env_file
from object_detection.follow import TargetFollower
from object_detection.session import ReachSession, hand_reader

FRAME_HEADER = struct.Struct(">Id")  # frame_id: uint32, client_ts: float64 (client clock, seconds)

_models: dict = {}
_gpu_lock = asyncio.Lock()  # one GPU, one frame at a time


def load_models() -> dict:
    cfg = load_config()
    if os.environ.get("GLANCE_FAKE_MODELS") == "1":
        from object_detection.fake import FakeDetector, FakeTracker

        return {"detector": FakeDetector(), "tracker": FakeTracker()}

    from object_detection.detector import GroundingDinoDetector
    from object_detection.tracker import make_tracker

    detector = GroundingDinoDetector(cfg["detector"])
    tracker = make_tracker(cfg["tracker"])
    detector.warmup()
    tracker.warmup()
    return {"detector": detector, "tracker": tracker}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env_file()
    _models.update(load_models())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    info = {"status": "ok", "models_loaded": bool(_models), "fake": os.environ.get("GLANCE_FAKE_MODELS") == "1"}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20)
    except ImportError:
        pass
    return info


def new_session(target: str, refer: str) -> ReachSession:
    cfg = load_config()
    follower = TargetFollower(
        _models["detector"],
        _models["tracker"],
        lost_frames=int(cfg["tracker"]["lost_frames"]),
        reacquire_min_iou=float(cfg["tracker"]["reacquire_min_iou"]),
        ambiguous_gap=float(cfg["detector"]["ambiguous_gap"]),
    )
    return ReachSession(target, refer, follower, cfg["reach_session"], hand=hand_reader())


async def receive_latest(websocket: WebSocket, timeout: Optional[float]) -> dict:
    """Wait for the next message, then drain any frames already buffered so only the
    newest frame is processed. A text (control) message stops the drain and wins."""
    message = await asyncio.wait_for(websocket.receive(), timeout=timeout)
    while message["type"] == "websocket.receive" and message.get("bytes") is not None:
        try:
            newer = await asyncio.wait_for(websocket.receive(), timeout=0.001)
        except asyncio.TimeoutError:
            break
        message = newer
    return message


def _authorized(websocket: WebSocket) -> bool:
    token = os.environ.get("GLANCE_TOKEN", "")
    return not token or websocket.query_params.get("token") == token


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    if not _authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    stale_link_s = float(load_config()["reach_session"]["stale_link_after_s"])
    session: Optional[ReachSession] = None

    try:
        while True:
            timeout = stale_link_s if session is not None and session.is_active else None
            try:
                message = await receive_latest(websocket, timeout)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "end", **session.end_as_stale_link().to_dict()})
                session = None
                continue

            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                payload = json.loads(message["text"])
                cmd = payload.get("cmd")
                if cmd == "start":
                    if session is not None and not session.done:
                        session.cancel()
                    session = new_session(payload["target"], payload.get("refer", ""))
                    await websocket.send_json({"type": "started", "session_id": session.session_id})
                elif session is None:
                    continue
                elif cmd == "pause":
                    session.pause()
                elif cmd == "resume":
                    session.resume()
                elif cmd == "cancel":
                    await websocket.send_json({"type": "end", **session.cancel().to_dict()})
                    session = None
                continue

            data = message.get("bytes")
            if data is None or session is None or not session.is_active:
                continue  # no active reach: drop the frame
            frame_id, client_ts = FRAME_HEADER.unpack_from(data)
            frame = decode_jpeg(data[FRAME_HEADER.size:])

            t0 = time.perf_counter()
            async with _gpu_lock:
                result = await asyncio.to_thread(session.handle_frame, frame, frame_id)
            server_ms = round((time.perf_counter() - t0) * 1000, 2)

            if result is not None:
                await websocket.send_json(
                    {"type": "progress", "client_ts": client_ts, "server_ms": server_ms, **asdict(result)}
                )
            if session.done:
                await websocket.send_json({"type": "end", **session.end_report.to_dict()})
                session = None
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
