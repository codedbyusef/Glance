"""FastAPI + WebSocket perception server, run on the Vultr GPU instance.

Processes only the newest frame per client and drops stale ones -- never queue,
a queue makes the glove lag the hand. Behind HTTPS with a token check in
production; see detection/CLAUDE.md "Perception server (WebSocket)".

Wire protocol for /ws (see detection/CLAUDE.md "Object detection and tracking (settled)"):
  client -> server, text (JSON): {"cmd": "start", "target": "...", "refer": "..."}
                                  {"cmd": "pause" | "resume" | "cancel"}
  client -> server, binary:      struct(">Id", frame_id, client_ts) + JPEG bytes
  server -> client, text (JSON): {"type": "started", "session_id": "..."}
                                  {"type": "progress", "session_id": "...", **FrameResult}
                                  {"type": "end", **ReachReport}
                                  {"type": "cancelled", "session_id": "..."}
                                  {"type": "error", "detail": "..."}
"""

from __future__ import annotations

import asyncio
import json
import struct
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from detection import detector as detector_model
from server.session import ReachSession
from shared import config

_FRAME_HEADER = struct.Struct(">Id")  # frame_id: uint32, client_ts: float64


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector_model.warmup()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


async def _receive_latest(websocket: WebSocket, timeout: Optional[float]) -> dict:
    """Waits for the next message (bounded by `timeout` if given), then drains
    any extra frames already buffered so we always act on the newest one. A
    control (text) message seen while draining is returned immediately
    instead of being discarded -- only extra binary frames get dropped.
    """
    if timeout is not None:
        message = await asyncio.wait_for(websocket.receive(), timeout=timeout)
    else:
        message = await websocket.receive()

    while message["type"] == "websocket.receive" and "bytes" in message:
        try:
            newer = await asyncio.wait_for(websocket.receive(), timeout=0)
        except asyncio.TimeoutError:
            break
        message = newer
    return message


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    cfg = config.load("settings.yaml")
    frame_width = cfg["server"]["frame_width"]
    frame_height = cfg["server"]["frame_height"]
    stale_link_s = cfg["reach_session"]["stale_link_after_s"]

    session: Optional[ReachSession] = None

    try:
        while True:
            timeout = stale_link_s if session is not None and session.is_active else None
            try:
                message = await _receive_latest(websocket, timeout)
            except asyncio.TimeoutError:
                report = session.end_as_stale_link()
                await websocket.send_json({"type": "end", **asdict(report)})
                session = None
                continue

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                payload = json.loads(message["text"])
                cmd = payload.get("cmd")
                if cmd == "start":
                    session = ReachSession(target=payload["target"], refer=payload.get("refer", ""))
                    await websocket.send_json({"type": "started", "session_id": session.session_id})
                elif cmd == "pause" and session is not None:
                    session.pause()
                elif cmd == "resume" and session is not None:
                    session.resume()
                elif cmd == "cancel" and session is not None:
                    session.cancel()
                    await websocket.send_json({"type": "cancelled", "session_id": session.session_id})
                    session = None
                continue

            if "bytes" in message:
                if session is None or not session.is_active:
                    continue  # no active reach -- drop the frame
                frame_id, client_ts = _FRAME_HEADER.unpack_from(message["bytes"])
                jpeg = message["bytes"][_FRAME_HEADER.size:]
                result = session.handle_frame(jpeg, frame_id, frame_width, frame_height)
                if session.done:
                    await websocket.send_json({"type": "end", **asdict(session.end_report)})
                    session = None
                else:
                    await websocket.send_json({"type": "progress", "session_id": session.session_id, **asdict(result)})
    except Exception as exc:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
