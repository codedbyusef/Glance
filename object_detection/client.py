"""Laptop-side client for the vision server (Vultr or local).

Where it connects, first match wins:
  1. GLANCE_VISION_URL                 full URL, e.g. ws://localhost:8000/ws
  2. config.yaml `vultr.use_tunnel: true`  -> ws://localhost:<local_port>/ws (run deploy/tunnel.sh first)
  3. config.yaml `vultr.use_tunnel: false` -> ws://<GLANCE_VULTR_HOST or vultr.host>:<server_port>/ws
GLANCE_TOKEN, if set, is sent as ?token=. Env values can live in object_detection/.env.

Library use:
    async with VisionClient() as vc:
        session_id = await vc.start("sheet of paper", "the paper on the desk")
        await vc.send_frame(jpeg_bytes)
        msg = await vc.next_message()      # progress / end / error dicts

Live check from the webcam (once the server is up):
    python -m object_detection.client --health
    python -m object_detection.client --target "sheet of paper"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
from typing import Optional
from urllib.parse import urlencode

from object_detection.common import load_config, load_env_file

FRAME_HEADER = struct.Struct(">Id")  # must match server.FRAME_HEADER


class ConfigError(RuntimeError):
    pass


def vultr_host() -> str:
    load_env_file()
    return os.environ.get("GLANCE_VULTR_HOST", "").strip() or str(load_config()["vultr"].get("host") or "").strip()


def server_url(path: str = "/ws", scheme: str = "ws") -> str:
    """Resolve the server URL. Raises ConfigError if the Vultr IP is needed but not set."""
    load_env_file()
    override = os.environ.get("GLANCE_VISION_URL", "").strip()
    if override:
        base = override.rstrip("/")
        for suffix in ("/ws", "/health"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if scheme == "http":
            base = base.replace("wss://", "https://").replace("ws://", "http://")
        url = base + path
    else:
        v = load_config()["vultr"]
        if v.get("use_tunnel", True):
            url = f"{scheme}://localhost:{v['local_port']}{path}"
        else:
            host = vultr_host()
            if not host:
                raise ConfigError(
                    "Vultr IPv4 not set. Put GLANCE_VULTR_HOST=<ip> in object_detection/.env "
                    "or set vultr.host in config.yaml."
                )
            url = f"{scheme}://{host}:{v['server_port']}{path}"

    token = os.environ.get("GLANCE_TOKEN", "").strip()
    if token and path == "/ws":
        url += "?" + urlencode({"token": token})
    return url


class VisionClient:
    def __init__(self, url: Optional[str] = None):
        self.url = url or server_url()
        self._ws = None
        self._frame_id = 0
        self.session_id: Optional[str] = None

    async def __aenter__(self) -> "VisionClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        import websockets

        # Large max_size is not needed (server sends JSON only); keep pings for dead-link detection.
        self._ws = await websockets.connect(self.url, ping_interval=5, ping_timeout=5)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def start(self, target: str, refer: str = "") -> str:
        await self._ws.send(json.dumps({"cmd": "start", "target": target, "refer": refer}))
        while True:
            msg = await self.next_message()
            if msg.get("type") == "started":
                self.session_id = msg["session_id"]
                return self.session_id
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("detail"))

    async def pause(self) -> None:
        await self._ws.send(json.dumps({"cmd": "pause"}))

    async def resume(self) -> None:
        await self._ws.send(json.dumps({"cmd": "resume"}))

    async def cancel(self) -> None:
        await self._ws.send(json.dumps({"cmd": "cancel"}))

    async def send_frame(self, jpeg: bytes) -> int:
        self._frame_id = (self._frame_id + 1) % 2**32
        await self._ws.send(FRAME_HEADER.pack(self._frame_id, time.time()) + jpeg)
        return self._frame_id

    async def next_message(self, timeout: Optional[float] = None) -> dict:
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        return json.loads(raw)


def check_health() -> int:
    import urllib.request

    url = server_url("/health", scheme="http")
    print(f"GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(resp.read().decode())
        return 0
    except Exception as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return 1


async def live(target: str, refer: str) -> None:
    """Stream the webcam to the server and print each progress/end message."""
    import cv2

    cfg = load_config()
    width, height = cfg["frame"]["frame_width"], cfg["frame"]["frame_height"]
    quality = int(cfg["frame"]["jpeg_quality"])
    cap = cv2.VideoCapture(cfg["camera"]["source"])
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {cfg['camera']['source']!r}")

    async with VisionClient() as vc:
        print(f"connected to {vc.url}")
        session_id = await vc.start(target, refer)
        print(f"session {session_id}")
        sent_at: dict[int, float] = {}
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                fid = await vc.send_frame(buf.tobytes())
                sent_at[fid] = time.perf_counter()
                try:
                    msg = await vc.next_message(timeout=2.0)
                except asyncio.TimeoutError:
                    print("no reply in 2 s")
                    continue
                if msg["type"] == "progress":
                    rtt = (time.perf_counter() - sent_at.pop(msg["frame_id"], time.perf_counter())) * 1000
                    print(
                        f"frame {msg['frame_id']:5d}  box={msg['target_box']}  conf={msg['confidence']:.2f}  "
                        f"hand={msg['hand_point']}  rtt={rtt:.0f}ms  server={msg['server_ms']:.0f}ms  "
                        f"net={rtt - msg['server_ms']:.0f}ms  {msg['timings_ms']}"
                    )
                else:
                    print(json.dumps(msg, indent=2))
                    if msg["type"] in ("end", "error"):
                        break
        finally:
            cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--health", action="store_true", help="only call /health")
    parser.add_argument("--target", default="sheet of paper")
    parser.add_argument("--refer", default="")
    args = parser.parse_args()
    try:
        if args.health:
            raise SystemExit(check_health())
        asyncio.run(live(args.target, args.refer))
    except ConfigError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
