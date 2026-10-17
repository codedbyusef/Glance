"""WebSocket client to the Vultr perception server.

Sends JPEG frames with frame_id + client timestamp; only ever keeps the newest
frame in flight, matching the server's drop-stale-frames contract.
"""

from __future__ import annotations

from glance.schemas import FrameResult


class PerceptionClient:
    def __init__(self, url: str):
        self.url = url

    async def send_frame(self, frame: bytes, frame_id: int) -> FrameResult:
        raise NotImplementedError
