"""Webcam-only / canned FrameResult stream so the vision loop and controller
can be built and tested before the Vultr perception server exists.
"""

from __future__ import annotations

from glance.schemas import FrameResult


class FakePerceptionClient:
    async def send_frame(self, frame: bytes, frame_id: int) -> FrameResult:
        raise NotImplementedError
