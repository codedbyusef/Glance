"""Per-frame reach loop: pulls frames, sends them to the perception server,
drives the haptic link from each FrameResult, and surfaces the ReachReport
when the server ends the session (see the settled session contract in
CLAUDE.md "Object detection and tracking (settled)").
"""

from __future__ import annotations

from glance.schemas import ReachCommand, ReachReport


class VisionLoop:
    def start_reach(self, cmd: ReachCommand) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def resume(self) -> None:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def poll_result(self) -> ReachReport | None:
        raise NotImplementedError
