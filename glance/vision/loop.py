"""Per-frame reach loop: pulls frames, calls the perception client and controller,
drives the haptic link, and emits events (step_done, target_lost, hand_lost, ...).
"""

from __future__ import annotations

from glance.schemas import Command, Event


class VisionLoop:
    def start_reach(self, cmd: Command) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def poll_event(self) -> Event | None:
        raise NotImplementedError
