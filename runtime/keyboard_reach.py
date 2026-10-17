"""KeyboardReach: the agent's HapticLoop with keys standing in for the vision loop (Step 1).

The controller calls start/pause/resume/cancel on its event loop thread. Keys come from the UI or stdin
thread and end the running reach with a status; the HapticResult goes to the controller through dispatch.

    s success   l target_lost   h hand_lost   n target_not_found   t timeout   k stale_link
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Callable

from guide_agent.schemas import HapticResult, LastSeen

from .glove_out import GloveOutput

KEY_STATUS = {"s": "success", "l": "target_lost", "h": "hand_lost", "n": "target_not_found", "t": "timeout",
              "k": "stale_link"}


class KeyboardReach:
    def __init__(self, dispatch: Callable, on_result: Callable[[HapticResult], None], camera=None,
                 glove: GloveOutput | None = None, clock=time.monotonic):
        self.dispatch = dispatch
        self.on_result = on_result
        self.camera = camera
        self.glove = glove
        self.clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.session_id: str | None = None
        self.target = ""
        self.refer = ""
        self.state = "idle"  # idle | running | paused
        self._started = 0.0
        self.last_event = ""  # shown on screen: the last start/pause/resume/cancel/result

    # ---------------------------------------------------------------- HapticLoop (controller's loop thread)

    def start(self, target: str, refer: str) -> str:
        with self._lock:
            self.session_id = f"kr-{next(self._ids)}"
            self.target, self.refer, self.state = target, refer, "running"
            self._started = self.clock()
            self.last_event = f"start {self.session_id}"
            return self.session_id

    def pause(self, session_id: str) -> None:
        if self._is_current(session_id, "running"):
            with self._lock:
                self.state, self.last_event = "paused", f"pause {session_id}"
            self._all_off()

    def resume(self, session_id: str) -> None:
        if self._is_current(session_id, "paused"):
            with self._lock:
                self.state, self.last_event = "running", f"resume {session_id}"

    def cancel(self, session_id: str) -> None:
        if self._is_current(session_id):
            with self._lock:
                self.session_id, self.state, self.last_event = None, "idle", f"cancel {session_id}"
            self._all_off()

    # ---------------------------------------------------------------- keys (any thread)

    def key(self, k: str) -> bool:
        """End the running reach with the status for key k. Returns whether a result was sent."""
        status = KEY_STATUS.get(k)
        if status is None:
            return False
        with self._lock:
            if self.session_id is None or self.state != "running":
                return False
            session_id, target = self.session_id, self.target
            elapsed = round(self.clock() - self._started, 1)
            self.session_id, self.state, self.last_event = None, "idle", f"{status} {session_id}"
        self._all_off()
        success = status == "success"
        frame = None if success or self.camera is None else self.camera.latest_frame()
        result = HapticResult(
            session_id=session_id, status=status, inferred=False, elapsed_s=elapsed, target=target,
            last_seen=None if status in ("success", "target_not_found") else LastSeen("center-middle", "near", 1.0),
            closest_approach="touching" if success else "near", frame=frame)
        self.dispatch(self.on_result, result)
        return True

    # ---------------------------------------------------------------- view

    def describe(self) -> str:
        with self._lock:
            if self.session_id is None:
                return f"reach: none ({self.last_event})" if self.last_event else "reach: none"
            elapsed = self.clock() - self._started
            return f"REACH: {self.target} ({self.refer}) {self.state.upper()} {elapsed:.0f}s"

    def _is_current(self, session_id: str, state: str | None = None) -> bool:
        with self._lock:
            return session_id == self.session_id and (state is None or self.state == state)

    def _all_off(self) -> None:
        if self.glove is not None:
            self.glove.all_off()
