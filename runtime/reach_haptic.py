"""ReachHaptic: the agent's HapticLoop on top of object_detection's ReachAPI (Step 3).

    ReachStartError -> HapticStartError    (the controller turns it into stale_link)
    ReachResult     -> HapticResult        (same field names; LastSeen and ReachFrame -> Frame map one to one)
    results         -> controller.on_haptic_result through dispatch; `cancelled` is dropped (the controller
                       ignores it anyway); `ambiguous` (off by default on the server) becomes target_not_found
    progress        -> runtime.guidance -> GloveOutput; pause, cancel, end and errors -> all-off

The controller calls start/pause/resume/cancel on its event loop thread. They are synchronous (the HapticLoop
contract): start blocks while the WebSocket connects (milliseconds; at most ReachAPI's start timeout).
ReachAPI's callbacks arrive on its own thread.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from guide_agent.interfaces import HapticStartError
from guide_agent.schemas import Frame, HapticResult, LastSeen

from .glove_out import GloveOutput
from .guidance import Guidance, GuidanceState

STATUS_MAP = {"ambiguous": "target_not_found"}


def to_haptic_result(r) -> HapticResult:
    """object_detection.reach_api.ReachResult -> guide_agent.schemas.HapticResult."""
    last_seen = None
    if r.last_seen is not None:
        last_seen = LastSeen(region=r.last_seen.region, depth=r.last_seen.depth, age_s=float(r.last_seen.age_s))
    frame = None
    if r.frame is not None:
        frame = Frame(id=r.frame.id, image=r.frame.image, captured_at=r.frame.captured_at)
    return HapticResult(session_id=r.session_id, status=STATUS_MAP.get(r.status, r.status), inferred=bool(r.inferred),
                        elapsed_s=float(r.elapsed_s), target=r.target, last_seen=last_seen,
                        closest_approach=r.closest_approach, frame=frame)


class ReachHaptic:
    def __init__(self, api_factory: Callable, dispatch: Callable, on_result: Callable[[HapticResult], None],
                 guidance: Guidance, glove: GloveOutput, on_frame: Optional[Callable[[dict, GuidanceState], None]] = None):
        """api_factory(on_result, on_progress) -> a ReachAPI (or a fake with its start/pause/resume/cancel/close).
        on_frame(progress, state): optional per-frame hook (timings and logs), on the API thread."""
        self.dispatch = dispatch
        self.on_result_cb = on_result
        self.guidance = guidance
        self.glove = glove
        self.on_frame = on_frame
        self._lock = threading.Lock()
        self.session_id: Optional[str] = None
        self.target = ""
        self.refer = ""
        self.paused = False
        self.progress: Optional[dict] = None
        self.state = GuidanceState(mode=guidance.mode)
        self.last_status = ""
        self.error = ""
        self.api = api_factory(self._on_result, self._on_progress)

    # ---------------------------------------------------------------- HapticLoop (controller's loop thread)

    def start(self, target: str, refer: str) -> str:
        self.glove.all_off()
        with self._lock:
            self.guidance.reset()
            self.progress, self.state, self.paused, self.error = None, GuidanceState(mode=self.guidance.mode), False, ""
            self.target, self.refer = target, refer
        try:
            session_id = self.api.start(target, refer)
        except Exception as e:  # ReachStartError (server down, bad URL or token) or anything else
            with self._lock:
                self.session_id, self.error = None, f"start failed: {e}"
            raise HapticStartError(str(e)) from e
        with self._lock:
            self.session_id, self.last_status = session_id, ""
        return session_id

    def pause(self, session_id: str) -> None:
        if session_id != self.session_id:
            return
        with self._lock:
            self.paused = True
        self.glove.all_off()
        self.api.pause(session_id)

    def resume(self, session_id: str) -> None:
        if session_id != self.session_id:
            return
        self.api.resume(session_id)
        with self._lock:
            self.paused = False

    def cancel(self, session_id: str) -> None:
        if session_id != self.session_id:
            return
        with self._lock:
            self.session_id, self.last_status = None, "cancelled"
        self.glove.all_off()
        self.api.cancel(session_id)  # its `cancelled` result is dropped in _on_result

    def close(self) -> None:
        self.glove.all_off()
        try:
            self.api.close()
        finally:
            self.glove.all_off()

    # ---------------------------------------------------------------- ReachAPI callbacks (API thread)

    def _on_progress(self, msg: dict) -> None:
        try:
            with self._lock:
                if self.paused or msg.get("session_id") != self.session_id:
                    return
                state = self.guidance.update(msg)
                self.progress, self.state = msg, state
            self.glove.apply(state.cmd)
            if self.on_frame is not None:
                self.on_frame(msg, state)
        except Exception as e:  # never leave a motor on after a bug
            self.glove.all_off()
            with self._lock:
                self.error = f"guidance: {e!r}"

    def _on_result(self, result) -> None:
        self.glove.all_off()
        with self._lock:
            if result.session_id == self.session_id:
                self.session_id = None
            self.last_status = result.status
        if result.status == "cancelled":
            return
        self.dispatch(self.on_result_cb, to_haptic_result(result))

    # ---------------------------------------------------------------- view

    def toggle_mode(self) -> str:
        with self._lock:
            mode = self.guidance.toggle_mode()
            self.state = GuidanceState(mode=mode)
        return mode

    def describe(self) -> str:
        with self._lock:
            if self.session_id is None:
                tail = f" ({self.last_status})" if self.last_status else ""
                return f"reach: none{tail}"
            return f"REACH: {self.target} ({self.refer}) {'PAUSED' if self.paused else 'RUNNING'}"

    def snapshot(self) -> dict:
        """Same shape as live_guidance.ReachLoop.snapshot(), for the shared drawing helpers."""
        with self._lock:
            return {"progress": self.progress if self.session_id else None, "state": self.state, "result": None,
                    "paused": self.paused, "session_id": self.session_id, "error": self.error, "target": self.target}
