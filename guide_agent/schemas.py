"""Messages that cross the controller boundary (CLAUDE.md sections 5 and 10)."""
from __future__ import annotations

from dataclasses import dataclass, field

HAPTIC_STATUSES = (
    "success", "target_not_found", "target_lost", "hand_lost", "stalled", "timeout", "stale_link",
)


@dataclass
class Frame:
    id: str
    image: bytes | None  # JPEG; never logged
    captured_at: float  # clock seconds


@dataclass
class LastSeen:
    region: str  # image thirds, e.g. "right-middle"
    depth: str  # near | mid | far
    age_s: float


@dataclass
class HapticResult:
    session_id: str
    status: str  # one of HAPTIC_STATUSES, or "cancelled" (never reaches the model)
    inferred: bool = False
    elapsed_s: float = 0.0
    target: str = ""
    last_seen: LastSeen | None = None
    closest_approach: str = "far"  # touching | near | far
    frame: Frame | None = None  # null on success

    def model_view(self) -> dict:
        """The event JSON shown to the model: everything except session_id and frame."""
        last_seen = None
        if self.last_seen is not None:
            last_seen = {
                "region": self.last_seen.region,
                "depth": self.last_seen.depth,
                "age_s": self.last_seen.age_s,
            }
        return {
            "event": "haptic_result",
            "status": self.status,
            "inferred": self.inferred,
            "elapsed_s": self.elapsed_s,
            "target": self.target,
            "last_seen": last_seen,
            "closest_approach": self.closest_approach,
        }

    def log_view(self) -> dict:
        view = self.model_view()
        view["session_id"] = self.session_id
        view["frame_id"] = self.frame.id if self.frame else None
        return view


# Triggers: what starts a turn (section 6).


@dataclass
class UtteranceTrigger:
    text: str
    frame: Frame | None
    arrived_at: float
    intercepted: str = "none"  # "halt" when the halt intercept runs a model turn
    pre_outputs: list = field(default_factory=list)  # output calls the intercept made before the turn


@dataclass
class HapticTrigger:
    event: HapticResult
    arrived_at: float


@dataclass
class IdleTrigger:
    checkin: int
    arrived_at: float


Trigger = UtteranceTrigger | HapticTrigger | IdleTrigger
