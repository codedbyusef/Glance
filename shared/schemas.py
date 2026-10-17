"""Dataclasses: single source of truth for messages passed between modules.

Do not pass loose dicts between modules -- construct and pass these instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# Reach session contract, settled in detection/CLAUDE.md "Object detection and
# tracking (settled)". Keep these names identical everywhere (agent, runtime
# vision loop, server, logs).
CommandType = Literal["start", "pause", "resume", "cancel"]
EndStatus = Literal[
    "success",
    "target_not_found",
    "target_lost",
    "hand_lost",
    "stalled",
    "timeout",
    "stale_link",
]
DepthBand = Literal["near", "mid", "far"]
Approach = Literal["touching", "near", "far"]


@dataclass
class ReachCommand:
    cmd: CommandType
    session_id: Optional[str] = None      # required for pause/resume/cancel
    target: Optional[str] = None          # required for "start"
    refer: Optional[str] = None           # required for "start"


@dataclass
class FrameResult:
    frame_id: int
    target_box: Optional[tuple[float, float, float, float]] = None
    hand_point: Optional[tuple[float, float]] = None
    hand_depth: Optional[float] = None
    target_depth: Optional[float] = None
    confidence: Optional[float] = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class LastSeen:
    region: str                           # image thirds, e.g. "right-middle"
    depth: DepthBand
    age_s: float


@dataclass
class ReachReport:
    """Terminal report for a reach session -- sent once, when it ends."""

    session_id: str
    status: EndStatus
    inferred: bool = False                # true for the "hand covered the target as it vanished" success case
    elapsed_s: float = 0.0
    last_seen: Optional[LastSeen] = None
    closest_approach: Optional[Approach] = None
    last_frame: Optional[FrameResult] = None   # omitted (None) on success
