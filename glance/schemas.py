"""Dataclasses: single source of truth for messages passed between modules.

Do not pass loose dicts between modules -- construct and pass these instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

StepType = Literal["audio", "haptic", "read"]

# Reach session contract, settled in CLAUDE.md "Object detection and tracking
# (settled)". Supersedes the earlier start_reach/stop + step_done/user_stop
# event sketch -- keep these names identical everywhere (planner prompt,
# orchestrator, vision loop, server, logs).
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
class Step:
    type: StepType
    text: Optional[str] = None            # for "audio"
    action: Optional[str] = None          # for "haptic", always "reach"
    target: Optional[str] = None          # for "haptic": 1-4 word noun phrase, no positions
    refer: Optional[str] = None           # for "haptic": fuller description, for disambiguation only
    region: Optional[str] = None          # for "read", e.g. "held_object"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)


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
