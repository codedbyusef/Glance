"""Dataclasses: single source of truth for messages passed between modules.

Do not pass loose dicts between modules -- construct and pass these instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

StepType = Literal["audio", "haptic", "read"]
EventName = Literal[
    "step_done",
    "target_not_found",
    "target_lost",
    "hand_lost",
    "timeout",
    "user_stop",
]


@dataclass
class Step:
    type: StepType
    text: Optional[str] = None            # for "audio"
    action: Optional[str] = None          # for "haptic", always "reach"
    target: Optional[str] = None          # for "haptic"
    success: Optional[str] = None         # e.g. "hand_on_target"
    region: Optional[str] = None          # for "read", e.g. "held_object"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)


@dataclass
class Command:
    cmd: str                              # "start_reach", "stop"
    target: Optional[str] = None
    success: Optional[str] = None


@dataclass
class Event:
    event: EventName


@dataclass
class FrameResult:
    frame_id: int
    target_box: Optional[tuple[float, float, float, float]] = None
    hand_point: Optional[tuple[float, float]] = None
    hand_depth: Optional[float] = None
    target_depth: Optional[float] = None
    confidence: Optional[float] = None
    timings_ms: dict[str, float] = field(default_factory=dict)
