"""Session state machine: idle, listening, planning, audio step, haptic step, replanning.

Runs plan steps in order and routes vision-loop events to the planner for a replan.
"""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PLANNING = auto()
    AUDIO_STEP = auto()
    HAPTIC_STEP = auto()
    REPLANNING = auto()


class Orchestrator:
    def __init__(self):
        self.state = State.IDLE

    def run(self) -> None:
        raise NotImplementedError
