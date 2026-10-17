"""Task state (CLAUDE.md section 8). The controller owns it; the model only sees it rendered."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SPEAK_STATES = ("speaking", "awaiting_reply")
REACH_STATES = ("starting", "running", "paused")


@dataclass
class Step:
    n: int
    type: str  # speak | reach
    issued_at: float
    # speak
    text: str = ""
    expect: str = ""  # done | answer
    # reach
    target: str = ""
    refer: str = ""
    say_first: str = ""
    session_id: str | None = None
    state: str = ""  # speak: speaking | awaiting_reply; reach: starting | running | paused
    heard_fully: bool = False
    result: str | None = None  # set when the step ends

    @property
    def speech(self) -> str:
        """What the user hears for this step: the speak text or the reach say_first."""
        return self.text if self.type == "speak" else self.say_first

    @property
    def pending(self) -> bool:
        if self.type == "speak":
            return self.state in SPEAK_STATES
        return self.state in REACH_STATES


@dataclass
class Task:
    id: str
    goal: str
    request: str
    started_at: float
    last_user_activity_at: float
    current_step: Step | None = None
    history: list[Step] = field(default_factory=list)
    notes: str = ""
    outcome: str | None = None  # success | failed | cancelled | superseded
    summary: str = ""

    def next_step_number(self) -> int:
        return len(self.history) + 1

    def end_current_step(self, result: str) -> None:
        step = self.current_step
        if step is None:
            return
        step.result = result
        self.history.append(step)
        self.current_step = None

    def snapshot(self) -> dict:
        return asdict(self)
