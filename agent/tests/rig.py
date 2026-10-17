"""Test rig: a controller wired to recording stubs, a fake clock and a scripted model."""
from __future__ import annotations

import asyncio

from guide_agent.config import Config
from guide_agent.controller import Controller
from guide_agent.models.base import ToolResults, UserInput
from guide_agent.models.scripted import ScriptedModel, call, plain, respond
from guide_agent.schemas import Frame, HapticResult
from guide_agent.state import Step, Task
from guide_agent.stubs import FakeClock, Recorder, StubAudio, StubCamera, StubHaptic
from guide_agent.turnlog import TurnLog

NOTES = "Step 1. Can on the table ahead."


def speak(text="Take two steps forward, then say done.", expect="done", notes=NOTES):
    args = {"text": text, "expect": expect}
    if notes is not None:
        args["notes"] = notes
    return call("speak", **args)


def reach(target="red soda can", refer="the red can on the table",
          say_first="Bring your right hand out in front of you, keep your head still, and follow the vibrations.",
          notes=NOTES):
    args = {"target": target, "refer": refer, "say_first": say_first}
    if notes is not None:
        args["notes"] = notes
    return call("reach", **args)


def start(goal="get the red can", replace_active=None):
    args = {"goal": goal}
    if replace_active is not None:
        args["replace_active"] = replace_active
    return call("start_task", **args)


def finish(outcome="success", summary="done", say=None):
    args = {"outcome": outcome, "summary": summary}
    if say is not None:
        args["say"] = say
    return call("finish_task", **args)


class Rig:
    def __init__(self, *script, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.clock = FakeClock()
        self.rec = Recorder(self.clock)
        self.audio = StubAudio(self.rec)
        self.haptic = StubHaptic(self.rec)
        self.camera = StubCamera()
        self.model = ScriptedModel(*script)
        self.log = TurnLog()
        self.ctl = Controller(self.cfg, self.model, self.audio, self.haptic, self.camera, self.clock,
                              system_prompt="SYSTEM", turnlog=self.log)
        self._frames = 0

    # inputs -----------------------------------------------------------------

    def frame(self, frame_id: str | None = None) -> Frame:
        self._frames += 1
        return Frame(id=frame_id or f"f-{self._frames}", image=b"jpeg", captured_at=self.clock.now())

    async def utter(self, text: str, frame: Frame | None | str = "auto", speech_start: bool = True,
                    settle: bool = True) -> None:
        """The user speaks: speech_start, then the utterance. settle also finishes the resulting playback."""
        if speech_start:
            self.ctl.on_speech_start()
        self.ctl.on_utterance(text, frame=self.frame() if frame == "auto" else frame)
        if settle:
            await self.settle()

    async def haptic_result(self, status: str, settle: bool = True, **fields) -> None:
        session_id = fields.pop("session_id", None) or self.ctl.step.session_id
        frame = fields.pop("frame", None if status == "success" else self.frame())
        self.ctl.on_haptic_result(HapticResult(session_id=session_id, status=status, frame=frame, **fields))
        if settle:
            await self.settle()

    async def drain(self) -> None:
        await asyncio.wait_for(self.ctl.drain(), timeout=2)

    async def settle(self) -> None:
        """Run turns to completion and let all playback finish, then check the section 8 invariant."""
        await self.drain()
        while self.audio.playing:
            self.audio.finish()
            await self.drain()
        assert self.ctl.invariant_holds(), f"invariant broken: {self.ctl.snapshot()}"

    async def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.ctl.tick()
        await self.drain()

    # setup helpers ------------------------------------------------------------

    def set_task(self, goal="get the red can", request="get me the red can", notes="", history=()) -> Task:
        now = self.clock.now()
        task = Task(id="t-setup", goal=goal, request=request, started_at=now, last_user_activity_at=now,
                    notes=notes, history=list(history))
        self.ctl.task = task
        return task

    async def pending_speak(self, text="Take two steps forward, then say done.", expect="done",
                            heard: bool = True) -> Step:
        """Put a speak step in place the way a real turn would: said, then heard (or cut off)."""
        task = self.ctl.task or self.set_task()
        step = Step(n=task.next_step_number(), type="speak", issued_at=self.clock.now(), text=text,
                    expect=expect, state="speaking")
        task.current_step = step
        self.ctl._say(text, expect, step=step, is_step_speech=True, then="arm_idle")
        if heard:
            self.audio.finish()
        else:
            self.ctl.on_speech_start()
            self.ctl.speech_start_deadline = None
        return step

    async def running_reach(self) -> Step:
        task = self.ctl.task or self.set_task()
        step = Step(n=task.next_step_number(), type="reach", issued_at=self.clock.now(), target="red soda can",
                    refer="the red can on the table", say_first="Bring your right hand out and follow the vibrations.",
                    state="starting")
        task.current_step = step
        self.ctl._say(step.say_first, "none", step=step, is_step_speech=True, then="go_reach")
        self.audio.finish()
        await self.drain()
        assert step.state == "running"
        return step

    # observations -------------------------------------------------------------

    def mark(self) -> int:
        return len(self.rec.calls)

    def names(self, start: int = 0) -> list[str]:
        return self.rec.names(start)

    def says(self, start: int = 0) -> list[tuple[str, str]]:
        return [(c.args["text"], c.args["expect"]) for c in self.rec.of("say", start)]

    def turns(self) -> list[dict]:
        return self.log.turns()

    def last_turn(self) -> dict:
        return self.turns()[-1]

    def warnings(self) -> list[str]:
        return [w for t in self.turns() for w in t["warnings"]]

    def input_text(self, request_index: int = -1) -> str:
        first = self.model.requests[request_index].messages[0]
        assert isinstance(first, UserInput)
        return first.text

    def tool_results(self, request_index: int = -1) -> list[dict]:
        messages = self.model.requests[request_index].messages
        results = [m for m in messages if isinstance(m, ToolResults)]
        return [r.result for r in results[-1].results] if results else []


__all__ = ["Rig", "speak", "reach", "start", "finish", "plain", "respond", "call", "NOTES"]
