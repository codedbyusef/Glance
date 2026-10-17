"""The controller (CLAUDE.md sections 6-9, 12 and 13).

Event handlers (on_speech_start, on_utterance, on_haptic_result, on_playback_done, tick) are synchronous and
must be called from the asyncio loop. Model turns run one at a time as asyncio tasks. Tool execution never
spans an await, so a turn is only ever abandoned or held at a model-call boundary.
"""
from __future__ import annotations

import asyncio
import itertools
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

from .compose import Entry, Transcript, compose_input, prepare_image
from .config import Config
from .intercepts import CANCEL, HALT, REPEAT, classify, normalize
from .models.base import ModelAdapter, ModelResponse, ToolCall, ToolResult, ToolResults, UserInput
from .schemas import Frame, HapticResult, HapticTrigger, IdleTrigger, Trigger, UtteranceTrigger
from .state import Step, Task
from .tools import TERMINAL_TOOLS, TOOL_SPECS, ToolError, validate, word_count
from .turnlog import TurnLog

SAY_AGAIN = "Sorry, I had a problem. Please say that again."
KEEP_GOING = "Sorry, I had a problem. Do you want to keep going?"
CANCELLED = "Okay, cancelled."
EMPTY_NUDGE = "Your last response was empty. Call a tool, or answer in plain text."
SKIPPED = {"status": "skipped", "reason": "not executed because an earlier call in this response failed"}

ARM_IDLE = "arm_idle"
GO_REACH = "go_reach"


@dataclass
class Speech:
    """A say() whose playback report is still due."""
    text: str
    expect: str
    task_id: str | None = None
    step_n: int | None = None  # the step this speech carries, if any
    is_step_speech: bool = False  # the step's own text or say_first
    then: str | None = None  # ARM_IDLE or GO_REACH, run when playback completes
    entry: Entry | None = None  # transcript line, marked when cut off


@dataclass
class Turn:
    id: int
    trigger: Trigger | None
    kind: str = "turn"  # "event" for log records that are not turns
    abandoned: bool = False
    held: bool = False
    release: asyncio.Event = field(default_factory=asyncio.Event)
    log: dict = field(default_factory=dict)
    first_output_at: float | None = None
    quiet: bool = False  # event records with no outputs are not written


class Controller:
    def __init__(
        self,
        cfg: Config,
        model: ModelAdapter,
        audio,
        haptic,
        camera,
        clock,
        system_prompt: str,
        turnlog: TurnLog | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.audio = audio
        self.haptic = haptic
        self.camera = camera
        self.clock = clock
        self.system_prompt = system_prompt
        self.turnlog = turnlog or TurnLog()

        self.task: Task | None = None
        self.closed_tasks: list[Task] = []
        self.transcript = Transcript()
        self.last_spoken: Speech | None = None
        self.speeches: dict[str, Speech] = {}
        self.queue: deque[Trigger] = deque()
        self.turn: Turn | None = None
        self.idle_deadline: float | None = None
        self.idle_checkins = 0
        self.speech_start_deadline: float | None = None
        self.last_user_text = ""
        self.errors: list[BaseException] = []

        self._cur: Turn | None = None
        self._running: set[asyncio.Task] = set()
        self._failed_sessions: set[str] = set()
        self._turn_ids = itertools.count(1)
        self._task_ids = itertools.count(1)
        audio.set_listener(self.on_playback_done)

    # ------------------------------------------------------------------ inputs

    def on_speech_start(self, at: float | None = None) -> None:
        now = self.clock.now()
        with self._instant("speech_start"):
            if self.task is not None:
                self.task.last_user_activity_at = now
            self.speech_start_deadline = now + self.cfg.speech_start_timeout_s
            self.idle_deadline = None
            self._stop_speech()
            step = self.step
            if step is not None and step.type == "reach" and step.state == "running":
                self._out("haptic.pause", session_id=step.session_id)
                self.haptic.pause(step.session_id)
                step.state = "paused"
            if self.turn is not None:  # decision 5: hold the in-flight turn
                self.turn.held = True
                self.turn.release.clear()

    def on_utterance(self, text: str, ended_at: float | None = None, frame: Frame | None = None) -> None:
        now = self.clock.now()
        self.speech_start_deadline = None
        trigger = UtteranceTrigger(text=text.strip(), frame=frame, arrived_at=now)
        if not normalize(text, self.cfg.assistant_name):
            with self._instant("empty_utterance"):
                self._release_or_restore()
            return

        self.idle_checkins = 0
        self.idle_deadline = None
        self.last_user_text = trigger.text
        if self.task is not None:
            self.task.last_user_activity_at = now
        if self.turn is not None:
            self._abandon_turn()

        intercept = classify(text, self.cfg.assistant_name)
        if intercept == CANCEL:
            self._intercept_cancel(trigger)
        elif intercept == REPEAT:
            self._intercept_repeat(trigger)
        elif intercept == HALT:
            self._intercept_halt(trigger)
        else:
            self.queue.appendleft(trigger)  # an utterance takes the next turn
        self._pump()

    def on_haptic_result(self, event: HapticResult) -> None:
        if event.status == "cancelled":
            self.turnlog.write({"kind": "event", "event": "haptic_result_dropped", "reason": "cancelled",
                                "session_id": event.session_id, "at": self.clock.now()})
            return
        self.queue.append(HapticTrigger(event=event, arrived_at=self.clock.now()))
        self._pump()

    def on_playback_done(self, speech_id: str, status: str) -> None:
        sp = self.speeches.pop(speech_id, None)
        if sp is None:
            return
        if status != "completed" and sp.entry is not None:
            sp.entry.cut_off = True
        step = self.step
        if step is None or sp.task_id != self.task.id or sp.step_n != step.n:
            return
        if sp.is_step_speech:
            step.heard_fully = status == "completed"
        if step.type == "speak" and not self._speech_pending(step):
            step.state = "awaiting_reply"
        if status == "completed" and sp.then:
            with self._instant("playback_done", quiet=True):
                self._then(sp.then, step)

    def tick(self) -> None:
        """Check the timers. The runtime calls this often; tests call it after advancing a fake clock."""
        now = self.clock.now()
        if self.speech_start_deadline is not None and now >= self.speech_start_deadline:
            self.speech_start_deadline = None
            with self._instant("speech_start_timeout"):
                self._release_or_restore()
        if self.turn is None and self.idle_deadline is not None and now >= self.idle_deadline:
            self.idle_deadline = None
            self._fire_idle()
        task = self.task
        if task is not None and self.turn is None and now - task.last_user_activity_at >= self.cfg.task_inactivity_close_s:
            with self._instant("inactivity_close"):
                self._close_task("cancelled", "closed after inactivity")

    def fire_idle_timeout(self) -> bool:
        """Expire the idle timer now (harness and eval). Returns whether an idle turn was queued."""
        self.idle_deadline = None
        return self._fire_idle()

    async def drain(self) -> None:
        """Wait until no turn is running or queued, including abandoned ones. Re-raises turn errors."""
        while True:
            pending = [t for t in self._running if not t.done()]
            if pending:
                await asyncio.wait(pending)
                continue
            if self.turn is None and self.queue:
                self._pump()
                continue
            if self.turn is None:
                break
            await asyncio.sleep(0)
        if self.errors:
            raise self.errors.pop(0)

    # ------------------------------------------------------------------ state helpers

    @property
    def step(self) -> Step | None:
        return self.task.current_step if self.task is not None else None

    def invariant_holds(self) -> bool:
        """Section 8: no task, or an active task with exactly one pending step."""
        return self.task is None or (self.task.current_step is not None and self.task.current_step.pending)

    def snapshot(self) -> dict:
        return {
            "task": self.task.snapshot() if self.task is not None else None,
            "idle_checkins": self.idle_checkins,
            "idle_armed": self.idle_deadline is not None,
            "queued": len(self.queue),
        }

    def _speech_pending(self, step: Step) -> bool:
        return any(s.task_id == self.task.id and s.step_n == step.n for s in self.speeches.values())

    # ------------------------------------------------------------------ intercepts (section 7)

    def _intercept_cancel(self, trigger: UtteranceTrigger) -> None:
        with self._instant(CANCEL, trigger, kind="turn"):
            self._record_trigger(trigger, CANCEL)
            self._stop_speech()
            if self.task is not None:
                self._close_task("cancelled", "cancelled by the user")
                self._say(CANCELLED, "none")

    def _intercept_repeat(self, trigger: UtteranceTrigger) -> None:
        with self._instant(REPEAT, trigger, kind="turn"):
            self._record_trigger(trigger, REPEAT)
            step = self.step
            last = self.last_spoken
            said = None
            if last is not None:
                is_step = bool(
                    step is not None and last.is_step_speech
                    and last.task_id == self.task.id and last.step_n == step.n
                )
                if is_step:
                    step.heard_fully = True
                said = self._say(last.text, last.expect, step=step if is_step else None, is_step_speech=is_step)
            self._restore(after=said)

    def _intercept_halt(self, trigger: UtteranceTrigger) -> None:
        if self.task is None:
            with self._instant(HALT, trigger, kind="turn"):
                self._record_trigger(trigger, HALT)
                self._stop_speech()
            return
        trigger.intercepted = HALT
        turn = self._new_turn(trigger)
        turn.log["intercept"] = HALT
        with self._acting(turn):
            self._stop_speech()
            step = self.step
            if step is not None:
                self._cancel_reach(step)
                self.task.end_current_step("interrupted")
        self._start_turn(turn)

    # ------------------------------------------------------------------ turns (section 6)

    def _pump(self) -> None:
        if self.turn is None and self.queue:
            self._start_turn(self._new_turn(self.queue.popleft()))

    def _start_turn(self, turn: Turn) -> None:
        self.turn = turn
        task = asyncio.get_running_loop().create_task(self._run_turn(turn))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    def _abandon_turn(self) -> None:
        turn = self.turn
        turn.abandoned = True
        turn.held = False
        turn.log["abandoned"] = True
        turn.release.set()
        self.turn = None

    def _release_or_restore(self) -> None:
        """An empty utterance or a speech_start timeout: release a held turn, else restore the pending step."""
        turn = self.turn
        if turn is not None:
            if turn.held:
                turn.held = False
                turn.release.set()
            return
        self._restore()

    async def _run_turn(self, turn: Turn) -> None:
        try:
            with self._acting(turn):
                messages = self._begin_turn(turn)
            if messages is not None:
                await self._model_loop(turn, messages)
        except Exception as exc:  # a controller bug: keep running, surface it through drain()
            self.errors.append(exc)
            turn.log["error"] = repr(exc)
        finally:
            self._finish_log(turn)
            if self.turn is turn:
                self.turn = None
                self._pump()

    def _begin_turn(self, turn: Turn) -> list | None:
        """Synchronous start of a turn: staleness checks, then the composed input. None drops the trigger."""
        trigger = turn.trigger
        now = self.clock.now()
        self.idle_deadline = None  # frozen during turns; playback of the turn's speech re-arms it
        if isinstance(trigger, HapticTrigger):
            event = trigger.event
            step = self.step
            if step is None or step.type != "reach" or step.session_id != event.session_id:
                turn.kind = turn.log["kind"] = "event"
                turn.log["event"] = "haptic_result_dropped"
                turn.log["reason"] = "stale session"
                return None
            self.task.end_current_step(event.status + (" (inferred)" if event.inferred else ""))
            frame = None if event.status == "success" else event.frame
        elif isinstance(trigger, IdleTrigger):
            step = self.step
            if step is None or step.type != "speak":
                turn.kind = turn.log["kind"] = "event"
                turn.log["event"] = "idle_timeout_dropped"
                return None
            frame = self.camera.latest_frame()
        else:
            frame = trigger.frame
            self._out("cue", name="thinking")
            self.audio.cue("thinking")

        text = compose_input(task=self.task, transcript=self.transcript, trigger=trigger, frame=frame, now=now,
                             cfg=self.cfg)
        intercept = trigger.intercepted if isinstance(trigger, UtteranceTrigger) else "none"
        self._record_trigger(trigger, None if intercept == "none" else intercept)
        turn.log["input"] = text
        turn.log["frame_id"] = frame.id if frame is not None else None
        image = prepare_image(frame.image, self.cfg.frame_max_long_edge_px) if frame is not None else None
        return [UserInput(text=text, image=image)]

    async def _model_loop(self, turn: Turn, messages: list) -> None:
        for _ in range(self.cfg.max_model_calls_per_turn):
            response: ModelResponse | None = None
            try:
                response = await self.model.generate(self.system_prompt, messages, TOOL_SPECS, self.cfg.temperature)
            except AssertionError:
                raise
            except Exception as exc:
                turn.log["model_calls"].append({"error": repr(exc)})
            else:
                turn.log["model_calls"].append({
                    "raw": response.raw_log if response.raw_log is not None else _response_summary(response),
                    "latency_ms": response.latency_ms,
                    "tokens": response.tokens,
                })
            if turn.held:
                await turn.release.wait()
            if turn.abandoned:
                return
            if response is None:
                continue
            with self._acting(turn):
                if self._handle_response(turn, response, messages):
                    return
        with self._acting(turn):
            self._turn_cap(turn)

    def _handle_response(self, turn: Turn, response: ModelResponse, messages: list) -> bool:
        """Execute one model response. Returns True when a terminal action executed."""
        text = (response.text or "").strip()
        if not response.tool_calls:
            if text:
                turn.log["terminal"] = "plain"
                turn.log["spoken"] = text
                self._plain_answer(text)
                return True
            self._info("empty_response")
            messages += [response.as_turn(), UserInput(EMPTY_NUDGE)]
            return False

        results: list[ToolResult] = []
        terminal: ToolCall | None = None
        failed = extra_terminal = False
        for tool_call in response.tool_calls:
            entry = {"name": tool_call.name, "args": tool_call.args}
            turn.log["tool_calls"].append(entry)
            if terminal is not None:
                entry["result"] = "dropped"
                extra_terminal = extra_terminal or tool_call.name in TERMINAL_TOOLS
                continue
            if failed:  # decision 2
                entry["result"] = "skipped"
                results.append(ToolResult(tool_call, SKIPPED))
                continue
            try:
                out = self._execute(turn, tool_call)
            except ToolError as err:
                entry["result"] = err.as_result()
                results.append(ToolResult(tool_call, err.as_result()))
                failed = True
                continue
            entry["result"] = out if out is not None else "ok"
            if tool_call.name in TERMINAL_TOOLS:
                terminal = tool_call
                turn.log["terminal"] = tool_call.name
            else:
                results.append(ToolResult(tool_call, out))
        if extra_terminal:
            self._warn("extra_terminal")
        if text:
            self._info("text_with_tool")
        if terminal is not None:
            return True
        messages += [response.as_turn(), ToolResults(results)]
        return False

    def _turn_cap(self, turn: Turn) -> None:
        self._warn("turn_cap")
        turn.log["terminal"] = "turn_cap"
        if self.task is not None and self.step is None:
            self._new_speak_step(KEEP_GOING, "answer")
        elif self.step is not None:
            self._restore(answer=SAY_AGAIN)
        else:
            self._say(SAY_AGAIN, "none")

    def _plain_answer(self, text: str) -> None:
        if word_count(text) > self.cfg.answer_max_words:
            self._warn("answer_too_long")
        if self.task is None:
            self._say(text, "none")
        elif self.step is not None:
            self._restore(answer=text)
        else:
            self._warn("no_pending_step")
            self._new_speak_step(text, "answer")

    # ------------------------------------------------------------------ tools (section 9)

    def _execute(self, turn: Turn, tool_call: ToolCall) -> dict | None:
        args = validate(tool_call.name, tool_call.args)
        if tool_call.name == "start_task":
            return self._start_task(turn, args)
        if self.task is None:
            raise ToolError("no_active_task", "No task is active. Answer in plain text, or call start_task first.")
        if tool_call.name == "speak":
            turn.log["spoken"] = args["text"]
            self._replace_step(turn)
            self._set_notes(args.get("notes"))
            self._new_speak_step(args["text"], args["expect"])
        elif tool_call.name == "reach":
            turn.log["spoken"] = args["say_first"]
            self._replace_step(turn)
            self._set_notes(args.get("notes"))
            self._new_reach_step(args["target"], args["refer"], args["say_first"])
        else:
            turn.log["spoken"] = args.get("say") or ""
            self._replace_step(turn)
            self._close_task(args["outcome"], args["summary"])
            say = args.get("say")
            if say:
                if word_count(say) > self.cfg.answer_max_words:
                    self._warn("answer_too_long")
                self._say(say, "none")
        return None

    def _start_task(self, turn: Turn, args: dict) -> dict:
        if self.task is not None and not args["replace_active"]:
            raise ToolError(
                "task_active",
                f"A task is already active (goal: {self.task.goal}). Give its next step with speak or reach, "
                "or set replace_active true only if the user clearly wants to switch tasks.",
            )
        if self.task is not None:
            self._close_task("superseded", "replaced by a new task")
        now = self.clock.now()
        trigger = turn.trigger
        request = trigger.text if isinstance(trigger, UtteranceTrigger) else self.last_user_text
        self.task = Task(id=f"t-{next(self._task_ids)}", goal=args["goal"], request=request,
                         started_at=now, last_user_activity_at=now)
        self.transcript.add(now, "event", f"task started: {args['goal']}")
        return {"task_id": self.task.id}

    def _new_speak_step(self, text: str, expect: str) -> None:
        step = Step(n=self.task.next_step_number(), type="speak", issued_at=self.clock.now(),
                    text=text, expect=expect, state="speaking")
        self.task.current_step = step
        if word_count(text) > self.cfg.step_max_words:
            self._warn("step_too_long")
        self._say(text, expect, step=step, is_step_speech=True, then=ARM_IDLE)

    def _new_reach_step(self, target: str, refer: str, say_first: str) -> None:
        step = Step(n=self.task.next_step_number(), type="reach", issued_at=self.clock.now(),
                    target=target, refer=refer, say_first=say_first, state="starting")
        self.task.current_step = step
        if word_count(say_first) > self.cfg.step_max_words:
            self._warn("step_too_long")
        self._say(say_first, "none", step=step, is_step_speech=True, then=GO_REACH)

    def _replace_step(self, turn: Turn) -> None:
        """End the current step because a new step or finish_task replaces it."""
        step = self.step
        if step is None:
            return
        self._cancel_reach(step)
        trigger = turn.trigger
        if isinstance(trigger, UtteranceTrigger):
            result = f"user: {trigger.text}"
        elif isinstance(trigger, IdleTrigger):
            result = "no reply"
        else:
            result = "replaced"
        self.task.end_current_step(result)

    def _set_notes(self, notes) -> None:
        if not isinstance(notes, str) or not notes.strip():
            self._warn("missing_notes")
            return
        if len(notes) > self.cfg.notes_max_chars:
            self._warn("notes_truncated")
            notes = notes[: self.cfg.notes_max_chars]
        self.task.notes = notes

    def _close_task(self, outcome: str, summary: str = "") -> None:
        task = self.task
        if task.current_step is not None:
            self._cancel_reach(task.current_step)
            task.end_current_step(outcome)
        task.outcome = outcome
        task.summary = summary
        self.closed_tasks.append(task)
        self.task = None
        self.idle_deadline = None
        self.transcript.add(self.clock.now(), "event", f"task closed: {outcome}")

    # ------------------------------------------------------------------ restore and reach (section 13)

    def _restore(self, answer: str | None = None, after: Speech | None = None) -> None:
        """Restore the pending step, optionally after a plain answer, or after speech already said (repeat)."""
        step = self.step
        if step is None:
            if answer:
                self._say(answer, "none")
            return
        then = ARM_IDLE if step.type == "speak" else GO_REACH
        expect = step.expect if step.type == "speak" else "none"
        if not step.heard_fully:
            if answer:
                self._say(answer, "none")
            step.heard_fully = True  # re-said, so it now counts as heard
            self._say(step.speech, expect, step=step, is_step_speech=True, then=then)
        elif answer:
            self._say(answer, expect, step=step, then=then)
        elif after is not None:
            after.task_id, after.step_n, after.then = self.task.id, step.n, then
        else:
            self._then(then, step)

    def _then(self, action: str, step: Step) -> None:
        if action == ARM_IDLE:
            self._arm_idle()
        elif action == GO_REACH:
            if step.session_id is None:
                self._start_reach(step)
            elif step.state == "paused":
                self._out("haptic.resume", session_id=step.session_id)
                self.haptic.resume(step.session_id)
                step.state = "running"

    def _arm_idle(self) -> None:
        step = self.step
        if step is None or step.type != "speak" or self.idle_checkins >= self.cfg.max_idle_checkins:
            self.idle_deadline = None
            return
        timeout = self.cfg.idle_timeout_done_s if step.expect == "done" else self.cfg.idle_timeout_answer_s
        self.idle_deadline = self.clock.now() + timeout

    def _fire_idle(self) -> bool:
        step = self.step
        if step is None or step.type != "speak" or self.idle_checkins >= self.cfg.max_idle_checkins:
            return False
        self.idle_checkins += 1
        self.queue.append(IdleTrigger(checkin=self.idle_checkins, arrived_at=self.clock.now()))
        self._pump()
        return True

    def _start_reach(self, step: Step) -> None:
        self._out("haptic.start", target=step.target, refer=step.refer)
        try:
            session_id = self.haptic.start(step.target, step.refer)
        except Exception as exc:
            session_id = f"failed-{self.task.id}-{step.n}"
            self._failed_sessions.add(session_id)
            step.session_id = session_id
            step.state = "running"
            self._info(f"haptic_start_failed: {exc}")
            self.on_haptic_result(HapticResult(
                session_id=session_id, status="stale_link", target=step.target,
                frame=self.camera.latest_frame(), last_seen=None, closest_approach="far", elapsed_s=0.0,
            ))
            return
        step.session_id = session_id
        step.state = "running"
        self.transcript.add(self.clock.now(), "event", f"reach started: {step.target}")

    def _cancel_reach(self, step: Step) -> None:
        if step.type != "reach" or step.session_id is None or step.state not in ("running", "paused"):
            return
        if step.session_id not in self._failed_sessions:
            self._out("haptic.cancel", session_id=step.session_id)
            self.haptic.cancel(step.session_id)
        step.state = "cancelled"

    # ------------------------------------------------------------------ outputs and logging

    def _say(self, text: str, expect: str, step: Step | None = None, is_step_speech: bool = False,
             then: str | None = None) -> Speech:
        self._out("say", text=text, expect=expect)
        speech_id = self.audio.say(text, expect)
        sp = Speech(text=text, expect=expect, is_step_speech=is_step_speech, then=then)
        if step is not None:
            sp.task_id, sp.step_n = self.task.id, step.n
            if step.type == "speak":
                step.state = "speaking"
        sp.entry = self.transcript.add(self.clock.now(), "assistant", text)
        self.speeches[speech_id] = sp
        self.last_spoken = sp
        return sp

    def _stop_speech(self) -> None:
        self._out("stop_speech")
        self.audio.stop_speech()

    def _record_trigger(self, trigger: Trigger, intercept: str | None = None) -> None:
        """Add the trigger to recent_turns. Called after composing, so a turn never sees its own trigger."""
        at = trigger.arrived_at
        if isinstance(trigger, UtteranceTrigger):
            self.transcript.add(at, "user", trigger.text)
            if intercept:
                self.transcript.add(at, "event", f"intercept: {intercept}")
        elif isinstance(trigger, HapticTrigger):
            status = trigger.event.status + (" (inferred)" if trigger.event.inferred else "")
            self.transcript.add(at, "event", f"haptic: {status}")
        elif isinstance(trigger, IdleTrigger):
            self.transcript.add(at, "event", f"idle check-in {trigger.checkin}")

    def _new_turn(self, trigger: Trigger | None, kind: str = "turn") -> Turn:
        turn = Turn(id=next(self._turn_ids), trigger=trigger, kind=kind)
        turn.log = {
            "kind": kind,
            "turn_id": turn.id,
            "task_id": self.task.id if self.task is not None else None,
            "trigger": _trigger_log(trigger),
            "intercept": None,
            "input": None,
            "frame_id": None,
            "model_calls": [],
            "tool_calls": [],
            "warnings": [],
            "info": [],
            "outputs": [],
            "state_before": self.snapshot(),
            "abandoned": False,
            "at": self.clock.now(),
        }
        return turn

    @contextmanager
    def _acting(self, turn: Turn):
        """Attribute output calls, warnings and info made inside the block to this turn."""
        previous, self._cur = self._cur, turn
        try:
            yield turn
        finally:
            self._cur = previous

    @contextmanager
    def _instant(self, name: str, trigger: Trigger | None = None, kind: str = "event", quiet: bool = False):
        """Synchronous work that is logged as one record: an intercept turn or an event."""
        turn = self._new_turn(trigger, kind=kind)
        turn.quiet = quiet
        turn.log["intercept" if kind == "turn" else "event"] = name
        with self._acting(turn):
            yield turn
        self._finish_log(turn)

    def _finish_log(self, turn: Turn) -> None:
        log = turn.log
        if turn.quiet and not log["outputs"] and not log["warnings"] and not log["info"]:
            return
        log["state_after"] = self.snapshot()
        if log["task_id"] is None and self.task is not None:
            log["task_id"] = self.task.id
        if turn.first_output_at is not None and turn.trigger is not None:
            log["latency_ms"] = round((turn.first_output_at - turn.trigger.arrived_at) * 1000, 1)
        self.turnlog.write(log)

    def _out(self, call: str, /, **args) -> None:
        at = self.clock.now()
        turn = self._cur
        if turn is None:
            self.turnlog.write({"kind": "event", "event": "output", "call": call, "args": args, "at": at})
            return
        turn.log["outputs"].append({"call": call, **args, "at": at})
        if call in ("say", "haptic.start") and turn.first_output_at is None:
            turn.first_output_at = at

    def _warn(self, code: str) -> None:
        self._note("warnings", code)

    def _info(self, code: str) -> None:
        self._note("info", code)

    def _note(self, key: str, code: str) -> None:
        if self._cur is None:
            self.turnlog.write({"kind": "event", "event": key, "code": code, "at": self.clock.now()})
        else:
            self._cur.log[key].append(code)


def _trigger_log(trigger: Trigger | None) -> dict | None:
    if trigger is None:
        return None
    if isinstance(trigger, UtteranceTrigger):
        return {"type": "user_utterance", "text": trigger.text, "intercepted": trigger.intercepted,
                "frame_id": trigger.frame.id if trigger.frame else None}
    if isinstance(trigger, HapticTrigger):
        return {"type": "haptic_result", "event": trigger.event.log_view()}
    return {"type": "idle_timeout", "checkin": trigger.checkin}


def _response_summary(response: ModelResponse) -> dict:
    return {"text": response.text, "tool_calls": [{"name": c.name, "args": c.args} for c in response.tool_calls]}
