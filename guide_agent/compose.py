"""Per-turn model input (CLAUDE.md section 11) and the recent-turns transcript."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass

from PIL import Image

from .config import Config
from .schemas import Frame, HapticTrigger, IdleTrigger, Trigger, UtteranceTrigger
from .state import Step, Task


@dataclass
class Entry:
    at: float
    role: str  # user | assistant | event
    text: str
    cut_off: bool = False

    def line(self) -> str:
        if self.role == "event":
            return f"[{self.text}]"
        if self.role == "assistant" and self.cut_off:
            return f"assistant (cut off): {self.text}"
        return f"{self.role}: {self.text}"


class Transcript:
    """User utterances, everything spoken, and bracketed events, newest last."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def add(self, at: float, role: str, text: str) -> Entry:
        entry = Entry(at, role, text)
        self.entries.append(entry)
        return entry

    def render(self, now: float, max_lines: int, max_age_s: float) -> list[str]:
        recent = [e for e in self.entries if now - e.at <= max_age_s][-max_lines:]
        return [f"[{round(now - e.at)}s] {e.line()}" for e in recent]


def _q(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_step(step: Step, now: float) -> list[str]:
    head = f"current_step: n={step.n} type={step.type} state={step.state}"
    if step.type == "speak":
        head += f" expect={step.expect}"
    head += f" heard_fully={str(step.heard_fully).lower()} age_s={round(now - step.issued_at)}"
    if step.type == "speak":
        return [head, f"  text: {_q(step.text)}"]
    return [head, f"  target: {_q(step.target)}", f"  refer: {_q(step.refer)}", f"  say_first: {_q(step.say_first)}"]


def render_history_line(step: Step) -> str:
    if step.type == "speak":
        body = f"speak {_q(step.text)}"
    else:
        body = f"reach target={_q(step.target)} refer={_q(step.refer)}"
    return f"  {step.n} {body} -> {step.result}"


def render_task_state(task: Task | None, now: float, cfg: Config) -> str:
    if task is None:
        return "<task_state>none</task_state>"
    lines = ["<task_state>", f"goal: {task.goal}", f"request: {_q(task.request)}"]
    if task.current_step is None:
        lines.append("current_step: none")
    else:
        lines.extend(render_step(task.current_step, now))
    shown = task.history[-cfg.history_max_steps:] if cfg.history_max_steps > 0 else []
    if not shown:
        lines.append("history: none")
    else:
        lines.append("history:")
        hidden = len(task.history) - len(shown)
        if hidden:
            lines.append(f"  ({hidden} earlier steps not shown)")
        lines.extend(render_history_line(s) for s in shown)
    lines.append(f"notes: {task.notes if task.notes else 'none'}")
    lines.append("</task_state>")
    return "\n".join(lines)


def render_trigger(trigger: Trigger) -> str:
    if isinstance(trigger, UtteranceTrigger):
        return f'<trigger type="user_utterance" intercepted="{trigger.intercepted}">{trigger.text}</trigger>'
    if isinstance(trigger, HapticTrigger):
        return f'<trigger type="haptic_result">{json.dumps(trigger.event.model_view())}</trigger>'
    if isinstance(trigger, IdleTrigger):
        return f'<trigger type="idle_timeout" checkin="{trigger.checkin}"/>'
    raise TypeError(f"unknown trigger {trigger!r}")


def render_frame(frame: Frame | None, now: float) -> str:
    if frame is None:
        return "<frame>none</frame>"
    return f'<frame id="{frame.id}" age_ms="{max(0, round((now - frame.captured_at) * 1000))}"/>'


def compose_input(
    *, task: Task | None, transcript: Transcript, trigger: Trigger, frame: Frame | None, now: float, cfg: Config
) -> str:
    recent = transcript.render(now, cfg.recent_turns_max, cfg.recent_turns_max_age_s)
    parts = [
        render_task_state(task, now, cfg),
        "<recent_turns>\n" + "\n".join(recent) + "\n</recent_turns>" if recent else "<recent_turns>none</recent_turns>",
        render_trigger(trigger),
        render_frame(frame, now),
    ]
    return "\n".join(parts)


def prepare_image(image: bytes | None, max_long_edge: int) -> bytes | None:
    """Downscale a JPEG so its long edge is at most max_long_edge. Undecodable bytes pass through."""
    if not image:
        return None
    try:
        with Image.open(io.BytesIO(image)) as img:
            if max(img.size) <= max_long_edge:
                return image
            img = img.convert("RGB")
            img.thumbnail((max_long_edge, max_long_edge))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()
    except Exception:
        return image
