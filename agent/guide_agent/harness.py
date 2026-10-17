"""Interactive harness (CLAUDE.md section 17.2): the real model with stubbed audio, haptics and camera.

    python -m guide_agent.harness                    type commands
    python -m guide_agent.harness --script FILE      replay commands from a file (one per line, # comments)

Playback model: speech keeps "playing" until the next command. Any command except /speech_start first lets
it finish, as if the user listened to the end; /speech_start cuts it off.
"""
from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from pathlib import Path

from .config import AGENT_ROOT, Config, load_config, load_dotenv
from .controller import Controller
from .fixtures import load_manifest
from .interfaces import RealClock
from .models.gemini import GeminiAdapter
from .prompt import build_system_prompt
from .schemas import Frame, HapticResult, LastSeen
from .stubs import OutputCall, Recorder, RecordingModel, StubAudio, StubCamera, StubHaptic
from .turnlog import TurnLog

HELP = """Commands:
  <text>                  the user says <text> (speech_start first, then the utterance with the current frame)
  /utter <text>           an utterance with no speech_start before it
  /frame <id|none|path>   frame for what follows (sticky): a fixture id, none, or an image path
  /speech_start           the user starts speaking: cuts off speech, pauses a running reach
  /timeout                the speech_start timeout expires now
  /idle                   the idle timer expires now
  /haptic <status> [inferred] [closest=near|far|touching] [region=right-middle] [depth=near] [age=1.5] [elapsed=6]
                          the running reach ends with <status>
  /fail_next_start        the next haptic.start fails
  /state                  print the task state
  /help, /quit"""


class Harness:
    def __init__(self, cfg: Config, model, image_dir: str | None = None, out=print):
        self.out = out
        self.cfg = cfg
        self.clock = RealClock()
        self.rec = Recorder(self.clock, echo=self._echo)
        self.audio = StubAudio(self.rec)
        self.haptic = StubHaptic(self.rec)
        self.camera = StubCamera()
        self.fixtures = load_manifest(image_dir)
        self.turnlog = TurnLog.daily(AGENT_ROOT / cfg.log_dir)
        self.ctl = Controller(cfg, RecordingModel(model, self.rec), self.audio, self.haptic, self.camera,
                              self.clock, build_system_prompt(cfg), self.turnlog)
        self.frame_source: str | None = None
        self._frames = 0
        self._logged = 0

    # ------------------------------------------------------------------ output

    def _echo(self, call: OutputCall) -> None:
        a = call.args
        if call.name == "say":
            self.out(f'  >> say(expect={a["expect"]}): "{a["text"]}"')
        elif call.name == "haptic.start":
            self.out(f'  >> haptic.start(target="{a["target"]}", refer="{a["refer"]}")')
        elif call.name.startswith("haptic."):
            self.out(f"  >> {call.name}({a['session_id']})")
        elif call.name == "cue":
            self.out(f"  >> cue({a['name']})")
        elif call.name == "stop_speech":
            if self.audio.playing:
                self.out("  >> stop_speech()")
        elif call.name == "model":
            self.out("  .. model call")

    def print_state(self) -> None:
        task = self.ctl.task
        if task is None:
            self.out("  state: no task")
            return
        step = task.current_step
        if step is None:
            desc = "no pending step"
        elif step.type == "speak":
            desc = f"step {step.n} speak {step.state} expect={step.expect} heard_fully={step.heard_fully}"
        else:
            desc = f"step {step.n} reach {step.state} target={step.target!r} session={step.session_id}"
        self.out(f"  state: task {task.id} {task.goal!r} | {desc}")
        for s in task.history:
            self.out(f"         history {s.n} {s.type} -> {s.result}")
        self.out(f"         notes: {task.notes or 'none'}")

    def _print_new_turns(self) -> None:
        records = self.turnlog.records[self._logged:]
        self._logged = len(self.turnlog.records)
        for r in records:
            if r.get("kind") != "turn":
                continue
            calls = [m.get("latency_ms") for m in r["model_calls"] if "latency_ms" in m]
            extras = []
            if r.get("abandoned"):
                extras.append("ABANDONED")
            if r["warnings"]:
                extras.append(f"warnings={r['warnings']}")
            if r["info"]:
                extras.append(f"info={r['info']}")
            if any("error" in m for m in r["model_calls"]):
                extras.append(f"model errors={[m['error'] for m in r['model_calls'] if 'error' in m]}")
            self.out(f"  turn {r['turn_id']}: terminal={r.get('terminal')} model_ms={calls} "
                     f"latency_ms={r.get('latency_ms')} {' '.join(extras)}".rstrip())

    # ------------------------------------------------------------------ frames

    def current_frame(self) -> Frame | None:
        source = self.frame_source
        if source is None:
            return None
        if source in self.fixtures:
            fixture = self.fixtures[source]
            if not fixture.available:
                self.out(f"  !! fixture {source} is missing ({fixture.path}); sending no frame")
                return None
            image = fixture.image()
        else:
            image = Path(source).read_bytes()
        self._frames += 1
        return Frame(id=f"{Path(source).stem}-{self._frames}", image=image, captured_at=self.clock.now())

    # ------------------------------------------------------------------ commands

    async def run_command(self, line: str) -> bool:
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        if line in ("/quit", "/exit"):
            return False
        if line != "/speech_start":
            self.audio.finish()  # the user heard everything that was playing
            await self.ctl.drain()
        word, _, rest = line.partition(" ")
        rest = rest.strip()
        if not line.startswith("/"):
            self.ctl.on_speech_start()
            self.ctl.on_utterance(line, frame=self.current_frame())
        elif word == "/utter":
            self.ctl.on_utterance(rest, frame=self.current_frame())
        elif word == "/frame":
            self.frame_source = None if rest in ("", "none") else rest
            self.out(f"  frame source: {self.frame_source or 'none'}")
        elif word == "/speech_start":
            self.ctl.on_speech_start()
        elif word == "/timeout":
            self.ctl.speech_start_deadline = self.clock.now()
            self.ctl.tick()
        elif word == "/idle":
            if not self.ctl.fire_idle_timeout():
                self.out("  (no idle check-in: no pending speak step, or the check-in limit was reached)")
        elif word == "/haptic":
            self._haptic(rest)
        elif word == "/fail_next_start":
            self.haptic.fail_next_start = True
        elif word == "/state":
            self.print_state()
            return True
        elif word == "/help":
            self.out(HELP)
            return True
        else:
            self.out(f"  unknown command {word}; /help lists them")
            return True
        await self.ctl.drain()
        self._print_new_turns()
        if word not in ("/frame",):
            self.print_state()
            step = self.ctl.step
            if step is not None and step.type == "reach" and step.state == "starting":
                self.out("  (say_first is playing; guidance starts when it finishes. Next: /haptic <status>, "
                         "or speak to interrupt)")
        return True

    def _haptic(self, rest: str) -> None:
        step = self.ctl.step
        if step is None or step.type != "reach" or step.session_id is None:
            self.out("  no running reach")
            return
        parts = shlex.split(rest)
        if not parts:
            self.out("  usage: /haptic <status> [inferred] [closest=near] [region=right-middle] [depth=near]")
            return
        status, opts = parts[0], dict(p.split("=", 1) for p in parts[1:] if "=" in p)
        last_seen = None
        if "region" in opts:
            last_seen = LastSeen(opts["region"], opts.get("depth", "near"), float(opts.get("age", 1.0)))
        self.ctl.on_haptic_result(HapticResult(
            session_id=step.session_id, status=status, inferred="inferred" in parts[1:],
            elapsed_s=float(opts.get("elapsed", 6.0)), target=step.target, last_seen=last_seen,
            closest_approach=opts.get("closest", "near"),
            frame=None if status == "success" else self.current_frame(),
        ))


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", help="file of commands to replay")
    parser.add_argument("--images", help="directory to load fixture photos from instead of fixtures/")
    parser.add_argument("--model", help="override the model id")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    load_dotenv()
    cfg = load_config()
    if args.model:
        cfg = load_config(model=args.model)
    model = GeminiAdapter(cfg.model, thinking_level=cfg.thinking_level)
    harness = Harness(cfg, model, image_dir=args.images)
    print(f"model {cfg.model}, assistant {cfg.assistant_name}, glove {cfg.glove_hand} hand. /help for commands.")

    if args.script:
        for line in Path(args.script).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.strip().startswith("#"):
                print(f"\n> {line.strip()}")
            if not await harness.run_command(line):
                break
        return
    while True:
        try:
            line = await asyncio.to_thread(input, "\n> ")
        except EOFError:
            break
        if not await harness.run_command(line):
            break


if __name__ == "__main__":
    asyncio.run(main())
