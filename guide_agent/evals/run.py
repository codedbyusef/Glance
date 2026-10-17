"""Eval runner (CLAUDE.md section 17.3): model cases against the real model.

    python -m guide_agent.evals.run                          every case, 3 runs each
    python -m guide_agent.evals.run --case done_reach --runs 1
    python -m guide_agent.evals.run --images DIR             look up fixture photos in DIR instead of fixture_frames/

Writes guide_agent/evals/reports/<timestamp>/report.md, results.json and turns.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from guide_agent.config import AGENT_ROOT, Config, load_config, load_dotenv
from guide_agent.controller import Controller, Speech
from guide_agent.fixtures import Fixture, load_manifest
from guide_agent.interfaces import RealClock
from guide_agent.models.gemini import GeminiAdapter
from guide_agent.prompt import build_system_prompt
from guide_agent.schemas import Frame, HapticResult, LastSeen
from guide_agent.state import Step, Task
from guide_agent.stubs import Recorder, RecordingModel, StubAudio, StubCamera, StubHaptic
from guide_agent.tools import SPECS_BY_NAME, word_count
from guide_agent.turnlog import TurnLog

from . import judge as judge_mod
from .checks import match, spoken_violations

CASE_DIR = Path(__file__).resolve().parent / "cases"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
SETTLE_TIMEOUT_S = 240


@dataclass
class StepOutcome:
    index: int
    trigger: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    terminal: str | None = None
    spoken: str = ""
    judge: str | None = None


@dataclass
class RunOutcome:
    case_id: str
    category: str
    run: int
    passed: bool = False
    steps: list[StepOutcome] = field(default_factory=list)
    global_violations: list[str] = field(default_factory=list)
    turns: int = 0
    turns_with_warnings: int = 0
    warnings: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    spoken_words: list[int] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------- case loading


def load_cases(selected: list[str] | None) -> list[dict]:
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CASE_DIR.glob("*.json"))]
    if selected:
        cases = [c for c in cases if c["id"] in selected]
        missing = set(selected) - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    return cases


def fixtures_needed(case: dict) -> set[str]:
    needed = set()
    if case.get("setup", {}).get("camera_frame"):
        needed.add(case["setup"]["camera_frame"])
    for step in case["steps"]:
        for key in ("frame", "camera_frame"):
            if step["trigger"].get(key):
                needed.add(step["trigger"][key])
    text = json.dumps(case)
    for fixture_id in _fixture_text_refs(text):
        needed.add(fixture_id)
    return needed


def _fixture_text_refs(text: str) -> list[str]:
    import re
    return re.findall(r"\{fixture:(\w+)\}", text)


def skip_reason(case: dict, fixtures: dict[str, Fixture]) -> str | None:
    reasons = []
    for fixture_id in sorted(fixtures_needed(case)):
        fixture = fixtures.get(fixture_id)
        if fixture is None:
            reasons.append(f"{fixture_id} (not in manifest)")
        elif not fixture.available:
            reasons.append(f"{fixture_id} (photo missing)")
    for fixture_id in _fixture_text_refs(json.dumps(case)):
        fixture = fixtures.get(fixture_id)
        if fixture is not None and fixture.available and not fixture.expected_text:
            reasons.append(f"{fixture_id} (expected_text not set)")
    return ", ".join(reasons) or None


def substitute_fixture_text(obj, fixtures: dict[str, Fixture]):
    if isinstance(obj, str):
        for fixture_id in _fixture_text_refs(obj):
            obj = obj.replace(f"{{fixture:{fixture_id}}}", fixtures[fixture_id].expected_text or "")
        return obj
    if isinstance(obj, list):
        return [substitute_fixture_text(v, fixtures) for v in obj]
    if isinstance(obj, dict):
        return {k: substitute_fixture_text(v, fixtures) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------- one run of one case


class CaseRun:
    def __init__(self, case: dict, run: int, cfg: Config, model, judge_adapter, fixtures: dict[str, Fixture]):
        self.case = substitute_fixture_text(case, fixtures)
        self.run = run
        self.cfg = cfg
        self.judge_adapter = judge_adapter
        self.fixtures = fixtures
        self.clock = RealClock()
        self.rec = Recorder(self.clock)
        self.audio = StubAudio(self.rec)
        self.haptic = StubHaptic(self.rec)
        self.camera = StubCamera()
        self.turnlog = TurnLog()
        self.ctl = Controller(cfg, RecordingModel(model, self.rec), self.audio, self.haptic, self.camera,
                              self.clock, build_system_prompt(cfg), self.turnlog)
        self.answer_limit = case.get("global", {}).get("answer_max_words", True)
        self._frames = 0

    def frame(self, fixture_id: str | None) -> Frame | None:
        if not fixture_id:
            return None
        self._frames += 1
        return Frame(id=f"{fixture_id}-{self._frames}", image=self.fixtures[fixture_id].image(),
                     captured_at=self.clock.now() - 0.3)

    def apply_setup(self) -> None:
        setup = self.case.get("setup", {})
        now = self.clock.now()
        task_d = setup.get("task")
        if task_d:
            task = Task(id="t-case", goal=task_d["goal"], request=task_d.get("request", ""), started_at=now - 90,
                        last_user_activity_at=now - 5, notes=task_d.get("notes", ""))
            for n, h in enumerate(task_d.get("history", []), 1):
                task.history.append(_step_from(h, n, now - 90 + 10 * n, heard=True))
            current = task_d.get("current_step")
            if current:
                step = _step_from(current, len(task.history) + 1, now - current.get("age_s", 8),
                                  heard=current.get("heard_fully", True))
                step.state = current["state"]
                if step.type == "reach" and step.state in ("running", "paused"):
                    step.session_id = "s-case"
                    self.haptic.sessions["s-case"] = step.state
                task.current_step = step
            self.ctl.task = task
        for entry in setup.get("recent_turns", []):
            self.ctl.transcript.add(now - entry["age_s"], entry["role"], entry["text"])
        if setup.get("last_spoken"):
            self.ctl.last_spoken = Speech(text=setup["last_spoken"], expect="none")
        self.camera.frame = self.frame(setup.get("camera_frame"))

    async def settle(self) -> None:
        async def _settle():
            await self.ctl.drain()
            while self.audio.playing:
                self.audio.finish()
                await self.ctl.drain()
        await asyncio.wait_for(_settle(), SETTLE_TIMEOUT_S)

    async def execute(self) -> RunOutcome:
        out = RunOutcome(case_id=self.case["id"], category=self.case["category"], run=self.run)
        self.apply_setup()
        for index, step in enumerate(self.case["steps"], 1):
            outcome = await self.run_step(index, step, out)
            out.steps.append(outcome)
            if not outcome.passed:
                break
        out.passed = all(s.passed for s in out.steps) and len(out.steps) == len(self.case["steps"]) \
            and not out.global_violations
        return out

    async def run_step(self, index: int, step: dict, out: RunOutcome) -> StepOutcome:
        trig = step["trigger"]
        expect = step.get("expect", {})
        rec_mark, log_mark = len(self.rec.calls), len(self.turnlog.records)
        failures: list[str] = []
        kind = trig["type"]
        if trig.get("camera_frame"):
            self.camera.frame = self.frame(trig["camera_frame"])

        if kind == "user_utterance":
            label = f'"{trig["text"]}"'
            if trig.get("speech_start", True):
                self.ctl.on_speech_start()
            self.ctl.on_utterance(trig["text"], frame=self.frame(trig.get("frame")))
        elif kind == "haptic_result":
            label = f"haptic_result {trig['status']}"
            current = self.ctl.step
            if current is None or current.type != "reach" or current.session_id is None:
                return StepOutcome(index, label, False, [f"no running reach to end (state: {self.ctl.snapshot()})"])
            last_seen = trig.get("last_seen")
            self.ctl.on_haptic_result(HapticResult(
                session_id=current.session_id, status=trig["status"], inferred=trig.get("inferred", False),
                elapsed_s=trig.get("elapsed_s", 6.0), target=current.target,
                last_seen=LastSeen(**last_seen) if last_seen else None,
                closest_approach=trig.get("closest_approach", "near"),
                frame=None if trig["status"] == "success" else self.frame(trig.get("frame")),
            ))
        elif kind == "idle_timeout":
            label = f"idle_timeout checkin {trig.get('checkin', 1)}"
            self.ctl.idle_checkins = trig.get("checkin", 1) - 1
            if not self.ctl.fire_idle_timeout():
                return StepOutcome(index, label, False, ["idle timeout did not fire (no pending speak step)"])
        else:
            raise ValueError(f"unknown trigger type {kind}")
        await self.settle()

        records = self.turnlog.records[log_mark:]
        turns = [r for r in records if r.get("kind") == "turn"]
        live = [r for r in turns if not r.get("abandoned")]
        main = live[-1] if live else None
        names = self.rec.names(rec_mark)
        says = [c.args["text"] for c in self.rec.of("say", rec_mark)]

        # bookkeeping for the report
        for t in turns:
            codes = [w for w in t["warnings"] if self.answer_limit or w != "answer_too_long"]
            out.turns += 1
            out.turns_with_warnings += bool(codes)
            out.warnings += codes
            if t.get("latency_ms") is not None:
                out.latencies_ms.append(t["latency_ms"])
        terminal = (main or {}).get("terminal") or "none"
        spoken = (main or {}).get("spoken") or ""
        if spoken:
            out.spoken_words.append(word_count(spoken))

        # global checks
        for text in says:
            out.global_violations += [f"step {index}: {v}" for v in spoken_violations(text)]
        if terminal in ("speak", "reach") and word_count(spoken) > self.cfg.step_max_words:
            out.global_violations.append(f"step {index}: step over {self.cfg.step_max_words} words: {spoken!r}")
        if terminal in ("plain", "finish_task") and self.answer_limit and word_count(spoken) > self.cfg.answer_max_words:
            out.global_violations.append(f"step {index}: answer over {self.cfg.answer_max_words} words: {spoken!r}")

        # expectations
        if "terminal" in expect and terminal not in expect["terminal"]:
            failures.append(f"terminal: expected {expect['terminal']}, got {terminal}")
        tool_calls = (main or {}).get("tool_calls", [])
        executed = [c for c in tool_calls if not _failed(c.get("result"))]
        if "args" in expect and terminal in SPECS_BY_NAME:
            call = next((c for c in executed if c["name"] == terminal), None)
            props = SPECS_BY_NAME[terminal]["parameters"]["properties"]
            for key, matcher in expect["args"].items():
                if key in props:  # keys the terminal tool doesn't have are skipped
                    failures += match((call or {}).get("args", {}).get(key), matcher, f"{terminal}.{key}")
        if "start_task" in expect:
            starts = [c for c in executed if c["name"] == "start_task"]
            want = expect["start_task"]
            if want is False and starts:
                failures.append(f"start_task: not expected, got {starts[0]['args']}")
            elif want is not False and not starts:
                failures.append("start_task: expected, not called")
            elif isinstance(want, dict) and starts:
                for key, matcher in want.items():
                    failures += match(starts[0]["args"].get(key), matcher, f"start_task.{key}")
        if "spoken" in expect:
            failures += match(spoken, expect["spoken"], "spoken")
        if "state_after" in expect:
            failures += self.check_state(expect["state_after"])
        if "outputs" in expect:
            failures += _check_outputs(names, expect["outputs"])
        judge_note = None
        if expect.get("judge"):
            try:
                ok, reason = await judge_mod.grade(self.judge_adapter, expect["judge"], (main or {}).get("input"),
                                                   tool_calls, says)
            except Exception as exc:
                ok, reason = False, f"judge error: {exc!r}"
            judge_note = f"{'pass' if ok else 'FAIL'}: {reason}"
            if not ok:
                failures.append(f"judge: {reason}")
        return StepOutcome(index, label, not failures, failures, terminal, spoken, judge_note)

    def check_state(self, want: dict) -> list[str]:
        task, step = self.ctl.task, self.ctl.step
        got = {
            "task": "active" if task else "none",
            "step_type": step.type if step else "none",
            "step_state": step.state if step else "none",
            "expect": step.expect if step and step.type == "speak" else "none",
            "step_n": step.n if step else None,
            "outcome": self.ctl.closed_tasks[-1].outcome if self.ctl.closed_tasks else "none",
            "goal": task.goal if task else "",
        }
        failures = []
        for key, value in want.items():
            if key not in got:
                failures.append(f"state_after: unknown key {key}")
            elif isinstance(value, dict):
                failures += match(got[key], value, f"state_after.{key}")
            elif got[key] != value:
                failures.append(f"state_after.{key}: expected {value!r}, got {got[key]!r}")
        return failures


def _step_from(d: dict, n: int, issued_at: float, heard: bool) -> Step:
    return Step(n=n, type=d["type"], issued_at=issued_at, text=d.get("text", ""),
                expect=d.get("expect", "done" if d["type"] == "speak" else ""), target=d.get("target", ""),
                refer=d.get("refer", ""), say_first=d.get("say_first", ""), heard_fully=heard,
                result=d.get("result"))


def _failed(result) -> bool:
    return result in ("dropped", "skipped") or (isinstance(result, dict) and "error" in result)


def _check_outputs(names: list[str], want: dict) -> list[str]:
    failures = []
    for name in want.get("includes", []):
        if name not in names:
            failures.append(f"outputs: expected {name} in {names}")
    for name in want.get("excludes", []):
        if name in names:
            failures.append(f"outputs: {name} must not appear in {names}")
    order = want.get("order", [])
    if order:
        positions = [names.index(n) if n in names else -1 for n in order]
        if -1 in positions or positions != sorted(positions):
            failures.append(f"outputs: expected order {order} in {names}")
    return failures


# ---------------------------------------------------------------------- report


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def write_report(path: Path, cfg: Config, runs_per_case: int, outcomes: list[RunOutcome],
                 skipped: dict[str, str], cases: list[dict], image_dir: str | None) -> str:
    by_case: dict[str, list[RunOutcome]] = defaultdict(list)
    for o in outcomes:
        by_case[o.case_id].append(o)
    by_cat: dict[str, list[RunOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.category].append(o)
    turns = sum(o.turns for o in outcomes)
    warned = sum(o.turns_with_warnings for o in outcomes)
    latencies = [x for o in outcomes for x in o.latencies_ms]
    words = [x for o in outcomes for x in o.spoken_words]
    violations = [(o.case_id, o.run, v) for o in outcomes for v in o.global_violations]
    warning_counts = Counter(w for o in outcomes for w in o.warnings)
    passed = sum(o.passed for o in outcomes)
    case_pass = {cid: sum(o.passed for o in os) / len(os) for cid, os in by_case.items()}

    lines = [
        f"# Eval report {path.name}",
        "",
        f"Model `{cfg.model}` (thinking {cfg.thinking_level}, temperature {cfg.temperature}), judge `{cfg.judge}`, "
        f"{runs_per_case} runs per case." + (f" Fixture photos from `{image_dir}`." if image_dir else ""),
        "",
        "## Summary",
        "",
        f"- Cases run: {len(by_case)} of {len(cases)}; skipped: {len(skipped)}",
        f"- Case runs passed: {passed}/{len(outcomes)}"
        + (f" ({100 * passed / len(outcomes):.0f}%)" if outcomes else ""),
        f"- Cases passing all runs: {sum(1 for v in case_pass.values() if v == 1)}/{len(case_pass)}",
        f"- Turns with a protocol warning: {warned}/{turns}" + (f" ({100 * warned / turns:.1f}%)" if turns else ""),
        f"- Global-check violations: {len(violations)}",
        f"- Turn latency ms: p50 {percentile(latencies, 0.5)}, p95 {percentile(latencies, 0.95)} "
        f"(n={len(latencies)})",
        f"- Words per spoken output: mean {round(statistics.mean(words), 1) if words else None}, "
        f"p50 {percentile(words, 0.5)}, max {max(words) if words else None}",
        "",
        "## By category",
        "",
        "| Category | Runs passed |",
        "|---|---|",
    ]
    for cat in sorted(by_cat):
        os_ = by_cat[cat]
        lines.append(f"| {cat} | {sum(o.passed for o in os_)}/{len(os_)} |")
    lines += ["", "## By case", "", "| Case | Category | Passed | First failure |", "|---|---|---|---|"]
    for case in cases:
        cid = case["id"]
        if cid in skipped:
            lines.append(f"| {cid} | {case['category']} | skipped | {skipped[cid]} |")
            continue
        os_ = by_case.get(cid, [])
        first = next((o for o in os_ if not o.passed), None)
        reason = ""
        if first is not None:
            failed_step = next((s for s in first.steps if not s.passed), None)
            reason = first.error or (failed_step.failures[0] if failed_step else
                                     (first.global_violations[0] if first.global_violations else ""))
        lines.append(f"| {cid} | {case['category']} | {sum(o.passed for o in os_)}/{len(os_)} | "
                     f"{_cell(reason)} |")
    if warning_counts:
        lines += ["", "## Protocol warnings", ""] + [f"- {code}: {n}" for code, n in warning_counts.most_common()]
    if violations:
        lines += ["", "## Global-check violations", ""] + [f"- {cid} run {r}: {v}" for cid, r, v in violations]
    failures = [o for o in outcomes if not o.passed]
    if failures:
        lines += ["", "## Failed runs", ""]
        for o in failures:
            lines.append(f"### {o.case_id} run {o.run}")
            if o.error:
                lines.append(f"- error: {o.error}")
            for s in o.steps:
                mark = "ok" if s.passed else "FAIL"
                lines.append(f"- step {s.index} {s.trigger}: {mark}, terminal {s.terminal}, spoken {s.spoken!r}")
                for f in s.failures:
                    lines.append(f"  - {f}")
                if s.judge:
                    lines.append(f"  - judge {s.judge}")
            lines.append("")
    if skipped:
        missing = sorted({part.split(" ")[0] for reason in skipped.values() for part in reason.split(", ")})
        lines += ["", "## Skipped", "", f"Fixtures needed: {', '.join(missing)}", ""]
        lines += [f"- {cid}: {reason}" for cid, reason in skipped.items()]
    text = "\n".join(lines) + "\n"
    (path / "report.md").write_text(text, encoding="utf-8")
    (path / "results.json").write_text(json.dumps({
        "model": cfg.model, "runs_per_case": runs_per_case, "skipped": skipped,
        "outcomes": [asdict(o) for o in outcomes],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    return text


def _cell(text: str) -> str:
    text = text.replace("|", "/").replace("\n", " ")
    return text if len(text) < 160 else text[:157] + "..."


# ---------------------------------------------------------------------- main


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the model cases against the real model.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--case", action="append", help="case id (repeatable)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--images", help="directory to load fixture photos from instead of fixture_frames/")
    parser.add_argument("--out", help="report directory name")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    load_dotenv()
    cfg = load_config()
    fixtures = load_manifest(args.images)
    cases = load_cases(args.case)
    skipped = {c["id"]: r for c in cases if (r := skip_reason(c, fixtures))}
    runnable = [c for c in cases if c["id"] not in skipped]
    model = GeminiAdapter(cfg.model, thinking_level=cfg.thinking_level, max_retries=5)
    judge_adapter = model if cfg.judge == cfg.model else GeminiAdapter(cfg.judge, thinking_level=cfg.thinking_level,
                                                                       max_retries=5)
    out_dir = REPORT_DIR / (args.out or datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    turn_file = TurnLog(out_dir / "turns.jsonl")

    print(f"{len(runnable)} cases to run x {args.runs}, {len(skipped)} skipped. Report: {out_dir}")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case: dict, run: int) -> RunOutcome:
        async with sem:
            runner = CaseRun(case, run, cfg, model, judge_adapter, fixtures)
            try:
                outcome = await runner.execute()
            except Exception as exc:
                outcome = RunOutcome(case_id=case["id"], category=case["category"], run=run,
                                     error=f"{exc!r}\n{traceback.format_exc(limit=3)}")
            for record in runner.turnlog.records:
                turn_file.write({"case": case["id"], "run": run, **record})
            print(f"  {case['id']} run {run}: {'pass' if outcome.passed else 'FAIL'}", flush=True)
            return outcome

    outcomes = await asyncio.gather(*(one(c, r) for c in runnable for r in range(1, args.runs + 1)))
    outcomes = sorted(outcomes, key=lambda o: ([c["id"] for c in cases].index(o.case_id), o.run))
    report = write_report(out_dir, cfg, args.runs, outcomes, skipped, cases, args.images)
    print()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
