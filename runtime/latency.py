"""Latency from end of speech to first audio, from the audio JSONL log (audio/logs/audio-YYYYMMDD.jsonl).

    python -m runtime.latency                    # the last voice_agent session today
    python -m runtime.latency --session all      # every voice_agent session today
    python -m runtime.latency --day 20261017 --session 164301

End of speech is the utterance's ended_at (utterance_delivered's wall time minus its latency_ms). Only
non-empty utterances count. Each one is paired with the first event of each kind that follows it, before the
next utterance:
    end -> utterance      the STT path (end-silence wait excluded: ended_at is the last voiced frame)
    end -> thinking cue   the agent's "thinking" tone actually played
    end -> first speech   playback_start of the first reply (what the user waits for)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def sessions(events: list[dict], cmd: str = "voice_agent") -> list[tuple[str, list[dict]]]:
    """(session id, events) for each session of cmd, in order."""
    out: list[tuple[str, list[dict]]] = []
    current: list[dict] | None = None
    for ev in events:
        if ev.get("event") == "session":
            current = [] if ev.get("cmd") == cmd else None
            if current is not None:
                out.append((str(ev.get("id")), current))
            continue
        if current is not None:
            current.append(ev)
    return out


def measure(events: list[dict]) -> dict[str, list[float]]:
    """ms per utterance for each measure (see the module docstring)."""
    out: dict[str, list[float]] = {"end -> utterance": [], "end -> thinking cue": [], "end -> first speech": []}
    pending: dict | None = None
    for ev in events:
        kind = ev.get("event")
        if kind == "utterance_delivered":
            pending = None
            if ev.get("chars", 0) > 0:
                end = ev["wall"] - ev.get("latency_ms", 0) / 1000
                out["end -> utterance"].append(ev.get("latency_ms", 0))
                pending = {"end": end, "cue": False, "speech": False}
        elif pending is not None and kind == "cue" and ev.get("name") == "thinking" and ev.get("played"):
            if not pending["cue"]:
                out["end -> thinking cue"].append((ev["wall"] - pending["end"]) * 1000)
                pending["cue"] = True
        elif pending is not None and kind == "playback_start":
            out["end -> first speech"].append((ev["wall"] - pending["end"]) * 1000)
            pending = None  # one reply per utterance
    return out


def report(values: dict[str, list[float]]) -> str:
    lines = []
    for name, ms in values.items():
        if ms:
            lines.append(f"{name:22s} n={len(ms):3d}  p50 {statistics.median(ms):6.0f} ms  max {max(ms):6.0f} ms")
        else:
            lines.append(f"{name:22s} n=  0")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from audio import config as audio_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default=time.strftime("%Y%m%d"), help="YYYYMMDD (default today)")
    parser.add_argument("--session", default="last", help="last, all, or a session id (HHMMSS)")
    args = parser.parse_args(argv)

    cfg = audio_config.load()
    path = cfg.path(cfg.log_dir) / f"audio-{args.day}.jsonl"
    if not path.exists():
        raise SystemExit(f"no audio log at {path}")
    found = sessions(load_events(path))
    if not found:
        raise SystemExit(f"no voice_agent session in {path}")
    if args.session == "last":
        chosen = [found[-1]]
    elif args.session == "all":
        chosen = found
    else:
        chosen = [s for s in found if s[0] == args.session]
        if not chosen:
            raise SystemExit(f"no session {args.session}; sessions: {', '.join(s for s, _ in found)}")
    events = [ev for _, evs in chosen for ev in evs]
    print(f"{path.name}, session(s) {', '.join(s for s, _ in chosen)}")
    print(report(measure(events)))


if __name__ == "__main__":
    main()
