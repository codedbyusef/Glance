"""Dev commands (CLAUDE.md section 14.2). Run from the repo root: python -m audio.cli <cmd>."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from . import config
from .capture import SAMPLE_RATE, device_rate, resample_to
from .log import EventLog
from .stt_client import FallbackTranscriber, LocalWhisper, SttError, VultrWhisper, to_wav

DEFAULT_CLIPS = config.AUDIO_DIR / "clips"
COMMANDS = {"done", "yes", "no", "stop", "wait", "cancel", "repeat"}


def bench_key(text: str) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed (section 14.2)."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


# ------------------------------------------------------------------ record

def record_until_enter(device, max_s: float = 15.0) -> np.ndarray:
    """Record from Enter to Enter (or max_s). Returns 16 kHz mono int16."""
    import sounddevice as sd

    try:
        sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        rate = SAMPLE_RATE
    except Exception:
        rate = device_rate(device, "input")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, t, status):
        chunks.append(indata[:, 0].copy())

    input("Press Enter, speak, then press Enter again. ")
    with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="int16", callback=callback):
        done = threading.Event()
        threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
        print("Recording...", flush=True)
        done.wait(max_s)
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return resample_to(samples, rate)


def cmd_record(args, cfg) -> int:
    clips = Path(args.dir)
    clips.mkdir(parents=True, exist_ok=True)
    samples = record_until_enter(cfg.input_device)
    slug = re.sub(r"[^a-z0-9]+", "-", args.label.lower()).strip("-")[:40] or "clip"
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}{'-' + args.tag if args.tag else ''}.wav"
    (clips / name).write_bytes(to_wav(samples))
    labels = clips / "labels.csv"
    new = not labels.exists()
    with open(labels, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label"])
        w.writerow([name, args.label])
    print(f"saved {clips / name} ({len(samples) / SAMPLE_RATE:.1f} s)")
    return 0


# ------------------------------------------------------------------ stt-bench

def load_labels(clips: Path) -> list[tuple[Path, str]]:
    """Labeled clips that exist on disk. Rows whose file is missing are skipped with a warning."""
    with open(clips / "labels.csv", newline="", encoding="utf-8") as f:
        rows = [(clips / row["file"], row["label"]) for row in csv.DictReader(f)]
    present = [(p, label) for p, label in rows if p.is_file()]
    for p, label in rows:
        if not p.is_file():
            print(f"warning: skipping missing clip {p.name} ({label!r})")
    return present


def bench_backends(names: list[str], cfg) -> list:
    out = []
    for name in names:
        if name == "vultr":
            url = config.stt_url()
            if not url:
                print("skip vultr: GLANCE_STT_URL is not set")
                continue
            b = VultrWhisper(url, config.stt_token(), cfg.stt_timeout_s)
            try:
                print(f"vultr health: {b.health()}")
            except SttError as e:
                print(f"vultr health failed: {e}")
            out.append(b)
        elif name == "local":
            b = LocalWhisper(cfg.fallback_model)
            t0 = time.monotonic()
            b.preload()
            print(f"local {cfg.fallback_model} loaded in {time.monotonic() - t0:.1f} s (not counted below)")
            out.append(b)
        else:
            raise SystemExit(f"unknown backend {name}")
    return out


def cmd_stt_bench(args, cfg) -> int:
    clips = Path(args.dir)
    items = load_labels(clips)
    rows = []
    for backend in bench_backends(args.backends.split(","), cfg):
        # One backend at a time, no fallback, with the normal filters.
        t = FallbackTranscriber(backend, None, cfg.hallucination_blocklist)
        for path, label in items:
            r = t.transcribe(path.read_bytes())
            ok = r.backend != "none" and bench_key(r.text) == bench_key(label)
            rows.append({"backend": backend.name, "file": path.name, "label": label, "text": r.text,
                         "ok": ok, "failed": r.backend == "none", "ms": r.ms, "dropped": r.dropped_reason or ""})
            if not ok:
                print(f"  [{backend.name}] MISS {path.name}: want {label!r} got {r.text!r} {r.dropped_reason or ''}")

    print(f"\n{'backend':8} {'clips':>5} {'exact':>7} {'cmds':>7} {'noise':>7} {'p50 ms':>7} {'p95 ms':>7}"
          f" {'failed':>6}")
    for name in dict.fromkeys(r["backend"] for r in rows):
        rs = [r for r in rows if r["backend"] == name]
        cmds = [r for r in rs if bench_key(r["label"]) in COMMANDS]
        ms = [r["ms"] for r in rs if not r["failed"]]
        acc = sum(r["ok"] for r in rs) / len(rs)
        cacc = f"{sum(r['ok'] for r in cmds) / len(cmds):.0%}" if cmds else "-"
        noise = [r for r in rs if not bench_key(r["label"])]  # labeled "": should come back empty
        nacc = f"{sum(r['ok'] for r in noise) / len(noise):.0%}" if noise else "-"
        print(f"{name:8} {len(rs):5} {acc:7.0%} {cacc:>7} {nacc:>7} {percentile(ms, 50):7.0f}"
              f" {percentile(ms, 95):7.0f} {sum(r['failed'] for r in rs):6}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["backend"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------ stt-check

CHECK_CLIP = config.AUDIO_DIR / "assets" / "stt_check.wav"
CHECK_TEXT = "get me the can"


def cmd_stt_check(args, cfg) -> int:
    url, token = config.stt_url(), config.stt_token()
    if not url:
        print("GLANCE_STT_URL is not set")
        return 2
    b = VultrWhisper(url, token, args.timeout)
    print(f"server: {b.base}  token: {'set' if token else 'NOT SET'}")
    t0 = time.monotonic()
    try:
        print(f"health ({(time.monotonic() - t0) * 1000:.0f} ms): {b.health()}")
    except SttError as e:
        print(f"health failed after {(time.monotonic() - t0) * 1000:.0f} ms: {e}")
        return 1
    wav = Path(args.clip).read_bytes()
    ok = True
    for i in range(args.n):
        t0 = time.monotonic()
        try:
            r = b.client.post(f"{b.base}/transcribe", content=wav, headers={"Content-Type": "audio/wav"})
        except httpx.HTTPError as e:
            print(f"transcribe #{i + 1}: {type(e).__name__} after {(time.monotonic() - t0) * 1000:.0f} ms")
            ok = False
            continue
        ms = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            print(f"transcribe #{i + 1}: HTTP {r.status_code} after {ms:.0f} ms: {r.text[:200]}")
            ok = False
            continue
        body = r.json()
        match = bench_key(body["text"]) == CHECK_TEXT if args.clip == str(CHECK_CLIP) else None
        print(f"transcribe #{i + 1}: {body['text']!r}  round trip {ms:.0f} ms, server infer {body['infer_ms']} ms"
              f"{'' if match is None else ('  OK' if match else '  MISMATCH, expected ' + repr(CHECK_TEXT))}")
        ok = ok and match is not False
    return 0 if ok else 1


# ------------------------------------------------------------------ say

def cmd_say(args, cfg) -> int:
    from .player import build_player

    if args.rate is not None:
        cfg.speaking_rate = args.rate
    log = EventLog(cfg.path(cfg.log_dir), keep=True)
    log.write("session", cmd="say", id=new_session_id(), pid=os.getpid())
    t0 = time.monotonic()
    player = build_player(cfg, log)
    done = threading.Event()
    reports: list[tuple[str, str]] = []
    said_at: dict[str, float] = {}

    def listener(speech_id: str, status: str) -> None:
        reports.append((speech_id, status))
        print(f"  report {speech_id} {status} after {time.monotonic() - said_at[speech_id]:.2f} s", flush=True)
        if len(reports) == len(args.text):
            done.set()

    player.set_listener(listener)
    player.start()
    player.warm.wait(30)
    player.ready.wait(10)
    print(f"voice loaded and warmed up in {time.monotonic() - t0:.1f} s (output {player.output.rate} Hz)")
    for text in args.text:
        sid = player.say(text, args.expect)
        said_at[sid] = time.monotonic()
        print(f"say {sid} expect={args.expect}: {text!r}")
    if args.cue:
        player.cue("thinking")
    if args.stop_after is not None:
        time.sleep(args.stop_after)
        t = time.monotonic()
        player.stop_speech()
        print(f"stop_speech returned in {(time.monotonic() - t) * 1000:.1f} ms")
    done.wait(120)
    starts = {e["speech_id"]: e for e in log.events if e["event"] == "playback_start"}
    synths = {e["speech_id"]: e for e in log.events if e["event"] == "synth"}
    for sid in said_at:
        if sid in starts:
            print(f"  {sid}: synth {synths[sid]['ms']} ms{' (cached)' if synths[sid]['cached'] else ''}, "
                  f"say to first audio {starts[sid]['wait_ms']} ms (+ output latency)")
    player.close()
    return 0


# ------------------------------------------------------------------ listen

LISTEN_EVENTS = {
    "state": lambda e: f"state -> {e['state']}",
    "wake": lambda e: f"wake word ({e['model']}) score {e['score']}"
                      + (f", inside a {e['during']} capture: dropping it up to the end of the phrase"
                         if e.get("during") else ""),
    "wake_end": lambda e: f"wake phrase ended ({e['by']}) after {e['waited_ms']} ms: capture starts",
    "wake_phrase_stripped": lambda e: "stripped a leading wake phrase from the transcript",
    "speech_start": lambda e: f"speech start ({e['source']})",
    "speech_end": lambda e: f"speech end: {e['reason']}, {e['duration_ms']} ms",
    "stt": lambda e: f"stt {e['backend']} {e['ms']} ms, {e['chars']} chars"
                     + (f", dropped: {e['dropped_reason']}" if e["dropped_reason"] else ""),
    "stt_fallback": lambda e: f"stt fallback from {e['backend']}: {e['reason']}",
    "stt_health": lambda e: f"stt server: {'ok ' + str(e.get('model')) if e['ok'] else 'unavailable: ' + e['error']}",
    "stt_fallback_ready": lambda e: f"local {e['model']} ready in {e['ms']} ms",
    "wake_model": lambda e: f"wake model: {e['model']}" + (" (STAND-IN)" if e["stand_in"] else ""),
    "mic_open": lambda e: f"mic open at {e['rate']} Hz" + (" (resampling to 16 kHz)" if e["resampling"] else ""),
    "ptt_takeover": lambda e: f"push-to-talk took over a {e['source']} capture",
    "playback_done": lambda e: f"tts {e['speech_id']} {e['status']}",
    "error": lambda e: f"ERROR {e['where']}: {e.get('error')}",
}


def event_printer(t0: float):
    def show(e: dict) -> None:
        fmt = LISTEN_EVENTS.get(e["event"])
        if fmt is not None:
            print(f"{e['mono'] - t0:8.2f}  {fmt(e)}", flush=True)
    return show


def cmd_listen(args, cfg, echo_back: bool = False) -> int:
    from .events import PrintSink
    from .listen import ReplyWindow, build_mic
    from .player import build_player

    if args.ptt:
        cfg.ptt = args.ptt
    t0 = time.monotonic()
    log = EventLog(cfg.path(cfg.log_dir), echo=event_printer(t0))
    sid = new_session_id()
    log.write("session", cmd="echo" if echo_back else "listen", id=sid, pid=os.getpid())
    window = ReplyWindow(cfg.assistant_name)
    player = None
    if not args.no_tts:
        player = build_player(cfg, log, window=window)
        player.start()

    class Sink(PrintSink):
        """Prints events; stands in for the agent by replying when asked to."""

        def on_speech_start(self, at=None):
            if player is not None:
                player.stop_speech()  # what the controller does
            super().on_speech_start(at)

        def on_utterance(self, text, ended_at=None, frame=None):
            super().on_utterance(text, ended_at, frame)
            if player is None or not text:
                return
            if echo_back:
                player.say(text, args.reply)
            elif args.reply != "none":
                player.say("Go on." if args.reply == "done" else "Yes?", args.reply)

    capture = None
    if args.wav:
        from .capture import FakeCapture, wav_items

        capture = FakeCapture(wav_items(args.wav), realtime=True)
    mic = build_mic(cfg, Sink(), player=player, window=window, log=log, capture=capture,
                    task_active=(lambda: True) if args.task_active else (lambda: False))
    if args.scores:
        peak = {"score": 0.0, "t": 0.0}

        def show_score(score, t):
            if score >= 0.2 and (score > peak["score"] or t - peak["t"] > 1.0):
                print(f"{time.monotonic() - t0:8.2f}  wake score {score:.2f}", flush=True)
                peak.update(score=score, t=t)
            elif t - peak["t"] > 1.0:
                peak["score"] = 0.0
        mic.listener.score_hook = show_score
    mic.start()
    how = {"none": "", "key": "  Push-to-talk: press Enter to start and again to stop."}.get(cfg.ptt, "")
    print(f"Listening. Say \"hey glance\" (or just talk while the reply window is open).{how}  Ctrl+C to quit.")
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else float("inf")
    try:
        while not (capture is not None and capture.done.is_set() and mic.idle.is_set() and mic.q.empty()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        if player is not None:
            player.close()
    print()
    print_stats(log_stats(pick_session(read_log(log_file(cfg)), sid) or []))
    return 0


# ------------------------------------------------------------------ stats (section 14.3 numbers from the JSONL log)

def log_file(cfg, day: str | None = None) -> Path:
    return cfg.path(cfg.log_dir) / f"audio-{day or time.strftime('%Y%m%d')}.jsonl"


def read_log(path: Path) -> list[dict]:
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue  # a line cut short by a crash
    except FileNotFoundError:
        pass
    return events


def new_session_id() -> str:
    return time.strftime("%H%M%S")  # unique within a day's log file, which is what stats reads


def split_sessions(events: list[dict]) -> list[tuple[str, str | None, list[dict]]]:
    """(id, cmd, events) per session marker. Events before the first marker form a session with id "-"."""
    out: list = []
    for e in events:
        if e.get("event") == "session":
            out.append((str(e.get("id") or f"#{len(out) + 1}"), e.get("cmd"), [e]))
        elif out:
            out[-1][2].append(e)
        else:
            out.append(("-", None, [e]))
    return out


def pick_session(events: list[dict], which: str = "last") -> list[dict] | None:
    """last: the last listen or echo session. all: everything. Otherwise a session id."""
    if which == "all":
        return events
    parts = split_sessions(events)
    if which == "last":
        listening = [evs for _, cmd, evs in parts if cmd in ("listen", "echo")]
        return listening[-1] if listening else None
    return next((evs for sid, _, evs in parts if sid == which), None)


def _pct(values: list[float]) -> dict:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "max": max(values) if values else float("nan")}


def log_stats(events: list[dict]) -> dict:
    """The section 14.3 numbers that can be read off the event log."""
    by = {}
    for e in events:
        by.setdefault(e.get("event"), []).append(e)
    get = lambda name: by.get(name, [])  # noqa: E731
    session = get("session")[-1] if get("session") else None
    out: dict = {"cmd": session.get("cmd") if session else None, "id": session.get("id") if session else None,
                 "minutes": (events[-1]["mono"] - events[0]["mono"]) / 60 if events else 0.0}
    wakes = get("wake")
    out["wakes"] = sum(1 for w in wakes if w.get("state", "wake") == "wake")
    out["wakes_in_open_capture"] = sum(1 for w in wakes if w.get("during"))
    out["wake_scores"] = sorted(w["score"] for w in wakes)
    starts = get("speech_start")
    out["speech_starts"] = {src: sum(1 for s in starts if s["source"] == src) for src in ("wake", "vad", "ptt")}
    delivered = get("utterance_delivered")
    out["utterances"] = len(delivered)
    out["empty_utterances"] = {}
    for d in delivered:
        if not d["chars"]:
            out["empty_utterances"][d["reason"]] = out["empty_utterances"].get(d["reason"], 0) + 1
    out["unanswered_starts"] = len(starts) - len(delivered)
    lat: dict = {}
    for d in delivered:
        if d.get("backend") and d.get("backend") != "none":
            lat.setdefault(d["backend"], []).append(d["latency_ms"])
    out["end_to_utterance_ms"] = {b: _pct(v) for b, v in lat.items()}
    stt: dict = {}
    for r in get("stt"):
        stt.setdefault(r["backend"], []).append(r["ms"])
    out["stt_ms"] = {b: _pct(v) for b, v in stt.items()}
    out["stt_fallbacks"] = len(get("stt_fallback"))
    # barge-in: the user's first voiced audio (speech_start at) to the sound actually stopping (playback_cut)
    barge = []
    for cut in get("playback_cut"):
        before = [s for s in starts if s.get("at") is not None and 0 <= cut["mono"] - s["mono"] <= 3.0]
        if before:
            barge.append(round((cut["wall"] - before[-1]["at"]) * 1000))
    out["barge_in_ms"] = _pct(barge)
    out["stop_to_silence_ms"] = _pct([c["ms"] for c in get("playback_cut")])
    cached = {e["speech_id"]: e.get("cached") for e in get("synth")}
    chars = {e["speech_id"]: e["chars"] for e in get("say_queued")}
    first = [(e["wait_ms"], chars.get(e["speech_id"])) for e in get("playback_start")
             if cached.get(e["speech_id"]) is False]
    out["say_to_first_audio_ms"] = _pct([w for w, _ in first])
    out["say_chars"] = sorted(c for _, c in first if c is not None)
    errors: dict = {}
    for e in get("error"):
        errors[e.get("where")] = errors.get(e.get("where"), 0) + 1
    out["errors"] = errors
    return out


def print_stats(st: dict) -> None:
    def fmt(p):
        return "none" if not p["n"] else f"n={p['n']} p50={p['p50']:.0f} p95={p['p95']:.0f} max={p['max']:.0f}"

    print(f"Session {st['id'] or '?'}: {st['cmd'] or '?'}, {st['minutes']:.1f} min")
    print(f"  wake detections: {st['wakes']} (scores {', '.join(f'{s:.2f}' for s in st['wake_scores']) or '-'})"
          + (f"; {st['wakes_in_open_capture']} inside open-state captures" if st["wakes_in_open_capture"] else ""))
    print(f"  speech starts: {st['speech_starts']}; utterances: {st['utterances']}"
          f" (empty: {st['empty_utterances'] or 0}); unanswered starts: {st['unanswered_starts']}")
    for b, p in st["end_to_utterance_ms"].items():
        print(f"  end of speech -> on_utterance, {b}: {fmt(p)} ms")
    for b, p in st["stt_ms"].items():
        print(f"  stt {b}: {fmt(p)} ms")
    print(f"  stt fallbacks: {st['stt_fallbacks']}")
    print(f"  user starts talking -> playback stopped: {fmt(st['barge_in_ms'])} ms")
    print(f"  stop_speech -> silence: {fmt(st['stop_to_silence_ms'])} ms")
    chars = st["say_chars"]
    print(f"  say -> first audio (uncached): {fmt(st['say_to_first_audio_ms'])} ms"
          + (f", median {chars[len(chars) // 2]} chars" if chars else ""))
    print(f"  errors: {st['errors'] or 0}")


def cmd_stats(args, cfg) -> int:
    path = Path(args.file) if args.file else log_file(cfg, args.day)
    events = read_log(path)
    if not events:
        print(f"no events in {path}")
        return 1
    chosen = pick_session(events, args.session)
    if chosen is None:
        print(f"no session {args.session!r} in {path}. Sessions:")
        for sid, cmd, evs in split_sessions(events):
            print(f"  {sid:8} {cmd or '?':9} {len(evs)} events")
        return 1
    print_stats(log_stats(chosen))
    return 0


# ------------------------------------------------------------------ selftest (section 12)

SELFTEST_TEXT = "Glance self test. After the tone, say: get me the can."


def record_seconds(cfg, seconds: float, log: EventLog) -> np.ndarray:
    """Record through the real capture path (16 kHz, or native rate resampled)."""
    from .capture import BLOCK_S, SoundDeviceCapture

    blocks, need = [], int(round(seconds / BLOCK_S))
    cap = SoundDeviceCapture(cfg.input_device, log)
    cap.start(lambda item: blocks.append(item[1]) if item[0] == "audio" else None)
    deadline = time.monotonic() + seconds + 3.0
    try:
        while len(blocks) < need and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        cap.close()
    if not blocks:
        raise RuntimeError("no audio from the mic")
    return np.concatenate(blocks[:need])


def run_selftest(cfg, player, record, backends, out=print) -> int:
    """Speaker, mic and every STT backend. Returns 0 when nothing failed."""
    statuses = []

    def row(status: str, name: str, detail: str) -> None:
        statuses.append(status)
        out(f"{status:4}  {name}: {detail}")

    wake = cfg.path(cfg.wake_model)
    row("PASS" if wake.is_file() else "WARN", "wake model",
        str(wake) if wake.is_file() else f"{wake} missing, the hey_jarvis stand-in will load")
    url = config.stt_url()
    row("PASS" if url else "WARN", "GLANCE_STT_URL", url or "not set: local STT only")
    if url:
        row("PASS" if config.stt_token() else "FAIL", "GLANCE_STT_TOKEN", "set" if config.stt_token() else "not set")

    if player is None:
        row("FAIL", "speaker", "the Player didn't load (see the error above)")
    else:
        reports, done = [], threading.Event()
        player.set_listener(lambda speech_id, status: (reports.append(status), done.set()))
        t0 = time.monotonic()
        player.say(SELFTEST_TEXT, "none")
        done.wait(20)
        status = reports[0] if reports else "no report"
        row("PASS" if status == "completed" else "FAIL", "speaker",
            f"{status} after {time.monotonic() - t0:.1f} s (you should have heard it)")
        player.cue("thinking")
        time.sleep(0.3)

    out('Recording 3 s: say "get me the can" now.')
    try:
        samples = record(3.0)
    except Exception as e:
        row("FAIL", "mic", repr(e))
        samples = None
    if samples is not None:
        peak = int(np.max(np.abs(samples.astype(np.int32)))) if len(samples) else 0
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
        status = "FAIL" if len(samples) < 2.5 * SAMPLE_RATE or peak == 0 else "WARN" if peak < 3000 else "PASS"
        row(status, "mic", f"{len(samples) / SAMPLE_RATE:.1f} s, peak {peak}, rms {rms:.0f}"
            + (" (speak louder or check the mic)" if status == "WARN" else ""))
        wav = to_wav(samples)
        for b in backends:
            t0 = time.monotonic()
            try:
                if isinstance(b, VultrWhisper):
                    health = b.health()
                    row("PASS", "stt vultr health", f"{health.get('model')} on {health.get('device')}")
                raw = b.transcribe(wav)
                ms = int((time.monotonic() - t0) * 1000)
                slow = b.name == "vultr" and ms > cfg.stt_timeout_s * 1000
                empty = not raw.text.strip()
                row("WARN" if slow or empty else "PASS", f"stt {b.name}", f"{ms} ms, transcript {raw.text!r}"
                    + (" (empty: nothing recognized as speech)" if empty else "")
                    + (" (over stt_timeout_s, so the Mic would have fallen back)" if slow else ""))
            except Exception as e:
                row("FAIL", f"stt {b.name}", f"{int((time.monotonic() - t0) * 1000)} ms: {e}")
    fails, warns = statuses.count("FAIL"), statuses.count("WARN")
    out(f"selftest: {'FAILED' if fails else 'ok'} ({fails} failed, {warns} warnings)")
    return 1 if fails else 0


def cmd_selftest(args, cfg) -> int:
    from .player import build_player

    log = EventLog(cfg.path(cfg.log_dir))
    log.write("session", cmd="selftest", id=new_session_id(), pid=os.getpid())
    player = None
    try:
        player = build_player(cfg, log)
        player.start()
    except Exception as e:
        print(f"Player failed to load: {e!r}")
    backends = []
    if config.stt_url():
        backends.append(VultrWhisper(config.stt_url(), config.stt_token(), timeout_s=10.0))
    local = LocalWhisper(cfg.fallback_model)
    t0 = time.monotonic()
    try:
        local.preload()
        print(f"local {cfg.fallback_model} loaded in {time.monotonic() - t0:.1f} s")
    except Exception as e:
        print(f"local {cfg.fallback_model} failed to load: {e!r}")
    backends.append(local)
    try:
        return run_selftest(cfg, player, lambda s: record_seconds(cfg, s, log), backends)
    finally:
        if player is not None:
            player.close()


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m audio.cli")
    p.add_argument("--config", help="config.yaml path (default audio/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--selftest", action="store_true", help="same as the selftest command")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("record", help="record one utterance to clips/ and label it")
    r.add_argument("--label", required=True)
    r.add_argument("--tag", default="", help="appended to the file name, e.g. quiet or noisy")
    r.add_argument("--dir", default=str(DEFAULT_CLIPS))

    b = sub.add_parser("stt-bench", help="transcribe every labeled clip with each backend")
    b.add_argument("dir", nargs="?", default=str(DEFAULT_CLIPS))
    b.add_argument("--backends", default="vultr,local")
    b.add_argument("--out", help="write per-clip results to this CSV")

    c = sub.add_parser("stt-check", help="call /health and transcribe a test clip on GLANCE_STT_URL")
    c.add_argument("--clip", default=str(CHECK_CLIP))
    c.add_argument("-n", type=int, default=3, help="number of transcriptions (the first may be slower)")
    c.add_argument("--timeout", type=float, default=10.0)

    s = sub.add_parser("say", help="speak through the real Player and print the reports")
    s.add_argument("text", nargs="+", help="one or more sentences, queued in order")
    s.add_argument("--expect", default="none", choices=["none", "done", "answer"])
    s.add_argument("--rate", type=float, help="override speaking_rate")
    s.add_argument("--stop-after", type=float, help="call stop_speech after this many seconds")
    s.add_argument("--cue", action="store_true", help="also request the thinking cue (dropped if speech is queued)")

    for name, helptext in (("listen", "real mic, wake word, VAD and STT; prints every event"),
                           ("echo", "listen, then say each transcript back")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--ptt", choices=["none", "key"], help="override the ptt config (key: Enter toggles)")
        sp.add_argument("--reply", default="none", choices=["none", "done", "answer"],
                        help="reply after each utterance with this expect, which opens the reply window")
        sp.add_argument("--no-tts", action="store_true", help="don't load the Player (no replies)")
        sp.add_argument("--task-active", action="store_true", help="pretend a task is running (no wake word)")
        sp.add_argument("--scores", action="store_true", help="print wake word scores of 0.2 and up")
        sp.add_argument("--wav", help="replay a 16 kHz mono WAV in real time instead of the mic")
        sp.add_argument("--minutes", type=float, help="stop after this many minutes (soak test)")

    st = sub.add_parser("stats", help="section 14.3 numbers from the event log")
    st.add_argument("--file", help="a JSONL log file (default: today's)")
    st.add_argument("--day", help="YYYYMMDD instead of today")
    st.add_argument("--session", default="last",
                    help="last (the last listen or echo session, the default), all, or a session id")

    sub.add_parser("selftest", help="play a sentence, record 3 s, transcribe with every backend (section 12)")

    args = p.parse_args(argv)
    if args.selftest:
        args.cmd = "selftest"
    if args.cmd is None:
        p.error("a command is required")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    return {"record": cmd_record, "stt-bench": cmd_stt_bench, "say": cmd_say, "stt-check": cmd_stt_check,
            "listen": cmd_listen, "echo": lambda a, c: cmd_listen(a, c, echo_back=True), "stats": cmd_stats,
            "selftest": cmd_selftest}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
