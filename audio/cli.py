"""Dev commands (CLAUDE.md section 14.2). Run from the repo root: python -m audio.cli <cmd>."""
from __future__ import annotations

import argparse
import csv
import logging
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

    print(f"\n{'backend':8} {'clips':>5} {'exact':>7} {'cmds':>7} {'p50 ms':>7} {'p95 ms':>7} {'failed':>6}")
    for name in dict.fromkeys(r["backend"] for r in rows):
        rs = [r for r in rows if r["backend"] == name]
        cmds = [r for r in rs if bench_key(r["label"]) in COMMANDS]
        ms = [r["ms"] for r in rs if not r["failed"]]
        acc = sum(r["ok"] for r in rs) / len(rs)
        cacc = f"{sum(r['ok'] for r in cmds) / len(cmds):.0%}" if cmds else "-"
        print(f"{name:8} {len(rs):5} {acc:7.0%} {cacc:>7} {percentile(ms, 50):7.0f} {percentile(ms, 95):7.0f}"
              f" {sum(r['failed'] for r in rs):6}")
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
    "wake": lambda e: f"wake word ({e['model']}) score {e['score']}",
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
    try:
        while not (capture is not None and capture.done.is_set() and mic.idle.is_set() and mic.q.empty()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        if player is not None:
            player.close()
    return 0


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m audio.cli")
    p.add_argument("--config", help="config.yaml path (default audio/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

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

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    return {"record": cmd_record, "stt-bench": cmd_stt_bench, "say": cmd_say, "stt-check": cmd_stt_check,
            "listen": cmd_listen, "echo": lambda a, c: cmd_listen(a, c, echo_back=True)}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
