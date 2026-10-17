import csv

import numpy as np
import pytest

from audio import cli, config
from audio.capture import resample_to
from audio.stt_client import RawTranscript, to_wav


def test_defaults_without_file(tmp_path):
    cfg = config.load(tmp_path / "missing.yaml")
    assert cfg.max_utterance_s == 10 and cfg.fallback_preload is True and cfg.assistant_name == "Glance"
    assert cfg.path("models/x.onnx") == config.AUDIO_DIR / "models" / "x.onnx"


def test_yaml_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("vad_aggressiveness: 3\nspeaking_rate: 1.2\nptt: gpio:17\n")
    cfg = config.load(p)
    assert (cfg.vad_aggressiveness, cfg.speaking_rate, cfg.ptt) == (3, 1.2, "gpio:17")


def test_unknown_key_is_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("vad_agressiveness: 3\n")
    with pytest.raises(ValueError, match="vad_agressiveness"):
        config.load(p)


def test_env_vars(monkeypatch):
    monkeypatch.setenv("GLANCE_STT_URL", "https://x")
    monkeypatch.delenv("GLANCE_STT_TOKEN", raising=False)
    assert config.stt_url() == "https://x" and config.stt_token() is None


def test_resample_48k_to_16k():
    t = np.arange(48000) / 48000
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    out = resample_to(tone, 48000)
    assert out.dtype == np.int16 and len(out) == 16000
    assert abs(np.abs(out[1000:-1000]).max() - 10000) < 500


class ScriptedBackend:
    def __init__(self, name, answers):
        self.name = name
        self.answers = answers

    def transcribe(self, wav_bytes):
        return RawTranscript(self.answers.pop(0))


def test_stt_bench(tmp_path, monkeypatch, capsys):
    labels = [("a.wav", "stop"), ("b.wav", "get me the can"), ("c.wav", "yes")]
    for name, _ in labels:
        (tmp_path / name).write_bytes(to_wav(np.zeros(1600, dtype=np.int16)))
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows(labels)
    monkeypatch.setattr(cli, "bench_backends", lambda names, cfg: [
        ScriptedBackend("vultr", ["Stop.", "Get me the can!", "Yes."]),
        ScriptedBackend("local", ["stop", "get me the cam", "You"]),
    ])
    out_csv = tmp_path / "out.csv"
    assert cli.main(["stt-bench", str(tmp_path), "--out", str(out_csv)]) == 0
    out = capsys.readouterr().out
    vultr_line = next(line for line in out.splitlines() if line.startswith("vultr "))
    local_line = next(line for line in out.splitlines() if line.startswith("local "))
    assert "100%" in vultr_line
    assert "33%" in local_line and "50%" in local_line  # 1 of 3 overall, 1 of 2 command clips
    assert "MISS c.wav" in out
    assert len(list(csv.DictReader(open(out_csv)))) == 6


def test_bench_key():
    assert cli.bench_key("  Get me, the CAN! ") == "get me the can"


def test_stt_bench_skips_missing_clips(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.wav").write_bytes(to_wav(np.zeros(1600, dtype=np.int16)))
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows([("a.wav", "stop"), ("gone.wav", "yes")])
    monkeypatch.setattr(cli, "bench_backends", lambda names, cfg: [ScriptedBackend("local", ["stop"])])
    assert cli.main(["stt-bench", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "skipping missing clip gone.wav" in out
    assert "100%" in next(line for line in out.splitlines() if line.startswith("local "))


def test_stt_check(monkeypatch, capsys):
    import httpx

    from audio.stt_client import VultrWhisper

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True, "model": "small.en"})
        return httpx.Response(200, json={"text": "Get me the can.", "infer_ms": 120, "duration_s": 1.5,
                                         "no_speech_prob": 0.0, "avg_logprob": -0.1, "model": "small.en"})

    monkeypatch.setenv("GLANCE_STT_URL", "http://gpu:8001")
    monkeypatch.setenv("GLANCE_STT_TOKEN", "t")
    monkeypatch.setattr(cli, "VultrWhisper",
                        lambda url, token, timeout: VultrWhisper(url, token, timeout, httpx.MockTransport(handler)))
    assert cli.main(["stt-check", "-n", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("  OK") == 2 and "server infer 120 ms" in out
    assert "token: set" in out and "'t'" not in out


# ------------------------------------------------------------------ stats

def ev(event, mono, **fields):
    return {"event": event, "mono": mono, "wall": 1000.0 + mono, **fields}


def test_log_stats_reads_the_acceptance_numbers():
    events = [
        ev("session", 0.0, cmd="old"), ev("wake", 0.5, score=0.9, state="wake"),
        ev("session", 1.0, cmd="echo"),
        ev("wake", 2.0, score=0.95, state="wake"), ev("speech_start", 2.0, source="wake", at=1002.0),
        ev("stt", 4.0, backend="vultr", ms=300), ev("utterance_delivered", 4.0, chars=5, reason="end",
                                                    backend="vultr", latency_ms=1000),
        ev("say_queued", 4.1, speech_id="s1", chars=40), ev("synth", 4.3, speech_id="s1", ms=200, cached=False),
        ev("playback_start", 4.4, speech_id="s1", wait_ms=300),
        ev("speech_start", 5.0, source="vad", at=1004.8),  # the user talks over the reply
        ev("stop_speech", 5.05), ev("playback_cut", 5.1, ms=50),
        ev("wake", 5.5, score=0.8, state="open", during="vad"),
        ev("stt_fallback", 6.0, backend="vultr", reason="timeout"), ev("stt", 6.5, backend="local", ms=900),
        ev("utterance_delivered", 6.5, chars=0, reason="end", backend="local", latency_ms=2900),
        ev("speech_start", 8.0, source="ptt", at=1008.0),
        ev("utterance_delivered", 8.1, chars=0, reason="too_short", backend=None, latency_ms=10),
        ev("error", 9.0, where="capture"),
    ]
    st = cli.log_stats(cli.last_session(events))
    assert st["cmd"] == "echo"
    assert st["wakes"] == 1 and st["wakes_in_open_capture"] == 1
    assert st["speech_starts"] == {"wake": 1, "vad": 1, "ptt": 1}
    assert st["utterances"] == 3 and st["unanswered_starts"] == 0
    assert st["empty_utterances"] == {"end": 1, "too_short": 1}
    assert st["end_to_utterance_ms"]["vultr"]["p50"] == 1000 and st["end_to_utterance_ms"]["local"]["n"] == 1
    assert st["stt_fallbacks"] == 1
    assert st["barge_in_ms"]["p50"] == pytest.approx(300)  # voice at 1004.8, sound cut at 1005.1
    assert st["stop_to_silence_ms"]["max"] == 50
    assert st["say_to_first_audio_ms"]["p50"] == 300 and st["say_chars"] == [40]
    assert st["errors"] == {"capture": 1}
    cli.print_stats(st)  # formats without error


def test_read_log_skips_broken_lines(tmp_path):
    f = tmp_path / "audio.jsonl"
    f.write_text('{"event": "a", "mono": 1, "wall": 1}\n{"event": "b", "mo\n', encoding="utf-8")
    assert [e["event"] for e in cli.read_log(f)] == ["a"]
    assert cli.read_log(tmp_path / "missing.jsonl") == []


def test_stats_command(tmp_path, capsys):
    f = tmp_path / "audio.jsonl"
    f.write_text("\n".join(cli.json.dumps(e) for e in [ev("session", 0.0, cmd="listen"),
                                                         ev("wake", 1.0, score=0.9, state="wake")]),
                 encoding="utf-8")
    assert cli.main(["stats", "--file", str(f)]) == 0
    assert "wake detections: 1" in capsys.readouterr().out


# ------------------------------------------------------------------ selftest

class FakeBackend:
    def __init__(self, name, text="get me the can", error=None):
        self.name, self.text, self.error = name, text, error

    def transcribe(self, wav):
        if self.error:
            raise self.error
        return RawTranscript(self.text)


def selftest_player():
    from audio.log import null_log
    from audio.player import FakeOutput, FakeSynth, Player

    p = Player(FakeSynth(), FakeOutput(realtime=False), log=null_log(), fixed_phrases=())
    p.start()
    return p


def speech(seconds=3.0, amp=3000):
    return (np.sin(np.arange(int(seconds * 16000)) / 5) * amp).astype(np.int16)


def test_selftest_passes(monkeypatch):
    monkeypatch.delenv("GLANCE_STT_URL", raising=False)
    lines = []
    p = selftest_player()
    try:
        code = cli.run_selftest(config.Config(), p, lambda s: speech(s), [FakeBackend("local")], out=lines.append)
    finally:
        p.close()
    assert code == 0
    text = "\n".join(lines)
    assert "PASS  speaker: completed" in text and "PASS  mic" in text
    assert "PASS  stt local" in text and "get me the can" in text


def test_selftest_reports_failures(monkeypatch):
    from audio.stt_client import SttError

    monkeypatch.setenv("GLANCE_STT_URL", "http://x")
    monkeypatch.delenv("GLANCE_STT_TOKEN", raising=False)
    lines = []
    code = cli.run_selftest(config.Config(), None, lambda s: np.zeros(48000, np.int16),
                            [FakeBackend("vultr", error=SttError("timeout")), FakeBackend("local")],
                            out=lines.append)
    assert code == 1
    text = "\n".join(lines)
    for row in ("FAIL  GLANCE_STT_TOKEN", "FAIL  speaker", "FAIL  mic", "FAIL  stt vultr", "PASS  stt local"):
        assert row in text


def test_selftest_mic_error_is_a_failure_not_a_crash(monkeypatch):
    monkeypatch.delenv("GLANCE_STT_URL", raising=False)

    def broken(seconds):
        raise OSError("no input device")

    lines = []
    assert cli.run_selftest(config.Config(), None, broken, [], out=lines.append) == 1
    assert any(line.startswith("FAIL  mic") for line in lines)


def test_selftest_flag(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_selftest", lambda args, cfg: called.append(args.cmd) or 0)
    assert cli.main(["--selftest"]) == 0 and called == ["selftest"]
    assert cli.main(["selftest"]) == 0 and called == ["selftest", "selftest"]
