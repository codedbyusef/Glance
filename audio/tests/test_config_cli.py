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
