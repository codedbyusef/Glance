import httpx
import numpy as np
import pytest

from audio.log import null_log
from audio.stt_client import (FallbackTranscriber, LocalWhisper, RawTranscript, SttError, VultrWhisper,
                              read_wav, summarize_segments, to_wav)

WAV = to_wav(np.zeros(16000, dtype=np.int16))


def ok_body(text="get me the can", no_speech=0.02, logprob=-0.21):
    return {"text": text, "duration_s": 1.0, "infer_ms": 100, "no_speech_prob": no_speech,
            "avg_logprob": logprob, "model": "small.en"}


def vultr(handler):
    return VultrWhisper("https://stt.example/", "secret", 2.0, transport=httpx.MockTransport(handler))


class FakeBackend:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = 0

    def transcribe(self, wav_bytes):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def make(primary, fallback=None, blocklist=("you", "thanks for watching", "thank you for watching")):
    log = null_log()
    return FallbackTranscriber(primary, fallback, list(blocklist), log), log


def test_vultr_success_sends_wav_and_bearer_token():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json=ok_body())

    t, log = make(vultr(handler), FakeBackend("local", error=AssertionError("must not be called")))
    r = t.transcribe(WAV)
    assert (r.text, r.backend, r.dropped_reason) == ("get me the can", "vultr", None)
    assert seen == {"url": "https://stt.example/transcribe", "auth": "Bearer secret", "body": WAV}
    assert log.of("stt")[0]["chars"] == len("get me the can")
    assert "secret" not in str(log.events)


def test_url_may_include_the_transcribe_path():
    assert VultrWhisper("http://h:8001/transcribe", None, 1).base == "http://h:8001"


@pytest.mark.parametrize("handler", [
    lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=req)),
    lambda req: (_ for _ in ()).throw(httpx.ConnectError("down", request=req)),
    lambda req: httpx.Response(500),
    lambda req: httpx.Response(401),
    lambda req: httpx.Response(200, text="not json"),
], ids=["timeout", "connect", "500", "401", "bad-json"])
def test_primary_failures_fall_back_to_local(handler):
    local = FakeBackend("local", RawTranscript("stop"))
    t, log = make(vultr(handler), local)
    r = t.transcribe(WAV)
    assert (r.text, r.backend) == ("stop", "local")
    assert local.calls == 1
    assert len(log.of("stt_fallback")) == 1


def test_timeout_is_reported_as_timeout():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(SttError, match="timeout"):
        vultr(handler).transcribe(WAV)


def test_both_failing_returns_empty_with_reason():
    t, log = make(FakeBackend("vultr", error=SttError("timeout")), FakeBackend("local", error=SttError("boom")))
    r = t.transcribe(WAV)
    assert (r.text, r.backend, r.dropped_reason) == ("", "none", "stt_failed")
    assert log.of("stt")[0]["dropped_reason"] == "stt_failed"


def test_unexpected_exception_in_backend_is_contained():
    t, log = make(FakeBackend("vultr", error=RuntimeError("bug")), FakeBackend("local", RawTranscript("yes")))
    assert t.transcribe(WAV).text == "yes"
    assert log.of("error")


def test_no_primary_configured_goes_straight_to_local():
    t, _ = make(None, FakeBackend("local", RawTranscript("yes")))
    assert t.transcribe(WAV).backend == "local"


@pytest.mark.parametrize("raw,expected,reason", [
    (RawTranscript("  get   me\tthe can  "), "get me the can", None),
    (RawTranscript("Stop."), "Stop.", None),  # case and punctuation kept
    (RawTranscript("noise", no_speech_prob=0.7, avg_logprob=-1.2), "", "no_speech"),
    (RawTranscript("ok", no_speech_prob=0.7, avg_logprob=-0.5), "ok", None),  # both conditions needed
    (RawTranscript("ok", no_speech_prob=0.5, avg_logprob=-1.5), "ok", None),
    (RawTranscript("   "), "", "empty"),
    (RawTranscript("[BLANK_AUDIO]"), "", "non_speech_tag"),
    (RawTranscript("(music)"), "", "non_speech_tag"),
    (RawTranscript(" [Music] (applause) "), "", "non_speech_tag"),
    (RawTranscript("[pause] get the can"), "[pause] get the can", None),
    (RawTranscript("You"), "", "blocklist"),
    (RawTranscript("you."), "", "blocklist"),
    (RawTranscript("Thanks for watching!"), "", "blocklist"),
    (RawTranscript("thank you for watching"), "", "blocklist"),
    (RawTranscript("thank you"), "thank you", None),
    (RawTranscript("you there"), "you there", None),
])
def test_filters(raw, expected, reason):
    t, _ = make(FakeBackend("vultr", raw))
    r = t.transcribe(WAV)
    assert (r.text, r.dropped_reason) == (expected, reason)


class Seg:
    def __init__(self, text, start, end, no_speech, logprob):
        self.text, self.start, self.end, self.no_speech_prob, self.avg_logprob = text, start, end, no_speech, logprob


def test_summarize_segments():
    raw = summarize_segments(iter([Seg(" get me", 0, 1, 0.1, -0.2), Seg(" the can", 1, 4, 0.3, -0.6)]))
    assert raw.text == "get me the can"
    assert raw.no_speech_prob == 0.3
    assert raw.avg_logprob == pytest.approx((-0.2 * 1 + -0.6 * 3) / 4)


def test_summarize_no_segments():
    raw = summarize_segments([])
    assert (raw.text, raw.no_speech_prob, raw.avg_logprob) == ("", 1.0, 0.0)


def test_wav_round_trip():
    samples = (np.arange(1600) % 100).astype(np.int16)
    out, rate, channels, width = read_wav(to_wav(samples))
    assert (rate, channels, width) == (16000, 1, 2)
    assert np.array_equal(out, samples)


class StubWhisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        assert audio.dtype == np.float32
        return iter([Seg(" yes", 0, 0.5, 0.01, -0.1)]), None


def test_local_whisper_loads_lazily_once():
    loads = []

    def factory():
        loads.append(1)
        return StubWhisper()

    local = LocalWhisper("tiny.en", model_factory=factory)
    assert loads == []
    assert local.transcribe(WAV).text == "yes"
    local.transcribe(WAV)
    assert loads == [1]


def test_local_whisper_load_failure_is_stt_error():
    def factory():
        raise OSError("no model files")

    with pytest.raises(SttError):
        LocalWhisper(model_factory=factory).transcribe(WAV)
