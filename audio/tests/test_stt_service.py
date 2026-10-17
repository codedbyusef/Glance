import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from audio.server.stt_service import create_app
from audio.stt_client import to_wav

AUTH = {"Authorization": "Bearer tok"}


class Seg:
    def __init__(self, text, start, end, no_speech, logprob):
        self.text, self.start, self.end, self.no_speech_prob, self.avg_logprob = text, start, end, no_speech, logprob


class StubModel:
    def __init__(self, segments=None):
        self.segments = segments if segments is not None else [Seg(" get me the can", 0, 1.9, 0.02, -0.21)]
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs))
        return iter(self.segments), None


def client(model=None, prompt=None, background=False, factory=None):
    model = model or StubModel()
    app = create_app(model_factory=factory or (lambda name: model), token="tok", model_name="small.en",
                     initial_prompt=prompt, load_in_background=background, device="cuda")
    return TestClient(app), model


def wav(seconds=1.0, rate=16000):
    return to_wav(np.zeros(int(seconds * rate), dtype=np.int16), rate)


def test_health_and_transcribe():
    c, model = client()
    with c:
        assert c.get("/health", headers=AUTH).json() == {"ready": True, "model": "small.en", "device": "cuda", "beam_size": 5,
                                                              "initial_prompt": "done, yes, no, stop, wait, cancel, repeat"}
        r = c.post("/transcribe", content=wav(1.9), headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "get me the can"
    assert body["duration_s"] == 1.9
    assert body["no_speech_prob"] == 0.02 and body["avg_logprob"] == -0.21 and body["model"] == "small.en"
    assert isinstance(body["infer_ms"], int)
    # warm-up on 1 s of silence with the decoder actually running, then the request; settings from section 9
    assert model.calls[0][0] == 16000 and model.calls[0][1]["vad_filter"] is False
    kw = model.calls[1][1]
    assert kw == {"language": "en", "beam_size": 5, "vad_filter": True, "condition_on_previous_text": False,
                  "initial_prompt": "done, yes, no, stop, wait, cancel, repeat"}


@pytest.mark.parametrize("prompt", ["", "none", "None"])
def test_empty_or_none_prompt_means_no_prompt(prompt):
    c, model = client(prompt=prompt)
    with c:
        c.post("/transcribe", content=wav(), headers=AUTH)
        assert c.get("/health", headers=AUTH).json()["initial_prompt"] is None
    assert model.calls[1][1]["initial_prompt"] is None


def test_no_segments():
    c, _ = client(StubModel(segments=[]))
    with c:
        body = c.post("/transcribe", content=wav(), headers=AUTH).json()
    assert (body["text"], body["no_speech_prob"], body["avg_logprob"]) == ("", 1.0, 0)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "tok"}])
def test_auth(headers):
    c, _ = client()
    with c:
        assert c.post("/transcribe", content=wav(), headers=headers).status_code == 401
        assert c.get("/health", headers=headers).status_code == 401


def test_too_long_is_413():
    c, _ = client()
    with c:
        assert c.post("/transcribe", content=wav(15.5), headers=AUTH).status_code == 413
        assert c.post("/transcribe", content=wav(15.0), headers=AUTH).status_code == 200


@pytest.mark.parametrize("body", [
    b"not a wav at all",
    to_wav(np.zeros(8000, dtype=np.int16), 8000),  # wrong rate
], ids=["garbage", "8khz"])
def test_bad_format_is_415(body):
    c, _ = client()
    with c:
        assert c.post("/transcribe", content=body, headers=AUTH).status_code == 415


def test_stereo_is_415():
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0" * 6400)
    c, _ = client()
    with c:
        assert c.post("/transcribe", content=buf.getvalue(), headers=AUTH).status_code == 415


def test_health_503_while_loading_then_200():
    release = threading.Event()
    model = StubModel()

    def slow_factory(name):
        release.wait(5)
        return model

    c, _ = client(factory=slow_factory, background=True)
    with c:
        r = c.get("/health", headers=AUTH)
        assert r.status_code == 503 and r.json()["ready"] is False
        assert c.post("/transcribe", content=wav(), headers=AUTH).status_code == 503
        release.set()
        for _ in range(100):
            if c.get("/health", headers=AUTH).status_code == 200:
                break
            threading.Event().wait(0.02)
        assert c.get("/health", headers=AUTH).json()["ready"] is True


def test_missing_token_refuses_to_start():
    app = create_app(model_factory=lambda n: StubModel(), token="", load_in_background=False)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_request_that_waited_too_long_is_dropped():
    gate = threading.Event()
    started = threading.Event()

    class SlowModel(StubModel):
        def transcribe(self, audio, **kwargs):
            if len(self.calls) >= 1:  # after warm-up
                started.set()
                gate.wait(5)
            return super().transcribe(audio, **kwargs)

    model = SlowModel()
    app = create_app(model_factory=lambda n: model, token="tok", load_in_background=False, stale_s=0.2)
    results = {}
    with TestClient(app) as c:
        t = threading.Thread(target=lambda: results.setdefault("first", c.post("/transcribe", content=wav(),
                                                                                 headers=AUTH)))
        t.start()
        assert started.wait(2)
        t2 = threading.Thread(target=lambda: results.setdefault("second", c.post("/transcribe", content=wav(),
                                                                                  headers=AUTH)))
        t2.start()
        threading.Event().wait(0.4)
        gate.set()
        t.join(5)
        t2.join(5)
    assert results["first"].status_code == 200
    assert results["second"].status_code == 503 and results["second"].json()["error"] == "stale"


def test_cuda_dll_dirs_is_noop_without_packages():
    from audio.server.stt_service import add_cuda_dll_dirs
    assert isinstance(add_cuda_dll_dirs(), list)
