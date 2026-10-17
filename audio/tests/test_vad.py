import pytest

from audio.config import Config
from audio.vad import FRAME_S, Endpointer, FakeVad, StartDetector, frames

import numpy as np


def run(ep, flags, t0=0.0):
    """Push frame flags; return (result, time of the frame that produced it)."""
    for i, v in enumerate(flags):
        t = t0 + i * FRAME_S
        res = ep.push(bool(v), t)
        if res is not None:
            return res, round(t + FRAME_S, 4)
    return None, None


def frames_of(seconds, voiced):
    return [voiced] * round(seconds / FRAME_S)


def test_start_needs_200ms_voiced_within_300ms():
    d = StartDetector(200)
    assert [d.push(True, i * FRAME_S) for i in range(9)] == [None] * 9
    assert d.push(True, 9 * FRAME_S) == 0.0


def test_start_reports_first_voiced_frame_in_window():
    d = StartDetector(200)
    flags = [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    out = [d.push(bool(v), i * FRAME_S) for i, v in enumerate(flags)]
    assert out[-1] == pytest.approx(2 * FRAME_S)
    assert all(o is None for o in out[:-1])


def test_clicks_do_not_start():
    d = StartDetector(200)
    flags = [1, 0] * 30  # 50% voiced: 7 or 8 of every 15 frames
    assert all(d.push(bool(v), i * FRAME_S) is None for i, v in enumerate(flags))


def test_end_needs_700ms_of_silence():
    cfg = Config()
    ep = Endpointer(cfg, 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, at = run(ep, frames_of(0.5, 1) + frames_of(2, 0))
    assert res.kind == "end"
    assert at == pytest.approx(0.5 + 0.7)
    assert (res.first_voiced, res.last_voiced_end) == (0.0, pytest.approx(0.5))


def test_brief_pause_does_not_end():
    ep = Endpointer(Config(), 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, at = run(ep, frames_of(0.5, 1) + frames_of(0.6, 0) + frames_of(0.5, 1) + frames_of(1, 0))
    assert res.kind == "end" and res.last_voiced_end == pytest.approx(1.6)


def test_too_short_burst_waits_then_reports_short():
    cfg = Config()  # min_utterance_ms 250, no_speech_after_wake_s 4
    ep = Endpointer(cfg, 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, at = run(ep, frames_of(0.2, 1) + frames_of(6, 0))
    assert res.kind == "short"
    assert at == pytest.approx(0.2 + 4.0)


def test_speech_after_short_burst_is_the_same_utterance():
    ep = Endpointer(Config(), 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, _ = run(ep, frames_of(0.2, 1) + frames_of(1.5, 0) + frames_of(0.5, 1) + frames_of(1, 0))
    assert res.kind == "end"
    assert res.first_voiced == 0.0  # the burst is part of it
    assert res.last_voiced_end == pytest.approx(2.2)


def test_exactly_min_length_is_enough():
    ep = Endpointer(Config(), 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, _ = run(ep, frames_of(0.26, 1) + frames_of(1, 0))
    assert res.kind == "end"


def test_max_utterance():
    ep = Endpointer(Config(), 0.0, first_voiced=0.0, last_voiced_end=0.02)
    res, at = run(ep, frames_of(12, 1))
    assert res.kind == "max" and at == pytest.approx(10.0)


def test_no_speech_after_wake():
    ep = Endpointer(Config(), 0.0)
    res, at = run(ep, frames_of(6, 0))
    assert res.kind == "no_speech" and at == pytest.approx(4.0)


def test_speech_after_wake_starts_with_start_rule():
    ep = Endpointer(Config(), 0.0)
    flags = frames_of(1.0, 0) + [1, 0] * 10 + frames_of(0.6, 1) + frames_of(1, 0)
    res, _ = run(ep, flags)
    assert res.kind == "end"
    assert res.first_voiced >= 1.0


def test_fake_vad_and_frames():
    block = np.concatenate([np.zeros(640, np.int16), np.full(640, 2000, np.int16)])
    fs = frames(block)
    assert [len(f) for f in fs] == [320] * 4
    assert [FakeVad().is_speech(f) for f in fs] == [False, False, True, True]
