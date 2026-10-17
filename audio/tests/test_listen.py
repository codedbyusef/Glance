import threading
import time

import numpy as np
import pytest

from audio.capture import FakeCapture
from audio.config import Config
from audio.events import RecordingSink
from audio.listen import Finish, Listener, Mic, ReplyWindow, SpeechStart, strip_wake_phrase, wake_phrase_pattern
from audio.log import null_log
from audio.stt_client import SttResult
from audio.vad import FakeVad
from audio.wake import FakeDetector

LOUD = np.full(1280, 3000, np.int16)
QUIET = np.zeros(1280, np.int16)
B = 0.08
SHORT_START = Config(speech_start_ms=200)  # at the default 300 ms, a burst under 250 ms can't start speech at all


def blocks(*parts):
    """blocks(("q", 10), ("l", 5)) -> list of 80 ms blocks."""
    out = []
    for kind, n in parts:
        out += [LOUD if kind == "l" else QUIET] * n
    return out


class Feeder:
    """Drives a Listener with consecutive blocks and collects actions."""

    def __init__(self, listener, t0=1000.0):
        self.l = listener
        self.t = t0
        self.actions = []

    def feed(self, bs):
        for b in bs:
            self.actions += self.l.feed(b, self.t)
            self.t += B
        return self.actions

    def ptt(self, down):
        self.actions += self.l.ptt(down, self.t)
        return self.actions


def make_listener(wake=False, scores=None, detector=True, cfg=None):
    det = FakeDetector(scores) if detector else None
    return Listener(cfg or Config(), FakeVad(), det, lambda: wake, null_log()), det


# ------------------------------------------------------------------ Listener

def test_open_state_vad_start_and_end_with_preroll():
    l, _ = make_listener(wake=False)
    f = Feeder(l)
    f.feed(blocks(("q", 10)))
    speech_t = f.t
    acts = f.feed(blocks(("l", 10), ("q", 12)))
    assert [type(a) for a in acts] == [SpeechStart, Finish]
    start, fin = acts
    assert start.source == "vad" and start.at == pytest.approx(speech_t)
    assert fin.reason == "end" and fin.ended_at == pytest.approx(speech_t + 0.8)
    assert fin.duration_ms == 800
    # pre-roll 300 ms + 800 ms of speech + 200 ms tail
    assert len(fin.audio) == pytest.approx((0.3 + 0.8 + 0.2) * 16000, abs=320)
    assert l.state() == "open"


def test_wake_state_ignores_speech_without_wake_word():
    l, _ = make_listener(wake=True)
    acts = Feeder(l).feed(blocks(("l", 20), ("q", 20)))
    assert acts == []
    assert l.state() == "wake"


def test_wake_detection_starts_capture_without_preroll():
    l, det = make_listener(wake=True, scores=[0.1] * 5 + [0.9])
    f = Feeder(l)
    f.feed(blocks(("l", 6)))  # "hey glance" spoken; detected on the 6th block
    assert [type(a) for a in f.actions] == [SpeechStart]
    assert f.actions[0].source == "wake"
    after_wake = f.t
    f.feed(blocks(("q", 2), ("l", 10), ("q", 12)))
    fin = f.actions[-1]
    assert isinstance(fin, Finish) and fin.reason == "end"
    # the clip starts after the wake block: 160 ms quiet + 800 ms speech + 200 ms tail, no wake phrase
    assert len(fin.audio) == pytest.approx((0.16 + 0.8 + 0.2) * 16000, abs=320)
    assert f.actions[0].at == pytest.approx(after_wake)
    assert det.blocks == 6 + 24


def clip_s(fin):
    return len(fin.audio) / 16000


def test_wake_waits_for_score_to_fall_when_speech_runs_on():
    # "hey glance get me the can" in one breath: no gap, so capture starts when the score falls below 0.3
    l, _ = make_listener(wake=True, scores=[0.1] * 5 + [0.6, 0.98, 1.0, 0.25])
    f = Feeder(l)
    f.feed(blocks(("l", 6)))
    detected = f.t
    assert [type(a) for a in f.actions] == [SpeechStart] and f.actions[0].at == pytest.approx(detected)
    f.feed(blocks(("l", 3)))  # the wake phrase tail; the score falls on the last of these blocks
    edge = f.t
    f.feed(blocks(("l", 10), ("q", 12)))
    fin = f.actions[-1]
    assert fin.reason == "end" and fin.ended_at == pytest.approx(edge + 0.8)
    assert clip_s(fin) == pytest.approx(0.8 + 0.2, abs=0.02)  # only the audio after the edge, plus the tail


def test_wake_gap_ends_the_phrase_before_the_score_falls():
    # the score lags: it stays high while the phrase's tail and the pause go by, but the first 60 ms gap ends it
    l, _ = make_listener(wake=True, scores=[0.1] * 5 + [0.6] + [0.99] * 12)
    f = Feeder(l)
    f.feed(blocks(("l", 6), ("l", 2)))  # detection, then 160 ms of the phrase's tail
    gap_start = f.t
    f.feed(blocks(("q", 6), ("l", 10), ("q", 12)))
    start, fin = f.actions
    assert fin.reason == "end"
    assert clip_s(fin) == pytest.approx(0.48 + 0.8 + 0.2, abs=0.02)  # from the gap: no tail of the wake phrase
    assert fin.ended_at == pytest.approx(gap_start + 0.48 + 0.8)


def test_wake_then_pause_waits_from_end_of_phrase():
    # "hey glance" ... pause: the tail plus silence must not end the capture; the no-speech window starts at the gap
    l, _ = make_listener(wake=True, scores=[0.1] * 5 + [0.6, 0.98, 1.0, 0.9, 0.5, 0.2])
    f = Feeder(l)
    f.feed(blocks(("l", 6), ("l", 3)))
    gap_start = f.t
    f.feed(blocks(("q", 30)))  # 2.4 s pause
    assert [type(a) for a in f.actions] == [SpeechStart]
    f.feed(blocks(("l", 10), ("q", 12)))
    fin = f.actions[-1]
    assert fin.reason == "end" and fin.audio is not None

    l, _ = make_listener(wake=True, scores=[0.1] * 5 + [0.6, 0.98, 1.0])
    f = Feeder(l)
    f.feed(blocks(("l", 6), ("l", 3)))
    gap_start = f.t
    f.feed(blocks(("q", 60)))
    fin = f.actions[-1]
    assert fin.reason == "no_speech" and fin.ended_at == pytest.approx(gap_start + 4.0, abs=0.03)


def test_wake_wait_is_capped():
    l, _ = make_listener(wake=True, scores=[0.1] * 5 + [0.6] + [0.99] * 13)  # falls after the cap
    f = Feeder(l)
    f.feed(blocks(("l", 6)))
    detected = f.t
    f.feed(blocks(("l", 20), ("q", 12)))  # no gap and the score never falls
    fin = f.actions[-1]
    assert fin.reason == "end"
    assert clip_s(fin) == pytest.approx(20 * B - 1.04 + 0.2, abs=0.02)  # starts at the first block end past 1.0 s
    assert fin.ended_at == pytest.approx(detected + 20 * B)


def test_wake_inside_open_capture_drops_the_wake_phrase():
    # open state: "hey glance" starts a VAD capture, the detection drops it, and "stop" is what's left
    l, _ = make_listener(wake=False, scores=[0.0] * 10 + [0.9] + [0.95] * 10)
    f = Feeder(l)
    f.feed(blocks(("q", 5), ("l", 6)))  # VAD start, then the detection on the 6th loud block
    f.feed(blocks(("q", 2)))
    stop_t = f.t
    f.feed(blocks(("l", 5), ("q", 12)))
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]  # still one speech_start
    start, fin = f.actions
    assert start.source == "vad"
    assert fin.reason == "end" and fin.ended_at == pytest.approx(stop_t + 0.4)
    assert clip_s(fin) == pytest.approx(0.16 + 0.4 + 0.2, abs=0.02)


def test_wake_in_open_state_before_vad_start_starts_a_wake_capture():
    l, _ = make_listener(wake=False, scores=[0.0] * 5 + [0.9])
    f = Feeder(l)
    f.feed(blocks(("q", 5), ("q", 1)))
    assert [(type(a), a.source) for a in f.actions] == [(SpeechStart, "wake")]


def test_ptt_during_wake_wait_takes_over():
    l, _ = make_listener(wake=True, scores=[0.9] + [0.99] * 10)
    f = Feeder(l)
    f.feed(blocks(("l", 2)))
    f.ptt(True)
    f.feed(blocks(("l", 5), ("q", 20)))
    f.ptt(False)
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]
    assert f.actions[1].reason == "ptt" and f.actions[1].audio is not None


def test_wake_ignored_during_ptt_capture():
    l, _ = make_listener(wake=True, scores=[0.0, 0.9, 0.9])
    f = Feeder(l)
    f.ptt(True)
    f.feed(blocks(("l", 8)))
    f.ptt(False)
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]
    assert clip_s(f.actions[1]) == pytest.approx(8 * B, abs=0.02)  # nothing dropped


@pytest.mark.parametrize("text,expected", [
    ("hey glance, stop", "stop"),
    ("Hey, Glance. Stop.", "Stop."),
    ("Hey Glance, get me the can.", "get me the can."),
    ("Glance, what is this?", "what is this?"),
    ("hey glance", ""),
    ("Hey, Glance.", ""),
    ("Glance", ""),
    ("what is on the desk", "what is on the desk"),
    ("hey there", "hey there"),
    ("Glancey said hi", "Glancey said hi"),
    ("stop, hey glance", "stop, hey glance"),
])
def test_strip_wake_phrase(text, expected):
    assert strip_wake_phrase(text, wake_phrase_pattern("Glance")) == expected


def test_wake_below_threshold_does_nothing():
    l, _ = make_listener(wake=True, scores=[0.49] * 10)
    assert Feeder(l).feed(blocks(("l", 10))) == []


def test_wake_cooldown():
    l, _ = make_listener(wake=True, scores=[0.9, 0.9, 0.9])
    f = Feeder(l)
    f.feed([QUIET])
    assert isinstance(f.actions[0], SpeechStart)
    f.actions += l.device_lost(f.t)  # ends the capture right away
    l.reset()
    f.feed([QUIET, QUIET])  # within 1.5 s: ignored
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]


def test_wake_detector_is_fed_in_every_state():
    l, det = make_listener(wake=False)
    Feeder(l).feed(blocks(("q", 5)))
    assert det.blocks == 5


def test_no_speech_after_wake_delivers_empty():
    l, _ = make_listener(wake=True, scores=[0.9])
    f = Feeder(l)
    f.feed(blocks(("q", 60)))
    start, fin = f.actions
    assert fin.audio is None and fin.reason == "no_speech"
    assert fin.ended_at - start.at == pytest.approx(4.0, abs=0.1)


def test_too_short_burst_delivers_empty_after_waiting():
    l, _ = make_listener(cfg=SHORT_START)
    f = Feeder(l)
    f.feed(blocks(("q", 5), ("l", 3), ("q", 70)))  # 240 ms of speech < 250 ms
    start, fin = f.actions
    assert fin.audio is None and fin.reason == "short"


def test_short_burst_then_speech_is_one_utterance():
    l, _ = make_listener(cfg=SHORT_START)
    f = Feeder(l)
    f.feed(blocks(("q", 5), ("l", 3), ("q", 15), ("l", 8), ("q", 12)))
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]
    assert f.actions[1].reason == "end" and f.actions[1].audio is not None


def test_max_utterance_cuts_and_transcribes():
    l, _ = make_listener()
    f = Feeder(l)
    f.feed(blocks(("q", 5), ("l", 140)))
    start, fin = f.actions[:2]  # (the Mic resets after a delivery; here speech goes on and starts again)
    assert fin.reason == "max" and fin.audio is not None
    assert fin.ended_at - start.at == pytest.approx(10.0, abs=0.05)


@pytest.mark.parametrize("wake", [True, False])
def test_ptt_works_in_wake_and_open_states(wake):
    l, _ = make_listener(wake=wake)
    f = Feeder(l)
    f.feed(blocks(("q", 10)))
    f.ptt(True)
    f.feed(blocks(("l", 5), ("q", 20)))  # silence doesn't end a PTT capture
    assert [type(a) for a in f.actions] == [SpeechStart]
    f.ptt(False)
    start, fin = f.actions
    assert start.source == "ptt" and fin.reason == "ptt"
    assert len(fin.audio) == pytest.approx((0.3 + 25 * 0.08) * 16000, abs=320)  # pre-roll included


def test_ptt_without_wake_model():
    l, _ = make_listener(wake=True, detector=False)
    f = Feeder(l)
    f.ptt(True)
    f.feed(blocks(("l", 10)))
    f.ptt(False)
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]


@pytest.mark.parametrize("wake", [True, False])
def test_ptt_only_ignores_wake_word_and_speech(wake):
    l, det = make_listener(wake=wake, scores=[0.9] * 40, cfg=Config(ptt_only=True))
    acts = Feeder(l).feed(blocks(("q", 5), ("l", 20), ("q", 15)))
    assert acts == []
    assert l.state() == "ptt"
    assert det.blocks == 0  # the wake model is not run at all


def test_ptt_only_press_and_release_is_one_utterance():
    l, _ = make_listener(wake=False, scores=[0.9] * 40, cfg=Config(ptt_only=True))
    f = Feeder(l)
    f.feed(blocks(("q", 10)))
    f.ptt(True)
    assert l.state() == "capture"
    f.feed(blocks(("l", 10), ("q", 15)))  # silence doesn't end it, and no wake detection cuts it
    f.ptt(False)
    start, fin = f.actions
    assert start.source == "ptt" and fin.reason == "ptt" and fin.audio is not None
    assert l.state() == "ptt"


def test_ptt_during_vad_capture_takes_over_without_second_start():
    l, _ = make_listener()
    f = Feeder(l)
    f.feed(blocks(("l", 5)))
    f.ptt(True)
    f.feed(blocks(("q", 20)))  # would have ended a VAD capture
    f.ptt(False)
    assert [type(a) for a in f.actions] == [SpeechStart, Finish]
    assert f.actions[0].source == "vad" and f.actions[1].reason == "ptt"


def test_ptt_too_short():
    l, _ = make_listener()
    f = Feeder(l)
    f.ptt(True)
    f.feed([LOUD, LOUD])
    f.ptt(False)
    assert f.actions[1].audio is None and f.actions[1].reason == "too_short"


def test_ptt_max():
    l, _ = make_listener()
    f = Feeder(l)
    f.ptt(True)
    f.feed(blocks(("l", 130)))
    assert f.actions[1].reason == "max"
    n = len(f.actions)
    f.ptt(False)  # release after the hard stop doesn't end anything
    assert len(f.actions) == n and not any(isinstance(a, Finish) for a in f.actions[2:])


def test_release_without_press_is_ignored():
    l, _ = make_listener()
    assert l.ptt(False, 1.0) == []


# ------------------------------------------------------------------ policy

class FakePlayer:
    def __init__(self):
        self.busy = False
        self.window = ReplyWindow()
        self.cues = []

    def cue(self, name):
        self.cues.append(name)


@pytest.mark.parametrize("busy,window_open,task,expected", [
    (False, False, False, True),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (True, True, True, False),
])
def test_wake_needed_only_when_all_hold(busy, window_open, task, expected):
    player = FakePlayer()
    player.busy = busy
    if window_open:
        player.window.on_say("answer")
    mic = Mic(Config(), RecordingSink(), FakeTranscriber(), FakeCapture([]), FakeVad(), player=player,
              task_active=lambda: task, log=null_log())
    assert mic.wake_needed() is expected
    assert mic.listener.state() == ("wake" if expected else "open")


def test_task_active_exception_counts_as_inactive():
    def boom():
        raise RuntimeError("controller")

    mic = Mic(Config(), RecordingSink(), FakeTranscriber(), FakeCapture([]), FakeVad(), task_active=boom,
              log=null_log())
    assert mic.wake_needed() is True


# ------------------------------------------------------------------ Mic contract

class FakeTranscriber:
    def __init__(self, results=None, order=None, error=None):
        self.results = list(results or [])
        self.order = order if order is not None else []
        self.error = error
        self.calls = 0

    def transcribe(self, wav):
        self.calls += 1
        self.order.append("stt")
        if self.error:
            raise self.error
        return self.results.pop(0) if self.results else SttResult("hello", "vultr", 10)


class RecordingDispatch:
    def __init__(self):
        self.calls = []

    def __call__(self, fn, *args):
        self.calls.append(fn.__name__)
        fn(*args)


def make_mic(transcriber=None, player=None, scores=None, wake=False, frame_source=None, order=None, cfg=None):
    """wake: True or False forces the policy; "real" uses section 6 with the fake player."""
    sink = RecordingSink()
    dispatch = RecordingDispatch()
    order = order if order is not None else []

    def frames():
        order.append("frame")
        return {"id": "f1"}

    mic = Mic(cfg or Config(), sink, transcriber or FakeTranscriber(order=order), FakeCapture([]), FakeVad(),
              detector=FakeDetector(scores), player=player or FakePlayer(), dispatch=dispatch,
              frame_source=frame_source or frames, log=null_log())
    if wake != "real":
        mic.wake_needed = lambda: wake
    return mic, sink, dispatch, order


def push(mic, bs, t0=1000.0, ptt=None):
    t = t0
    for i, b in enumerate(bs):
        if ptt and i in ptt:
            mic.handle(("ptt", ptt[i], t))
        mic.handle(("audio", b, t))
        t += B
    return t


def assert_paired(sink):
    names = sink.names()
    assert names.count("speech_start") == names.count("utterance")
    for a, b in zip(names[::2], names[1::2]):
        assert (a, b) == ("speech_start", "utterance")


def test_vad_utterance_end_to_end():
    mic, sink, dispatch, order = make_mic()
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    assert sink.names() == ["speech_start", "utterance"]
    text, ended_at, frame = sink.calls[1].args
    assert text == "hello" and frame == {"id": "f1"}
    assert order == ["frame", "stt"]  # frame grabbed at end of speech, before STT
    assert dispatch.calls == ["on_speech_start", "on_utterance"]


def test_log_carries_speech_start_time_and_backend():
    mic, sink, _, _ = make_mic()
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    start, = mic.log.of("speech_start")
    assert start["at"] == pytest.approx(sink.calls[0].args[0], abs=1e-3)
    delivered, = mic.log.of("utterance_delivered")
    assert delivered["backend"] == "vultr" and delivered["chars"] == 5


def test_every_start_gets_one_utterance_in_all_empty_cases():
    order = []
    tr = FakeTranscriber(results=[SttResult("", "none", 5, "stt_failed"), SttResult("", "vultr", 5, "blocklist")],
                         order=order)
    mic, sink, dispatch, _ = make_mic(transcriber=tr, scores=[0.9], wake=True, order=order, cfg=SHORT_START)
    t = push(mic, blocks(("q", 60)))  # wake, then no speech
    mic.wake_needed = lambda: False
    t = push(mic, blocks(("q", 5), ("l", 3), ("q", 70)), t0=t)  # too short
    t = push(mic, blocks(("q", 5), ("l", 10), ("q", 12)), t0=t)  # STT failed
    t = push(mic, blocks(("q", 5), ("l", 10), ("q", 12)), t0=t)  # filtered
    assert sink.utterances() == ["", "", "", ""]
    assert_paired(sink)
    assert tr.calls == 2  # no STT for the no-speech and too-short cases
    no_speech, short = sink.calls[1].args, sink.calls[3].args
    assert no_speech[2] is None and short[2] is None  # no frame without STT
    assert set(dispatch.calls) == {"on_speech_start", "on_utterance"} and len(dispatch.calls) == 8


def test_stt_exception_delivers_empty():
    mic, sink, _, _ = make_mic(transcriber=FakeTranscriber(error=RuntimeError("bug")))
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    assert sink.utterances() == [""]


def test_frame_source_exception_still_transcribes():
    def bad():
        raise RuntimeError("camera")

    mic, sink, _, _ = make_mic(frame_source=bad)
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    assert sink.utterances() == ["hello"]
    assert sink.calls[1].args[2] is None


def test_device_lost_during_capture_delivers_empty():
    mic, sink, _, _ = make_mic()
    t = push(mic, blocks(("q", 5), ("l", 5)))
    mic.handle(("device_lost", None, t))
    assert sink.names() == ["speech_start", "utterance"] and sink.utterances() == [""]


def test_worker_exception_recovers_and_keeps_pairing():
    mic, sink, _, _ = make_mic()
    push(mic, blocks(("q", 5), ("l", 5)))
    assert sink.names() == ["speech_start"]

    def broken(block, t):
        raise RuntimeError("detector bug")

    mic.listener.feed = broken
    try:
        mic.handle(("audio", LOUD, 2000.0))
    except RuntimeError as e:
        mic.log.error("test", e)
        mic._recover()
    assert sink.names() == ["speech_start", "utterance"]
    assert mic.listener.capture is None


def test_reply_window_closes_on_delivered_text_but_not_on_name():
    player = FakePlayer()
    tr = FakeTranscriber(results=[SttResult("Glance", "vultr", 5), SttResult("yes", "vultr", 5)])
    mic, sink, _, _ = make_mic(transcriber=tr, player=player, wake="real")
    player.window.on_say("answer")
    t = push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    assert player.window.is_open
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)), t0=t)
    assert not player.window.is_open
    assert sink.utterances() == ["", "yes"]  # the wake phrase alone is stripped to nothing


def test_hey_glance_stop_in_open_state_arrives_as_stop():
    tr = FakeTranscriber(results=[SttResult("hey glance, stop", "vultr", 5)])
    mic, sink, _, _ = make_mic(transcriber=tr, scores=[0.0] * 10 + [0.9] + [0.95] * 10, wake=False)
    push(mic, blocks(("q", 5), ("l", 6), ("q", 2), ("l", 5), ("q", 12)))
    assert sink.utterances() == ["stop"]
    assert_paired(sink)


def test_wake_tone_on_every_detection():
    player = FakePlayer()
    mic, sink, _, _ = make_mic(player=player, scores=[0.9], wake=True)
    push(mic, blocks(("q", 3)))
    assert player.cues == ["wake"]
    # also when "hey glance" interrupts an open-state capture
    player = FakePlayer()
    mic, sink, _, _ = make_mic(player=player, scores=[0.0] * 10 + [0.9], wake=False)
    push(mic, blocks(("q", 5), ("l", 6), ("q", 2)))
    assert player.cues == ["wake"] and sink.names() == ["speech_start"]


def test_wake_tone_can_be_turned_off_and_never_breaks_the_mic():
    player = FakePlayer()
    mic, sink, _, _ = make_mic(player=player, scores=[0.9], wake=True, cfg=Config(wake_tone=False))
    push(mic, blocks(("q", 3)))
    assert player.cues == [] and sink.names() == ["speech_start"]

    class BrokenPlayer(FakePlayer):
        def cue(self, name):
            raise RuntimeError("output gone")

    mic, sink, _, _ = make_mic(player=BrokenPlayer(), scores=[0.9], wake=True)
    push(mic, blocks(("q", 3)))
    assert sink.names() == ["speech_start"] and mic.log.of("error")[-1]["where"] == "mic.wake_tone"


def test_empty_utterance_keeps_reply_window_open():
    player = FakePlayer()
    mic, sink, _, _ = make_mic(transcriber=FakeTranscriber(results=[SttResult("", "vultr", 5, "empty")]),
                               player=player, wake="real")
    player.window.on_say("done")
    push(mic, blocks(("q", 5), ("l", 10), ("q", 12)))
    assert player.window.is_open


def test_ptt_through_the_mic():
    mic, sink, _, _ = make_mic(wake=True)
    push(mic, blocks(("q", 3), ("l", 10), ("q", 3)), ptt={3: True, 15: False})
    assert sink.names() == ["speech_start", "utterance"]


def test_ptt_only_through_the_mic_pairs_and_ignores_voice():
    mic, sink, dispatch, _ = make_mic(wake=False, scores=[0.9] * 60, cfg=Config(ptt_only=True))
    push(mic, blocks(("l", 20), ("q", 15)))  # talking (or a speaker) without the key: nothing
    assert sink.names() == []
    push(mic, blocks(("q", 3), ("l", 10), ("q", 3)), t0=1010.0, ptt={3: True, 15: False})
    assert sink.names() == ["speech_start", "utterance"]
    assert set(dispatch.calls) == {"on_speech_start", "on_utterance"}


def test_threaded_mic_with_fake_capture_and_flush():
    items = [("audio", b, 1000.0 + i * B) for i, b in enumerate(blocks(("q", 5), ("l", 10), ("q", 12)))]
    # audio arriving during STT is dropped; a later utterance still works
    slow = FakeTranscriber()
    order = []
    orig = slow.transcribe

    def slow_transcribe(wav):
        time.sleep(0.05)
        return orig(wav)

    slow.transcribe = slow_transcribe
    sink = RecordingSink()
    threads = []

    def dispatch(fn, *args):
        threads.append(threading.current_thread().name)
        fn(*args)

    mic = Mic(Config(), sink, slow, FakeCapture(items), FakeVad(), detector=FakeDetector(), player=FakePlayer(),
              dispatch=dispatch, log=null_log())
    mic.wake_needed = lambda: False
    mic.start()
    try:
        assert sink.wait_for(2, timeout=3)
        assert sink.names() == ["speech_start", "utterance"]
        assert set(threads) == {"mic-worker"}
    finally:
        mic.close()


def test_queue_overflow_drops_oldest():
    mic, _, _, _ = make_mic()
    for i in range(300):
        mic.put(("audio", QUIET, float(i)))
    assert mic.q.qsize() == 250
    assert mic.q.get_nowait()[2] == 50.0


def test_flush_drops_audio_but_keeps_ptt_and_device_events():
    mic, _, _, _ = make_mic()
    mic.put(("audio", QUIET, 1.0))
    mic.put(("ptt", True, 1.1))
    mic.put(("audio", QUIET, 1.2))
    mic.put(("device_lost", None, 1.3))
    mic._flush_audio()
    assert [mic.q.get_nowait()[0] for _ in range(mic.q.qsize())] == ["ptt", "device_lost"]
