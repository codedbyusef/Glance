import threading
import time

import numpy as np
import pytest

from audio import player as player_mod
from audio.listen import ReplyWindow
from audio.log import null_log
from audio.player import FakeOutput, FakeSynth, Player, clean_text


class Reports:
    def __init__(self):
        self.items: list[tuple[str, str]] = []
        self.threads: list[str] = []
        self.cond = threading.Condition()

    def __call__(self, speech_id, status):
        with self.cond:
            self.items.append((speech_id, status))
            self.threads.append(threading.current_thread().name)
            self.cond.notify_all()

    def wait(self, n, timeout=3.0):
        with self.cond:
            assert self.cond.wait_for(lambda: len(self.items) >= n, timeout), self.items
        return self.items


class QueuedDispatch:
    """Stands in for loop.call_soon_threadsafe: calls run only when run() is called."""

    def __init__(self):
        self.pending = []
        self.lock = threading.Lock()

    def __call__(self, fn, *args):
        with self.lock:
            self.pending.append((fn, args))

    def run(self):
        with self.lock:
            calls, self.pending = self.pending, []
        for fn, args in calls:
            fn(*args)


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def make():
    players = []

    def _make(synth=None, output=None, dispatch=None, fixed=(), window=None):
        reports = Reports()
        kwargs = {"dispatch": dispatch} if dispatch else {}
        p = Player(synth or FakeSynth(), output or FakeOutput(), log=null_log(), window=window,
                   fixed_phrases=fixed, **kwargs)
        p.set_listener(reports)
        p.start()
        assert p.ready.wait(2)
        players.append(p)
        return p, reports

    yield _make
    for p in players:
        p.close()


def test_queue_order_unique_ids_one_report_each(make):
    p, reports = make()
    ids = [p.say(t, "none") for t in ("one", "two", "three")]
    assert len(set(ids)) == 3
    assert reports.wait(3) == [(i, "completed") for i in ids]
    time.sleep(0.1)
    assert len(reports.items) == 3


def test_nothing_reported_inside_say(make):
    p, reports = make()
    main = threading.current_thread().name
    for t in ("hello", "", "  * _ ", "again"):
        p.say(t, "none")
        assert reports.items == [] or main not in reports.threads
    reports.wait(4)
    assert main not in reports.threads


def test_empty_text_completes_in_queue_order(make):
    p, reports = make()
    a = p.say("first sentence", "none")
    b = p.say("**", "none")
    c = p.say("last", "none")
    assert reports.wait(3) == [(a, "completed"), (b, "completed"), (c, "completed")]


def test_stop_speech_reports_synchronously_in_order(make):
    out = FakeOutput()
    p, reports = make(output=out)
    out.gate.clear()  # writer blocks inside its first write
    ids = [p.say(t, "none") for t in ("aaaaaaaaaa", "bbbb", "cccc")]
    assert wait_until(lambda: p._writing)
    assert p.busy
    p.stop_speech()
    assert reports.items == [(i, "interrupted") for i in ids]
    assert reports.threads == [threading.current_thread().name] * 3
    assert not p.busy
    out.gate.set()
    time.sleep(0.2)
    assert len(reports.items) == 3
    assert out.aborts >= 1


def test_stop_cuts_audio_within_one_chunk(make):
    out = FakeOutput(rate=16000)
    p, reports = make(output=out)
    p.say("x" * 300, "none")  # 3 s of audio
    assert out.wrote.wait(2)
    time.sleep(0.12)
    p.stop_speech()
    at_stop = out.samples_written
    time.sleep(0.2)
    assert out.samples_written - at_stop <= int(16000 * 0.05)


def test_wake_cue_is_shorter_and_unknown_cues_are_ignored(make):
    out = FakeOutput(rate=16000)
    p, reports = make(output=out)
    p.cue("wake")
    assert wait_until(lambda: out.samples_written >= int(16000 * 0.09))
    time.sleep(0.05)
    assert out.samples_written == int(16000 * 0.09)
    p.cue("ding")
    time.sleep(0.05)
    assert out.samples_written == int(16000 * 0.09)
    assert p.log.of("error")[-1]["where"] == "player.cue"


def test_stop_logs_when_the_sound_was_cut(make):
    out = FakeOutput(rate=16000)
    p, reports = make(output=out)
    p.say("x" * 300, "none")
    assert out.wrote.wait(2)
    p.stop_speech()
    assert wait_until(lambda: p.log.of("playback_cut"))
    assert 0 <= p.log.of("playback_cut")[0]["ms"] <= 100


def test_completed_just_before_stop_is_reported_once_and_first():
    out = FakeOutput(realtime=False)
    synth = FakeSynth()
    dispatch = QueuedDispatch()
    reports = Reports()
    p = Player(synth, out, dispatch=dispatch, log=null_log(), fixed_phrases=())
    p.set_listener(reports)
    p.start()
    try:
        a = p.say("short", "none")
        assert wait_until(lambda: dispatch.pending)  # a completed; its report sits in the loop queue
        out.gate.clear()
        b = p.say("second one", "none")
        assert wait_until(lambda: p._writing)
        p.stop_speech()
        assert reports.items == [(a, "completed"), (b, "interrupted")]
        dispatch.run()  # the scheduled drain finds nothing left
        out.gate.set()
        time.sleep(0.1)
        dispatch.run()
        assert reports.items == [(a, "completed"), (b, "interrupted")]
    finally:
        out.gate.set()
        p.close()


def test_stop_during_latency_wait_is_interrupted(make):
    out = FakeOutput(latency_s=0.5, realtime=False)
    p, reports = make(output=out)
    a = p.say("hi", "none")
    assert out.wrote.wait(2)
    time.sleep(0.05)
    p.stop_speech()
    time.sleep(0.6)
    assert reports.items == [(a, "interrupted")]


def test_completed_waits_for_output_latency(make):
    out = FakeOutput(latency_s=0.3, realtime=False)
    p, reports = make(output=out)
    t0 = time.monotonic()
    p.say("hi", "none")
    reports.wait(1)
    assert time.monotonic() - t0 >= 0.3


def test_cue_plays_when_idle_and_is_dropped_while_speaking(make):
    out = FakeOutput(rate=16000)
    p, reports = make(output=out)
    p.cue("thinking")
    assert wait_until(lambda: out.samples_written >= int(16000 * 0.15))
    assert out.samples_written == int(16000 * 0.15)
    p.say("x" * 50, "none")
    assert out.wrote.wait(1)
    before = len(p.log.of("cue"))
    p.cue("thinking")
    assert p.log.of("cue")[before]["played"] is False
    reports.wait(1)
    assert out.samples_written == int(16000 * 0.15) + int(16000 * 0.5)
    p.cue("beep")
    assert p.log.of("error")[-1]["where"] == "player.cue"
    assert reports.items and all(len(r) == 2 for r in reports.items)  # cues never report


def test_cache_hits(make):
    synth = FakeSynth()
    p, reports = make(synth=synth, fixed=("Okay, cancelled.",))
    assert p.warm.wait(2)
    assert synth.calls == ["Okay, cancelled."]
    p.say("repeat me", "none")
    p.say("repeat me", "none")
    p.say("Okay, cancelled.", "none")
    reports.wait(3)
    assert synth.calls == ["Okay, cancelled.", "repeat me"]


def test_lru_evicts_oldest(make, monkeypatch):
    monkeypatch.setattr(player_mod, "CACHE_SIZE", 2)
    synth = FakeSynth(per_char_s=0.001)
    p, reports = make(synth=synth)
    for t in ("a", "b", "c", "a"):
        p.say(t, "none")
    reports.wait(4)
    assert synth.calls == ["a", "b", "c", "a"]


def test_text_is_cleaned_before_synthesis(make):
    synth = FakeSynth()
    p, reports = make(synth=synth)
    p.say("**Hello**   _there_ `x` #1", "none")
    reports.wait(1)
    assert synth.calls == ["Hello there x 1"]
    assert clean_text(" a\n\tb ") == "a b"


def test_synth_failure_reports_interrupted_and_continues(make):
    synth = FakeSynth()
    synth.fail_on.add("bad")
    p, reports = make(synth=synth)
    a = p.say("bad", "none")
    b = p.say("good", "none")
    assert reports.wait(2) == [(a, "interrupted"), (b, "completed")]


def test_output_failure_interrupts_queue_then_recovers(make, monkeypatch):
    monkeypatch.setattr(player_mod, "REOPEN_S", 0.1)
    out = FakeOutput()
    out.fail_writes = 1
    p, reports = make(output=out)
    a = p.say("first", "none")
    b = p.say("second", "none")
    assert reports.wait(2) == [(a, "interrupted"), (b, "interrupted")]
    assert wait_until(lambda: p._device_ok, 2)
    c = p.say("third", "none")
    assert reports.wait(3)[-1] == (c, "completed")
    assert out.opens == 2


def test_output_missing_at_startup(make, monkeypatch):
    monkeypatch.setattr(player_mod, "REOPEN_S", 0.3)
    out = FakeOutput()
    out.fail_opens = 2
    p, reports = make(output=out)
    main = threading.current_thread().name
    a = p.say("nobody hears this", "none")
    assert reports.wait(1) == [(a, "interrupted")]
    assert main not in reports.threads
    assert wait_until(lambda: p._device_ok, 3)
    b = p.say("now it works", "none")
    assert reports.wait(2)[-1] == (b, "completed")


def test_resamples_to_output_rate(make):
    out = FakeOutput(rate=48000, realtime=False)
    p, reports = make(synth=FakeSynth(rate=22050), output=out)
    p.say("x" * 100, "none")  # 1 s at 22050
    reports.wait(1)
    assert abs(out.samples_written - 48000) <= 2


def test_busy(make):
    out = FakeOutput()
    p, reports = make(output=out)
    assert not p.busy
    p.say("x" * 20, "none")
    assert p.busy
    reports.wait(1)
    assert wait_until(lambda: not p.busy)


def test_invalid_expect_is_none(make):
    window = ReplyWindow()
    p, reports = make(window=window)
    p.say("q", "answer")
    assert window.is_open
    p.say("q", "maybe")
    assert not window.is_open
    assert "invalid expect" in p.log.of("error")[-1]["error"]


def test_listener_exception_does_not_break_reports(make):
    p, reports = make()
    calls = []

    def bad(speech_id, status):
        calls.append(speech_id)
        raise RuntimeError("controller bug")

    p.set_listener(bad)
    p.say("a", "none")
    p.say("b", "none")
    assert wait_until(lambda: len(calls) == 2)


def test_reply_window_rules():
    w = ReplyWindow("Glance")
    assert not w.is_open
    w.on_say("done")
    assert w.is_open
    w.on_delivered("")
    assert w.is_open
    w.on_delivered("Hey, Glance.")
    w.on_delivered("glance")
    assert w.is_open
    w.on_say("none")
    assert not w.is_open
    w.on_say("answer")
    w.on_delivered("yes")
    assert not w.is_open


def test_tone_shape():
    t = player_mod.tone(16000)
    assert len(t) == 2400 and t.dtype == np.float32 and np.abs(t).max() <= 0.12 and abs(t[0]) < 1e-3
