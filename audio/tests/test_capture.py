import sys
import threading
import time
import types

import numpy as np

from audio import capture
from audio.log import null_log


class FakeStream:
    """Calls the callback from a thread until told to go silent (a stalled or unplugged mic)."""

    instances = []

    def __init__(self, device=None, samplerate=None, channels=None, dtype=None, blocksize=None, callback=None):
        if FakeSd.fail_opens:
            FakeSd.fail_opens -= 1
            raise OSError("no such device")
        self.callback, self.blocksize, self.samplerate = callback, blocksize, samplerate
        self.active = False
        self.silent = threading.Event()
        FakeStream.instances.append(self)

    def start(self):
        self.active = True

        def run():
            while self.active and not self.silent.is_set():
                self.callback(np.zeros((self.blocksize, 1), np.int16), self.blocksize, None, None)
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def abort(self, ignore_errors=True):
        self.active = False

    def close(self, ignore_errors=True):
        self.active = False


class FakeSd(types.ModuleType):
    fail_opens = 0
    native_rate = None  # set to e.g. 48000 to refuse 16 kHz

    def __init__(self):
        super().__init__("sounddevice")
        self.InputStream = FakeStream

    def check_input_settings(self, device=None, samplerate=None, channels=None, dtype=None):
        if FakeSd.native_rate:
            raise ValueError("invalid sample rate")

    def query_devices(self, device=None, kind=None):
        return {"default_samplerate": FakeSd.native_rate or 16000}


def run_capture(monkeypatch, until, timeout=5.0):
    FakeStream.instances = []
    monkeypatch.setitem(sys.modules, "sounddevice", FakeSd())
    monkeypatch.setattr(capture, "STALL_S", 0.2)
    monkeypatch.setattr(capture, "REOPEN_S", 0.1)
    items, log = [], null_log()
    cap = capture.SoundDeviceCapture(log=log)
    cap.start(items.append)
    end = time.monotonic() + timeout
    try:
        while not until(items, log) and time.monotonic() < end:
            time.sleep(0.02)
    finally:
        cap.close()
    return items, log


def test_stalled_mic_reports_device_lost_and_reopens(monkeypatch):
    FakeSd.fail_opens, FakeSd.native_rate = 0, None

    def until(items, log):
        if FakeStream.instances and len(FakeStream.instances) == 1 and len(items) > 3:
            FakeStream.instances[0].silent.set()  # the mic goes quiet: no more callbacks
        return len(log.of("mic_open")) >= 2 and items[-1][0] == "audio"

    items, log = run_capture(monkeypatch, until)
    kinds = [k for k, _, _ in items]
    assert "device_lost" in kinds
    assert kinds[-1] == "audio"  # audio flows again after the reopen
    assert len(log.of("mic_open")) >= 2


def test_open_failure_is_retried(monkeypatch):
    FakeSd.fail_opens, FakeSd.native_rate = 2, None
    items, log = run_capture(monkeypatch, lambda items, log: items)
    assert items and items[0][0] == "audio"
    assert len([e for e in log.of("error") if e["where"] == "capture.open"]) == 2


def test_native_rate_is_resampled_to_16k_blocks(monkeypatch):
    FakeSd.fail_opens, FakeSd.native_rate = 0, 48000
    items, log = run_capture(monkeypatch, lambda items, log: len(items) >= 2)
    FakeSd.native_rate = None
    assert log.of("mic_open")[0]["rate"] == 48000 and log.of("mic_open")[0]["resampling"] is True
    assert all(len(block) == capture.BLOCK_SAMPLES and block.dtype == np.int16 for _, block, _ in items)
