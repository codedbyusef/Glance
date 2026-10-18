"""Step 4 and 5 pieces that run without hardware: per-machine config, the default view, the glove-check
sequence, and the selftest's offline paths."""
from __future__ import annotations

import pytest

from haptics import protocol
from runtime import config as runtime_config
from runtime import selftest
from runtime.glove_check import PULSE_RATES, check_steps
from runtime.glove_out import OFF


def test_local_config_overrides_the_shared_one(tmp_path):
    shared = tmp_path / "config.yaml"
    shared.write_text("camera:\n  source: 1\n")
    local = tmp_path / "config.local.yaml"
    local.write_text("camera:\n  source: picam\nglove:\n  kind: hardware\n  port: /dev/ttyUSB0\n")
    cfg = runtime_config.load(shared, local)
    assert cfg["camera"]["source"] == "picam" and cfg["camera"]["width"] == 1280
    assert cfg["glove"] == {"kind": "hardware", "port": "/dev/ttyUSB0", "on_level": 255, "firmware": "s3"}
    local.write_text("glove:\n  prot: COM5\n")
    with pytest.raises(ValueError, match="glove.prot"):
        runtime_config.load(shared, local)


def test_default_view(monkeypatch):
    monkeypatch.setattr(runtime_config.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert runtime_config.default_view() == "web"  # the headless Pi
    monkeypatch.setenv("DISPLAY", ":0")
    assert runtime_config.default_view() == "window"
    monkeypatch.setattr(runtime_config.sys, "platform", "win32")
    assert runtime_config.default_view() == "window"


def test_glove_check_covers_every_channel_then_every_pulse_rate():
    steps = [s for s in check_steps(hold_s=1.0) if s.label != "off"]
    steady = steps[:8]
    assert [next(iter(s.cmd.on)) for s in steady] == list(range(8))
    assert all(s.cmd.pulse_hz == {} for s in steady)
    pulses = steps[8:]  # proximity rates on a direction motor (index), not the wrist (decision 24)
    assert [s.cmd.pulse_hz[protocol.INDEX] for s in pulses] == list(PULSE_RATES)
    assert all(s.cmd.on == frozenset() for s in pulses)
    assert all(s.cmd == OFF for s in check_steps() if s.label == "off")


def test_selftest_offline_paths(monkeypatch):
    selftest.RESULTS.clear()
    monkeypatch.delenv("GLANCE_STT_URL", raising=False)
    selftest.check_stt()
    selftest.check_glove("fake", None, 255)
    selftest.check_glove("hardware", None, 255)  # no port: a clear FAIL, not a crash
    statuses = [(c, s) for c, s, _ in selftest.RESULTS]
    assert statuses == [("stt", "FAIL"), ("glove", "PASS"), ("glove", "FAIL")]
    assert "glove.port" in selftest.RESULTS[-1][2]
