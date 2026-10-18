"""The 6-motor ESP32-S3 glove (firmware/glove/code.py) over the runtime path, with no hardware: wire mapping,
port auto-detection, the glove commands, the glove check, and runtime.run's glove choice and env file."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from haptics import protocol
from haptics.link import GloveLink
from runtime import config as runtime_config
from runtime import glove as glove_cmd
from runtime import run as run_mod
from runtime.glove_check import check_steps
from runtime.glove_out import (FIRMWARE_CHANNELS, OFF, FakeGlove, GloveCommand, HardwareGlove, find_glove_port,
                               make_glove)


class DryLink(GloveLink):
    """A real GloveLink in dry-run mode (no port), recording the wire frames instead of printing them."""

    def __init__(self, channels):
        super().__init__(dry_run=True, channels=channels)
        self.sent = []

    def start(self):  # no background thread in tests
        pass

    def _write(self, data):
        self.sent.append(bytes(data))


def wire(glove):
    """The 6 levels the S3 would receive right now."""
    return list(glove.link.wire_frame_at(0.0)[1:-1])


def test_s3_glove_sends_six_levels_in_code_py_order():
    g = HardwareGlove(firmware="s3", link=DryLink(protocol.S3_CHANNELS), on_level=255)
    assert g.channels == protocol.S3_CHANNELS
    # code.py's channel order: 0 thumb, 1 index, 2 ring, 3 pinky, 4 back, 5 palm
    for logical, wire_index in ((protocol.THUMB, 0), (protocol.INDEX, 1), (protocol.RING, 2),
                                (protocol.PINKY, 3), (protocol.BACK, 4), (protocol.PALM, 5)):
        g.apply(GloveCommand(on=frozenset({logical})))
        expected = [0] * 6
        expected[wire_index] = 255
        assert wire(g) == expected, protocol.CHANNELS[logical]


def test_s3_glove_ignores_middle_and_wrist():
    g = HardwareGlove(firmware="s3", link=DryLink(protocol.S3_CHANNELS))
    g.apply(GloveCommand(on=frozenset({protocol.MIDDLE, protocol.LEFT}), pulse_hz={protocol.WRIST: 4.0}))
    assert wire(g) == [255, 0, 0, 0, 0, 0]  # only the thumb (LEFT); no middle, no wrist motor
    frame = g.link.wire_frame_at(0.0)
    assert len(frame) == 8 and frame[-1] == 255  # 0xAA + 6 levels + XOR checksum
    g.all_off()
    assert wire(g) == [0] * 6 and g.link.sent[-1] == bytes([0xA5, 0x00])


def test_unknown_firmware_is_rejected():
    with pytest.raises(ValueError, match="firmware"):
        HardwareGlove(firmware="s2", link=DryLink(None))


def test_find_glove_port():
    port = lambda dev, vid=None, desc="", man="": SimpleNamespace(device=dev, vid=vid, description=desc,  # noqa
                                                                    manufacturer=man)
    assert find_glove_port([port("COM3", 0x8086, "Intel"), port("COM7", 0x239A, "Feather ESP32-S3")]) == "COM7"
    assert find_glove_port([port("/dev/ttyACM0", 0x303A)]) == "/dev/ttyACM0"
    assert find_glove_port([port("/dev/ttyACM1", None, "CircuitPython CDC data")]) == "/dev/ttyACM1"
    assert find_glove_port([port("COM3", 0x8086, "Intel")]) is None
    assert find_glove_port([]) is None


def test_auto_port_without_a_glove_says_how_to_fix_it(monkeypatch):
    monkeypatch.setattr("runtime.glove_out.find_glove_port", lambda ports=None: None)
    with pytest.raises(ValueError, match="glove ports"):
        HardwareGlove(port="auto")


def test_fake_panel_crosses_out_motors_the_glove_lacks():
    full, s3 = np.zeros((220, 200, 3), np.uint8), np.zeros((220, 200, 3), np.uint8)
    cmd = GloveCommand(on=frozenset({protocol.MIDDLE}), pulse_hz={protocol.WRIST: 2.0})
    a, b = FakeGlove(clock=lambda: 0.1), FakeGlove(clock=lambda: 0.1, channels=protocol.S3_CHANNELS)
    a.apply(cmd)
    b.apply(cmd)
    a.draw(full, (0, 0))
    b.draw(s3, (0, 0))
    assert (full != s3).any()  # middle and wrist lit on the 8-motor panel, crossed out on the S3 panel


def test_glove_check_steps_follow_the_firmware():
    steps = [s for s in check_steps(channels=FIRMWARE_CHANNELS["s3"]) if s.label != "off"]
    assert [next(iter(s.cmd.on)) for s in steps] == list(protocol.S3_CHANNELS)  # 6 motors, no pulse steps
    assert "wire channel 2" in steps[2].label and "ring" in steps[2].label
    assert len([s for s in check_steps() if s.label != "off"]) == 8 + 4  # glove.ino: 8 motors + 4 pulse rates


@pytest.mark.parametrize("name,channel", [("LEFT", protocol.THUMB), ("forward", protocol.INDEX),
                                          ("BACK", protocol.RING), ("right", protocol.PINKY),
                                          ("UP", protocol.BACK), ("down", protocol.PALM),
                                          ("palm", protocol.PALM), ("all", None)])
def test_channel_for(name, channel):
    assert glove_cmd.channel_for(name) == channel


def test_channel_for_unknown():
    with pytest.raises(SystemExit):
        glove_cmd.channel_for("elbow")


def test_sweep_closes_in_and_ends_all_off(monkeypatch):
    monkeypatch.setattr(glove_cmd.time, "sleep", lambda s: None)
    panel = FakeGlove()
    words = glove_cmd.run_sweep(panel, "table", seconds=4.0, out=lambda s: None)
    assert set(words[2]) == {"RIGHT", "FORWARD"}       # starts bottom-left of the target
    assert words[-1] == ["DOWN"]                        # ends resting on it (table mode)
    assert panel.command == OFF


@pytest.mark.parametrize("firmware,link_channels", [("s3", protocol.S3_CHANNELS), ("ino8", None)])
def test_make_glove_opens_the_link_for_the_firmware(monkeypatch, firmware, link_channels):
    opened = []

    def fake_link(port, channels=None):
        opened.append((port, channels))
        return DryLink(channels)

    monkeypatch.setattr("haptics.link.GloveLink", fake_link)
    tee = make_glove("hardware", "/dev/ttyACM0", 255, firmware)
    hw, panel = tee.outputs
    assert opened == [("/dev/ttyACM0", link_channels)]
    assert hw.port == "/dev/ttyACM0" and panel.channels == set(FIRMWARE_CHANNELS[firmware])


# ------------------------------------------------------------------ runtime.run

def test_pick_glove(monkeypatch):
    rt = runtime_config.load(local=os.devnull)
    monkeypatch.setattr("runtime.glove_out.find_glove_port", lambda ports=None: "/dev/ttyACM0")
    assert run_mod.pick_glove(None, None, rt) == ("hardware", "/dev/ttyACM0")   # plugged in -> hardware
    assert run_mod.pick_glove("fake", None, rt) == ("fake", None)
    assert run_mod.pick_glove(None, "COM9", rt) == ("hardware", "COM9")
    monkeypatch.setattr("runtime.glove_out.find_glove_port", lambda ports=None: None)
    assert run_mod.pick_glove(None, None, rt) == ("fake", None)                  # nothing plugged in -> fake
    with pytest.raises(SystemExit, match="no glove port"):
        run_mod.pick_glove("hardware", None, rt)


def test_load_env_sets_missing_names_only(tmp_path, monkeypatch):
    env = tmp_path / "glance.env"
    env.write_text("# comment\nGLANCE_TEST_A=one\nGLANCE_TEST_B='two'\nnot a line\n")
    monkeypatch.delenv("GLANCE_TEST_A", raising=False)
    monkeypatch.setenv("GLANCE_TEST_B", "shell wins")
    assert runtime_config.load_env(env) == ["GLANCE_TEST_A"]
    assert os.environ["GLANCE_TEST_A"] == "one" and os.environ["GLANCE_TEST_B"] == "shell wins"
    monkeypatch.delenv("GLANCE_TEST_A")
    assert runtime_config.load_env(tmp_path / "missing.env") == []
