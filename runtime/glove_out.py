"""The one glove output interface (CLAUDE.md section 5). Motors are on or off; proximity is the pulse rate of the
active direction motors (decision 24).

Channel indices and names come from haptics/protocol.py. All-off happens on session end, pause, cancel, any
exception and process exit; callers use all_off() on every one of those paths.

The real glove is the 6-motor ESP32-S3 (firmware/glove/code.py): one motor per direction (thumb, index, ring,
pinky, back, palm), no middle-finger and no wrist motor.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from haptics import protocol


@dataclass(frozen=True)
class GloveCommand:
    on: frozenset[int] = frozenset()  # channel indices from haptics/protocol.py that are steadily on
    pulse_hz: dict[int, float] = field(default_factory=dict)  # channel -> pulse rate (guidance: the direction motors)


OFF = GloveCommand()


class GloveOutput(Protocol):
    def apply(self, cmd: GloveCommand) -> None: ...  # called each vision frame; must not block

    def all_off(self) -> None: ...  # called on every exit path


def channel_on(cmd: GloveCommand, ch: int, t: float) -> bool:
    """Whether a channel is energized at time t with a fixed rate: steady channels are on, pulsing ones are on for
    the first half of each period (FakeGlove uses PulsePhase so rate changes stay smooth)."""
    if ch in cmd.on:
        return True
    hz = cmd.pulse_hz.get(ch, 0.0)
    return hz > 0 and (t * hz) % 1.0 < 0.5


class PulsePhase:
    """Per-channel pulse phase advanced by hz * elapsed time (like haptics.link.GloveLink.frame_at), so a rate
    that changes every frame speeds the blinking up smoothly; a channel that starts pulsing starts "on"."""

    def __init__(self):
        self._phase: dict[int, float] = {}
        self._t: float | None = None

    def on(self, cmd: GloveCommand, t: float) -> set[int]:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        self._phase = {ch: (self._phase.get(ch, 0.0) + hz * dt) % 1.0 for ch, hz in cmd.pulse_hz.items() if hz > 0}
        return set(cmd.on) | {ch for ch, ph in self._phase.items() if ph < 0.5}


# The logical channels each firmware has (haptics/protocol.py), in wire order.
FIRMWARE_CHANNELS = {"s3": protocol.S3_CHANNELS, "ino8": tuple(range(8))}

# Panel layout: a right-hand glove seen from the back, fingers up. (x, y) in a 150 x 170 box.
_SPOTS = {
    protocol.THUMB: (18, 95), protocol.INDEX: (45, 30), protocol.MIDDLE: (75, 20), protocol.RING: (105, 30),
    protocol.PINKY: (132, 50), protocol.BACK: (75, 85), protocol.PALM: (75, 120), protocol.WRIST: (75, 158),
}
_MEANING = {
    protocol.THUMB: "L", protocol.INDEX: "FWD", protocol.MIDDLE: "-", protocol.RING: "BACK", protocol.PINKY: "R",
    protocol.BACK: "UP", protocol.PALM: "DOWN", protocol.WRIST: "prox",
}


class FakeGlove:
    """Draws the motors on the live view, blinking at their pulse rates. No hardware.

    channels: the motors the real glove has; the others are drawn crossed out (they get nothing on hardware)."""

    def __init__(self, clock=time.monotonic, channels=None, label: str = "glove"):
        self.clock = clock
        self.channels = set(channels) if channels is not None else set(range(8))
        self.label = label
        self._pulse = PulsePhase()
        self._lock = threading.Lock()
        self._cmd = OFF
        self.applied = 0  # for tests

    def apply(self, cmd: GloveCommand) -> None:
        with self._lock:
            self._cmd = cmd
            self.applied += 1

    def all_off(self) -> None:
        self.apply(OFF)

    @property
    def command(self) -> GloveCommand:
        with self._lock:
            return self._cmd

    def draw(self, img: np.ndarray, origin: tuple[int, int]) -> None:
        """Draw the panel with its top-left corner at origin."""
        cmd, t = self.command, self.clock()
        lit = self._pulse.on(cmd, t)
        x0, y0 = origin
        cv2.rectangle(img, (x0, y0), (x0 + 150, y0 + 190), (30, 30, 30), -1)
        cv2.putText(img, self.label, (x0 + 6, y0 + 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for ch, (x, y) in _SPOTS.items():
            if ch not in self.channels:  # not on this glove: shown, but crossed out
                cv2.circle(img, (x0 + x, y0 + y), 11, (60, 60, 60), 1)
                cv2.line(img, (x0 + x - 8, y0 + y - 8), (x0 + x + 8, y0 + y + 8), (60, 60, 60), 1)
                continue
            on = ch in lit
            color = (0, 220, 255) if on else (90, 90, 90)
            cv2.circle(img, (x0 + x, y0 + y), 11, color, -1 if on else 2)
            cv2.putText(img, _MEANING[ch], (x0 + x - 12, y0 + y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0, 0, 0) if on else (200, 200, 200), 1)
        hz = max(cmd.pulse_hz.values(), default=0.0)
        if hz > 0:
            cv2.putText(img, f"{hz:.1f} Hz", (x0 + 98, y0 + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)


class HardwareGlove:
    """The real glove over USB serial (Step 4), through haptics.link.GloveLink and the firmware protocol in
    haptics/protocol.py: a steady channel is sent at on_level, an off channel at 0, and only the channels the
    firmware has go on the wire, in its order (firmware "s3": code.py's 6 motors; "ino8": glove.ino's 8).
    GloveLink resends the latest levels at 50 Hz on its own thread and gates each pulsing channel on/off at its
    rate (the proximity pulse on the direction motors), so apply() only stores the command and never blocks.
    The firmware turns every motor off if no frame arrives for 300 ms, so a crashed process can't leave a
    motor on.

    port "auto" (or None) finds the glove with find_glove_port()."""

    def __init__(self, port: str | None = None, on_level: int = 255, firmware: str = "s3", link=None):
        if firmware not in FIRMWARE_CHANNELS:
            raise ValueError(f"glove firmware must be one of {sorted(FIRMWARE_CHANNELS)}, not {firmware!r}")
        self.firmware = firmware
        self.channels = FIRMWARE_CHANNELS[firmware]
        if link is None:
            if not port or port == "auto":
                port = find_glove_port()
            if not port:
                raise ValueError("no glove serial port found: plug the glove in, or set glove.port in "
                                 "runtime/config.local.yaml / pass --glove-port (e.g. /dev/ttyACM0, COM5). "
                                 "python -m runtime.glove ports lists them")
            link = open_link(port, None if firmware == "ino8" else self.channels)
        self.port = port
        self.link = link
        self.on_level = int(on_level)
        self.link.start()

    def apply(self, cmd: GloveCommand) -> None:
        levels = [0] * 8
        pulse = {ch: hz for ch, hz in cmd.pulse_hz.items() if hz > 0}
        for ch in cmd.on | set(pulse):
            levels[ch] = self.on_level
        self.link.set(levels, 0.0, pulse_hz=pulse)

    def all_off(self) -> None:
        self.link.off()  # zero levels now, plus the firmware's all-off command

    def close(self) -> None:
        self.link.stop()  # sends all-off, closes the port

    def read_lines(self) -> list[str]:
        """What the firmware printed (GLANCE GLOVE READY, TEST lines)."""
        read = getattr(self.link, "read_lines", None)
        return read() if read is not None else []


class GlovePortError(RuntimeError):
    pass


def open_link(port: str, channels, attempts: int = 3, wait_s: float = 1.5, sleep=time.sleep):
    """GloveLink on `port`, retrying briefly (CircuitPython drops and re-adds its USB port for a moment after a file
    on CIRCUITPY is saved). Errors say what they usually mean: busy (another program has the port; Windows gives
    a serial port to one process at a time) or missing (unplugged, or the board is in safe mode)."""
    from haptics.link import GloveLink

    last = None
    for i in range(attempts):
        try:
            return GloveLink(port, channels=channels)
        except Exception as e:  # serial.SerialException
            last = e
            if i + 1 < attempts:
                sleep(wait_s)
    text = str(last)
    if "Access is denied" in text or "PermissionError" in text or "Device or resource busy" in text:
        hint = (f"{port} is busy: another program has it open (another Glance run, a VS Code/Mu/Thonny serial "
                f"monitor, firmware/glove/send_test.py). Close it and retry. On Linux, 'Permission denied' instead "
                f"means you are not in the dialout group yet (log out and back in after install_pi.sh).")
    elif "FileNotFoundError" in text or ("could not open port" in text and "No such file" in text):
        hint = (f"{port} does not exist right now: the glove is unplugged or not running (check that CIRCUITPY "
                f"shows up and the board is not in safe mode). python -m runtime.glove ports lists the ports.")
    else:
        hint = f"could not open {port}"
    raise GlovePortError(f"{hint} ({text})") from last


# USB vendor ids: Adafruit (CircuitPython on the Feather ESP32-S3) and Espressif (bare ESP32-S3 / Arduino).
GLOVE_USB_VIDS = (0x239A, 0x303A)


def find_glove_port(ports=None) -> str | None:
    """The glove's serial port: the first port from Adafruit/Espressif, or whose description names
    CircuitPython or an ESP32. ports: serial.tools.list_ports.comports() (injected in tests)."""
    if ports is None:
        try:
            from serial.tools import list_ports
        except ImportError:
            return None
        ports = list_ports.comports()
    for p in ports:
        if getattr(p, "vid", None) in GLOVE_USB_VIDS:
            return p.device
    for p in ports:
        text = f"{getattr(p, 'description', '')} {getattr(p, 'manufacturer', '')}".lower()
        if "circuitpython" in text or "esp32" in text:
            return p.device
    return None


class TeeGlove:
    """Several outputs as one: the real glove plus the FakeGlove panel that shows what it was told."""

    def __init__(self, *outputs):
        self.outputs = outputs

    def apply(self, cmd: GloveCommand) -> None:
        for o in self.outputs:
            o.apply(cmd)

    def all_off(self) -> None:
        for o in self.outputs:
            o.all_off()

    def close(self) -> None:
        for o in self.outputs:
            close = getattr(o, "close", None)
            if close is not None:
                close()

    def draw(self, img: np.ndarray, origin: tuple[int, int]) -> None:
        for o in self.outputs:
            if hasattr(o, "draw"):
                o.draw(img, origin)
                return


def make_glove(kind: str, port: str | None = None, on_level: int = 255, firmware: str = "s3"):
    """The live scripts' glove: fake -> FakeGlove; hardware -> HardwareGlove mirrored on a FakeGlove panel
    that crosses out the motors this glove doesn't have."""
    if kind == "fake":
        return FakeGlove(channels=FIRMWARE_CHANNELS[firmware])  # draws the glove the hardware would be
    if kind == "hardware":
        hw = HardwareGlove(port, on_level, firmware)
        return TeeGlove(hw, FakeGlove(channels=hw.channels, label=f"glove ({firmware}, {hw.port})"))
    raise ValueError(f"glove kind must be fake or hardware, not {kind!r}")


def glove_from_config(cfg: dict, kind: str | None = None, port: str | None = None):
    """make_glove from runtime config["glove"], with optional --glove / --glove-port overrides."""
    g = cfg["glove"]
    return make_glove(kind or g["kind"], port or g["port"], g["on_level"], g["firmware"])
