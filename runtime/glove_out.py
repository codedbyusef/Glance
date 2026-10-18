"""The one glove output interface (CLAUDE.md section 5). Motors are on or off; proximity is the wrist's pulse rate.

Channel indices and names come from haptics/protocol.py. All-off happens on session end, pause, cancel, any
exception and process exit; callers use all_off() on every one of those paths.
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
    pulse_hz: dict[int, float] = field(default_factory=dict)  # channel -> pulse rate (today only WRIST); 0 = none


OFF = GloveCommand()


class GloveOutput(Protocol):
    def apply(self, cmd: GloveCommand) -> None: ...  # called each vision frame; must not block

    def all_off(self) -> None: ...  # called on every exit path


def channel_on(cmd: GloveCommand, ch: int, t: float) -> bool:
    """Whether a channel is energized at time t: steady channels are on, pulsing ones are on for half a period."""
    if ch in cmd.on:
        return True
    hz = cmd.pulse_hz.get(ch, 0.0)
    return hz > 0 and (t * hz) % 1.0 < 0.5


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
    """Draws the 8 motors on the live view, blinking at their pulse rates. No hardware."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
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
        x0, y0 = origin
        cv2.rectangle(img, (x0, y0), (x0 + 150, y0 + 190), (30, 30, 30), -1)
        cv2.putText(img, "glove (fake)", (x0 + 6, y0 + 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for ch, (x, y) in _SPOTS.items():
            on = channel_on(cmd, ch, t)
            color = (0, 220, 255) if on else (90, 90, 90)
            cv2.circle(img, (x0 + x, y0 + y), 11, color, -1 if on else 2)
            cv2.putText(img, _MEANING[ch], (x0 + x - 12, y0 + y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0, 0, 0) if on else (200, 200, 200), 1)
        hz = cmd.pulse_hz.get(protocol.WRIST, 0.0)
        if hz > 0:
            cv2.putText(img, f"{hz:.1f} Hz", (x0 + 98, y0 + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)


class HardwareGlove:
    """The real glove over USB serial (Step 4), through haptics.link.GloveLink and the firmware protocol in
    haptics/protocol.py: a steady channel is sent at on_level, an off channel at 0. GloveLink resends the
    latest levels at 50 Hz on its own thread and gates the wrist on/off at its pulse rate in software
    (link.frame_at), so apply() only stores the command and never blocks. The firmware turns every motor
    off if no frame arrives for 300 ms, so a crashed process can't leave a motor on.

    Only the wrist can pulse (GloveLink gates only WRIST); any other channel in pulse_hz is sent steady on."""

    def __init__(self, port: str | None = None, on_level: int = 255, link=None):
        if link is None:
            if not port:
                raise ValueError("HardwareGlove needs a serial port: set glove.port in runtime/config.local.yaml "
                                 "or pass --glove-port (e.g. COM5, /dev/ttyUSB0)")
            from haptics.link import GloveLink

            link = GloveLink(port)
        self.link = link
        self.on_level = int(on_level)
        self.link.start()

    def apply(self, cmd: GloveCommand) -> None:
        levels = [0] * 8
        for ch in cmd.on | set(ch for ch, hz in cmd.pulse_hz.items() if hz > 0):
            levels[ch] = self.on_level
        self.link.set(levels, cmd.pulse_hz.get(protocol.WRIST, 0.0))

    def all_off(self) -> None:
        self.link.off()  # zero levels now, plus the firmware's all-off command

    def close(self) -> None:
        self.link.stop()  # sends all-off, closes the port


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


def make_glove(kind: str, port: str | None = None, on_level: int = 255):
    """The live scripts' glove: fake -> FakeGlove; hardware -> HardwareGlove mirrored on a FakeGlove panel."""
    if kind == "fake":
        return FakeGlove()
    if kind == "hardware":
        return TeeGlove(HardwareGlove(port, on_level), FakeGlove())
    raise ValueError(f"glove kind must be fake or hardware, not {kind!r}")
