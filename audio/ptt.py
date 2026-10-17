"""Push-to-talk (CLAUDE.md section 6, decision 4). Both put ("ptt", down, t) on the Mic's queue."""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from .log import EventLog, null_log

Put = Callable[[tuple], None]


class KeyPtt:
    """Dev only: Enter toggles. A terminal can't see key releases."""

    def __init__(self, stream=None, echo: bool = True):
        self.stream = stream or sys.stdin
        self.echo = echo
        self.down = False
        self._thread: threading.Thread | None = None

    def start(self, put: Put) -> None:
        def run():
            while True:
                line = self.stream.readline()
                if not line:  # stdin closed
                    return
                self.down = not self.down
                if self.echo:
                    print("[ptt] talk now, Enter to stop" if self.down else "[ptt] released", flush=True)
                put(("ptt", self.down, time.time()))

        self._thread = threading.Thread(target=run, name="ptt-key", daemon=True)
        self._thread.start()

    def close(self) -> None:
        pass  # the thread is blocked on readline; it is a daemon


class GpioPtt:
    """A momentary button between the GPIO pin and ground. gpiozero ships with Raspberry Pi OS."""

    def __init__(self, pin: int, log: EventLog | None = None):
        self.pin = pin
        self.log = log or null_log()
        self.button = None

    def start(self, put: Put) -> None:
        try:
            from gpiozero import Button

            self.button = Button(self.pin, pull_up=True, bounce_time=0.03)
            self.button.when_pressed = lambda: put(("ptt", True, time.time()))
            self.button.when_released = lambda: put(("ptt", False, time.time()))
            self.log.write("ptt_ready", kind="gpio", pin=self.pin)
        except Exception as e:  # no GPIO (laptop) or pin busy: PTT is unavailable, keep running
            self.log.error("ptt.gpio", e, pin=self.pin)

    def close(self) -> None:
        if self.button is not None:
            self.button.close()


def build_ptt(spec: str, log: EventLog | None = None):
    """none | key | gpio:<pin>"""
    if spec == "none":
        return None
    if spec == "key":
        return KeyPtt()
    if spec.startswith("gpio:"):
        return GpioPtt(int(spec.split(":", 1)[1]), log)
    raise ValueError(f"ptt must be none, key or gpio:<pin>, not {spec!r}")
