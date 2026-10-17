"""Sends the 8-value motor array to the glove at ~20 Hz over BLE or WiFi (undecided).

Never resends stale values when vision stalls -- the ESP32 failsafe (no packet for
300 ms -> all motors off) depends on that.
"""

from __future__ import annotations


class HapticLink:
    def send(self, values: list[int]) -> None:
        raise NotImplementedError

    def stop_all(self) -> None:
        raise NotImplementedError
