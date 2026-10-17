"""GloveLink: sends motor levels to the glove over USB serial on a background thread.

The sender runs at a fixed rate (default 50 Hz) independent of the vision frame rate,
so the wrist motor can pulse smoothly and the firmware's 300 ms failsafe never trips
while the program is alive. Call set() whenever the controller produces a new command.

    link = GloveLink("/dev/cu.usbserial-0001")   # or dry_run=True to print instead
    link.start()
    link.set(cmd.levels, cmd.wrist_hz)
    ...
    link.stop()                                   # sends all-off
"""
import threading
import time

from haptics.protocol import BAUD, CMD_OFF, CMD_TEST, WRIST, encode_command, encode_frame


class GloveLink:
    def __init__(self, port=None, baud=BAUD, rate_hz=50.0, dry_run=False):
        self.dry_run = dry_run or port is None
        self.rate_hz = rate_hz
        self._levels = [0] * 8
        self._wrist_hz = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._serial = None
        if not self.dry_run:
            import serial  # pyserial
            self._serial = serial.Serial(port, baud, timeout=0)
            time.sleep(2.0)  # many ESP32 boards reset when the port opens

    def set(self, levels, wrist_hz=0.0):
        if len(levels) != 8:
            raise ValueError("need 8 levels")
        with self._lock:
            self._levels = [max(0, min(255, int(v))) for v in levels]
            self._wrist_hz = max(0.0, float(wrist_hz))

    def off(self):
        self.set([0] * 8, 0.0)
        self._write(encode_command(CMD_OFF))

    def test(self):
        """Ask the firmware to pulse each channel once (about 5 s)."""
        self._write(encode_command(CMD_TEST))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.off()
        if self._serial:
            self._serial.close()

    def frame_at(self, t):
        """Levels actually sent at time t (wrist gated on/off at wrist_hz, 50% duty)."""
        with self._lock:
            out = list(self._levels)
            hz = self._wrist_hz
        if hz > 0 and (t * hz) % 1.0 >= 0.5:
            out[WRIST] = 0
        return out

    def _loop(self):
        period = 1.0 / self.rate_hz
        next_t = time.monotonic()
        while self._running:
            self._write(encode_frame(self.frame_at(time.monotonic())))
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _write(self, data):
        if self.dry_run:
            if data[0] == 0xAA:
                print("frame", list(data[1:9]))
            else:
                print("cmd", list(data))
            return
        self._serial.write(data)
