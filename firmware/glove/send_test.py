"""Bench test for the 6-motor glove firmware (code.py on the ESP32-S3, or glove.ino). Needs pyserial.

PORT on the laptop is the S3's data serial port, usually /dev/ttyACM0 (boot.py turns the REPL console off).

    python3 firmware/glove/send_test.py PORT test            # firmware pulses each channel once
    python3 firmware/glove/send_test.py PORT one CH LEVEL    # buzz one channel (0-5) at LEVEL (1-255)
    python3 firmware/glove/send_test.py PORT all LEVEL       # buzz every channel at LEVEL
    python3 firmware/glove/send_test.py PORT off             # all motors off

`one` and `all` resend the frame every 100 ms until Ctrl+C (the firmware stops the motors
if no frame arrives for 300 ms). Channels: 0 thumb, 1 index, 2 ring, 3 pinky, 4 back, 5 palm.
"""
import sys
import time
from functools import reduce

import serial

NUM_CH = 6
FRAME_HEADER = 0xAA
CMD_HEADER = 0xA5
CMD_OFF = 0x00
CMD_TEST = 0x01


def encode_frame(levels):
    data = bytes(int(v) for v in levels)
    return bytes([FRAME_HEADER]) + data + bytes([reduce(lambda a, b: a ^ b, data, 0)])


def stream(ser, levels):
    print("sending", levels, "- Ctrl+C to stop")
    try:
        while True:
            ser.write(encode_frame(levels))
            time.sleep(0.1)
    except KeyboardInterrupt:
        ser.write(bytes([CMD_HEADER, CMD_OFF]))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    port, mode, args = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
    ser = serial.Serial(port, 115200, timeout=0.1)
    time.sleep(2.0)  # many ESP32 boards reset when the port opens
    if mode == "test":
        ser.write(bytes([CMD_HEADER, CMD_TEST]))
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                print(line)
            if line == "TEST DONE":
                break
    elif mode == "one":
        levels = [0] * NUM_CH
        levels[args[0]] = args[1]
        stream(ser, levels)
    elif mode == "all":
        stream(ser, [args[0]] * NUM_CH)
    elif mode == "off":
        ser.write(bytes([CMD_HEADER, CMD_OFF]))
    else:
        sys.exit(__doc__)
    ser.close()


if __name__ == "__main__":
    main()
