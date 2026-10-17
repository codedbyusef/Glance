"""Wire protocol between the laptop and the glove firmware (firmware/glove/glove.ino).

Frame:   0xAA, 8 levels (0-255), checksum = XOR of the 8 levels
Command: 0xA5, 0x00 = all off | 0xA5, 0x01 = test mode (pulse each channel once)
"""
from functools import reduce

CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "back", "palm", "wrist")
THUMB, INDEX, MIDDLE, RING, PINKY, BACK, PALM, WRIST = range(8)

# Meaning of each channel in the guidance mapping.
LEFT, FORWARD, CONFIRM, BACKWARD, RIGHT, UP, DOWN, PROXIMITY = range(8)

FRAME_HEADER = 0xAA
CMD_HEADER = 0xA5
CMD_OFF = 0x00
CMD_TEST = 0x01
BAUD = 115200


def encode_frame(levels):
    """Return the 10-byte frame for 8 motor levels."""
    if len(levels) != 8:
        raise ValueError(f"need 8 levels, got {len(levels)}")
    data = bytes(int(v) for v in levels)  # raises ValueError if any level is outside 0-255
    checksum = reduce(lambda a, b: a ^ b, data, 0)
    return bytes([FRAME_HEADER]) + data + bytes([checksum])


def encode_command(cmd):
    return bytes([CMD_HEADER, cmd])
