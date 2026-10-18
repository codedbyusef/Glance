"""Wire protocol between the laptop/Pi and the glove firmware.

Frame:   0xAA, N levels (0-255), checksum = XOR of the N levels
Command: 0xA5, 0x00 = all off | 0xA5, 0x01 = test mode (pulse each channel once)

The 8 channels below are the logical layout everything above the wire uses (GuidanceController, the
runtime's GloveCommand). The firmware decides which of them physically exist and in what order:
    firmware/glove/glove.ino               8 levels, in CHANNELS order
    firmware/glove/code.py (ESP32-S3)      6 levels, in S3_CHANNELS order: no middle finger, no wrist motor
"""
from functools import reduce

CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "back", "palm", "wrist")
THUMB, INDEX, MIDDLE, RING, PINKY, BACK, PALM, WRIST = range(8)

# Meaning of each channel in the guidance mapping.
LEFT, FORWARD, CONFIRM, BACKWARD, RIGHT, UP, DOWN, PROXIMITY = range(8)

# The ESP32-S3 glove (firmware/glove/code.py): the logical channel behind each of its 6 wire levels.
S3_CHANNELS = (THUMB, INDEX, RING, PINKY, BACK, PALM)

FRAME_HEADER = 0xAA
CMD_HEADER = 0xA5
CMD_OFF = 0x00
CMD_TEST = 0x01
BAUD = 115200


def encode_frame(levels):
    """Return the 10-byte frame for 8 motor levels (glove.ino)."""
    if len(levels) != 8:
        raise ValueError(f"need 8 levels, got {len(levels)}")
    return encode_levels(levels)


def encode_levels(levels):
    """Return the frame for any number of levels: 0xAA, levels, XOR checksum (code.py takes 6)."""
    data = bytes(int(v) for v in levels)  # raises ValueError if any level is outside 0-255
    checksum = reduce(lambda a, b: a ^ b, data, 0)
    return bytes([FRAME_HEADER]) + data + bytes([checksum])


def wire_levels(levels, channels=None):
    """The levels actually sent: the 8 logical levels, or only `channels` of them, in that order."""
    return list(levels) if channels is None else [levels[ch] for ch in channels]


def encode_command(cmd):
    return bytes([CMD_HEADER, cmd])
