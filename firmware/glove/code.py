# Glance glove firmware (CircuitPython 9, Adafruit Feather ESP32-S3 + L293D + SN754410)
#
# Receives motor levels from the laptop over the USB "data" serial port (enabled in boot.py)
# and drives 6 coin motors. Same protocol as glove.ino:
#   Frame:   0xAA, b0..b5, checksum      levels 0-255 per channel, checksum = XOR of b0..b5
#   Command: 0xA5, 0x00                  all motors off
#            0xA5, 0x01                  test mode: pulse each channel once, in order
#
# Every frame carries all 6 levels; 0 = off. Send frames continuously (e.g. 20-50 Hz).
#
# Channels:
#   ch  motor          Feather pin
#   0   thumb          5
#   1   index          11
#   2   ring           12
#   3   pinky          6
#   4   back of hand   9
#   5   palm           10
#
# Wiring: driver VCC1 and VCC2 (motor supply) at 5 V, EN pins tied high, each motor between a
# driver output and GND (motors spin one way only). 10k pulldown on each driver input.
#
# Failsafe: no valid frame for FAILSAFE_S -> every motor off.

import time

import board
import pwmio
import usb_cdc

# Feather silkscreen labels, in channel order (not pin order).
PINS = (board.D5, board.D11, board.D12, board.D6, board.D9, board.D10)
NAMES = ("thumb", "index", "ring", "pinky", "back", "palm")
NUM_CH = len(PINS)

PWM_FREQ = 5000    # Hz
MIN_DUTY_PCT = 60  # motors stall below ~2.3 V, so level 1 starts here
MAX_DUTY_PCT = 85  # comfort + voltage cap; L293D already drops ~1.4 V
FAILSAFE_S = 0.3

FRAME_HEADER = 0xAA
CMD_HEADER = 0xA5
CMD_OFF = 0x00
CMD_TEST = 0x01

WAIT_HEADER, READ_FRAME, READ_CMD = range(3)

motors = [pwmio.PWMOut(p, frequency=PWM_FREQ, duty_cycle=0) for p in PINS]
port = usb_cdc.data
state = WAIT_HEADER
buf = bytearray(NUM_CH + 1)  # levels + checksum
buf_len = 0
last_frame = 0.0
motors_on = False


def level_to_duty(level):
    """Level 0 -> off, 1-255 -> MIN_DUTY_PCT..MAX_DUTY_PCT of full 16-bit duty."""
    if level == 0:
        return 0
    pct = MIN_DUTY_PCT + (level - 1) * (MAX_DUTY_PCT - MIN_DUTY_PCT) / 254
    return int(65535 * pct / 100)


def say(line):
    print(line)
    if port:
        port.write((line + "\n").encode())


def all_off():
    global motors_on
    for m in motors:
        m.duty_cycle = 0
    motors_on = False


def apply_frame(levels):
    global motors_on, last_frame
    for ch in range(NUM_CH):
        motors[ch].duty_cycle = level_to_duty(levels[ch])
    motors_on = any(levels[:NUM_CH])
    last_frame = time.monotonic()


def run_test():
    """Pulses each channel for 0.4 s, printing its name, so the mapping can be checked by touch."""
    global last_frame
    all_off()
    for ch in range(NUM_CH):
        say("TEST {} {}".format(ch, NAMES[ch]))
        motors[ch].duty_cycle = level_to_duty(255)
        time.sleep(0.4)
        motors[ch].duty_cycle = 0
        time.sleep(0.25)
    say("TEST DONE")
    last_frame = time.monotonic()


def handle_byte(b):
    global state, buf_len
    if state == WAIT_HEADER:
        if b == FRAME_HEADER:
            state = READ_FRAME
            buf_len = 0
        elif b == CMD_HEADER:
            state = READ_CMD
    elif state == READ_FRAME:
        buf[buf_len] = b
        buf_len += 1
        if buf_len == NUM_CH + 1:
            chk = 0
            for i in range(NUM_CH):
                chk ^= buf[i]
            if chk == buf[NUM_CH]:
                apply_frame(buf)  # bad checksum -> drop the frame silently
            state = WAIT_HEADER
    elif state == READ_CMD:
        if b == CMD_OFF:
            all_off()
        elif b == CMD_TEST:
            run_test()
        state = WAIT_HEADER


if port is None:
    print("No USB data port: copy boot.py to CIRCUITPY and press RESET.")
    while True:
        time.sleep(1)

port.timeout = 0
say("GLANCE GLOVE READY")

while True:
    n = port.in_waiting
    if n:
        for b in port.read(n):
            handle_byte(b)
    if motors_on and time.monotonic() - last_frame > FAILSAFE_S:
        all_off()
