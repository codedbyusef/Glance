# Glove firmware

| File | What |
|---|---|
| `code.py`, `boot.py` | **The glove in use**: CircuitPython 9 on the Adafruit Feather ESP32-S3, 6 coin motors (thumb, index, ring, pinky, back of hand, palm). No middle-finger or wrist motor. |
| `send_test.py` | Bench test with raw frames (`test`, `one CH LEVEL`, `all LEVEL`, `off`). Needs pyserial. |
| `glove.ino` | The earlier 8-motor Arduino sketch (adds middle and wrist, no longer used). Same protocol with 8 levels. |

**Flash:** copy `boot.py` and `code.py` to the CIRCUITPY drive and press RESET. `boot.py` turns the REPL console
off and enables a raw USB "data" serial port; that becomes the only serial port, `/dev/ttyACM0` on the Pi.
To get the REPL back, rename `boot.py` and press RESET.

**Proximity** is in the pulsing: during guidance the active direction motors pulse at 1 Hz far up to 6 Hz near,
timed on the Pi/laptop (frames are sent at 50 Hz), so the firmware needs no change.

**Protocol** (`haptics/protocol.py`): frame `0xAA`, 6 levels (0-255, wire order thumb, index, ring, pinky,
back, palm), XOR checksum. `0xA5 0x00` is all off, `0xA5 0x01` the test mode. Level 1-255 maps to 60-85% duty.
No frame for 300 ms turns every motor off.

**Laptop or Pi:** both work the same. On Windows the glove is a COM port (no driver needed); on the Pi it is
`/dev/ttyACM0` (the install script adds you to `dialout`; log in again once).

**From the runtime** (Pi or laptop; the glove is found automatically by its USB vendor id):

```bash
python -m runtime.glove ports          # which port is the glove
python -m runtime.glove test           # firmware test mode, prints its TEST lines
python -m runtime.glove buzz LEFT      # through the runtime's mapping (LEFT = thumb, FORWARD = index, ...)
python -m runtime.glove sweep          # a simulated reach: guidance commands as a fake hand closes in
python -m runtime.glove_check --glove hardware      # each motor in turn, with the live view
bash runtime/scripts/run_glance.sh --ptt-only       # the whole system on the Pi
```

`python3 firmware/glove/send_test.py /dev/ttyACM0 test` (and its other modes) still works as before.
