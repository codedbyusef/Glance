# haptics/ — guidance controller, glove link, fake stream

Owner: Jasper. Firmware lives in `firmware/glove/glove.ino`.

## Contract
- Channel order everywhere: 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky, 5 back of hand, 6 palm, 7 wrist
  (= J1 plug pins 1-8). Middle (2) is unused for now; reserved for a fingertip touch sensor.
- Meaning: thumb = left, pinky = right, index = forward, ring = back, back of hand = up, palm = down,
  wrist = proximity (pulse rate: faster = closer).
- Link: USB serial, 115200 baud. Frame = 0xAA + 8 levels (0-255) + XOR checksum. 0xA5 0x00 = all off,
  0xA5 0x01 = firmware test mode. Firmware turns every motor off if no frame arrives for 300 ms.

## Files
- `protocol.py` — byte format shared with the firmware.
- `controller.py` — `GuidanceController.update(hand, target_box, frame_size)` -> `HapticCommand`
  (8 levels, wrist pulse Hz, status). Pure function of its inputs plus smoothing/hysteresis state.
  Tunables are in `ControllerConfig`. Set `success_frames=None` if the perception server decides success,
  so there is one owner of "done".
- `link.py` — `GloveLink` sends the latest command at 50 Hz on a background thread and gates the wrist
  motor at the pulse rate. `dry_run=True` prints instead of sending.
- `fake_stream.py` — test the glove with no vision: `python -m haptics.fake_stream --port PORT test|sweep|one N|reach`.

## Usage
```python
from haptics.controller import GuidanceController
from haptics.link import GloveLink
ctrl, link = GuidanceController(), GloveLink("/dev/cu.usbserial-0001"); link.start()
cmd = ctrl.update(hand_xy, target_box, (w, h))   # per vision frame
link.set(cmd.levels, cmd.wrist_hz)
```

## Tests
`pip install -r haptics/requirements.txt && python -m pytest haptics -q` (24 tests, no hardware needed).
