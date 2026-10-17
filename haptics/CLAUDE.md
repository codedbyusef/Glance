# Haptics: guidance controller, glove link, firmware

Covers `haptics/` (guidance controller, glove link, fake stream) and `firmware/glove/` (ESP32 sketch). Read the root
`CLAUDE.md` first for priorities, ownership and conventions.

Code map: `haptics/controller.py` turns hand-to-target error into the 8 motor values, `haptics/link.py` sends the
array to the glove at ~20 Hz, `haptics/fake_stream.py` is a scripted sweep so hardware never waits on vision,
`firmware/glove/glove.ino` drives the motors. Tunables: `config/mapping.yaml`, `config/controller.yaml`.

## Guidance controller (pure rules, per frame)

- Errors, normalised: `dx = (target_cx - hand_x) / W`, `dy = (target_cy - hand_y) / H`, `dz` = normalised target depth minus hand depth.
- Depth Anything V2 gives relative depth (larger = closer), so normalise per frame (patch medians divided by the frame's 5th-95th percentile range). Verify the forward/back sign by hand-waving at the camera.
- Positive dx -> right channel; negative -> left. Positive dy (target lower in image) -> down; negative -> up. Target farther than hand -> forward.
- Wrist channel = proximity, stronger as total distance shrinks.
- Deadband per axis, hysteresis, EMA smoothing, and at most TWO direction channels active at once.
- Success itself is decided server-side (see `detection/CLAUDE.md` "Object detection and tracking (settled)"): the
  controller just drives motors from dx/dy/dz every frame; once the runtime gets a `success` `ReachReport`, it fires
  the confirm pulse and stops, and the agent takes over with a touch check.
- Starting values (tune, not measured): deadband 0.05, hysteresis 0.03, EMA alpha 0.3. Keep them in `config/controller.yaml`.
- No hand in frame: zero all motors. Tracker mask empty for several frames: re-detect once; the session's
  own `hand_lost`/`target_lost` timeouts (config/settings.yaml: `reach_session`) are what actually end the reach.
- No fine-tuned or learned policy. Rules only.
- No hand-rotation compensation. The mapping is fixed in the camera's image frame; the user learns it.

## Channel order (the 8-value array, ints 0-255, ~20 Hz, BLE or WiFi: undecided)

`[thumb, index, middle, ring, pinky, top_of_hand, bottom_of_hand, wrist]` (indices 0-7)

## Cue-to-channel mapping is PROVISIONAL. Keep it in `config/mapping.yaml`, never hardcode it.

| Cue | Mapping A (proposed for first test) | Mapping B (earlier planning chat) |
|---|---|---|
| left | thumb | index |
| right | pinky | pinky |
| forward | index | middle |
| back | ring | ring |
| up | top of hand | top of hand |
| down | bottom of hand (palm) | bottom of hand (palm) |
| proximity | wrist | wrist |
| confirm | middle, short double pulse | thumb, short double pulse |

The design doc also lists a "teammate version" with no forward/back channels; it cannot guide depth, so it is not used.
Decide between A and B after the first blindfolded tester. Right hand only for the demo. The glove's physical wiring does not change between mappings.

## Safety and comfort (firmware + software)

- Failsafe: no packet for 300 ms -> all motors off. Do not resend stale values when vision stalls.
- "stop" (voice) cancels the step and zeroes all motors. All motors pulsing together is the haptic stop signal.
- Cap motor intensity and continuous duty cycle; check with testers that sustained vibration does not numb the hand.
- Firmware test mode: pulse one channel at a time. Software ships a fake stream (scripted sweep) so hardware never waits on vision.

## Hardware summary

- 8x 10 mm x 2.7 mm 3 V coin ERM motors, right-hand glove. Motors only spin one way.
- ESP32 DevKit (classic) on a forearm strap with a USB power bank (5V). Glove connects through a 10-pin plug at the wrist (8 signals + 2 shared returns).
- Two 16-pin half-H drivers (SN754410 and/or L293D, same pinout, both fed 5V). U1 drives channels 0-3, U2 drives channels 4-7.
- ESP32 GPIO assumed: thumb 13, index 27, middle 16, ring 17, pinky 18, top 19, bottom 21, wrist 22 (avoid strapping pins 0, 2, 5, 12, 15 and GPIO14). Change if the board is not a classic ESP32.
- 10k pulldown on every driver input line so motors stay off during boot. 470 uF across 5V/GND near the drivers.
- PWM ~20 kHz, 8-bit. Motors do not spin below ~2.3 V, so firmware maps 0 -> off and 1-255 -> roughly 60%-85% duty, with a per-channel max duty so the two driver chips feel matched. Flash the ESP32 with the main switch off.
