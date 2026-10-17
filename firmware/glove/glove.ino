/*
 * ESP32 glove firmware (sketch stub).
 *
 * Receives the 8-value motor array over BLE/WiFi (transport undecided) and
 * drives two SN754410/L293D half-H drivers via PWM (~20 kHz, 8-bit).
 *
 * Channel order: [thumb, index, middle, ring, pinky, top_of_hand, bottom_of_hand, wrist]
 * GPIO (classic ESP32, change if board differs):
 *   thumb 13, index 27, middle 16, ring 17, pinky 18, top 19, bottom 21, wrist 22
 *
 * Failsafe: no packet for 300 ms -> all motors off. Do not resend stale values.
 * "stop" / all-channels pulse together is the haptic stop signal.
 * Test mode: pulse one channel at a time (see testMode()).
 *
 * Motors don't spin below ~2.3V: map 0 -> off, 1-255 -> ~60%-85% duty,
 * with a per-channel max duty so both driver chips feel matched.
 *
 * Flash with the main switch off. 10k pulldown on every driver input line.
 */

void setup() {
  // TODO: configure PWM channels, pulldowns, BLE/WiFi link.
}

void loop() {
  // TODO: receive motor array, apply failsafe timeout, write PWM.
}

void testMode() {
  // TODO: pulse one channel at a time for hardware bring-up.
}
