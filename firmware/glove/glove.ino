// Glance glove firmware (ESP32 DevKit + 2x L293D)
//
// Receives motor levels from the laptop over USB serial and drives 8 coin motors.
//
// Protocol (115200 baud):
//   Frame:   0xAA, b0..b7, checksum      levels 0-255 per channel, checksum = XOR of b0..b7
//   Command: 0xA5, 0x00                  all motors off
//            0xA5, 0x01                  test mode: pulse each channel once, in order
//
// Channel order (matches haptics/protocol.py and the wiring diagram):
//   0 thumb, 1 index, 2 middle, 3 ring, 4 pinky, 5 back of hand, 6 palm, 7 wrist
//
// Failsafe: no valid frame for FAILSAFE_MS -> every motor off.

#include <Arduino.h>
#include <esp_arduino_version.h>
#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 2
#endif

// GPIO per channel. On an ESP32-WROVER board, change 16 and 17 to 25 and 26.
const uint8_t PINS[8] = {13, 27, 16, 17, 18, 19, 21, 22};
const char *NAMES[8] = {"thumb", "index", "middle", "ring", "pinky", "back", "palm", "wrist"};

const uint32_t PWM_FREQ = 5000;   // Hz
const uint8_t PWM_BITS = 8;       // duty 0-255
const uint8_t MAX_DUTY_PCT = 85;  // comfort + voltage cap; L293D already drops ~1.4 V
const uint32_t FAILSAFE_MS = 300;
const uint32_t BAUD = 115200;

const uint8_t FRAME_HEADER = 0xAA;
const uint8_t CMD_HEADER = 0xA5;
const uint8_t CMD_OFF = 0x00;
const uint8_t CMD_TEST = 0x01;

enum ParseState { WAIT_HEADER, READ_FRAME, READ_CMD };
ParseState state = WAIT_HEADER;
uint8_t buf[9];   // 8 levels + checksum
uint8_t bufLen = 0;
uint32_t lastFrameMs = 0;
bool motorsOn = false;

void pwmAttach(uint8_t ch) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PINS[ch], PWM_FREQ, PWM_BITS);
#else
  ledcSetup(ch, PWM_FREQ, PWM_BITS);
  ledcAttachPin(PINS[ch], ch);
#endif
}

void pwmWrite(uint8_t ch, uint8_t level) {
  uint32_t duty = (uint32_t)level * MAX_DUTY_PCT / 100;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PINS[ch], duty);
#else
  ledcWrite(ch, duty);
#endif
}

void allOff() {
  for (uint8_t ch = 0; ch < 8; ch++) pwmWrite(ch, 0);
  motorsOn = false;
}

void applyFrame(const uint8_t *levels) {
  bool any = false;
  for (uint8_t ch = 0; ch < 8; ch++) {
    pwmWrite(ch, levels[ch]);
    if (levels[ch]) any = true;
  }
  motorsOn = any;
  lastFrameMs = millis();
}

// Pulses each channel for 400 ms, printing its name, so the mapping can be checked by touch.
void runTest() {
  allOff();
  for (uint8_t ch = 0; ch < 8; ch++) {
    Serial.printf("TEST %u %s\n", ch, NAMES[ch]);
    pwmWrite(ch, 255);
    delay(400);
    pwmWrite(ch, 0);
    delay(250);
  }
  Serial.println("TEST DONE");
  lastFrameMs = millis();
}

void handleByte(uint8_t b) {
  switch (state) {
    case WAIT_HEADER:
      if (b == FRAME_HEADER) { state = READ_FRAME; bufLen = 0; }
      else if (b == CMD_HEADER) { state = READ_CMD; }
      break;
    case READ_FRAME:
      buf[bufLen++] = b;
      if (bufLen == 9) {
        uint8_t chk = 0;
        for (uint8_t i = 0; i < 8; i++) chk ^= buf[i];
        if (chk == buf[8]) applyFrame(buf);  // bad checksum -> drop the frame silently
        state = WAIT_HEADER;
      }
      break;
    case READ_CMD:
      if (b == CMD_OFF) allOff();
      else if (b == CMD_TEST) runTest();
      state = WAIT_HEADER;
      break;
  }
}

void setup() {
  Serial.begin(BAUD);
  for (uint8_t ch = 0; ch < 8; ch++) {
    pwmAttach(ch);
    pwmWrite(ch, 0);
  }
  Serial.println("GLANCE GLOVE READY");
}

void loop() {
  while (Serial.available() > 0) handleByte((uint8_t)Serial.read());
  if (motorsOn && millis() - lastFrameMs > FAILSAFE_MS) allOff();
}
