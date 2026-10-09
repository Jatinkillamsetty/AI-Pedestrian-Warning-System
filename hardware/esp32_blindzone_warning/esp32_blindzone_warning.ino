/*
  =============================================================================
  AI Pedestrian Detection & Heavy-Vehicle Blind-Zone Warning System
  ESP32 Hardware Controller Firmware (.ino)
  =============================================================================
  Components Integrated:
  - ESP32 Microcontroller (DevKit V1)
  - Green LED (GPIO 2)  -> Safe Indicator
  - Red LED (GPIO 4)    -> Danger Warning Indicator
  - HC-SR04 Ultrasonic Sensor (Trig: GPIO 5, Echo: GPIO 18) -> Rangefinder
  =============================================================================
*/

#include <Arduino.h>

// Pin Definitions for ESP32
#define GREEN_LED_PIN  2
#define RED_LED_PIN    4
#define TRIG_PIN       5
#define ECHO_PIN       18

// Ultrasonic Safety Threshold (in centimeters)
#define ULTRASONIC_DANGER_THRESHOLD_CM 30.0

// System State Variables
bool aiDangerState = false;           // State received from Python AI software
unsigned long lastSensorReadTime = 0;
const unsigned long SENSOR_INTERVAL_MS = 200; // Read sensor every 200ms

// Function declarations
float readHCSR04Distance();
void updateOutputs(bool dangerActive);
void processSerialCommand(String cmd);

void setup() {
  // Initialize Serial Communication with Host Computer
  Serial.begin(115200);

  // Configure Pin Modes
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Initial Pin State: SAFE (Green LED ON, Red LED OFF, Trig LOW)
  digitalWrite(GREEN_LED_PIN, HIGH);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(TRIG_PIN, LOW);

  Serial.println("SYSTEM_READY: ESP32 Hardware Initialized.");
}

void loop() {
  // 1. Process Incoming Serial Commands from Python AI Software
  while (Serial.available() > 0) {
    String incomingCmd = Serial.readStringUntil('\n');
    incomingCmd.trim();
    if (incomingCmd.length() > 0) {
      processSerialCommand(incomingCmd);
    }
  }

  // 2. Periodically Read HC-SR04 Ultrasonic Distance & Telemetry
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorReadTime >= SENSOR_INTERVAL_MS) {
    lastSensorReadTime = currentMillis;

    float distanceCm = readHCSR04Distance();

    // HC-SR04 Danger Logic:
    // distance > 0 && distance <= threshold -> ultrasonic danger
    // distance == -1 -> invalid sensor reading, NEVER treat -1 as danger!
    bool ultrasonicDanger = (distanceCm > 0.0 && distanceCm <= ULTRASONIC_DANGER_THRESHOLD_CM);

    // Master Danger State: AI Danger OR Valid Ultrasonic Danger
    bool masterDanger = aiDangerState || ultrasonicDanger;

    // Drive Physical LEDs
    updateOutputs(masterDanger);

    // 3. Serial Monitor Output Formatting
    // Format required by user:
    // Distance: 24.9 cm | AI: DANGER | LED: RED
    // Distance: 100.0 cm | AI: SAFE | LED: GREEN
    // Distance: INVALID | Sensor: ERROR | AI: SAFE | LED: GREEN
    String aiLabel = aiDangerState ? "DANGER" : "SAFE";
    String ledLabel = masterDanger ? "RED" : "GREEN";

    if (distanceCm < 0.0) {
      // Sensor Error / Out of range
      Serial.print("Distance: INVALID | Sensor: ERROR | AI: ");
      Serial.print(aiLabel);
      Serial.print(" | LED: ");
      Serial.print(ledLabel);
      Serial.print(" | JSON:{\"dist_cm\":-1.0,\"danger\":");
      Serial.print(masterDanger ? "true" : "false");
      Serial.println("}");
    } else {
      // Valid Distance Measurement
      Serial.print("Distance: ");
      Serial.print(distanceCm, 1);
      Serial.print(" cm | AI: ");
      Serial.print(aiLabel);
      Serial.print(" | LED: ");
      Serial.print(ledLabel);
      Serial.print(" | JSON:{\"dist_cm\":");
      Serial.print(distanceCm, 1);
      Serial.print(",\"danger\":");
      Serial.print(masterDanger ? "true" : "false");
      Serial.println("}");
    }
  }
}

/**
 * Reads actual distance in centimeters from the HC-SR04 Ultrasonic Sensor.
 * Returns -1.0 if pulseIn times out or reading is invalid.
 */
float readHCSR04Distance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(4);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read pulse duration (timeout at 40,000us = ~6.8 meters)
  long duration = pulseIn(ECHO_PIN, HIGH, 40000);

  if (duration <= 0) {
    return -1.0; // Invalid reading / sensor timeout
  }

  // Speed of sound calculation (0.0343 cm/us / 2)
  float distanceCm = (duration * 0.0343) / 2.0;

  if (distanceCm < 2.0 || distanceCm > 400.0) {
    return -1.0; // Out of reliable operational bounds
  }

  return distanceCm;
}

/**
 * Updates physical Green LED and Red LED states based on master danger state.
 */
void updateOutputs(bool dangerActive) {
  if (dangerActive) {
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
  } else {
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(RED_LED_PIN, LOW);
  }
}

/**
 * Parses explicit serial commands from Python AI software.
 */
void processSerialCommand(String cmd) {
  cmd.toUpperCase();
  if (cmd == "DANGER" || cmd == "ALERT") {
    aiDangerState = true;
  } else if (cmd == "SAFE" || cmd == "CLEAR") {
    aiDangerState = false;
  }
}
