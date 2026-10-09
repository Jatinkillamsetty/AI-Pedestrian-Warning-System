"""
ESP32 Serial & Hardware Emulation Interface Module.
Handles PySerial communication with ESP32 board driving Red/Green LEDs
and HC-SR04 Ultrasonic Distance Sensor.
"""

import time
import json
import re
import threading
from typing import List, Dict, Any, Optional

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

class ESP32Interface:
    """
    Manages USB Serial connection to ESP32 board with built-in hardware simulation mode.
    """
    def __init__(self, default_port: str = "AUTO", baud_rate: int = 115200):
        self.default_port = default_port
        self.baud_rate = baud_rate
        
        self.ser: Optional[Any] = None
        self.is_connected = False
        self.use_simulation = False
        
        self.latest_distance_cm: float = -1.0
        self.hardware_danger_state: bool = False
        self.last_state_sent: Optional[bool] = None

        # Lock for thread-safe serial operations
        self._lock = threading.Lock()

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns list of detected serial ports on the host computer."""
        ports_list = ["SIMULATOR (Emulated Hardware)"]
        if PYSERIAL_AVAILABLE:
            try:
                available = serial.tools.list_ports.comports()
                for p in available:
                    ports_list.append(p.device)
            except Exception as e:
                print(f"[ESP32Interface] Error listing serial ports: {e}")
        return ports_list

    def connect(self, port_name: str, baud_rate: int = 115200) -> bool:
        """Connects to specified serial port or enables hardware simulation mode."""
        self.disconnect()
        self.baud_rate = baud_rate

        if "SIMULATOR" in port_name.upper():
            self.use_simulation = True
            self.is_connected = True
            self.hardware_danger_state = False
            self.last_state_sent = False
            print("[ESP32Interface] Hardware Simulation Mode Enabled.")
            return True

        if not PYSERIAL_AVAILABLE:
            print("[ESP32Interface Warning] pyserial is not installed! Defaulting to simulator mode.")
            self.use_simulation = True
            self.is_connected = True
            self.hardware_danger_state = False
            self.last_state_sent = False
            return True

        try:
            target_port = port_name
            if target_port == "AUTO":
                available = serial.tools.list_ports.comports()
                if len(available) > 0:
                    target_port = available[0].device
                else:
                    print("[ESP32Interface Warning] No physical serial port found. Falling back to SIMULATOR.")
                    self.use_simulation = True
                    self.is_connected = True
                    self.hardware_danger_state = False
                    self.last_state_sent = False
                    return True

            self.ser = serial.Serial(target_port, self.baud_rate, timeout=0.1)
            time.sleep(1.5)  # Wait for ESP32 serial reset
            self.is_connected = True
            self.use_simulation = False
            self.hardware_danger_state = False
            self.last_state_sent = False
            print(f"[ESP32Interface] Connected to ESP32 on port {target_port} at {baud_rate} baud.")
            return True

        except Exception as err:
            print(f"[ESP32Interface Error] Could not connect to serial port {port_name}: {err}")
            self.use_simulation = True
            self.is_connected = True
            self.hardware_danger_state = False
            self.last_state_sent = False
            return False

    def disconnect(self):
        """Closes serial connection."""
        with self._lock:
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            self.is_connected = False
            self.use_simulation = False
            self.hardware_danger_state = False
            self.last_state_sent = None

    def send_alert_state(self, is_danger: bool):
        """
        Sends explicit state command to ESP32 over serial.
        is_danger = True  -> Sends 'DANGER\n' (Red LED ON, Green LED OFF)
        is_danger = False -> Sends 'SAFE\n'   (Green LED ON, Red LED OFF)
        """
        self.hardware_danger_state = is_danger

        if self.last_state_sent == is_danger:
            return

        self.last_state_sent = is_danger

        if self.use_simulation:
            return

        if self.is_connected and self.ser is not None and self.ser.is_open:
            with self._lock:
                try:
                    cmd_str = "DANGER\n" if is_danger else "SAFE\n"
                    self.ser.write(cmd_str.encode('utf-8'))
                    self.ser.flush()
                except Exception as e:
                    print(f"[ESP32Interface Error] Serial write failed: {e}")

    def read_telemetry(self) -> Dict[str, Any]:
        """
        Reads actual HC-SR04 ultrasonic distance sensor telemetry from ESP32 over Serial.
        Parses JSON data: JSON:{"dist_cm": 24.9, "danger": true}
        """
        if self.use_simulation:
            import random
            if self.hardware_danger_state:
                self.latest_distance_cm = round(random.uniform(18.0, 28.0), 1)
            else:
                self.latest_distance_cm = round(random.uniform(140.0, 220.0), 1)

            is_sensor_danger = (self.latest_distance_cm > 0.0 and self.latest_distance_cm <= 30.0)

            return {
                "distance_cm": self.latest_distance_cm,
                "is_danger": is_sensor_danger,
                "connected": True,
                "mode": "SIMULATOR",
                "green_led": not self.hardware_danger_state,
                "red_led": self.hardware_danger_state
            }

        if self.is_connected and self.ser is not None and self.ser.is_open:
            with self._lock:
                try:
                    while self.ser.in_waiting > 0:
                        raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        json_match = re.search(r'\{.*\}', raw_line)
                        if json_match:
                            json_str = json_match.group(0)
                            data = json.loads(json_str)
                            if "dist_cm" in data:
                                self.latest_distance_cm = float(data["dist_cm"])
                except Exception as err:
                    print(f"[ESP32Interface Warning] Telemetry read error: {err}")

        is_sensor_danger = (self.latest_distance_cm > 0.0 and self.latest_distance_cm <= 30.0)

        return {
            "distance_cm": self.latest_distance_cm,
            "is_danger": is_sensor_danger,
            "connected": self.is_connected,
            "mode": "PHYSICAL" if not self.use_simulation else "SIMULATOR",
            "green_led": not self.hardware_danger_state,
            "red_led": self.hardware_danger_state
        }
