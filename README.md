# 🚛 AI-Based Pedestrian Detection & Heavy-Vehicle Blind-Zone Warning System

A real-time computer-vision safety application designed for heavy-vehicle (trucks, buses, excavators, forklifts) blind-zone monitoring, integrated with an **ESP32 microcontroller**, **HC-SR04 Ultrasonic Distance Sensor**, **Red & Green LEDs**, and a **Physical Audio Buzzer**.

---

## 🎯 1. Project Objective

Heavy vehicles (trucks, construction equipment, buses) suffer from large driver blind spots where pedestrians are hidden from direct mirror view. This project provides an intelligent computer-vision safety system suitable for a **college project demonstration**:

1. Observes the area around a vehicle via camera.
2. AI detects pedestrians in real time using YOLOv8.
3. Renders a semi-transparent **Danger/Blind Zone** overlay on the camera view.
4. Highlights pedestrians entering the zone in RED with a **DANGER** badge.
5. Plays a spoken/siren audio warning through computer speakers.
6. **ESP32 Hardware Integration**:
   - 🟢 **Green LED** lit during **SAFE** state.
   - 🔴 **Red LED** lit during **DANGER** warning state.
   - 🔊 **Physical Buzzer** sounds alarm on hardware when danger is detected.
   - 📡 **HC-SR04 Ultrasonic Sensor** measures physical distance in centimeters.
7. Implements an **Alert Manager with debouncing and cooldown** to prevent audio spam.
8. Displays a real-time monitoring dashboard with system status, total people, distance telemetry, and FPS.

---

## 🔌 2. ESP32 Hardware Wiring Schematic & Pinout

### Component Pin Mapping (ESP32 DevKit V1)
| Component | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **Green LED** | `GPIO 2` | Lit when Area is SAFE (with 220Ω resistor) |
| **Red LED** | `GPIO 4` | Lit when Pedestrian in DANGER Zone (with 220Ω resistor) |
| **Active/Passive Buzzer**| `GPIO 15` | Sounds physical alarm during DANGER state |
| **HC-SR04 Trig Pin** | `GPIO 5` | Ultrasonic Trigger Pulse Output |
| **HC-SR04 Echo Pin** | `GPIO 18` | Ultrasonic Echo Input Pulse |
| **VCC (HC-SR04)** | `5V / VIN` | Sensor Power Supply |
| **GND** | `GND` | Common Ground |

```
                       ┌─────────────────────────┐
                       │      HOST COMPUTER      │
                       │   (Python AI + YOLO)    │
                       └────────────┬────────────┘
                                    │
                             USB Serial Cable
                                    │
                       ┌────────────▼────────────┐
                       │    ESP32 MICROCONTROLLER│
                       └─┬──────┬───────┬──────┬─┘
                         │      │       │      │
            ┌────────────┴─┐ ┌──┴──┐ ┌──┴──┐ ┌─┴────────────┐
            │   HC-SR04    │ │RED  │ │GREEN│ │  PHYSICAL    │
            │  ULTRASONIC  │ │LED  │ │LED  │ │   BUZZER     │
            │  (Distance)  │ │(G4) │ │(G2) │ │   (G15)      │
            └──────────────┘ └─────┘ └─────┘ └──────────────┘
```

---

## 🏗 3. System Architecture & Processing Pipeline

```
          USB WEBCAM / DEMO STREAM
                     │
                     ▼
             LIVE VIDEO FRAME
                     │
                     ▼
          YOLO PERSON DETECTION (YOLOv8n)
                     │
                     ▼
            DANGER-ZONE CHECK & HC-SR04 ULTRASONIC TELEMETRY
                     │
         ┌───────────┴───────────┐
         │                       │
     [SAFE]                  [DANGER]
         │                       │
         ▼                       ▼
    - Green LED ON          - Red LED ON
    - Red LED OFF           - Green LED OFF
    - Buzzer SILENT         - Buzzer ALARM
    - No Speaker Sound      - Speaker Audio Alert
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
           STREAMLIT LIVE DASHBOARD
                     │
                     ▼
                 Next Frame
                     ↺
```

---

## 💻 4. Installation & Setup

### Step 1: Navigate to Project Directory
```bash
cd /home/ubuntu/.gemini/antigravity/scratch/pedestrian_warning_system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Upload Firmware to ESP32 (Arduino IDE)
1. Connect your ESP32 board to your computer via USB cable.
2. Open Arduino IDE.
3. Open the sketch file:
   [`hardware/esp32_blindzone_warning/esp32_blindzone_warning.ino`](file:///home/ubuntu/.gemini/antigravity/scratch/pedestrian_warning_system/hardware/esp32_blindzone_warning/esp32_blindzone_warning.ino)
4. Select Board: **ESP32 Dev Module** (or your specific ESP32 board).
5. Select Port: `/dev/ttyUSB0` (Linux) or `COM3` (Windows).
6. Click **Upload**.

---

## 🚀 5. How to Run the Application

Launch the Streamlit dashboard using:

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`.

### Hardware Connection Steps:
1. In the sidebar under **🔌 ESP32 Hardware Interface**:
   - Select your serial port (e.g. `/dev/ttyUSB0` or `COM3`) or pick `SIMULATOR (Emulated Hardware)` to test without physical hardware.
   - Click **🔌 Connect ESP32**.
2. Click **▶ Start Camera** to begin live camera monitoring.

---

## 📂 6. Project Structure

```
pedestrian_warning_system/
├── app.py                   # Streamlit UI & main frame execution loop
├── camera.py                # Camera capture & synthetic demo stream manager
├── detector.py              # YOLO person detector & PyTorch GPU/CPU manager
├── danger_zone.py           # Polygonal blind-zone logic & semi-transparent drawing
├── alert_manager.py         # Audio debouncer, cooldown & Pygame sound player
├── esp32_interface.py       # USB Serial bridge & hardware simulator manager
├── config.py                # System settings, hardware pins, and color palettes
├── requirements.txt         # Dependencies list (includes pyserial)
├── README.md                # Documentation & guide
├── models/                  # Directory storing YOLO model weights
├── assets/
│   └── sounds/              # Audio alert files (speech & siren beeps)
├── utils/
│   ├── drawing.py           # Bounding box & HUD overlay renderer
│   └── fps.py               # Rolling FPS counter utility
└── hardware/
    └── esp32_blindzone_warning/
        └── esp32_blindzone_warning.ino  # Ready-to-upload Arduino C++ sketch for ESP32
```

---

## ⚠️ 7. Safety Disclaimer

> [!CAUTION]
> This application is a **research and college project demonstration prototype**. It is **NOT** a certified automotive safety device and should **NOT** be used as a primary safety system on active commercial vehicles.
