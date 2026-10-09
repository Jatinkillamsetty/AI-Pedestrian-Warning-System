"""
Configuration settings for the AI Pedestrian Detection & Heavy-Vehicle Blind-Zone Warning System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
HARDWARE_DIR = BASE_DIR / "hardware"

# Ensure required directories exist (with read-only filesystem safety for Vercel Serverless)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    HARDWARE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Model Settings
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
PERSON_CLASS_ID = 0  # COCO class index for person

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# Detection & Alert Defaults
DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_ALERT_COOLDOWN = 3.0  # Seconds between audio alert repetitions
DEFAULT_AUDIO_ENABLED = True
DEFAULT_VOICE_TYPE = "Voice Warning"  # Options: "Voice Warning", "Siren Beep"

# ESP32 Hardware Settings
DEFAULT_SERIAL_PORT = "SIMULATOR"
DEFAULT_BAUD_RATE = 115200
ULTRASONIC_DANGER_THRESHOLD_CM = 30.0
ESP32_PIN_MAP = {
    "GREEN_LED": 2,
    "RED_LED": 4,
    "TRIG": 5,
    "ECHO": 18
}

# Danger Zone Presets (Normalized Coordinates 0.0 to 1.0)
ZONE_PRESETS = {
    "Front Blind Spot": [
        [0.15, 0.35],
        [0.85, 0.35],
        [0.99, 1.00],
        [0.01, 1.00]
    ],
    "Side Mirror Blind Spot": [
        [0.45, 0.25],
        [0.99, 0.25],
        [0.99, 1.00],
        [0.55, 1.00]
    ],
    "Rear Danger Zone": [
        [0.05, 0.45],
        [0.95, 0.45],
        [0.95, 1.00],
        [0.05, 1.00]
    ],
    "Custom Rectangular": [
        [0.15, 0.30],
        [0.85, 0.30],
        [0.85, 1.00],
        [0.15, 1.00]
    ]
}

# UI Color Palette (OpenCV BGR Format)
COLOR_SAFE_BGR = (100, 220, 0)       # Green
COLOR_DANGER_BGR = (40, 40, 240)      # Bright Red
COLOR_ZONE_NORMAL_BGR = (255, 170, 0) # Cyan / Amber Blue
COLOR_ZONE_DANGER_BGR = (0, 0, 255)   # Bright Red
COLOR_TEXT_BGR = (255, 255, 255)      # White
COLOR_BG_DARK_BGR = (15, 23, 42)      # Dark Slate

# Audio Sound File Paths
VOICE_ALERT_PATH = SOUNDS_DIR / "warning_speech.mp3"
SIREN_ALERT_PATH = SOUNDS_DIR / "alert_beep.wav"
