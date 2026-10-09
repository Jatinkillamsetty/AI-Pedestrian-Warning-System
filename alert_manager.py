"""
Alert Manager for handling audio notifications, debouncing frame alerts,
cooldown timers, Pygame Sound playback, and ESP32 Hardware signals.
"""

import time
import os
import threading
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import pygame

from config import (
    SOUNDS_DIR,
    VOICE_ALERT_PATH,
    SIREN_ALERT_PATH,
    DEFAULT_ALERT_COOLDOWN,
    DEFAULT_AUDIO_ENABLED,
    DEFAULT_VOICE_TYPE
)

class AlertManager:
    """
    Manages non-blocking audio alerts and ESP32 hardware signals with debouncing.
    """
    def __init__(
        self,
        cooldown_seconds: float = DEFAULT_ALERT_COOLDOWN,
        audio_enabled: bool = DEFAULT_AUDIO_ENABLED,
        voice_type: str = DEFAULT_VOICE_TYPE
    ):
        self.cooldown_seconds = cooldown_seconds
        self.audio_enabled = audio_enabled
        self.voice_type = voice_type
        
        self.last_alert_time = 0.0
        self.in_danger_state = False
        self._mixer_initialized = False

        # Ensure audio assets exist
        self._ensure_sound_assets()

        # Initialize Pygame Mixer
        self._init_mixer()

    def _init_mixer(self):
        """Safely initializes pygame mixer for audio playback."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._mixer_initialized = True
        except Exception as e:
            print(f"[AlertManager Warning] Audio mixer init error: {e}")
            self._mixer_initialized = False

    def _ensure_sound_assets(self):
        """Generates default audio warning files if they don't exist."""
        SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Generate Synthetic Siren Beep (.wav)
        if not SIREN_ALERT_PATH.exists():
            try:
                sample_rate = 44100
                duration = 1.0  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = np.sin(2 * np.pi * 880 * t) * (np.sin(2 * np.pi * 5 * t) > 0)
                audio = (0.6 * tone * 32767).astype(np.int16)
                wav.write(str(SIREN_ALERT_PATH), sample_rate, audio)
            except Exception as e:
                print(f"[AlertManager] Error creating siren beep: {e}")

        # 2. Generate Voice Warning MP3 via gTTS if internet is available
        if not VOICE_ALERT_PATH.exists():
            try:
                from gtts import gTTS
                tts = gTTS(text="Warning! Pedestrian detected in danger zone.", lang="en")
                tts.save(str(VOICE_ALERT_PATH))
            except Exception as e:
                print(f"[AlertManager] gTTS unavailable or offline ({e}). Using siren fallback.")

    def play_sound(self):
        """Plays selected warning sound asynchronously in background thread."""
        if not self.audio_enabled or not self._mixer_initialized:
            return

        def _play_thread():
            try:
                target_path = VOICE_ALERT_PATH if (self.voice_type == "Voice Warning" and VOICE_ALERT_PATH.exists()) else SIREN_ALERT_PATH
                if target_path.exists():
                    sound = pygame.mixer.Sound(str(target_path))
                    sound.play()
            except Exception as err:
                print(f"[AlertManager] Playback error: {err}")

        threading.Thread(target=_play_thread, daemon=True).start()

    def update(self, is_in_danger: bool, esp32_interface=None) -> bool:
        """
        Updates danger state, triggers computer speaker audio, and sends ESP32 hardware signals.
        """
        now = time.time()
        triggered = False

        # 1. Send hardware signal to ESP32 (Red LED, Green LED, Buzzer)
        if esp32_interface is not None:
            esp32_interface.send_alert_state(is_in_danger)

        # 2. Process audio cooldown and speaker warning
        if is_in_danger:
            if not self.in_danger_state:
                # Transition: Entered danger zone
                self.in_danger_state = True
                self.last_alert_time = now
                self.play_sound()
                triggered = True
            else:
                # Continuous danger state: check cooldown
                if (now - self.last_alert_time) >= self.cooldown_seconds:
                    self.last_alert_time = now
                    self.play_sound()
                    triggered = True
        else:
            # Zone cleared -> reset danger state immediately
            self.in_danger_state = False

        return triggered
