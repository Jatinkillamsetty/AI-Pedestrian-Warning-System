"""
Sound Generator Utility
Pre-generates TTS audio warning files and fallback WAV sound effects
so the system can play audio alerts offline without network dependency.
"""

import sys
import math
import struct
import wave
from pathlib import Path

# Add parent directory to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import WARNING_SOUND_PATH, WARNING_WAV_PATH, SOUNDS_DIR, DEFAULT_WARNING_TEXT


def generate_beep_wav(output_path: Path, freq1=880.0, freq2=1760.0, duration=0.8, sample_rate=44100):
    """
    Generates a dual-tone urgent warning beep WAV file using pure math wave synthesis.
    Works 100% offline without external audio files or network.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            t = i / sample_rate
            # Pulsing alarm pattern: switch frequency every 0.15 seconds
            freq = freq2 if (int(t / 0.15) % 2 == 0) else freq1
            # Sine wave sample with slight fade-out envelope
            sample_val = math.sin(2 * math.pi * freq * t)
            # Amplitude modulation for siren effect
            amplitude = 32767 * 0.8
            packed_sample = struct.pack('<h', int(sample_val * amplitude))
            wav_file.writeframesraw(packed_sample)
            
    print(f"✅ Created synthetic warning audio beep: {output_path}")


def generate_tts_mp3(output_path: Path, text: str = DEFAULT_WARNING_TEXT):
    """
    Tries to generate spoken TTS mp3 via gTTS if internet is available.
    """
    try:
        from gtts import gTTS
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(output_path))
        print(f"✅ Generated spoken TTS warning audio: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ gTTS online TTS generation skipped/unavailable ({e}). Using offline WAV sound.")
        return False


def ensure_audio_assets():
    """Ensures audio files exist in assets/sounds/."""
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Always generate offline WAV beep
    if not WARNING_WAV_PATH.exists():
        generate_beep_wav(WARNING_WAV_PATH)
        
    # Attempt spoken MP3 via gTTS
    if not WARNING_SOUND_PATH.exists():
        success = generate_tts_mp3(WARNING_SOUND_PATH)
        if not success and WARNING_WAV_PATH.exists():
            # Copy or point MP3 path fallback
            print(f"ℹ️ Audio alert ready via {WARNING_WAV_PATH}")


if __name__ == "__main__":
    ensure_audio_assets()
