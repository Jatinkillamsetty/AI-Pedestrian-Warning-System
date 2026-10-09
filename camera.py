"""
Camera Manager module for handling USB webcam video capture, resolution tracking,
and synthetic demo video stream fallback.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional

from config import DEFAULT_CAMERA_INDEX, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT

class CameraManager:
    """
    Manages live USB webcam input, camera status, resolution, and synthetic video fallback.
    """
    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.use_demo_mode = False
        
        # Synthetic demo animation variables
        self._anim_t = 0.0

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Starts video capture from specified webcam index."""
        if camera_index is not None:
            self.camera_index = camera_index

        self.stop()
        self.cap = cv2.VideoCapture(self.camera_index)

        # Check if camera opened successfully
        if self.cap is not None and self.cap.isOpened():
            # Try setting standard resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            
            # Read test frame
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_connected = True
                self.use_demo_mode = False
                print(f"[CameraManager] Webcam {self.camera_index} connected successfully.")
                return True

        # Camera failed to open
        self.is_connected = False
        print(f"[CameraManager Warning] Could not access webcam at index {self.camera_index}.")
        return False

    def enable_demo_mode(self):
        """Switches stream to synthetic animated demo video generator."""
        self.stop()
        self.use_demo_mode = True
        self.is_connected = True

    def stop(self):
        """Releases the camera capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Reads next frame from webcam or synthetic generator.
        
        Returns:
            Tuple[success, frame_bgr]
        """
        if self.use_demo_mode:
            return True, self._generate_synthetic_demo_frame()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame

        # If live camera fails while running
        self.is_connected = False
        return False, self._generate_error_frame()

    def get_info(self) -> Dict[str, Any]:
        """Returns camera telemetry information dictionary."""
        if self.use_demo_mode:
            return {
                "status": "DEMO MODE",
                "resolution": f"{DEFAULT_FRAME_WIDTH} × {DEFAULT_FRAME_HEIGHT}",
                "width": DEFAULT_FRAME_WIDTH,
                "height": DEFAULT_FRAME_HEIGHT,
                "index": "Synthetic Test Stream"
            }

        if self.is_connected and self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return {
                "status": "CONNECTED",
                "resolution": f"{w} × {h}",
                "width": w,
                "height": h,
                "index": str(self.camera_index)
            }

        return {
            "status": "NOT DETECTED",
            "resolution": "N/A",
            "width": 0,
            "height": 0,
            "index": str(self.camera_index)
        }

    def _generate_synthetic_demo_frame(self) -> np.ndarray:
        """
        Generates a realistic synthetic video frame containing an animated person
        walking back and forth into the heavy-vehicle danger zone for testing.
        """
        w, h = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background: Dark Road Surface
        frame[:] = (35, 38, 45)

        # Draw Perspective Vehicle Hood & Lane Lines
        cv2.line(frame, (0, h), (w // 3, int(h * 0.4)), (80, 85, 95), 2)
        cv2.line(frame, (w, h), (2 * w // 3, int(h * 0.4)), (80, 85, 95), 2)

        # Advance animation time
        self._anim_t += 0.04
        
        # Pedestrian 1 Position (Oscillates horizontally across danger zone)
        # Danger zone is typically x=0.2 to 0.8, y=0.5 to 0.9
        ped1_x = int(w * (0.15 + 0.70 * (0.5 + 0.5 * np.sin(self._anim_t * 0.7))))
        ped1_y = int(h * 0.75)

        # Pedestrian 2 Position (Static / Secondary)
        ped2_x = int(w * 0.85)
        ped2_y = int(h * 0.55)

        # Draw Pedestrian 1 Mannequin (head, torso, legs)
        self._draw_synthetic_person(frame, ped1_x, ped1_y, height=140, color=(200, 200, 200))

        # Draw Pedestrian 2 Mannequin
        self._draw_synthetic_person(frame, ped2_x, ped2_y, height=100, color=(170, 170, 170))

        return frame

    def _draw_synthetic_person(self, img: np.ndarray, px: int, py: int, height: int, color: Tuple[int, int, int]):
        """Renders a simple synthetic human figure (head, body, limbs) on image."""
        head_radius = height // 7
        head_y = py - height + head_radius
        torso_top = head_y + head_radius
        torso_bottom = py - int(height * 0.35)

        # Head
        cv2.circle(img, (px, head_y), head_radius, color, -1, cv2.LINE_AA)
        # Torso
        cv2.line(img, (px, torso_top), (px, torso_bottom), color, 4, cv2.LINE_AA)
        # Arms
        cv2.line(img, (px - head_radius * 2, torso_top + head_radius), (px + head_radius * 2, torso_top + head_radius), color, 3, cv2.LINE_AA)
        # Legs
        cv2.line(img, (px, torso_bottom), (px - head_radius * 2, py), color, 4, cv2.LINE_AA)
        cv2.line(img, (px, torso_bottom), (px + head_radius * 2, py), color, 4, cv2.LINE_AA)

    def _generate_error_frame(self) -> np.ndarray:
        """Returns error frame displayed when webcam is unavailable."""
        w, h = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 20, 35)

        msg = "⚠ CAMERA NOT DETECTED"
        submsg = "Please connect USB webcam or click 'Enable Demo Mode'"

        cv2.putText(frame, msg, (w // 2 - 180, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, submsg, (w // 2 - 220, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame
