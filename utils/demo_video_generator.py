"""
Demo Video Frame Generator
Creates a synthetic frame stream with walking pedestrian silhouettes
entering and leaving a blind zone. Useful for testing and demonstration
on systems without a physical USB webcam attached.
"""

import math
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class DemoVideoGenerator:
    """Generates synthetic video frames containing animated pedestrian figures."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0

    def get_frame(self) -> np.ndarray:
        """Renders and returns a synthetic frame with animated pedestrians."""
        # Create dark road background with lane markings
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 45)  # Asphalt dark grey

        # Draw road lane dashed lines
        for y in range(0, self.height, 40):
            cv2.line(frame, (self.width // 2, y), (self.width // 2, y + 20), (180, 180, 180), 2)
            
        # Draw vehicle hood / dash perspective at bottom
        hood_pts = np.array([
            [0, self.height],
            [self.width // 4, self.height - 40],
            [3 * self.width // 4, self.height - 40],
            [self.width, self.height]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [hood_pts], (20, 25, 30))
        cv2.polylines(frame, [hood_pts], True, (70, 80, 90), 2)
        
        cv2.putText(frame, "HEAVY VEHICLE BLIND SPOT DEMO FEED", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Animate Pedestrian 1: Walking horizontally across the lower danger zone
        # Cycle duration: ~10 seconds (300 frames)
        cycle_t = (self.frame_count % 300) / 300.0
        
        # Pedestrian 1 position (enters danger zone around middle of cycle)
        p1_x = int(50 + cycle_t * (self.width - 100))
        p1_y = int(360 + math.sin(cycle_t * math.pi * 4) * 15)  # slight bobbing
        
        self._draw_pedestrian_figure(frame, p1_x, p1_y, scale=1.0, label="Pedestrian 1")

        # Animate Pedestrian 2: Walking safely in top background area
        cycle2_t = ((self.frame_count + 150) % 360) / 360.0
        p2_x = int(self.width - 80 - cycle2_t * (self.width - 160))
        p2_y = int(140 + math.cos(cycle2_t * math.pi * 2) * 10)
        self._draw_pedestrian_figure(frame, p2_x, p2_y, scale=0.6, label="Pedestrian 2")

        self.frame_count += 1
        return frame

    def _draw_pedestrian_figure(self, frame: np.ndarray, x: int, y: int, scale: float = 1.0, label: str = ""):
        """Draws a simple realistic human figure silhouette onto the frame."""
        head_radius = int(12 * scale)
        body_h = int(50 * scale)
        arm_w = int(24 * scale)
        
        head_center = (x, y - body_h - head_radius)
        
        # Color: Natural skin/jacket tones
        color_body = (220, 160, 60)   # Blue jacket
        color_head = (190, 190, 230)  # Head tone
        
        # Head
        cv2.circle(frame, head_center, head_radius, color_head, -1)
        
        # Torso
        torso_top = (x, y - body_h)
        torso_bottom = (x, y - int(body_h * 0.4))
        cv2.line(frame, torso_top, torso_bottom, color_body, int(8 * scale))
        
        # Arms (walking animation)
        arm_swing = math.sin(self.frame_count * 0.2) * 15 * scale
        cv2.line(frame, torso_top, (int(x - arm_w + arm_swing), torso_bottom[1]), color_body, int(4 * scale))
        cv2.line(frame, torso_top, (int(x + arm_w - arm_swing), torso_bottom[1]), color_body, int(4 * scale))
        
        # Legs
        leg_swing = math.sin(self.frame_count * 0.25) * 18 * scale
        cv2.line(frame, torso_bottom, (int(x - leg_swing), y), color_body, int(5 * scale))
        cv2.line(frame, torso_bottom, (int(x + leg_swing), y), color_body, int(5 * scale))
