"""
Danger / Blind-Zone Polygon module for Heavy-Vehicle safety detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from config import ZONE_PRESETS, COLOR_ZONE_NORMAL_BGR, COLOR_ZONE_DANGER_BGR, COLOR_TEXT_BGR

class DangerZone:
    """
    Manages and renders the configurable heavy-vehicle blind-zone polygon.
    """
    def __init__(self, preset_name: str = "Front Blind Spot"):
        self.preset_name = preset_name
        self.normalized_points: List[List[float]] = list(
            ZONE_PRESETS.get(preset_name, ZONE_PRESETS["Front Blind Spot"])
        )

    def set_preset(self, preset_name: str):
        """Updates zone coordinates to a preset configuration."""
        if preset_name in ZONE_PRESETS:
            self.preset_name = preset_name
            self.normalized_points = list(ZONE_PRESETS[preset_name])

    def set_custom_rectangle(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Sets a rectangular danger zone using normalized min/max bounds (0.0 to 1.0)."""
        self.preset_name = "Custom Rectangular"
        self.normalized_points = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]

    def get_pixel_polygon(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Converts normalized [0..1] points to absolute pixel coordinates [x, y]."""
        h, w = frame_shape[:2]
        pixel_pts = []
        for pt in self.normalized_points:
            px = int(pt[0] * w)
            py = int(pt[1] * h)
            pixel_pts.append([px, py])
        return np.array(pixel_pts, dtype=np.int32)

    def is_point_in_zone(self, point: Tuple[int, int], frame_shape: Tuple[int, ...]) -> bool:
        """
        Determines whether a ground point (x, y) falls inside the danger polygon.
        Uses OpenCV cv2.pointPolygonTest.
        """
        pts = self.get_pixel_polygon(frame_shape)
        pts_contour = pts.reshape((-1, 1, 2))
        res = cv2.pointPolygonTest(pts_contour, (float(point[0]), float(point[1])), False)
        return res >= 0

    def is_person_in_zone(self, bbox: List[int], frame_shape: Tuple[int, ...]) -> Tuple[bool, Tuple[int, int]]:
        """
        Comprehensive check to determine if a person is inside the danger zone.
        Checks:
        1. Bottom-center point ((x1+x2)//2, y2)
        2. Clamped bottom-center ((x1+x2)//2, h - 10)
        3. Center point ((x1+x2)//2, (y1+y2)//2)
        
        Returns:
            Tuple[is_in_danger, test_point]
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bc_x = cx
        bc_y = y2

        # 1. Direct bottom-center point check
        if self.is_point_in_zone((bc_x, bc_y), frame_shape):
            return True, (bc_x, bc_y)

        # 2. Clamped bottom-center (for when person box reaches bottom edge of screen)
        clamped_bc = (cx, min(y2 - 10, h - 10))
        if self.is_point_in_zone(clamped_bc, frame_shape):
            return True, clamped_bc

        # 3. Center point check
        if self.is_point_in_zone((cx, cy), frame_shape):
            return True, (cx, cy)

        # 4. Quarter points along person's height
        q3_y = int(y1 + 0.75 * (y2 - y1))
        if self.is_point_in_zone((cx, q3_y), frame_shape):
            return True, (cx, q3_y)

        return False, (bc_x, bc_y)

    def draw(self, frame: np.ndarray, is_danger: bool = False) -> np.ndarray:
        """
        Draws semi-transparent danger-zone overlay polygon on the frame.
        Turns RED when is_danger is True!
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        pts = self.get_pixel_polygon(frame.shape)

        # Color: RED if active warning, BLUE/AMBER if normal monitoring
        color = COLOR_ZONE_DANGER_BGR if is_danger else COLOR_ZONE_NORMAL_BGR

        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], color)

        # Blend fill overlay with original frame (40% opacity in danger, 25% normal)
        alpha = 0.40 if is_danger else 0.25
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw thick boundary border
        border_thickness = 4 if is_danger else 2
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=border_thickness, lineType=cv2.LINE_AA)

        # Label inside zone
        label = "⚠ DANGER / BLIND ZONE ALERT!" if is_danger else "BLIND ZONE AREA"
        
        # Find centroid of polygon for label placement
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, int(h * 0.7)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg_box_color = (0, 0, 0)
        text_draw_color = (255, 255, 255) if is_danger else color
        
        cv2.rectangle(output, (cx - tw // 2 - 10, cy - th // 2 - 8), (cx + tw // 2 + 10, cy + th // 2 + 8), bg_box_color, -1)
        if is_danger:
            cv2.rectangle(output, (cx - tw // 2 - 10, cy - th // 2 - 8), (cx + tw // 2 + 10, cy + th // 2 + 8), color, 2)

        cv2.putText(output, label, (cx - tw // 2, cy + th // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_draw_color, 2, cv2.LINE_AA)

        return output
