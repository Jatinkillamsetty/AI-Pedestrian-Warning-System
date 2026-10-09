"""
OpenCV Drawing helper utilities for visual overlays, bounding boxes,
bottom-center point markers, and safety status HUD banners.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from config import COLOR_SAFE_BGR, COLOR_DANGER_BGR, COLOR_TEXT_BGR, COLOR_BG_DARK_BGR

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draws bounding boxes and bottom-center contact points for detected pedestrians.
    Highlighting red if person is in danger zone, green if safe.
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        is_in_danger = det["is_in_danger"]
        person_x, person_y = det["bottom_center"]

        # Select color scheme based on danger status
        color = COLOR_DANGER_BGR if is_in_danger else COLOR_SAFE_BGR
        thickness = 3 if is_in_danger else 2

        # 1. Bounding Box Corner Reticle or Rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # 2. Top Label (e.g. "PERSON 95%" or "⚠ DANGER 95%")
        label_prefix = "⚠ DANGER: PERSON" if is_in_danger else "PERSON"
        label_text = f"{label_prefix} {int(conf * 100)}%"

        # Draw label background box
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_bg_y1 = max(0, y1 - text_h - 10)
        label_bg_y2 = max(text_h + 10, y1)

        cv2.rectangle(
            output,
            (x1, label_bg_y1),
            (x1 + text_w + 12, label_bg_y2),
            color,
            -1
        )
        cv2.putText(
            output,
            label_text,
            (x1 + 6, label_bg_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT_BGR,
            2,
            cv2.LINE_AA
        )

        # 3. Bottom-Center Ground Contact Marker Point
        # Outer pulsing circle ring + center filled dot
        dot_radius = 8 if is_in_danger else 6
        cv2.circle(output, (person_x, person_y), dot_radius + 4, color, 2, cv2.LINE_AA)
        cv2.circle(output, (person_x, person_y), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(output, (person_x, person_y), 2, COLOR_TEXT_BGR, -1, cv2.LINE_AA)

        # Optional vertical dashed line connecting center of box to bottom-center
        center_x = (x1 + x2) // 2
        cv2.line(output, (center_x, (y1 + y2) // 2), (person_x, person_y), color, 1, cv2.LINE_AA)

    return output


def draw_hud_banner(
    frame: np.ndarray,
    is_danger: bool,
    people_count: int,
    danger_count: int,
    fps: float
) -> np.ndarray:
    """
    Renders top status banner HUD over the frame.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # Banner height
    banner_h = 50
    overlay = output.copy()

    # Background color: Red tint if danger, dark slate if safe
    bg_color = (20, 20, 180) if is_danger else (30, 41, 59)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)

    # Blend banner transparency
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

    # Status Badge Text
    if is_danger:
        status_str = "⚠ PEDESTRIAN IN DANGER ZONE!"
        status_color = (50, 50, 255)
    else:
        status_str = "✓ AREA SAFE"
        status_color = (50, 220, 100)

    # Left: Status badge
    cv2.putText(
        output,
        status_str,
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA
    )

    # Right: Telemetry metrics (People, Danger count, FPS)
    telemetry_str = f"People: {people_count} | In Zone: {danger_count} | FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.putText(
        output,
        telemetry_str,
        (max(15, w - tw - 15), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 225, 230),
        1,
        cv2.LINE_AA
    )

    return output
