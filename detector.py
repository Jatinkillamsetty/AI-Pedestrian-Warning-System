"""
YOLO Pedestrian Detector Module using PyTorch and Ultralytics YOLOv8.
Automatically selects CUDA GPU if available, fallback to CPU.
Calculates bottom-center contact points and evaluates danger-zone status.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from config import MODEL_PATH, DEFAULT_MODEL_NAME, PERSON_CLASS_ID, DEFAULT_CONF_THRESHOLD

class PedestrianDetector:
    """
    Handles YOLO model loading, person detection, bottom-center calculation,
    and danger-zone checks.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        
        # Determine execution device (GPU vs CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PedestrianDetector] Initializing model on device: {self.device.upper()}")

        # Load YOLO model
        try:
            if MODEL_PATH.exists():
                self.model = YOLO(str(MODEL_PATH))
            else:
                self.model = YOLO(model_name)
                self.model.save(str(MODEL_PATH))
        except Exception as e:
            print(f"[PedestrianDetector Error] Failed loading model {model_name}: {e}")
            self.model = YOLO("yolov8n.pt")

    def set_confidence(self, threshold: float):
        """Updates confidence threshold dynamically."""
        self.conf_threshold = max(0.10, min(0.95, threshold))

    def detect(self, frame: np.ndarray, danger_zone=None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Runs person detection on frame.
        
        Returns:
            Tuple[detections_list, is_any_in_danger]:
            - detections_list: list of dicts with bbox, conf, bottom_center, is_in_danger
            - is_any_in_danger: True if at least one detected person is inside danger zone.
        """
        if frame is None or self.model is None:
            return [], False

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],  # Filter person class only
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        is_any_in_danger = False

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for box in boxes:
                # Bounding box pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Evaluate danger zone status using comprehensive check
                in_danger = False
                target_point = ((x1 + x2) // 2, y2)

                if danger_zone is not None:
                    in_danger, target_point = danger_zone.is_person_in_zone([x1, y1, x2, y2], frame.shape)
                    if in_danger:
                        is_any_in_danger = True

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "bottom_center": target_point,
                    "is_in_danger": in_danger
                })

        return detections, is_any_in_danger
