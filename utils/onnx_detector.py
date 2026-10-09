import os
import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None

class ONNXPedestrianDetector:
    """
    Lightweight ONNX Runtime inference engine for YOLOv8n.
    Designed for serverless environments (Vercel, AWS Lambda) without PyTorch dependency.
    Total package footprint: < 60 MB.
    """
    def __init__(self, model_path: str = "yolov8n.onnx", conf_threshold: float = 0.50):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.session = None

        if ort is not None and os.path.exists(model_path):
            try:
                # Initialize ONNX Runtime session
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                print(f"[ONNXPedestrianDetector] Successfully initialized ONNX model: {model_path}")
            except Exception as e:
                print(f"[ONNXPedestrianDetector Error] Failed to load ONNX session: {e}")

    def set_confidence(self, threshold: float):
        self.conf_threshold = max(0.10, min(0.95, threshold))

    def preprocess(self, img: np.ndarray, target_size=(640, 640)):
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        # Transpose HWC -> CHW and add batch dimension (1, 3, 640, 640)
        tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
        return tensor, w, h

    def detect(self, frame: np.ndarray, danger_zone=None):
        if frame is None or self.session is None:
            return [], False

        tensor, orig_w, orig_h = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})
        output = outputs[0]  # Shape: (1, 84, 8400)

        # Output format: 84 rows -> 4 bbox coords (cx, cy, w, h), 80 class scores
        # Extract predictions for Class 0 ('person')
        predictions = np.squeeze(output)  # Shape: (84, 8400)
        boxes_cxcywh = predictions[0:4, :]  # (4, 8400)
        person_scores = predictions[4, :]   # (8400,)

        # Filter by confidence threshold
        mask = person_scores >= self.conf_threshold
        if not np.any(mask):
            return [], False

        filtered_boxes = boxes_cxcywh[:, mask].T  # (N, 4)
        filtered_scores = person_scores[mask]      # (N,)

        # Convert (cx, cy, w, h) in 640x640 space to (x1, y1, x2, y2) in original frame dimensions
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        boxes_xywh = []
        scores_list = []

        for box, score in zip(filtered_boxes, filtered_scores):
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            bw = int(w * scale_x)
            bh = int(h * scale_y)
            boxes_xywh.append([x1, y1, bw, bh])
            scores_list.append(float(score))

        # Perform Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_list, self.conf_threshold, 0.45)

        detections = []
        is_any_in_danger = False

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_xywh[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                conf = scores_list[i]

                target_point = ((x1 + x2) // 2, y2)
                in_danger = False

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
