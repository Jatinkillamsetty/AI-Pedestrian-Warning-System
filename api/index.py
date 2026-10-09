import sys
import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from danger_zone import DangerZone
from utils.drawing import draw_danger_zones, draw_detections, draw_telemetry_hud

# Import ONNX detector first for lightweight Vercel serverless footprint
try:
    from utils.onnx_detector import ONNXPedestrianDetector
    USE_ONNX = True
except Exception:
    from detector import PedestrianDetector
    USE_ONNX = False

app = FastAPI(title="AI Pedestrian Warning System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lazy instances
detector_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None:
        if USE_ONNX:
            onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.onnx")
            detector_instance = ONNXPedestrianDetector(model_path=onnx_path)
        else:
            detector_instance = PedestrianDetector()
    return detector_instance

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "system": "AI Pedestrian & Heavy-Vehicle Blind-Zone Warning System",
        "backend": "ONNX Runtime (Serverless)" if USE_ONNX else "PyTorch",
        "version": "1.0.0"
    }

@app.post("/api/detect")
async def detect_pedestrians(
    file: UploadFile = File(...),
    confidence: float = Form(0.50),
    preset: str = Form("Front Blind Spot")
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Invalid image file uploaded."}, status_code=400)

        dz = DangerZone(preset_name=preset)
        det = get_detector()
        det.set_confidence(confidence)

        polygon = dz.get_pixel_polygon(frame.shape)
        detections, is_danger = det.detect(frame, dz)

        # Draw overlays
        annotated_frame = draw_danger_zones(frame.copy(), polygon, is_danger)
        annotated_frame = draw_detections(annotated_frame, detections)
        annotated_frame = draw_telemetry_hud(annotated_frame, len(detections), is_danger, 0.0, 15.0, "VERCEL_SERVERLESS")

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            "is_danger": is_danger,
            "detection_count": len(detections),
            "detections": detections,
            "preset": preset,
            "confidence_threshold": confidence,
            "annotated_image": f"data:image/jpeg;base64,{jpg_as_text}"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Pedestrian & Blind-Zone Warning System</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0d1117;
                --card-bg: #161b22;
                --border: #30363d;
                --text: #c9d1d9;
                --heading: #f0f6fc;
                --primary: #238636;
                --danger: #da3633;
                --accent: #58a6ff;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            header {
                text-align: center;
                margin-bottom: 32px;
                border-bottom: 1px solid var(--border);
                padding-bottom: 16px;
            }
            h1 { color: var(--heading); font-size: 2rem; margin: 0 0 8px 0; }
            p.sub { color: #8b949e; margin: 0; }
            .grid {
                display: grid;
                grid-template-columns: 320px 1fr;
                gap: 24px;
            }
            .panel {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 20px;
            }
            label { display: block; margin-top: 12px; font-weight: 600; font-size: 0.9rem; color: var(--heading); }
            select, input[type="file"], input[type="range"] {
                width: 100%;
                margin-top: 6px;
                padding: 8px;
                background: #0d1117;
                border: 1px solid var(--border);
                color: var(--text);
                border-radius: 6px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                margin-top: 20px;
                padding: 12px;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                font-size: 1rem;
            }
            button:hover { background: #2ea043; }
            .status-badge {
                padding: 12px;
                border-radius: 6px;
                text-align: center;
                font-weight: 700;
                margin-top: 16px;
                font-size: 1.1rem;
            }
            .status-safe { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }
            .status-danger { background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #da3633; }
            .preview-box {
                min-height: 400px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px dashed var(--border);
                border-radius: 8px;
                overflow: hidden;
                background: #000;
            }
            .preview-box img {
                max-width: 100%;
                max-height: 550px;
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🛡️ Heavy-Vehicle Blind-Zone Pedestrian Detector</h1>
                <p class="sub">Vercel Serverless AI Vision & Geometric Danger-Zone Telemetry</p>
            </header>

            <div class="grid">
                <div class="panel">
                    <h3>Control Panel</h3>
                    <label for="imageInput">Upload Frame Image</label>
                    <input type="file" id="imageInput" accept="image/*">

                    <label for="presetSelect">Blind Zone Preset</label>
                    <select id="presetSelect">
                        <option value="Front Blind Spot">Front Blind Spot (A-Pillar)</option>
                        <option value="Right Side Mirror">Right Side Mirror Zone</option>
                        <option value="Rear Wide Zone">Rear Wide Danger Zone</option>
                    </select>

                    <label for="confRange">Confidence Threshold: <span id="confVal">0.50</span></label>
                    <input type="range" id="confRange" min="0.30" max="0.95" step="0.05" value="0.50" oninput="document.getElementById('confVal').innerText=this.value">

                    <button onclick="runDetection()">Run AI Detection</button>

                    <div id="statusBadge" class="status-badge status-safe" style="display:none;">
                        SAFE - NO BLIND ZONE CONFLICT
                    </div>
                </div>

                <div class="panel">
                    <h3>Detection Output</h3>
                    <div class="preview-box" id="previewContainer">
                        <span style="color:#8b949e;">Upload an image and click "Run AI Detection"</span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function runDetection() {
                const fileInput = document.getElementById('imageInput');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select an image file first.');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('confidence', document.getElementById('confRange').value);
                formData.append('preset', document.getElementById('presetSelect').value);

                const container = document.getElementById('previewContainer');
                container.innerHTML = '<span style="color:#58a6ff;">Processing frame with ONNX YOLOv8...</span>';

                try {
                    const res = await fetch('/api/detect', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();

                    if (data.error) {
                        alert('Error: ' + data.error);
                        container.innerHTML = '<span style="color:#f85149;">Detection failed.</span>';
                        return;
                    }

                    container.innerHTML = `<img src="${data.annotated_image}" alt="Detection Output">`;
                    
                    const badge = document.getElementById('statusBadge');
                    badge.style.display = 'block';
                    if (data.is_danger) {
                        badge.className = 'status-badge status-danger';
                        badge.innerText = '🚨 DANGER - PEDESTRIAN IN BLIND ZONE!';
                    } else {
                        badge.className = 'status-badge status-safe';
                        badge.innerText = '✅ SAFE - NO BLIND ZONE CONFLICT';
                    }
                } catch (err) {
                    alert('Network error: ' + err.message);
                    container.innerHTML = '<span style="color:#f85149;">Connection error.</span>';
                }
            }
        </script>
    </body>
    </html>
    """
