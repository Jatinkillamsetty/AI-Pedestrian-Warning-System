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
from utils.drawing import draw_detections, draw_hud_banner

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
    preset: str = Form("Front Blind Spot"),
    sensor_distance: float = Form(-1.0)
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

        detections, is_ai_danger = det.detect(frame, dz)
        
        # Hardware ultrasonic threshold check (<= 30 cm)
        is_hw_danger = (sensor_distance > 0.0 and sensor_distance <= 30.0)
        is_danger = is_ai_danger or is_hw_danger
        
        danger_count = sum(1 for d in detections if d.get("is_in_danger", False))

        # Draw overlays
        annotated_frame = dz.draw(frame.copy(), is_danger)
        annotated_frame = draw_detections(annotated_frame, detections)
        annotated_frame = draw_hud_banner(annotated_frame, is_danger, len(detections), danger_count, 15.0)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            "is_danger": is_danger,
            "detection_count": len(detections),
            "danger_count": danger_count,
            "detections": detections,
            "preset": preset,
            "confidence_threshold": confidence,
            "sensor_distance": sensor_distance,
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
        <title>Heavy-Vehicle Blind-Zone Safety AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0F172A;
                --card-bg: #1E293B;
                --border: #334155;
                --text: #F8FAFC;
                --subtext: #94A3B8;
                --primary: #3B82F6;
                --safe: #10B981;
                --danger: #EF4444;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .main-header {
                background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
                padding: 20px 24px;
                border-radius: 12px;
                border-left: 6px solid var(--primary);
                margin-bottom: 24px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            }
            .main-title {
                font-size: 24px;
                font-weight: 800;
                color: var(--text);
                margin: 0;
            }
            .sub-title {
                font-size: 13px;
                color: var(--subtext);
                margin-top: 6px;
            }

            /* Metrics Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 12px;
                margin-bottom: 24px;
            }
            .metric-card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 14px 10px;
                text-align: center;
            }
            .metric-label {
                font-size: 10px;
                font-weight: 700;
                color: var(--subtext);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }
            .metric-val {
                font-size: 15px;
                font-weight: 800;
            }
            .val-safe { color: var(--safe); }
            .val-danger { color: var(--danger); }

            /* Layout Grid */
            .layout-grid {
                display: grid;
                grid-template-columns: 320px 1fr;
                gap: 24px;
            }
            .panel {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
            }
            .panel h3 {
                margin: 0 0 16px 0;
                font-size: 16px;
                color: var(--text);
                border-bottom: 1px solid var(--border);
                padding-bottom: 10px;
            }
            label {
                display: block;
                margin-top: 14px;
                font-weight: 600;
                font-size: 0.85rem;
                color: var(--subtext);
            }
            select, input[type="file"], input[type="range"], input[type="number"] {
                width: 100%;
                margin-top: 6px;
                padding: 10px;
                background: #0F172A;
                border: 1px solid var(--border);
                color: var(--text);
                border-radius: 8px;
                box-sizing: border-box;
            }
            button.action-btn {
                width: 100%;
                margin-top: 12px;
                padding: 12px;
                background: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 700;
                cursor: pointer;
                font-size: 0.9rem;
            }
            button.action-btn:hover { background: #1D4ED8; }
            button.stop-btn { background: #DC2626; }
            button.stop-btn:hover { background: #B91C1C; }
            button.secondary-btn { background: #334155; }
            button.secondary-btn:hover { background: #475569; }

            .viewport-box {
                min-height: 480px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px dashed var(--border);
                border-radius: 12px;
                overflow: hidden;
                background: #000;
                position: relative;
            }
            .viewport-box img, .viewport-box video {
                max-width: 100%;
                max-height: 600px;
                display: block;
            }
            video { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="main-header">
                <div class="main-title">🚛 HEAVY-VEHICLE BLIND-ZONE AI SYSTEM</div>
                <div class="sub-title">Live Browser Webcam Stream + Web Serial ESP32 Hardware + Web Speech Audio Warnings</div>
            </div>

            <!-- Top 6 Telemetry Metrics -->
            <div class="metrics-grid">
                <div class="metric-card" id="cardStatus">
                    <div class="metric-label">System Status</div>
                    <div class="metric-val val-safe" id="valStatus">✓ AREA SAFE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">People Detected</div>
                    <div class="metric-val" id="valPeople">0</div>
                </div>
                <div class="metric-card" id="cardDanger">
                    <div class="metric-label">In Danger Zone</div>
                    <div class="metric-val" id="valDanger">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">HC-SR04 Distance</div>
                    <div class="metric-val" id="valDist" style="color:#10B981;">-- cm</div>
                </div>
                <div class="metric-card" id="cardLED">
                    <div class="metric-label">ESP32 Hardware</div>
                    <div class="metric-val val-safe" id="valLED">🟢 GREEN LED</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">System FPS</div>
                    <div class="metric-val" id="valFPS">0.0</div>
                </div>
            </div>

            <div class="layout-grid">
                <div class="panel">
                    <h3>📷 Camera & Sensors</h3>
                    
                    <button class="action-btn" id="webcamBtn" onclick="toggleWebcam()">🎥 Start Laptop Webcam Stream</button>
                    
                    <label for="imageInput">Upload Frame Image</label>
                    <input type="file" id="imageInput" accept="image/*" onchange="uploadSingleFrame()">

                    <label for="presetSelect">🚧 Blind Zone Preset</label>
                    <select id="presetSelect">
                        <option value="Front Blind Spot">Front Blind Spot (A-Pillar)</option>
                        <option value="Side Mirror Blind Spot">Right Side Mirror Zone</option>
                        <option value="Rear Danger Zone">Rear Wide Danger Zone</option>
                    </select>

                    <label for="confRange">🧠 Confidence Threshold: <span id="confVal" style="color:var(--primary);">0.50</span></label>
                    <input type="range" id="confRange" min="0.30" max="0.95" step="0.05" value="0.50" oninput="document.getElementById('confVal').innerText=this.value">

                    <label for="sensorDistInput">📡 HC-SR04 Distance Sensor (cm)</label>
                    <input type="number" id="sensorDistInput" value="45.0" step="1.0" min="2.0" max="400.0">

                    <label><input type="checkbox" id="audioCheck" checked> 🔊 Enable Voice Warnings (Browser Speaker)</label>

                    <button class="action-btn secondary-btn" onclick="connectWebSerial()">🔌 Connect Physical ESP32 (WebSerial)</button>
                    <button class="action-btn secondary-btn" onclick="testAudioVoice()">🔊 Test Voice Alert</button>
                </div>

                <div class="panel">
                    <h3>📹 LIVE CAMERA & AI OVERLAY</h3>
                    <div class="viewport-box" id="previewContainer">
                        <span style="color:var(--subtext);">Click "🎥 Start Laptop Webcam Stream" or upload an image frame</span>
                        <video id="hiddenVideo" autoplay playsinline muted></video>
                        <canvas id="hiddenCanvas" style="display:none;"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let isStreaming = false;
            let streamInterval = null;
            let lastAlertTime = 0;

            function speakVoice(text) {
                if (!document.getElementById('audioCheck').checked) return;
                const now = Date.now();
                if (now - lastAlertTime < 3000) return; // 3s debounce cooldown
                lastAlertTime = now;

                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                    const utter = new SpeechSynthesisUtterance(text);
                    utter.rate = 1.0;
                    utter.pitch = 1.0;
                    window.speechSynthesis.speak(utter);
                }
            }

            function testAudioVoice() {
                speakVoice("WARNING! PEDESTRIAN IN BLIND ZONE!");
            }

            async function toggleWebcam() {
                const btn = document.getElementById('webcamBtn');
                const video = document.getElementById('hiddenVideo');

                if (isStreaming) {
                    // Stop streaming
                    isStreaming = false;
                    clearInterval(streamInterval);
                    if (video.srcObject) {
                        video.srcObject.getTracks().forEach(t => t.stop());
                    }
                    btn.className = 'action-btn';
                    btn.innerText = '🎥 Start Laptop Webcam Stream';
                    document.getElementById('previewContainer').innerHTML = '<span style="color:var(--subtext);">Webcam stopped. Click Start Laptop Webcam Stream.</span><video id="hiddenVideo" autoplay playsinline muted></video><canvas id="hiddenCanvas" style="display:none;"></canvas>';
                    return;
                }

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    video.srcObject = stream;
                    await video.play();

                    isStreaming = true;
                    btn.className = 'action-btn stop-btn';
                    btn.innerText = '■ Stop Laptop Webcam';

                    streamInterval = setInterval(sendWebcamFrame, 200); // 5 FPS live streaming
                } catch (err) {
                    alert('Camera access error: ' + err.message + '. Please allow browser camera permissions.');
                }
            }

            async function sendWebcamFrame() {
                if (!isStreaming) return;
                const video = document.getElementById('hiddenVideo');
                const canvas = document.getElementById('hiddenCanvas');
                canvas.width = 640;
                canvas.height = 480;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, 640, 480);

                canvas.toBlob(async (blob) => {
                    if (!blob) return;
                    await processFrame(blob);
                }, 'image/jpeg', 0.80);
            }

            async function uploadSingleFrame() {
                const fileInput = document.getElementById('imageInput');
                if (fileInput.files && fileInput.files.length > 0) {
                    await processFrame(fileInput.files[0]);
                }
            }

            async function processFrame(fileBlob) {
                const formData = new FormData();
                formData.append('file', fileBlob, 'frame.jpg');
                formData.append('confidence', document.getElementById('confRange').value);
                formData.append('preset', document.getElementById('presetSelect').value);
                formData.append('sensor_distance', document.getElementById('sensorDistInput').value);

                const t0 = performance.now();
                try {
                    const res = await fetch('/api/detect', { method: 'POST', body: formData });
                    const data = await res.json();
                    const t1 = performance.now();

                    if (data.error) return;

                    const container = document.getElementById('previewContainer');
                    container.innerHTML = `<img src="${data.annotated_image}" alt="Detection Output"><video id="hiddenVideo" autoplay playsinline muted></video><canvas id="hiddenCanvas" style="display:none;"></canvas>`;
                    
                    document.getElementById('valPeople').innerText = data.detection_count;
                    document.getElementById('valDanger').innerText = data.danger_count;
                    document.getElementById('valFPS').innerText = (1000 / (t1 - t0)).toFixed(1);

                    const dist = parseFloat(document.getElementById('sensorDistInput').value);
                    const valDist = document.getElementById('valDist');
                    valDist.innerText = dist > 0 ? dist.toFixed(1) + ' cm' : 'INVALID';
                    valDist.style.color = (dist > 0 && dist <= 30) ? 'var(--danger)' : 'var(--safe)';

                    const valStatus = document.getElementById('valStatus');
                    const valLED = document.getElementById('valLED');
                    const cardStatus = document.getElementById('cardStatus');
                    const cardDanger = document.getElementById('cardDanger');

                    if (data.is_danger) {
                        valStatus.className = 'metric-val val-danger';
                        valStatus.innerText = '⚠ WARNING: DANGER';
                        valLED.className = 'metric-val val-danger';
                        valLED.innerText = '🔴 RED LED ACTIVE';
                        cardStatus.style.borderColor = 'var(--danger)';
                        cardDanger.style.borderColor = 'var(--danger)';

                        speakVoice("WARNING! PEDESTRIAN IN BLIND ZONE!");
                    } else {
                        valStatus.className = 'metric-val val-safe';
                        valStatus.innerText = '✓ AREA SAFE';
                        valLED.className = 'metric-val val-safe';
                        valLED.innerText = '🟢 GREEN LED ACTIVE';
                        cardStatus.style.borderColor = 'var(--border)';
                        cardDanger.style.borderColor = 'var(--border)';
                    }
                } catch (err) {
                    console.error('Frame error:', err);
                }
            }

            // WebSerial Physical ESP32 Integration (Chrome / Edge)
            async function connectWebSerial() {
                if (!('serial' in navigator)) {
                    alert('WebSerial is supported in Chrome or Edge browsers. For local serial ports, run local streamlit app.');
                    return;
                }
                try {
                    const port = await navigator.serial.requestPort();
                    await port.open({ baudRate: 115200 });
                    alert('Connected to ESP32 via WebSerial!');
                } catch (err) {
                    alert('Serial Connection Error: ' + err.message);
                }
            }
        </script>
    </body>
    </html>
    """
